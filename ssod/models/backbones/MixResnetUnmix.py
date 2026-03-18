import random
import torch

from mmcv.runner import BaseModule
from mmrotate.models import ROTATED_BACKBONES
from mmdet.models.backbones.resnet import ResNet

@ROTATED_BACKBONES.register_module()
class MixResNetUnmix(BaseModule):
    def __init__(self, nt=None, ng=None, tile_prop=1.0, depth=50, **kwargs):
        super().__init__()
        self.nt = nt
        self.ng = ng
        self.tile_prop = tile_prop
        self.resnet50 = ResNet(depth=depth, **kwargs)
        self._inv_mask = None

    def _mix_tile(self, x):
        bs, c, h, w = x.shape
        if bs == 0 or bs < self.ng:
            return x, None, None

        rot_angles = torch.randint(0, 4, (bs,), device=x.device)    
        tile_h, tile_w = h // self.nt, w // self.nt
        x_tiles = x.view(bs, c, self.nt, tile_h, self.nt, tile_w).permute(0, 2, 4, 1, 3, 5)

        for b in range(bs):
            for i in range(self.nt):
                for j in range(self.nt):
                    x_tiles[b, i, j] = torch.rot90(x_tiles[b, i, j], k=rot_angles[b].item(), dims=[-2, -1])
        x_rotated = x_tiles.permute(0, 3, 1, 4, 2, 5).contiguous().view(bs, c, h, w)

        
        mix_mask = torch.argsort(torch.rand(bs // self.ng, self.ng, self.nt, self.nt), dim=1).to(x.device)
        inv_mask = torch.argsort(mix_mask, dim=1)

        img_mask = mix_mask.view(bs // self.ng, self.ng, 1, self.nt, self.nt)
        img_mask = img_mask.repeat_interleave(3, dim=2)
        img_mask = img_mask.repeat_interleave(h // self.nt, dim=3)
        img_mask = img_mask.repeat_interleave(w // self.nt, dim=4)

        x_mixed = x_rotated.view(bs // self.ng, self.ng, c, h, w)
        x_mixed = torch.gather(x_mixed, dim=1, index=img_mask)
        return x_mixed.view(bs, c, h, w), inv_mask, rot_angles
    
    def _unmix_tiles(self, features, inv_mask, rot_angles):
        unmixed_features = {}
        for pn, feat in features.items():
            bs, c, h, w = feat.shape
            h_ = (h // self.nt) * self.nt or self.nt
            w_ = (w // self.nt) * self.nt or self.nt
            
            feat_resized = torch.nn.functional.interpolate(feat, size=(h_, w_), mode='bilinear') if (h != h_ or w != w_) else feat
            
            feat_unmixed = feat_resized.view(bs // self.ng, self.ng, c, h_, w_)
            feat_mask = inv_mask.view(bs // self.ng, self.ng, 1, self.nt, self.nt)
            feat_mask = feat_mask.repeat_interleave(c, dim=2)
            feat_mask = feat_mask.repeat_interleave(h_ // self.nt, dim=3)
            feat_mask = feat_mask.repeat_interleave(w_ // self.nt, dim=4)
            
            feat_unmixed = torch.gather(feat_unmixed, dim=1, index=feat_mask)
            feat_unmixed = feat_unmixed.view(bs, c, h_, w_)

            tile_h, tile_w = h_ // self.nt, w_ // self.nt
            tiles = feat_unmixed.view(bs, c, self.nt, tile_h, self.nt, tile_w).permute(0, 2, 4, 1, 3, 5)
            for b in range(bs):
                for i in range(self.nt):
                    for j in range(self.nt):
                        k = rot_angles[b].item()
                        tiles[b, i, j] = torch.rot90(tiles[b, i, j], k=(4 - k) % 4, dims=[-2, -1])
            feat_unrotated = tiles.permute(0, 3, 1, 4, 2, 5).contiguous().view(bs, c, h_, w_)
            
            if h != h_ or w != w_:
                feat_unrotated = torch.nn.functional.interpolate(feat_unrotated, size=(h, w), mode='bilinear')

            unmixed_features[pn] = feat_unrotated

        return unmixed_features

    def forward(self, x):
        bs, c, h, w = x.shape
        p = random.random()

        if p < self.tile_prop:
            x_mixed, self._inv_mask, self._rot_angles = self._mix_tile(x)
        else:
            x_mixed = x
            self._inv_mask = None
            self._rot_angles = None

        features = self.resnet50(x_mixed)
        features = {f'res{i+2}': feat for i, feat in enumerate(features)}

        if self._inv_mask is not None and self._rot_angles is not None:
            features = self._unmix_tiles(features, self._inv_mask, self._rot_angles)

        return [features[f'res2'], features[f'res3'], features[f'res4'], features[f'res5']]

