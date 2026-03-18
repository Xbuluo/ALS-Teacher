import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from geomloss import SamplesLoss
from mmrotate.models import ROTATED_LOSSES


@ROTATED_LOSSES.register_module()
class RotatedSoftEMDLoss(nn.Module):
    def __init__(self,
                 distance_type=None,
                 fallback_blur=0.05,
                 fallback_scaling=0.9,
                 p=2):
        super().__init__()
        self.distance_type = distance_type

        if distance_type in ['sinkhorn', 'hausdorff', 'Multiscale', 'energy']:
            self.dist_func = SamplesLoss(distance_type, p=p, blur=fallback_blur, scaling=fallback_scaling)
        else:
            self.dist_func = SamplesLoss("sinkhorn", p=p, blur=fallback_blur, scaling=fallback_scaling)


    def forward(self, t_logits, s_logits):
        weights = self.generate_soft_weights(t_logits)
        mask = weights > 0
        if mask.sum() <= 10:
            return dict(loss_emd=torch.tensor(0.0, device=weights.device))
        
        t_scores = t_logits['score_pred']
        s_scores = s_logits['score_pred']
        t_bbox = t_logits['bbox'][mask]
        s_bbox = s_logits['bbox'][mask]
        pt_valid = t_logits['bbox'][:, :2][mask]

        t_scores = t_scores[mask]
        s_scores = s_scores[mask]
        
        t_prob = F.softmax(t_scores, dim=0)
        s_prob = F.softmax(s_scores, dim=0)

        loss_emd = self.dist_func(t_prob, pt_valid, s_prob, pt_valid)

        weight_dist = torch.abs(t_bbox[:, -1] - s_bbox[:, -1]) / np.pi + 1
        loss_emd = torch.log1p(loss_emd) * weight_dist.mean().detach() * 0.01

        return dict(loss_emd=loss_emd)



    def generate_soft_weights(self, logits, cls_threshold=0.5, score_threshold=0.5):
        cls = logits['cls_score']
        score = logits['score_pred']

        conf_cls = cls.softmax(dim=1).max(dim=1)[0]
        conf_score = torch.sigmoid(score.squeeze(-1))
        soft_weight = 0.5 * conf_cls + 0.5 * conf_score 

        mask = (conf_cls > cls_threshold) & (conf_score > score_threshold)
        soft_weight = soft_weight * mask.float()

        return soft_weight / (soft_weight.sum() + 1e-6)
