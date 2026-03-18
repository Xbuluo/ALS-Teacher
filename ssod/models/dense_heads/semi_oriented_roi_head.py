import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrotate.models import OrientedStandardRoIHead, ROTATED_HEADS
from mmrotate.core import rbbox2roi
from mmrotate.models.roi_heads.bbox_heads import RotatedShared2FCBBoxHead
from mmcv.runner import force_fp32
# from mmrotate.core import multiclass_nms_rotated
from ..utils.multiclass_nms_rotated import multiclass_nms_rotated


@ROTATED_HEADS.register_module()
class ScoredRotatedShared2FCBBoxHead(RotatedShared2FCBBoxHead):
    def __init__(self, *args, predict_score=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_score = predict_score
        if self.predict_score:
            self.score_head = nn.Sequential(
                nn.Linear(self.fc_out_channels, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1)
            )

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.flatten(1)

        for fc in self.shared_fcs:
            x = self.relu(fc(x))
        fc_feat = x

        cls_score = self.fc_cls(fc_feat) if self.with_cls else None
        bbox_pred = self.fc_reg(fc_feat) if self.with_reg else None
        score_pred = self.score_head(fc_feat).squeeze(-1) if self.predict_score else None

        return cls_score, bbox_pred, score_pred
    
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes_(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = bboxes.view(bboxes.size(0), -1, 5)
            bboxes[..., :4] = bboxes[..., :4] / scale_factor
            bboxes = bboxes.view(bboxes.size(0), -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels, inds = multiclass_nms_rotated(
                bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img, return_inds=True)
            return det_bboxes, det_labels, inds
        


@ROTATED_HEADS.register_module()
class SemiOrientedRoiHead(OrientedStandardRoIHead):

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      get_data=False,
                      **kwargs):

        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(len(img_metas))]

        if get_data:
            if self.with_bbox:
                if all([len(p) == 0 for p in proposal_list]):
                    return dict()
                rois = rbbox2roi(proposal_list)
                bbox_results = self._bbox_forward(x, rois)
                return bbox_results

        losses = dict()
        bbox_results = dict()

        if self.with_bbox:
            num_imgs = len(img_metas)
            sampling_results = []

            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                )
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x]
                )
                if gt_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i].new_zeros((0, gt_bboxes[0].size(-1)))
                else:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]

                sampling_results.append(sampling_result)

            bbox_results = self._bbox_forward_train(x, sampling_results, gt_bboxes, gt_labels, img_metas)
            losses.update(bbox_results['loss_bbox'])

            rois = rbbox2roi(proposal_list)
            bbox_results = self._bbox_forward(x, rois)

        return losses, bbox_results

    def _bbox_forward(self, x, rois):
        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        cls_score, bbox_pred, score_pred = self.bbox_head(bbox_feats)
        decoded_bboxes = self.bbox_head.bbox_coder.decode(rois[:, 1:], bbox_pred)
        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox=decoded_bboxes, 
            bbox_feats=bbox_feats,
            score_pred=score_pred 
        )
        return bbox_results
    
    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                gt_labels, self.train_cfg)
        labels, _, _, _ = bbox_targets

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        if self.bbox_head.predict_score:
            fg_mask = labels < self.bbox_head.num_classes
            if fg_mask.any():
                pred_score = bbox_results['score_pred']
                gt_score = fg_mask.float()
                loss_score = F.binary_cross_entropy_with_logits(pred_score, gt_score, reduction='mean')
            else:
                loss_score = bbox_results['score_pred'].sum() * 0.0
            loss_bbox['loss_score'] = loss_score
        
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
    
    def simple_test_bboxes_(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False):
        rois = rbbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        bbox_score = bbox_results['score_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_score = bbox_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        det_bbox_scores = []
        det_cls_scores = []
        for i in range(len(proposals)):
            det_bbox, det_label, inds = self.bbox_head.get_bboxes_(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            bbox_score_is = torch.sigmoid(bbox_score[i])
            det_bbox_scores.append(bbox_score_is[inds])
            det_cls_scores.append(cls_score[i][inds])
        return det_bboxes, det_labels, det_bbox_scores, det_cls_scores

