import torch
import numpy as np

from .rotated_semi_detector import RotatedSemiDetector
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models import build_detector
from mmrotate.models import OrientedRPNHead

@ROTATED_DETECTORS.register_module()
class RotatedTwoStageTeacher(RotatedSemiDetector):
    def __init__(self, model: dict, semi_loss, train_cfg=None, test_cfg=None, symmetry_aware=False,
                 pretrained=None):
        """
        Rotated Single Stage Dense Teacher.
        Args:
            model:
            semi_loss:
            train_cfg:
            test_cfg:
            symmetry_aware:
            pretrained:
        """
        teacher_model = model.get('teacher_model', None)
        teacher_model.update(pretrained=pretrained)
        student_model = model.get('student_model', None)
        student_model.update(pretrained=pretrained)
        super(RotatedTwoStageTeacher, self).__init__(
            dict(teacher=build_detector(teacher_model), student=build_detector(student_model)),
            semi_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            # ugly manner to get start iteration, to fit resume mode
            self.iter_count = train_cfg.get("iter_count", 0)
            # Prepare semi-training config
            # step to start training student (not include EMA update)
            self.burn_in_steps = train_cfg.get("burn_in_steps", 5000)
            # prepare super & un-super weight
            self.sup_weight = train_cfg.get("sup_weight", 1.0)
            self.unsup_weight = train_cfg.get("unsup_weight", 1.0)
            self.weight_suppress = train_cfg.get("weight_suppress", "linear")
            self.logit_specific_weights = train_cfg.get("logit_specific_weights")
            self.region_ratio = train_cfg.get("region_ratio")
        self.symmetry_aware = symmetry_aware

    def forward_train(self, imgs, img_metas, **kwargs):
        super(RotatedTwoStageTeacher, self).forward_train(imgs, img_metas, **kwargs)
        gt_bboxes = kwargs.get('gt_bboxes')
        gt_labels = kwargs.get('gt_labels')
        
        format_data = dict()
        for idx, img_meta in enumerate(img_metas):
            tag = img_meta['tag']
            if tag in ['sup_strong', 'sup_weak']:
                tag = 'sup'
            if tag not in format_data.keys():
                format_data[tag] = dict()
                format_data[tag]['img'] = [imgs[idx]]
                format_data[tag]['img_metas'] = [img_metas[idx]]
                format_data[tag]['gt_bboxes'] = [gt_bboxes[idx]]
                format_data[tag]['gt_labels'] = [gt_labels[idx]]
            else:
                format_data[tag]['img'].append(imgs[idx])
                format_data[tag]['img_metas'].append(img_metas[idx])
                format_data[tag]['gt_bboxes'].append(gt_bboxes[idx])
                format_data[tag]['gt_labels'].append(gt_labels[idx])
        for key in format_data.keys():
            format_data[key]['img'] = torch.stack(format_data[key]['img'], dim=0)
        losses = dict()
        # supervised forward
        sup_losses, _ , _ = self.student.forward_train(**format_data['sup'])
        for key, val in sup_losses.items():
            if key[:4] == 'loss':
                if isinstance(val, list):
                    losses[f"{key}_sup"] = [self.sup_weight * x for x in val]
                else:
                    losses[f"{key}_sup"] = self.sup_weight * val
            else:
                losses[f"{key}_sup"] = val
        if self.iter_count > self.burn_in_steps:
            # unsupervised forward
            unsup_weight = self.unsup_weight
            if self.weight_suppress == 'exp':
                target = self.burn_in_steps + 2000
                if self.iter_count <= target:
                    scale = np.exp((self.iter_count - target) / 1000)
                    unsup_weight *= scale
            elif self.weight_suppress == 'step':
                target = self.burn_in_steps * 2
                if self.iter_count <= target:
                    unsup_weight *= 0.25
            elif self.weight_suppress == 'linear':
                target = self.burn_in_steps * 2
                if self.iter_count <= target:
                    unsup_weight *= (self.iter_count - self.burn_in_steps) / self.burn_in_steps

            with torch.no_grad():
                # get teacher data
                _, teacher_logits, teacher_pseudo = self.teacher.forward_train(
                    get_data=True, **format_data['unsup_weak'], get_pred=True)
                
            # get student data
            teacher_pseudo = self.active_select(teacher_pseudo)
            format_data['unsup_strong']['gt_bboxes'] = teacher_pseudo["bboxes"]
            format_data['unsup_strong']['gt_labels'] = teacher_pseudo["labels"]
            unsup_losses, student_logits, _ = self.student.forward_train(**format_data['unsup_strong'])

            for key, val in unsup_losses.items():
                if key[:4] == 'loss':
                    if isinstance(val, list):
                        losses[f"{key}_unsup"] = [self.sup_weight * x for x in val]
                    else:
                        losses[f"{key}_unsup"] = self.sup_weight * val
                else:
                    losses[f"{key}_unsup"] = val
            loss_emd = self.semi_loss(teacher_logits, student_logits)
            losses.update(loss_emd)

            for key, val in self.logit_specific_weights.items():
                if key in losses:
                    losses[key] *= val

        self.iter_count += 1

        for key, val in losses.items():
            if isinstance(val, list):
                new_val = []
                for x in val:
                    if not torch.isfinite(x) or (x > 100).item():
                        new_val.append(torch.tensor(0.0, device=x.device))
                    else:
                        new_val.append(x)
                losses[key] = new_val
            elif isinstance(val, torch.Tensor):
                if not torch.isfinite(val) or (val > 100).item():
                    losses[key] = torch.tensor(0.0, device=val.device)

        return losses

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    
    def active_select(self, pseudo):
        bs = len(pseudo["bboxes"])
        selected_bboxes = []
        selected_labels = []
        
        for i in range(bs):
            # final_bboxes = pseudo["bboxes"][i]
            # final_labels = torch.argmax(pseudo["cls_scores"][i], dim=1)
            bboxes = pseudo["bboxes"][i]
            scores = pseudo["scores"][i]
            cls_scores = pseudo["cls_scores"][i]
            
            if len(bboxes) == 0:
                selected_bboxes.append(bboxes)
                selected_labels.append(torch.empty((0,), dtype=torch.long, device=bboxes.device))
                continue
            
            prelim_mask = scores >= 0.1 
            if prelim_mask.sum() == 0:
                selected_bboxes.append(bboxes.new_empty((0, 5)))
                selected_labels.append(torch.empty((0,), dtype=torch.long, device=bboxes.device))
                continue
            
            filtered_bboxes = bboxes[prelim_mask]
            filtered_scores = scores[prelim_mask]
            filtered_cls_scores = cls_scores[prelim_mask]
            
            entropy = -torch.sum(filtered_cls_scores * torch.log(filtered_cls_scores + 1e-8), dim=1)
            m_diff = entropy
            
            m_info = filtered_scores.clone()
            
            pred_labels = torch.argmax(filtered_cls_scores, dim=1)
            unique_classes = torch.unique(pred_labels)
            diversity_score = len(unique_classes)
            m_dive = torch.full_like(filtered_scores, diversity_score, dtype=torch.float)
            
            angles = filtered_bboxes[:, -1]
            if len(angles) > 1:
                mean_angle = angles.mean()
                angle_variance = torch.mean((angles - mean_angle) ** 2)
            else:
                angle_variance = torch.tensor(0.0, device=angles.device)
            m_var = torch.full_like(filtered_scores, float(angle_variance.item()), dtype=torch.float)
            
            def min_max_normalize(tensor):
                if tensor.max() - tensor.min() < 1e-6:
                    return torch.ones_like(tensor)
                return (tensor - tensor.min()) / (torch.clamp(tensor.max() - tensor.min(), min=1e-6))
            
            m_diff_norm = min_max_normalize(m_diff)
            m_info_norm = min_max_normalize(m_info)
            m_dive_norm = min_max_normalize(m_dive)
            m_var_norm = min_max_normalize(m_var)
            
            weights = torch.tensor([1, 1, 1, 1], device=bboxes.device)
            combined_scores = (weights[0] * m_diff_norm + weights[1] * m_info_norm + weights[2] * m_dive_norm + weights[3] * m_var_norm)
            
            keep_k = max(1, int(len(combined_scores) * 0.9))
            _, topk_indices = torch.topk(combined_scores, k=min(keep_k, len(combined_scores)), largest=True)
            
            final_mask = filtered_scores[topk_indices] >= 0.3
            final_indices = topk_indices[final_mask]
            
            if len(final_indices) < len(filtered_scores) * 0.8:
                final_bboxes = filtered_bboxes
                final_labels = pred_labels
            else:
                final_bboxes = filtered_bboxes[final_indices]
                final_labels = pred_labels[final_indices]

            selected_bboxes.append(final_bboxes)
            selected_labels.append(final_labels)
            
        return {
            "bboxes": selected_bboxes,
            "labels": selected_labels
        }
