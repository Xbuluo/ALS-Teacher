
import torch
import torch.nn.functional as F
from mmrotate.models import OrientedRCNN, ROTATED_DETECTORS


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)


@ROTATED_DETECTORS.register_module()
class SemiOrientedRCNN(OrientedRCNN):
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      get_data=False,
                      get_pred=False,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # from tools.draw_bbox import draw
        # img_to_draw = cv2.imread(img_metas[1]['filename'])
        # boxes = gt_bboxes[1].cpu().numpy()
        # for box in boxes:
        #     cx, cy, w, h, a= box
        #     angle = math.degrees(a)  # 旋转框的旋转角度
        #     draw(img_to_draw, cx, cy, w, h,angle)
        # cv2.imwrite('D:/SOOD/tools/unsup.png', img_to_draw)
        x = self.extract_feat(img)
        loss = dict()

        # sup
        if not get_pred and not get_data:
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    x, img_metas, gt_bboxes, gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    **kwargs)
                loss.update(rpn_losses)
            else:
                proposal_list = proposals

            loss_roi, bbox_list = self.roi_head.forward_train(x, img_metas, proposal_list,
                gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, get_data=False, **kwargs)
            loss.update(loss_roi)

            return loss, bbox_list, None


        # unsup
        with torch.no_grad():
            self.eval()
            if proposals is None:
                proposal_results = self.rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_results = proposals

            det_bboxes, _, det_scores, det_cls_scores =self.roi_head.simple_test_bboxes_(x, img_metas, proposal_results, self.test_cfg.rcnn)
            self.train()

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            _, proposal_list_ = self.rpn_head.forward_train(
                x, img_metas, gt_bboxes, gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
        else:
            proposal_list_ = proposals    

        bbox_list = self.roi_head.forward_train(x, img_metas, proposal_list_, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, get_data=get_data)

        for i in range(len(det_bboxes)):
            det_cls_scores[i] = F.softmax(det_cls_scores[i], dim=1)
            
        return None, bbox_list, dict(
            bboxes = det_bboxes,
            scores = det_scores,
            cls_scores = det_cls_scores
        )
