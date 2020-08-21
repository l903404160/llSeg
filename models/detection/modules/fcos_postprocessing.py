import torch
import torch.nn as nn
from layers import ml_nms
from structures import Instances, Boxes


class FCOSPostProcesser(nn.Module):
    def __init__(self, cfg):
        super(FCOSPostProcesser, self).__init__()
        # nms
        self.pre_nms_thresh_train = cfg.MODEL.FCOS_HEADS.INFERENCE_THRESH_TRAIN
        self.pre_nms_topk_train = cfg.MODEL.FCOS_HEADS.PRE_NMS_TOPK_TRAIN
        self.post_nms_topk_train = cfg.MODEL.FCOS_HEADS.POST_NMS_TOPK_TRAIN

        self.pre_nms_thresh_test = cfg.MODEL.FCOS_HEADS.INFERENCE_THRESH_TEST
        self.pre_nms_topk_test = cfg.MODEL.FCOS_HEADS.PRE_NMS_TOPK_TEST
        self.post_nms_topk_test = cfg.MODEL.FCOS_HEADS.POST_NMS_TOPK_TEST
        self.thresh_with_ctr = cfg.MODEL.FCOS_HEADS.THRESH_WITH_CTR
        self.nms_thresh = cfg.MODEL.FCOS_HEADS.NMS_THRESH

        self.fpn_strides = cfg.MODEL.FCOS_HEADS.FPN_STRIDES

        self.num_refine = 256

    def forward(self, preds, image_sizes):
        logits, bbox_reg, ctrness, top_feats, bbox_towers, locations = preds
        sampled_boxes = []
        bundle = {
            "l": locations, "o": logits,
            "r": bbox_reg, "c": ctrness,
            "s": self.fpn_strides
        }
        if len(top_feats) > 0:
            bundle["t"] = top_feats

        for i, per_bundle in enumerate(zip(*bundle.values())):
            per_bundle = dict(zip(bundle.keys(), per_bundle))
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.
            l = per_bundle["l"]
            o = per_bundle["o"]
            r = per_bundle["r"] * per_bundle["s"]
            c = per_bundle["c"]
            t = per_bundle["t"] if "t" in bundle else None
            sampled_boxes.append(
                self._forward_for_single_feature_map(
                    l, o, r, c, image_sizes, t
                )
            )
            for per_im_smapled_boxes in sampled_boxes[-1]:
                per_im_smapled_boxes.fpn_levels = l.new_ones(len(per_im_smapled_boxes), dtype=torch.long) * i

        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        # if self.training:
        boxlists = self._select_top_k_over_all_levels(boxlists)
        # else:
        # boxlists = self._select_over_all_levels(boxlists)
        return boxlists

    def _forward_for_single_feature_map(self, locations, logits_pred, reg_pred, ctrness_pred, image_sizes, top_feat=None):
        N,C,H,W = logits_pred.shape

        # put in the same format as locations
        logits_pred = logits_pred.view(N,C,H,W).permute(0,2,3,1)
        logits_pred = logits_pred.reshape(N, -1, C).sigmoid()
        box_regisssion = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regisssion = box_regisssion.reshape(N, -1, 4)
        ctrness_pred = ctrness_pred.view(N, 1, H, W).permute(0, 2, 3, 1)
        ctrness_pred = ctrness_pred.reshape(N, -1).sigmoid()

        # if self.thresh_with_ctr is True, we multiply the classification scores with centerness scores before appling the threshold
        if self.thresh_with_ctr:
            logits_pred = logits_pred * ctrness_pred[:, :, None]
        candidate_inds = logits_pred > self.pre_nms_thresh_test

        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_topk_test)

        if not self.thresh_with_ctr:
            logits_pred = logits_pred * ctrness_pred[:, :, None]
        results = []
        for i in range(N):
            per_box_cls = logits_pred[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = box_regisssion[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.scores = torch.sqrt(per_box_cls)
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations
            results.append(boxlist)
        return results

    def _select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # Multiclass nms
            result = ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # limit to max per image detections "over all classes
            if number_of_detections > self.post_nms_topk_test > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(cls_scores.cpu(), number_of_detections - self.post_nms_topk_test + 1)
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results

    def _select_top_k_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            result = boxlists[i]
            number_of_detections = len(result)

            if number_of_detections > self.num_refine:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(cls_scores.cpu(), number_of_detections - self.num_refine + 1)
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results
