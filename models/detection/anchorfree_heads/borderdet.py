import math
import torch

import torch.nn as nn
import torch.nn.functional as F
from layers import ShapeSpec, cat, generalized_batched_nms
import torch.distributed as dist

from models.detection.anchorfree_heads import DET_ANCHORFREE_HEADS_REGISRY
from layers import BorderAlign
from models.detection.anchorfree_heads.anchorfree_heads import AnchorFreeHeadBase
from models.detection.anchorfree_heads.fcos_head import Scale
from models.detection.modules.anchor_generator import ShiftGenerator
from utils.nn.focal_loss import sigmoid_focal_loss_jit
from utils.nn.fcos_loss import iou_loss
from utils.nn.smoothL1Loss import smooth_l1_loss
from structures import Boxes, pairwise_iou, Instances
from models.detection.modules.box_regression import Shift2BoxTransform


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls, box_delta, box_center, border_cls, border_delta, num_classes=80):
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]

    border_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in border_cls]
    border_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in border_delta]
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    box_center = cat(box_center_flattened, dim=1).view(-1, 1)

    border_cls = cat(border_cls_flattened, dim=1).view(-1, num_classes)
    border_delta = cat(border_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta, box_center, border_cls, border_delta


class BorderDetHead(AnchorFreeHeadBase):
    def __init__(self, cfg, input_shape=None):
        super(BorderDetHead, self).__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.num_classes = cfg.MODEL.FCOS_HEADS.NUM_CLASSES
        self.in_features = cfg.MODEL.FCOS_HEADS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS_HEADS.FPN_STRIDES

        shapes = [ShapeSpec(stride=s) for s in self.fpn_strides]
        self.shift_generator = ShiftGenerator(cfg, shapes)

        in_channels = 256
        num_convs = cfg.MODEL.FCOS_HEADS.NUM_CONVS
        prior_prob = cfg.MODEL.FCOS_HEADS.PRIOR_PROB

        # loss
        self.focal_loss_alpha = cfg.MODEL.FCOS_HEADS.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS_HEADS.LOSS_GAMMA
        self.iou_los_type = cfg.MODEL.FCOS_HEADS.LOC_LOSS_TYPE
        self.center_sampling_radius = cfg.MODEL.FCOS_HEADS.POS_RADIUS
        self.border_iou_thresh = cfg.MODEL.FCOS_HEADS.BORDER_IOU_THRESH
        self.border_bbox_std = cfg.MODEL.FCOS_HEADS.BORDER_BBOX_STD

        # inference
        self.score_threshold  = cfg.MODEL.FCOS_HEADS.BORDER_IOU_THRESH
        self.topk_candidates = cfg.MODEL.FCOS_HEADS.PRE_NMS_TOPK_TEST
        self.nms_threshold = cfg.MODEL.FCOS_HEADS.NMS_THRESH
        self.nms_type = cfg.MODEL.FCOS_HEADS.NMS_TYPE
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

        self.centerness_on_reg = cfg.MODEL.FCOS_HEADS.BORDER_CENTER_ON_REG
        self.norm_reg_targets = cfg.MODEL.FCOS_HEADS.NORM_REG_TARGETS

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            cls_subnet.append(nn.GroupNorm(32, in_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            bbox_subnet.append(nn.GroupNorm(32, in_channels))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)

        self.cls_score = nn.Conv2d(
            in_channels, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1)

        # BorderDet
        self.add_module("border_cls_subnet", BorderBranch(in_channels, 256))
        self.add_module("border_bbox_subnet", BorderBranch(in_channels, 128))

        self.border_cls_score = nn.Conv2d(
            in_channels, self.num_classes, kernel_size=1, stride=1)
        self.border_bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=1, stride=1)

        # initialization
        bias_value = -math.log((1-prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)
        torch.nn.init.constant_(self.border_cls_score.bias, bias_value)

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])

        # Matching and Loss
        self.shift2box_transform = Shift2BoxTransform(
            weights=cfg.MODEL.FCOS_HEADS.BBOX_REG_WEIGHTS)
        self.object_sizes_of_interest = cfg.MODEL.FCOS_HEADS.OBJECT_SIZES_OF_INTEREST

    def forward(self, in_features):
        feats = [in_features[f] for f in self.in_features]
        ori_shifts = self.shift_generator(feats)

        logits = []
        bbox_reg = []
        centerness = []
        border_logits = []
        border_bbox_reg = []
        pre_bbox = []

        shifts = [
            torch.cat([shi.unsqueeze(0) for shi in shift], dim=0)
            for shift in list(zip(*ori_shifts))
        ]

        for level, (feature, shifts_i) in enumerate(zip(feats, shifts)):
            cls_subnet = self.cls_subnet(feature)
            bbox_subnet = self.bbox_subnet(feature)

            logits.append(self.cls_score(cls_subnet))
            if self.centerness_on_reg:
                centerness.append(self.centerness(bbox_subnet))
            else:
                centerness.append(self.centerness(cls_subnet))

            bbox_pred = self.scales[level](self.bbox_pred(bbox_subnet))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred) * self.fpn_strides[level]
            else:
                bbox_pred = torch.exp(bbox_pred) * self.fpn_strides[level]
            bbox_reg.append(bbox_pred)

            # border
            N,C,H,W = feature.shape
            pre_off = bbox_pred.clone().detach()
            with torch.no_grad():
                pre_off = pre_off.permute(0, 2, 3, 1).reshape(N, -1, 4)
                pre_boxes = self.compute_box(shifts_i, pre_off)
                align_boxes, wh = self.compute_border(pre_boxes, level, H, W)
                pre_bbox.append(pre_boxes)

            border_cls_conv = self.border_cls_subnet(cls_subnet, align_boxes, wh)
            border_cls_logits = self.border_cls_score(border_cls_conv)
            border_logits.append(border_cls_logits)

            border_reg_conv = self.border_bbox_subnet(bbox_subnet, align_boxes, wh)
            border_bbox_pred = self.border_bbox_pred(border_reg_conv)
            border_bbox_reg.append(border_bbox_pred)

        if self.training:
            pre_bbox = torch.cat(pre_bbox, dim=1)
        return (logits, bbox_reg, centerness, border_logits, border_bbox_reg, pre_bbox, ori_shifts)

    def forward_bbox(self, features, targets):
        gt_instances = [x["instances"].to(self.device) for x in targets]
        bd_based_box, ori_shifts = features[-2], features[-1]
        gts = self.get_ground_truth(ori_shifts, gt_instances, bd_based_box)
        return self.losses(features, gts)

    def losses(self, pred, targets):
        box_cls, box_delta, box_center, bd_box_cls, bd_box_delta, bd_based_box, shifts = pred
        gt_classes, gt_shifts_reg_deltas, gt_centerness, gt_border_classes, gt_border_shifts_deltas = targets

        pred_cls_logits, pred_shift_deltas, pred_centerness, border_cls_ligits, border_shift_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            box_cls, box_delta, box_center, bd_box_cls, bd_box_delta, self.num_classes
        )

        # FCOS
        gt_classes = gt_classes.flatten()
        gt_shifts_deltas = gt_shifts_reg_deltas.view(-1, 4)
        gt_centerness = gt_centerness.view(-1, 1)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()
        acc_centerness_num = gt_centerness[foreground_idxs].sum()

        gt_classes_target = torch.zeros_like(pred_cls_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        dist.all_reduce(num_foreground)
        num_foreground /= dist.get_world_size()
        dist.all_reduce(acc_centerness_num)
        acc_centerness_num /= dist.get_world_size()

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_cls_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum"
        )/ max(1, num_foreground)

        loss_box_reg = iou_loss(
            pred_shift_deltas[foreground_idxs],
            gt_shifts_deltas[foreground_idxs],
            box_mode='ltrb',
            loss_type=self.iou_los_type,
            reduction='sum',
        )/max(1, acc_centerness_num)

        loss_centerness = F.binary_cross_entropy_with_logits(
            pred_centerness[foreground_idxs],
            gt_centerness[foreground_idxs],
            reduction='sum'
        ) / max(1, num_foreground)

        # borderDet
        gt_class_border = gt_border_classes.flatten()
        gt_deltas_border = gt_border_shifts_deltas.view(-1, 4)

        valid_idxs_border = gt_class_border >= 0
        foreground_idxs_border = (gt_class_border >= 0) & (gt_class_border != self.num_classes)
        num_foreground_border = foreground_idxs_border.sum()

        gt_classes_border_target = torch.zeros_like(border_cls_ligits)
        gt_classes_border_target[
            foreground_idxs_border, gt_class_border[foreground_idxs_border]
        ] = 1

        dist.all_reduce(num_foreground_border)
        num_foreground_border /= dist.get_world_size()

        num_foreground_border = max(num_foreground_border, 1.0)
        loss_border_cls = sigmoid_focal_loss_jit(
            border_cls_ligits[valid_idxs_border],
            gt_classes_border_target[valid_idxs_border],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction='sum'
        ) / num_foreground_border
        if foreground_idxs_border.numel() > 0:
            loss_border_reg = (
                smooth_l1_loss(
                    border_shift_deltas[foreground_idxs_border],
                    gt_deltas_border[foreground_idxs_border],
                    beta=0,
                    reduction='sum'
                ) / num_foreground_border
            )
        else:
            loss_border_reg = 0 * border_shift_deltas.sum()

        return {
            "loss_cls": loss_cls,
            "loss_box_reg": loss_box_reg,
            "loss_centerness": loss_centerness,
            "loss_border_cls": loss_border_cls,
            "loss_border_reg": loss_border_reg,
        }

    def inference(self, features, image_sizes):
        logits, bbox_reg, centerness, border_logits, border_bbox_reg, pre_bbox, ori_shifts = features
        del bbox_reg
        del ori_shifts

        results = []
        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in logits]
        box_center = [permute_to_N_HWA_K(x, 1) for x in centerness]
        border_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in border_logits]
        border_delta = [permute_to_N_HWA_K(x, 4) for x in border_bbox_reg]


        for img_idx, image_size_per_image in enumerate(image_sizes):
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_ctr_per_image = [box_ctr_per_level[img_idx] for box_ctr_per_level in box_center]
            border_cls_per_image = [border_cls_per_level[img_idx] for border_cls_per_level in border_cls]
            border_reg_per_image = [border_reg_per_level[img_idx] for border_reg_per_level in border_delta]
            bd_based_box_per_image = [box_loc_per_level[img_idx] for box_loc_per_level in pre_bbox]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_ctr_per_image, border_cls_per_image,
                border_reg_per_image, bd_based_box_per_image, tuple(image_size_per_image)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_center, border_cls, border_delta, bd_based_box, image_size):
        boxes_all = []
        scores_all = []
        class_idxs_all = []
        border_bbox_std = bd_based_box[0].new_tensor(self.border_bbox_std)

        # Iterate over every feature level
        for box_cls_i, box_ctr_i, bd_box_cls_i, bd_box_reg_i, bd_based_box_i in zip(
                box_cls, box_center, border_cls, border_delta, bd_based_box):
            # (HxWxK,)
            box_cls_i = box_cls_i.sigmoid_()
            box_ctr_i = box_ctr_i.sigmoid_()
            bd_box_cls_i = bd_box_cls_i.sigmoid_()

            predicted_prob = (box_cls_i * box_ctr_i).sqrt()

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold

            predicted_prob = predicted_prob * bd_box_cls_i

            predicted_prob = predicted_prob[keep_idxs]
            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, predicted_prob.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = predicted_prob.sort(descending=True)
            topk_idxs = topk_idxs[:num_topk]

            keep_idxs = keep_idxs.nonzero()
            keep_idxs = keep_idxs[topk_idxs]
            keep_box_idxs = keep_idxs[:, 0]
            classes_idxs = keep_idxs[:, 1]

            predicted_prob = predicted_prob[:num_topk]
            bd_box_reg_i = bd_box_reg_i[keep_box_idxs]
            bd_based_box_i = bd_based_box_i[keep_box_idxs]

            det_wh = (bd_based_box_i[..., 2:4] - bd_based_box_i[..., :2])
            det_wh = torch.cat([det_wh, det_wh], dim=1)
            predicted_boxes = bd_based_box_i + (bd_box_reg_i * border_bbox_std * det_wh)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob.sqrt())
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]

        keep = generalized_batched_nms(boxes_all, scores_all, class_idxs_all,
                                       self.nms_threshold, nms_type=self.nms_type)
        boxes_all = boxes_all[keep]
        scores_all = scores_all[keep]
        class_idxs_all = class_idxs_all[keep]

        number_of_detections = len(keep)
        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.max_detections_per_image > 0:
            image_thresh, _ = torch.kthvalue(
                scores_all,
                number_of_detections - self.max_detections_per_image + 1
            )
            keep = scores_all >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            boxes_all = boxes_all[keep]
            scores_all = scores_all[keep]
            class_idxs_all = class_idxs_all[keep]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all)
        result.scores = scores_all
        result.pred_classes = class_idxs_all
        return result

    def compute_box(self, location, pred_offset):
        detections = torch.stack([
            location[:,:,0] - pred_offset[:,:,0],
            location[:,:,1] - pred_offset[:,:,1],
            location[:,:,0] + pred_offset[:,:,2],
            location[:,:,1] + pred_offset[:,:,3]
        ], dim=2)
        return detections

    def compute_border(self, _boxes, fm_i, height, width):
        boxes = _boxes / self.fpn_strides[fm_i]
        boxes[:,:,0].clamp_(min=0, max=width-1)
        boxes[:,:,1].clamp_(min=0, max=height-1)
        boxes[:,:,2].clamp_(min=0, max=width-1)
        boxes[:,:,3].clamp_(min=0, max=height-1)

        wh = (boxes[:,:,2:] - boxes[:,:,:2]).contiguous()
        return boxes, wh

    @torch.no_grad()
    def get_ground_truth(self, shifts, targets, pre_boxes_list):
        """
        Args:
            shifts (list[list[Tensor]]): a list of N=#image elements. Each is a
                list of #feature level tensors. The tensors contains shifts of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each shift.
                R is the total number of shifts, i.e. the sum of Hi x Wi for all levels.
                Shifts in the valid boxes are assigned their corresponding label in the
                [0, K-1] range. Shifts in the background are assigned the label "K".
                Shifts in the ignore areas are assigned a label "-1", i.e. ignore.
            gt_shifts_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth shift2box transform
                targets (dl, dt, dr, db) that map each shift to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                shift is labeled as foreground.
            gt_centerness (Tensor):
                An float tensor (0, 1) of shape (N, R) whose values in [0, 1]
                storing ground-truth centerness for each shift.
            border_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each shift.
                R is the total number of shifts, i.e. the sum of Hi x Wi for all levels.
                Shifts in the valid boxes are assigned their corresponding label in the
                [0, K-1] range. Shifts in the background are assigned the label "K".
                Shifts in the ignore areas are assigned a label "-1", i.e. ignore.
            border_shifts_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth shift2box transform
                targets (dl, dt, dr, db) that map each shift to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                shift is labeled as foreground.
        """
        gt_classes = []
        gt_shifts_deltas = []
        gt_centerness = []

        border_classes = []
        border_shifts_deltas = []

        for shifts_per_image, targets_per_image, pre_boxes in zip(shifts, targets, pre_boxes_list):
            object_sizes_of_interest = torch.cat([
                shifts_i.new_tensor(size).unsqueeze(0).expand(
                    shifts_i.size(0), -1) for shifts_i, size in zip(
                    shifts_per_image, self.object_sizes_of_interest)
            ], dim=0)

            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0)

            gt_boxes = targets_per_image.gt_boxes

            deltas = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, gt_boxes.tensor.unsqueeze(1))

            if self.center_sampling_radius > 0:
                centers = gt_boxes.get_centers()
                is_in_boxes = []
                for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                    radius = stride * self.center_sampling_radius
                    center_boxes = torch.cat((
                        torch.max(centers - radius, gt_boxes.tensor[:, :2]),
                        torch.min(centers + radius, gt_boxes.tensor[:, 2:]),
                    ), dim=-1)
                    center_deltas = self.shift2box_transform.get_deltas(
                        shifts_i, center_boxes.unsqueeze(1))
                    is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
                is_in_boxes = torch.cat(is_in_boxes, dim=1)
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = deltas.min(dim=-1).values > 0

            max_deltas = deltas.max(dim=-1).values
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_deltas >= object_sizes_of_interest[None, :, 0]) & \
                (max_deltas <= object_sizes_of_interest[None, :, 1])

            gt_positions_area = gt_boxes.area().unsqueeze(1).repeat(
                1, shifts_over_all_feature_maps.size(0))
            gt_positions_area[~is_in_boxes] = math.inf
            gt_positions_area[~is_cared_in_the_level] = math.inf

            # if there are still more than one objects for a position,
            # we choose the one with minimal area
            positions_min_area, gt_matched_idxs = gt_positions_area.min(dim=0)

            # ground truth box regression
            gt_shifts_reg_deltas_i = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, gt_boxes[gt_matched_idxs].tensor)

            # ground truth classes
            has_gt = len(targets_per_image) > 0
            if has_gt:
                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                # Shifts with area inf are treated as background.
                gt_classes_i[positions_min_area == math.inf] = self.num_classes
            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes

            # ground truth centerness
            left_right = gt_shifts_reg_deltas_i[:, [0, 2]]
            top_bottom = gt_shifts_reg_deltas_i[:, [1, 3]]
            gt_centerness_i = torch.sqrt(
                (left_right.min(dim=-1).values / left_right.max(dim=-1).values).clamp_(min=0)
                * (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values).clamp_(min=0)
            )

            gt_classes.append(gt_classes_i)
            gt_shifts_deltas.append(gt_shifts_reg_deltas_i)
            gt_centerness.append(gt_centerness_i)

            # border
            iou = pairwise_iou(Boxes(pre_boxes), gt_boxes)
            (max_iou, argmax_iou) = iou.max(dim=1)
            invalid = max_iou < self.border_iou_thresh
            gt_target = gt_boxes[argmax_iou].tensor

            border_cls_target = targets_per_image.gt_classes[argmax_iou]
            border_cls_target[invalid] = self.num_classes

            border_bbox_std = pre_boxes.new_tensor(self.border_bbox_std)
            pre_boxes_wh = pre_boxes[:, 2:4] - pre_boxes[:, 0:2]
            pre_boxes_wh = torch.cat([pre_boxes_wh, pre_boxes_wh], dim=1)
            border_off_target = (gt_target - pre_boxes) / (pre_boxes_wh * border_bbox_std)

            border_classes.append(border_cls_target)
            border_shifts_deltas.append(border_off_target)

        return (
            torch.stack(gt_classes),
            torch.stack(gt_shifts_deltas),
            torch.stack(gt_centerness),
            torch.stack(border_classes),
            torch.stack(border_shifts_deltas),
        )


class BorderBranch(nn.Module):
    def __init__(self, in_channels, border_channels):
        super(BorderBranch, self).__init__()
        self.cur_point_conv = nn.Sequential(
            nn.Conv2d(in_channels, border_channels, kernel_size=1),
            nn.InstanceNorm2d(border_channels),
            nn.ReLU()
        )
        self.ltrb__conv = nn.Sequential(
            nn.Conv2d(in_channels, border_channels * 4, kernel_size=1),
            nn.InstanceNorm2d(border_channels*4),
            nn.ReLU()
        )
        # TODO BorderAlign
        self.border_align = BorderAlign(pool_size=10)
        self.border_conv = nn.Sequential(
            nn.Conv2d(5*border_channels, in_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, feature, boxes, wh):
        N, C, H, W = feature.size()
        fm_short = self.cur_point_conv(feature)
        feature = self.ltrb__conv(feature)
        ltrb_conv = self.border_align(feature, boxes)
        ltrb_conv = ltrb_conv.permute(0, 3, 1, 2).reshape(N, -1, H, W)
        align_conv = torch.cat([ltrb_conv, fm_short], dim=1)
        align_conv = self.border_conv(align_conv)
        return align_conv


@DET_ANCHORFREE_HEADS_REGISRY.register()
def borderdet_head_builder(cfg, input_shape=None):
    borderdet_head = BorderDetHead(cfg, input_shape)
    return borderdet_head