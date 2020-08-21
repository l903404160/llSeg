# FCOS head
# Ref : https://github.com/aim-uofa/AdelaiDet/tree/master/adet/modeling/fcos
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.detection.anchorfree_heads.two_fcso_head import TwoFCOSP3Head
from layers import ShapeSpec, DeformableConv, cat
from layers import get_norm_with_channels  as get_norm
from models.detection.modules.fcos_postprocessing import FCOSPostProcesser
from layers.anchorfree.fcos_layers import compute_locations
from structures import pairwise_iou

from typing import Dict
from models.detection.anchorfree_heads.anchorfree_heads import AnchorFreeHeadBase, DET_ANCHORFREE_HEADS_REGISRY
from structures import Instances, Boxes
from utils.nn.fcos_loss import FCOSLoss

from layers import ml_nms


INF = 100000000


class FCOSHead(AnchorFreeHeadBase):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]=None):
        super(FCOSHead, self).__init__()
        self.in_features = cfg.MODEL.FCOS_HEADS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS_HEADS.FPN_STRIDES
        self.yield_proposal = cfg.MODEL.FCOS_HEADS.YIELD_PROPOSAL

        self.num_classes = cfg.MODEL.FCOS_HEADS.NUM_CLASSES
        self.num_levels = len(self.in_features)
        self.in_channels = cfg.MODEL.FCOS_HEADS.FPN_CHANNELS

        self.radius = cfg.MODEL.FCOS_HEADS.POS_RADIUS

        self.fcos_box_head = FCOSBoxHead(cfg)
        # postprocesser
        self.post_processer = FCOSPostProcesser(cfg)
        # two stage head
        self.two_stage_fcos_head = TwoFCOSP3Head(cfg)
        # generate soi
        soi = []
        prev_size = -1
        for s in cfg.MODEL.FCOS_HEADS.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi

        # loss
        self.loss_func = self._init_loss(cfg)
        self.center_sample = cfg.MODEL.FCOS_HEADS.CENTER_SAMPLE

        # Device
        self.device = cfg.MODEL.DEVICE

    def _init_loss(self, cfg):
        return FCOSLoss(cfg)

    def forward(self, in_features, image_sizes=None):
        features = self.fcos_box_head(in_features)
        with torch.no_grad():
            proposals = self.post_processer(features, image_sizes)
            num_instances = [len(proposals[i]) for i in range(len(proposals))]
            image_indexes = [torch.ones(len(proposals[i])) * i for i in range(len(proposals))]
            image_indexes = torch.cat(image_indexes, dim=0)[:, None].cuda()
            boxes = torch.cat([proposals[i].pred_boxes.tensor for i in range(len(proposals))], dim=0)
            boxes = torch.cat([image_indexes, boxes], dim=1)
        cls_pred, box_pred = self.two_stage_fcos_head(in_features, boxes)
        cls_pred = torch.split(cls_pred[0], num_instances, dim=0) # [N, proposals]
        box_pred = torch.split(box_pred[0], num_instances, dim=0)

        del boxes
        del image_indexes
        del num_instances

        two_pred = cls_pred, box_pred, proposals
        return features, two_pred

    def forward_bbox(self, features, targets):
        """
            1. 计算fcos的loss
            2. 计算二次回归分类的loss
                1. 生成proposals   -- 完成
                2. 计算每个proposal对应的定位索引   -- 完成
                3. 根据索引构建loss计算过程   -- 完成，但是现在回归loss有问题收敛不了
        Args:
            features:
            targets:
            image_sizes:
        Returns:
        """
        # 1. 计算FCOS loss
        # 2. 计算二次回归分类的loss
            # 1. 生成proposals  (当前设置256个proposal用作二次回归和分类)
            # 2. 计算每个proposal对应的定位索引
            # 3. 根据索引构建loss计算过程  目前有预测有GT
        # TODO 改变二阶段回归的方式

        fcos_features, two_pred = features
        two_cls, two_box, proposals = two_pred
        #
        gt_instances = [x["instances"].to(self.device) for x in targets]
        locations = fcos_features[-1]
        training_targets = self._get_ground_truth_back(locations, gt_instances)
        prop_targets = self._get_proposal_targets(proposals, gt_instances)
        # # Two loss
        two_loss = self.two_stage_fcos_head.losses(two_pred, prop_targets)
        # FCOS loss
        fcos_loss = self.losses(fcos_features, training_targets)
        fcos_loss.update(two_loss)
        return fcos_loss

    def losses(self, preds, training_targets):
        logits, bbox_reg, ctrness, top_feats, bbox_towers, locations = preds
        instances = Instances((0, 0))
        instances.labels = cat([x.reshape(-1) for x in training_targets['labels']], dim=0)
        instances.gt_inds = cat([x.reshape(-1) for x in training_targets['target_inds']], dim=0)
        instances.im_inds = cat([x.reshape(-1) for x in training_targets['im_inds']], dim=0)
        instances.reg_targets = cat([x.reshape(-1, 4) for x in training_targets['reg_targets']], dim=0)
        instances.locations = cat([x.reshape(-1, 2) for x in training_targets['locations']], dim=0)
        instances.fpn_levels = cat([x.reshape(-1) for x in training_targets['fpn_levels']], dim=0)

        instances.logits_pred = cat([x.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for x in logits], dim=0)
        instances.reg_pred = cat([x.permute(0, 2, 3, 1).reshape(-1, 4) for x in bbox_reg], dim=0)
        instances.ctrness_pred = cat([x.permute(0, 2, 3, 1).reshape(-1) for x in ctrness], dim=0)
        if len(top_feats) > 0:
            instances.top_feats = cat([x.permute(0, 2, 3, 1).reshape(-1, x.size(1)) for x in top_feats], dim=0)

        return self.loss_func(instances)

    def inference(self, preds, image_sizes, in_features=None):
        # 合并两阶段的预测框并做NMS
        # 1. 生成two 的预测框。
        # 2. 合并做NMS，得出最终的框。
        feats, two_pred = preds
        cls_pred, box_pred, proposals = two_pred

        boxes = proposals[0].pred_boxes.tensor
        det_whs = boxes[:, 2:] - boxes[:, :2]
        det_whs = torch.cat([det_whs, det_whs], dim=1)
        boxes = boxes + (box_pred[0] * det_whs * 0.5)
        cls_pred = F.softmax(cls_pred[0], dim=1)
        cls_score, cls = torch.max(cls_pred, dim=1)

        detections = Instances(image_sizes[0])
        detections.pred_boxes = Boxes(boxes)
        detections.scores = cls_score
        detections.pred_classes = cls
        # NMS
        detections = ml_nms(detections, 0.7)
        return [detections]

    # Other functions
    def _get_ground_truth(self, locations, gt_instances, proposal_indexes, proposal_strides):
        num_loc_list = [len(loc) for loc in locations]
        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[l], -1)
            )

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(locations, dim=0)

        training_targets = self._compute_targets_for_locations(
            locations, gt_instances, loc_to_size_range, num_loc_list
        )

        proposal_labels = [
            torch.index_select(training_targets['labels'][i], dim=0, index=proposal_indexes[i].long())
            for i in range(len(proposal_indexes))
        ]
        proposal_regs = [
            torch.index_select(training_targets['reg_targets'][i], dim=0, index=proposal_indexes[i].long()) / proposal_strides[i]
            for i in range(len(proposal_indexes))
        ]

        proposal_targets = {
            'proposal_labels': proposal_labels,
            'proposal_regs': proposal_regs
        }

        training_targets["locations"] = [locations.clone() for _ in range(len(gt_instances))]
        training_targets["im_inds"] = [
            locations.new_ones(locations.size(0), dtype=torch.long) * i for i in range(len(gt_instances))
        ]

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(v, num_loc_list) for k,v in training_targets.items()
        }

        training_targets["fpn_levels"] = [
            loc.new_ones(len(loc), dtype=torch.long) * level
            for level, loc in enumerate(training_targets["locations"])
        ]

        reg_targets = training_targets["reg_targets"]
        for l in range(len(reg_targets)):
            reg_targets[l] = reg_targets[l] / float(self.fpn_strides[l])
        return training_targets, proposal_targets

    def _get_ground_truth_back(self, locations, gt_instances):
        num_loc_list = [len(loc) for loc in locations]
        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[l], -1)
            )

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(locations, dim=0)

        training_targets = self._compute_targets_for_locations(
            locations, gt_instances, loc_to_size_range, num_loc_list
        )

        training_targets["locations"] = [locations.clone() for _ in range(len(gt_instances))]
        training_targets["im_inds"] = [
            locations.new_ones(locations.size(0), dtype=torch.long) * i for i in range(len(gt_instances))
        ]

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(v, num_loc_list) for k,v in training_targets.items()
        }

        training_targets["fpn_levels"] = [
            loc.new_ones(len(loc), dtype=torch.long) * level
            for level, loc in enumerate(training_targets["locations"])
        ]

        reg_targets = training_targets["reg_targets"]
        for l in range(len(reg_targets)):
            reg_targets[l] = reg_targets[l] / float(self.fpn_strides[l])
        return training_targets

    def _get_proposal_targets(self, proposals, gt_instances):
        num_images = len(gt_instances)
        proposal_cls_targets = []
        proposal_reg_targets = []

        for im_ind in range(num_images):
            proposal = proposals[im_ind]
            if len(proposal) == 0:
                reg_targets = torch.zeros_like(proposal.pred_boxes.tensor)
                two_gt_classes = torch.ones_like(proposal.pred_classes) * self.num_classes
                proposal_reg_targets.append(reg_targets)
                proposal_cls_targets.append(two_gt_classes)
                continue
            gt_target = gt_instances[im_ind]
            gt_boxes = gt_target.gt_boxes
            gt_classes = gt_target.gt_classes
            candidate_boxes = proposal.pred_boxes
            iou = pairwise_iou(candidate_boxes, gt_boxes)
            (max_iou, argmax_iou) = iou.max(dim=1)
            invalid = max_iou < 0.6
            two_gt_classes = gt_classes[argmax_iou]
            two_gt_classes[invalid] = self.num_classes

            gt_boxes = gt_boxes[argmax_iou].tensor
            candidate_boxes = candidate_boxes.tensor
            whs = candidate_boxes[:, 2:4] - candidate_boxes[:, 0:2]
            whs = torch.cat([whs, whs], dim=1)

            reg_targets = (gt_boxes - candidate_boxes) / (whs * 0.5)

            proposal_reg_targets.append(reg_targets)
            proposal_cls_targets.append(two_gt_classes)

        return {
            'cls_targets': proposal_cls_targets,
            'reg_targets': proposal_reg_targets
        }

    def _compute_targets_for_locations(self, locations, targets, size_ranges, num_loc_list):
        labels = []
        reg_targets = []
        target_inds = []

        xs, ys = locations[:, 0], locations[:, 1]

        num_targets = 0
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                target_inds.append(labels_per_im.new_zeros(locations.size(0)) - 1)
                continue

            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]

            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sample:
                if targets_per_im.has("gt_bitmasks_full"):
                    bitmasks = targets_per_im.gt_bitmaksks_full
                else:
                    bitmasks = None
                is_in_bboxes = self._get_sample_region(
                    bboxes, self.fpn_strides, num_loc_list, xs, ys,
                    bitmasks=bitmasks, radius=self.radius
                )
            else:
                is_in_bboxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = (max_reg_targets_per_im >= size_ranges[:, [0]]) & (max_reg_targets_per_im <= size_ranges[:, [1]])
            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_bboxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            target_inds_per_im = locations_to_gt_inds + num_targets
            num_targets += len(targets_per_im)

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes
            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            target_inds.append(target_inds_per_im)
        return {
            "labels": labels,
            "reg_targets": reg_targets,
            "target_inds": target_inds
        }

    def _get_sample_region(self, boxes, strides, num_loc_list, loc_xs, loc_ys, bitmasks=None, radius=1):
        if bitmasks is not None:
            _, h, w = bitmasks.size()
            ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
            xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

            m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
            m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
            center_x = m10 / m00
            center_y = m01 / m00
        else:
            center_x = boxes[..., [0, 2]].sum(dim=-1) * 0.5
            center_y = boxes[..., [1, 3]].sum(dim=-1) * 0.5
        num_gts = boxes.size(0)
        K = len(loc_xs)
        boxes = boxes[None].expand(K, num_gts, 4)
        center_x = center_x[None].expand(K, num_gts)
        center_y = center_y[None].expand(K, num_gts)
        center_gt = boxes.new_zeros(boxes.shape)

        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)

        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > boxes[beg:end, :, 0], xmin, boxes[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > boxes[beg:end, :, 1], ymin, boxes[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > boxes[beg:end, :, 2], boxes[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > boxes[beg:end, :, 3], boxes[beg:end, :, 3], ymax)
            beg=end
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), dim=-1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def _transpose(self, training_targets, num_loc_list):
        """
        Args:
            training_targets:
            num_loc_list:
        Returns:
        """
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0)
            )

        return targets_level_first

    def _computation_indexes_for_proposals(self, proposals, locations):
        loc_indexes = []
        loc_strides = []
        num_loc_list = [len(loc) for loc in locations]
        index_increament = torch.stack(
            [torch.tensor(num_loc_list[:i]).sum().float() for i in range(len(num_loc_list))]).cuda()
        max_locations = [location.max(dim=0)[0] for location in locations]
        ws_hs = torch.stack(max_locations)
        fpn_strides = torch.tensor(self.fpn_strides)[:, None].cuda()
        ws_hs = (ws_hs - fpn_strides / 2) / fpn_strides
        ws = ws_hs[:, 0]

        for i in range(len(proposals)):
            proposals_per_im = proposals[i]

            proposals_locations = proposals_per_im.locations
            fpn_levels = proposals_per_im.fpn_levels
            proposals_strides = fpn_strides[fpn_levels]

            proposals_locations = (proposals_locations - proposals_strides / 2) / proposals_strides
            proposals_locations = proposals_locations[:, 1][:, None] * (ws + 1) + proposals_locations[:, 0][:, None]
            proposals_locations = proposals_locations + index_increament
            proposals_locations = torch.diag(proposals_locations[:, fpn_levels])
            loc_indexes.append(proposals_locations)
            loc_strides.append(proposals_strides)
        return loc_indexes, loc_strides


class FCOSBoxHead(nn.Module):
    def __init__(self, cfg):
        super(FCOSBoxHead, self).__init__()
        self.in_features = cfg.MODEL.FCOS_HEADS.IN_FEATURES
        norm = cfg.MODEL.FCOS_HEADS.NORM
        in_channels = cfg.MODEL.FCOS_HEADS.FPN_CHANNELS
        self.num_classes = cfg.MODEL.FCOS_HEADS.NUM_CLASSES
        # cls convs
        num_cls_convs = cfg.MODEL.FCOS_HEADS.NUM_CLS_CONVS
        cls_deformable = cfg.MODEL.FCOS_HEADS.USE_DEFORMABLE
        self.cls_convs = self._init_convs(in_channels, num_cls_convs, norm, cls_deformable)

        # box convs
        num_bbox_convs = cfg.MODEL.FCOS_HEADS.NUM_BBOX_CONVS
        bbox_deformable = cfg.MODEL.FCOS_HEADS.USE_DEFORMABLE
        self.bbox_convs = self._init_convs(in_channels, num_bbox_convs, norm, bbox_deformable)

        # share convs
        num_share_convs = cfg.MODEL.FCOS_HEADS.NUM_SHARED_CONVS
        share_deformable = False
        self.share_convs = self._init_convs(in_channels, num_share_convs, norm, share_deformable)

        self.num_levels = len(self.in_features)
        self.fpn_strides = cfg.MODEL.FCOS_HEADS.FPN_STRIDES

        # classifier
        self.classifier = nn.Conv2d(
            in_channels, self.num_classes, kernel_size=3, padding=1, stride=1
        )

        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1, padding=1
        )

        self.ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        # initialize model weights
        for modules in [self.cls_convs, self.bbox_convs, self.share_convs, self.classifier, self.bbox_pred, self.ctrness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.normal_(l.weight, std=0.01)
                    nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS_HEADS.PRIOR_PROB
        bias_value = -math.log((1-prior_prob) / prior_prob)
        nn.init.constant_(self.classifier.bias, bias_value)

        if cfg.MODEL.FCOS_HEADS.USE_SCALE:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(self.num_levels)])
        else:
            self.scales = None

    def _init_convs(self, in_channels, num_convs, norm="BN", deformable=False):
        ops = []
        for i in range(num_convs):
            if deformable and i == num_convs - 1:
                conv = DeformableConv
            else:
                conv = nn.Conv2d
            ops.append(conv(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True))
            ops.append(get_norm(norm, in_channels))
            ops.append(nn.ReLU())
        return nn.Sequential(*ops)

    def _compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def forward(self, in_features):
        logits = []
        bbox_reg = []
        ctrness = []
        top_feats = []
        bbox_towers = []
        feats = [in_features[f] for f in self.in_features]
        locations = self._compute_locations(features=feats)

        for l, feature in enumerate(feats):
            feature = self.share_convs(feature)
            cls_tower = self.cls_convs(feature)
            bbox_tower = self.bbox_convs(feature)
            logits.append(self.classifier(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                reg = self.scales[l](reg)
            bbox_reg.append(F.relu(reg))

        return logits, bbox_reg, ctrness, top_feats, bbox_towers, locations


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


def fcos_head_builder(cfg, input_shape=None):
    head = FCOSHead(cfg)
    return head


if __name__ == '__main__':
    from configs.config import get_detection_config
    cfg = get_detection_config()

    model = FCOSHead(cfg)
    p3 = torch.randn(2, 256, 64, 64)
    p4 = torch.randn(2, 256, 32, 32)
    p5 = torch.randn(2, 256, 16, 16)
    p6 = torch.randn(2, 256, 8, 8)
    p7 = torch.randn(2, 256, 4, 4)

    features = {
        'p3': p3,
        'p4': p4,
        'p5': p5,
        'p6': p6,
        'p7': p7,
    }

    print(model)
    image_sizes = [(512, 512), (512, 512)]

    out = model(features)
    print(out[0][0].size())
    print(out[1][1].size())
    print(out[2][2].size())
    print(out[-1][0].size())

    detections = model.inference(out, image_sizes)
    print(detections)

