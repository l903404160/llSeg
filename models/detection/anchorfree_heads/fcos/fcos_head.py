import torch
import math
import torch.nn.functional as F

from layers import ShapeSpec, DeformableConv, NaiveSyncBatchNorm, NaiveGroupNorm
from typing import List, Dict
import torch.nn as nn

from .fcos_tools import compute_locations
from .fcos_output import FCOSOutputs
from models.detection.anchorfree_heads import DET_ANCHORFREE_HEADS_REGISRY

from models.detection.anchorfree_heads.search_head.model_head import SearchFCOSHead, SimFCOSHead

INF = 100000000


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


class FCOSAnchorFreeHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # TODO change this configs
        self.in_features = cfg.MODEL.FCOS_HEADS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS_HEADS.FPN_STRIDES
        self.yield_proposal = cfg.MODEL.FCOS_HEADS.YIELD_PROPOSAL

        #Debug
        self.search_head = False

        if self.search_head:
            self.fcos_head = SearchFCOSHead()
        else:
            self.fcos_head = FCOSHead(cfg, [input_shape[f] for f in self.in_features])

        self.in_channels_to_top_module = self.fcos_head.in_channels_to_top_module

        self.fcos_outputs = FCOSOutputs(cfg)

    def forward_head(self, features, top_module=None):
        features = [features[f] for f in self.in_features]
        pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers = self.fcos_head(
            features, top_module, self.yield_proposal)
        return pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers

    def forward(self, images, features, gt_instances=None, top_module=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        features = [features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        if self.search_head:
            logits_pred, reg_pred, ctrness_pred, top_feats, bbox_towers = self.fcos_head(features)
        else:
            logits_pred, reg_pred, ctrness_pred, top_feats, bbox_towers = self.fcos_head(
                features, top_module, self.yield_proposal
            )

        results = {}
        if self.yield_proposal:
            results["features"] = {
                f: b for f, b in zip(self.in_features, bbox_towers)
            }

        if self.training:
            results, losses = self.fcos_outputs.losses(
                logits_pred, reg_pred, ctrness_pred,
                locations, gt_instances, top_feats
            )

            if self.yield_proposal:
                with torch.no_grad():
                    results["proposals"] = self.fcos_outputs.predict_proposals(
                        logits_pred, reg_pred, ctrness_pred,
                        locations, images.image_sizes, top_feats
                    )
            return results, losses
        else:
            results = self.fcos_outputs.predict_proposals(
                logits_pred, reg_pred, ctrness_pred,
                locations, images.image_sizes, top_feats
            )

            return results, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations


class FCOSHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: change this configs
        self.num_classes = cfg.MODEL.FCOS_HEADS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS_HEADS.FPN_STRIDES
        head_configs = {"cls": (cfg.MODEL.FCOS_HEADS.NUM_CLS_CONVS,
                                cfg.MODEL.FCOS_HEADS.USE_DEFORMABLE),
                        "bbox": (cfg.MODEL.FCOS_HEADS.NUM_BBOX_CONVS,
                                 cfg.MODEL.FCOS_HEADS.USE_DEFORMABLE),
                        "share": (cfg.MODEL.FCOS_HEADS.NUM_SHARED_CONVS,
                                  False)}
        norm = None if cfg.MODEL.FCOS_HEADS.NORM == "none" else cfg.MODEL.FCOS_HEADS.NORM
        self.num_levels = len(input_shape)

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        self.in_channels_to_top_module = in_channels

        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            for i in range(num_convs):
                if use_deformable and i == num_convs - 1:
                    conv_func = DeformableConv
                else:
                    conv_func = nn.Conv2d
                tower.append(conv_func(
                    in_channels, in_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=True
                ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, in_channels))
                elif norm == "NaiveGN":
                    tower.append(NaiveGroupNorm(32, in_channels))
                elif norm == "BN":
                    tower.append(ModuleListDial([
                        nn.BatchNorm2d(in_channels) for _ in range(self.num_levels)
                    ]))
                elif norm == "SyncBN":
                    tower.append(ModuleListDial([
                        NaiveSyncBatchNorm(in_channels) for _ in range(self.num_levels)
                    ]))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        # self.cls_shape_conv = ShapeConvs()
        # self.box_shape_conv = ShapeConvs()

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes,
            kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3,
            stride=1, padding=1
        )
        self.ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3,
            stride=1, padding=1
        )

        if cfg.MODEL.FCOS_HEADS.USE_SCALE:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(self.num_levels)])
        else:
            self.scales = None

        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower, self.cls_logits,
            self.bbox_pred, self.ctrness
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS_HEADS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x, top_module=None, yield_bbox_towers=False):
        logits = []
        bbox_reg = []
        ctrness = []
        top_feats = []
        bbox_towers = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            # cls_tower = self.cls_shape_conv(cls_tower)
            bbox_tower = self.bbox_tower(feature)
            # bbox_tower = self.box_shape_conv(bbox_tower)

            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)
            logits.append(self.cls_logits(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)

            if self.scales is not None:
                reg = self.scales[l](reg)
            bbox_reg.append(F.relu(reg))

            if top_module is not None:
                top_feats.append(top_module(bbox_tower))
                # top_feats.append(top_module(cls_tower))
        return logits, bbox_reg, ctrness, top_feats, bbox_towers


class ShapeConvs(nn.Module):
    def __init__(self):
        super(ShapeConvs, self).__init__()
        self.conv_tb = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3,1), stride=2, padding=(1, 0), dilation=1),
            nn.GroupNorm(16, 128),
            nn.ReLU()
        )
        self.conv_md = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(1,1), stride=1, dilation=1),
            nn.GroupNorm(16, 128),
            nn.ReLU()
        )
        self.conv_rl = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(1,3), stride=2, padding=(0, 1), dilation=1),
            nn.GroupNorm(16, 128),
            nn.ReLU()
        )

        self.aggre = nn.Sequential(
            nn.Conv2d(128*3, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU()
        )

    def forward(self, x):
        _,_, h,w = x.size()
        tb = F.interpolate(self.conv_tb(x), size=(h, w), mode='bilinear', align_corners=True)
        rl = F.interpolate(self.conv_rl(x), size=(h, w), mode='bilinear', align_corners=True)
        md = self.conv_md(x)
        out = self.aggre(torch.cat([tb, md, rl], dim=1))
        return out


@DET_ANCHORFREE_HEADS_REGISRY.register()
def fcos_head_builder(cfg, input_shape=None):
    head = FCOSAnchorFreeHead(cfg, input_shape)
    return head


