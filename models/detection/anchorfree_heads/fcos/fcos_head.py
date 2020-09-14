import torch

from layers import ShapeSpec
from typing import List, Dict
import torch.nn as nn

from .fcos_tools import compute_locations
from .fcos_output import FCOSOutputs
from models.detection.anchorfree_heads import DET_ANCHORFREE_HEADS_REGISRY
from models.detection.anchorfree_heads.fcos.fcos_predictor import FCOSHead
from models.detection.anchorfree_heads.fcos.fcos_tools import SearchFPN, BalancedFeaturePyramids
from models.detection.anchorfree_heads.search_head.model_head import SearchFCOSHead

INF = 100000000


class FCOSAnchorFreeHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # TODO change this configs
        self.in_features = cfg.MODEL.FCOS_HEADS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS_HEADS.FPN_STRIDES
        self.yield_proposal = cfg.MODEL.FCOS_HEADS.YIELD_PROPOSAL

        # self.searchfpn1 = SearchFPN(dim_in=256, dim_mid=256, dim_out=256)
        # self.searchfpn2 = SearchFPN(dim_in=256, dim_mid=256, dim_out=256)
        # self.bfp = BalancedFeaturePyramids(dim_in=256, dim_mid=256, dim_out=256)

        # self.fcos_head = FCOSHead(cfg, [input_shape[f] for f in self.in_features])

        # use searched head
        self.fcos_head = SearchFCOSHead(dim_in=256, dim_mid=128, dim_out=256)

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
        # features = self.searchfpn1(features)
        # features = self.searchfpn2(features)

        features = [features[f] for f in self.in_features]

        # features = self.bfp(features)

        locations = self.compute_locations(features)
        # logits_pred, reg_pred, ctrness_pred, top_feats, bbox_towers = self.fcos_head(
        #     features, top_module, self.yield_proposal
        # )
        logits_pred, reg_pred, ctrness_pred, top_feats, bbox_towers = self.fcos_head(features)

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


@DET_ANCHORFREE_HEADS_REGISRY.register()
def fcos_head_builder(cfg, input_shape=None):
    head = FCOSAnchorFreeHead(cfg, input_shape)
    return head

