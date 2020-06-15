import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nn.smoothL1Loss import smooth_l1_loss
from utils.nn.focal_loss import sigmoid_focal_loss_jit

from layers import ShapeSpec, batched_nms, cat
from structures import Boxes, ImageList, Instances, pairwise_iou
from utils.events import get_event_storage

from models.detection.modules.anchor_generator import build_anchor_generator
from models.detection.backbone import backbone_builder
from models.detection.modules.box_regression import Box2BoxTransform
from models.detection.modules.matcher import Matcher
from models.detection.modules.postprocessing import detector_postprocess

__all__ = ['RetinaNet']


def permute_to_N_HWA_K(tensor:torch.Tensor, K):
    """
    Transpose / reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    Args:
        tensor:
        K:

    Returns:

    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K) # Size = (N, HWA, K)
    return tensor


class RetinaNet(nn.Module):
    """
    Implementation of RetinaNet
    """
    def __init__(self, cfg):
        super(RetinaNet, self).__init__()
        self.num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        self.in_features = cfg.MODEL.RETINANET.IN_FEATURES

        # Loss Parameters:
        self.focal_loss_alpha = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA

        # Inference parameters:
        self.score_threshold = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.RETINANET.NMS_THRESH_TEST
        self.max_detections_pre_image = cfg.TEST.DETECTIONS_PER_IMAGE

        # No Vis parameters for fast training

        self.backbone = backbone_builder(cfg)
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        # TODO RetinaNetHEAD
        self.head = RetinaHead(cfg, feature_shapes)
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)

        # Mathing and loss
        self.box2bbox_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_mather = Matcher(
            cfg.MODEL.RETINANET.IOU_THRESHOLDS,
            cfg.MODEL.RETINANET.IOU_LABELS,
            allow_low_quality_matches=True
        )

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        self.loss_normalizer = 100
        self.loss_normalizer_momentum = 0.9

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """

        Args:
            batched_inputs: a list, batched outputs of :class:'DatasetMapper'
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                image: Tensor, image in (C,H,W) format
                instances: Instances

                other information that's included in the original dicts, such as :

                height, width: the output resolution of the model, used in inference


        Returns:
            dict [str: Tensor]
                mapping from a named loss to a tensor stroing the loss. Used during training only
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]

        anchors = self.anchor_generator(features)
        # TODO head
        pred_logits, pred_anchor_deltas = self.head(features)

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are issing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)

            losses = self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes)

            return losses
        else:
            results = self.inference(anchors, pred_logits, pred_anchor_deltas, images.image_size)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_size):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes):
        """

        Args:
            anchors:
            pred_logits:
            gt_labels:
            pred_anchor_deltas:
            gt_boxes:

        Returns:

        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels) # N,R
        anchors = type(anchors[0]).cat(anchors).tensor  # R, 4
        gt_anchor_deltas = [self.box2bbox_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # N, R, 4

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (1 - self.loss_normalizer_momentum) * max(num_pos_anchors, 1)

        # classification and regission loss
        gt_labels_target = F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[:, :-1]
        loss_cls = sigmoid_focal_loss_jit(
            cat(pred_logits, dim=1)[valid_mask],
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum"
        )

        loss_box_reg = smooth_l1_loss(
            cat(pred_anchor_deltas, dim=1)[pos_mask],
            gt_anchor_deltas[pos_mask],
            beta=self.smooth_l1_loss_beta,
            reduction="sum"
        )
        return {
            "loss_cls": loss_cls / self.loss_normalizer,
            "loss_box_reg": loss_box_reg / self.loss_normalizer
        }

    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        """

        Args:
            anchors:
            gt_instances:

        Returns:

        """
        anchors = Boxes.cat(anchors) # R 4

        gt_labels = []
        matched_gt_boxes = []
        for gt_per_image in gt_instances:
            match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)
            matched_idxs, anchor_labels = self.anchor_mather(match_quality_matrix)
            del match_quality_matrix

            if len(gt_per_image) > 0:
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]
                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_labels_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_labels_i[anchor_labels == -1] = -1
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes

    def inference(self, anchors, pred_logits, pred_anchor_deltas, image_size):
        """

        Args:
            anchors:
            pred_logits:
            pred_anchor_deltas:
            image_size:

        Returns:

        """
        results = []
        for img_idx, image_size in enumerate(image_size):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            results_per_image = self.inference_single_image(anchors, pred_logits_per_image, deltas_per_image, tuple(image_size))
            results.append(results_per_image)

        return results

    def inference_single_image(self, anchors, box_cls, box_delta, image_size):
        """
        Single-image inference
        Args:
            anchors:
            box_cls:
            box_delta:
            image_size:

        Returns:

        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (H W A K)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k scoreing indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk(at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[: num_topk]

            # Filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            # ?????
            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2bbox_transform.apply_deltas(box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class RetinaHead(nn.Module):
    """
        The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super(RetinaHead, self).__init__()
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        num_convs = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors

        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported"
        num_anchors = num_anchors[0]

        cls_subnet = []
        bbox_subnet = []

        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)

        self.cls_score = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors*4, kernel_size=3, stride=1, padding=1
        )
        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg