# CornerNet Head
import torch
import torch.nn as nn

from layers import TopLeftCornerPooling, BottomRightCornerPooling
from models.detection.anchorfree_heads import DET_ANCHORFREE_HEADS_REGISRY

from utils.nn.anchorfree_focal_loss import CornerNetFocalLoss

from layers import batched_nms, batched_soft_nms
from structures import Instances, Boxes, BoxMode

from .anchorfree_heads import AnchorFreeHeadBase


class CornerNetHead(AnchorFreeHeadBase):
    def __init__(self, cfg, input_shape=None):
        super(CornerNetHead, self).__init__()
        # pool and classifier
        self.nstack = cfg.MODEL.HOURGLASS.NSTACK
        self.channels = cfg.MODEL.HOURGLASS.CNV_DIM
        self.num_classes = cfg.MODEL.ANCHORFREE_HEADS.NUM_CLASSES

        # NMS config
        self.nms_thresh = cfg.MODEL.ANCHORFREE_HEADS.NMS_THRESH
        self.soft_nms_enabled = cfg.MODEL.ANCHORFREE_HEADS.SOFT_NMS
        self.soft_nms_thresh = cfg.MODEL.ANCHORFREE_HEADS.SOFT_NMS_THRESH
        self.soft_nms_method = cfg.MODEL.ANCHORFREE_HEADS.SOFT_NMS_METHOD
        self.soft_nms_sigma = cfg.MODEL.ANCHORFREE_HEADS.SOFT_NMS_SIGMA
        self.soft_nms_prune = cfg.MODEL.ANCHORFREE_HEADS.SOFT_NMS_PRUNE

        self.K = cfg.MODEL.ANCHORFREE_HEADS.K
        self.kernel = cfg.MODEL.ANCHORFREE_HEADS.MAX_KERNEL
        self.ae_threshold = cfg.MODEL.ANCHORFREE_HEADS.AE_THRESH
        self.num_dets = cfg.MODEL.ANCHORFREE_HEADS.NUM_DETS

        self.loss_func = self._init_loss(cfg)

        self.tl_cnvs = nn.ModuleList(
            [self._make_tl_layer(self.channels) for _ in range(self.nstack)]
        )

        self.br_cnvs = nn.ModuleList(
            [self._make_br_layer(self.channels) for _ in range(self.nstack)]
        )

        # Keypoint Heatmaps
        self.tl_heats = nn.ModuleList(
            [self._make_tag_layer(in_channels=self.channels, inter_channels=self.channels, out_channels=self.num_classes) for _ in range(self.nstack)]
        )

        self.br_heats = nn.ModuleList(
            [self._make_tag_layer(in_channels=self.channels, inter_channels=self.channels, out_channels=self.num_classes) for _ in range(self.nstack)]
        )

        # tags
        self.tl_tags = nn.ModuleList(
            [self._make_tag_layer(in_channels=self.channels, inter_channels=self.channels, out_channels=1) for _ in range(self.nstack)]
        )

        self.br_tags = nn.ModuleList(
            [self._make_tag_layer(in_channels=self.channels, inter_channels=self.channels, out_channels=1) for _ in range(self.nstack)]
        )

        # offsets
        self.tl_regrs = nn.ModuleList(
            [self._make_tag_layer(in_channels=self.channels, inter_channels=self.channels, out_channels=2) for _ in range(self.nstack)]
        )

        self.br_regrs = nn.ModuleList(
            [self._make_tag_layer(in_channels=self.channels, inter_channels=self.channels, out_channels=2) for _ in range(self.nstack)]
        )

        # initialize weight
        for tl_heat, br_heat in zip(self.tl_heats, self.br_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)

    def _make_tl_layer(self, in_channels):
        return TopLeftCornerPooling(in_channels)

    def _make_br_layer(self, in_channels):
        return BottomRightCornerPooling(in_channels)

    def _make_tag_layer(self, in_channels, inter_channels, out_channels):
        # with_bn = False
        return nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1)
        )

    def _init_loss(self, cfg):
        return CornerNetFocalLoss(cfg)

    def forward(self, in_features):
        # need improvement
        outputs = {}
        if self.training:
            # hg1
            tl_cnv_a = self.tl_cnvs[0](in_features['hg1'])
            br_cnv_a = self.br_cnvs[0](in_features['hg1'])

            tl_heat_a, br_heat_a = self.tl_heats[0](tl_cnv_a), self.br_heats[0](br_cnv_a)
            tl_tag_a, br_tag_a = self.tl_tags[0](tl_cnv_a), self.br_tags[0](br_cnv_a)
            tl_regr_a, br_regr_a = self.tl_regrs[0](tl_cnv_a), self.br_regrs[0](br_cnv_a)
            outputs['hg1'] = [tl_heat_a, br_heat_a, tl_tag_a, br_tag_a, tl_regr_a, br_regr_a]
        # hg2
        tl_cnv_b = self.tl_cnvs[1](in_features['hg2'])
        br_cnv_b = self.br_cnvs[1](in_features['hg2'])

        tl_heat_b, br_heat_b = self.tl_heats[1](tl_cnv_b), self.br_heats[1](br_cnv_b)
        tl_tag_b, br_tag_b = self.tl_tags[1](tl_cnv_b), self.br_tags[1](br_cnv_b)
        tl_regr_b, br_regr_b = self.tl_regrs[1](tl_cnv_b), self.br_regrs[1](br_cnv_b)
        outputs['hg2'] = [tl_heat_b, br_heat_b, tl_tag_b, br_tag_b, tl_regr_b, br_regr_b]
        return outputs

    def forward_bbox(self, features, targets):
        tl_inds = torch.stack([targets[i]['tl_tag'] for i in range(len(targets))], dim=0).cuda()
        br_inds = torch.stack([targets[i]['br_tag'] for i in range(len(targets))], dim=0).cuda()

        loss = {}

        for key, value in features.items():
            name = key + '_loss'
            tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr = value
            # transpose and gather features
            tl_heat, br_heat = self._sigmoid_func(tl_heat), self._sigmoid_func(br_heat)
            tl_tag = self._transpose_and_gather_feature(tl_tag, tl_inds)
            br_tag = self._transpose_and_gather_feature(br_tag, br_inds)
            tl_regr = self._transpose_and_gather_feature(tl_regr, tl_inds)
            br_regr = self._transpose_and_gather_feature(br_regr, br_inds)
            preds = [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]
            loss[name] = self.losses(preds, targets) / 2
        return loss

    def inference(self, features, image_sizes):
        bboxes, scores, clses = self._decode_predictions(features)
        predictions = self._post_process_bboxes(bboxes, scores, clses, image_sizes)
        return predictions

    def losses(self, preds, targets):
        loss = self.loss_func(preds, targets)
        return loss

    def _transpose_and_gather_feature(self, features, inds):
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(features.size(0), -1, features.size(3))
        features = self._gather_features(features, inds)
        return features

    @staticmethod
    def _gather_features(features, inds, mask=None):
        dim = features.size(2)
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
        features = features.gather(1, inds)

        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(features)
            features = features[mask]
            features = features.view(-1, dim)
        return features

    @staticmethod
    def _sigmoid_func(features):
        features = torch.clamp(features.sigmoid_(), min=1e-4, max=1-1e-4)
        return features

    @staticmethod
    def _nms(heatmap, kernel=1):
        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(heatmap, kernel, stride=1, padding=pad)
        keep = (hmax == heatmap).float()
        return heatmap * keep

    @staticmethod
    def _topk(scores, K=20):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

        topk_clses = (topk_inds / (height * width)).int()
        topk_inds = topk_inds % (height * width)

        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    def _decode_predictions(self, features):
        # TODO change to single image inference
        tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr = features['hg2']
        tl_heat, br_heat = self._sigmoid_func(tl_heat), self._sigmoid_func(br_heat)

        batch, cat, height, width = tl_heat.size()

        # perform nms on heatmaps
        tl_heat = self._nms(tl_heat, kernel=self.kernel)
        br_heat = self._nms(br_heat, kernel=self.kernel)

        tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = self._topk(tl_heat, K=self.K)
        br_scores, br_inds, br_clses, br_ys, br_xs = self._topk(br_heat, K=self.K)

        tl_ys = tl_ys.view(batch, self.K, 1).expand(batch, self.K, self.K)
        tl_xs = tl_xs.view(batch, self.K, 1).expand(batch, self.K, self.K)

        br_ys = br_ys.view(batch, 1, self.K).expand(batch, self.K, self.K)
        br_xs = br_xs.view(batch, 1, self.K).expand(batch, self.K, self.K)

        if tl_regr is not None and br_regr is not None:
            tl_regr = self._transpose_and_gather_feature(tl_regr, tl_inds)
            tl_regr = tl_regr.view(batch, self.K, 1, 2)
            br_regr = self._transpose_and_gather_feature(br_regr, br_inds)
            br_regr = br_regr.view(batch, 1, self.K, 2)

            tl_xs = tl_xs + tl_regr[..., 0]
            tl_ys = tl_ys + tl_regr[..., 1]
            br_xs = br_xs + br_regr[..., 0]
            br_ys = br_ys + br_regr[..., 1]

        # all possible boxes based on top k corners ( ignoring class)
        bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

        tl_tag = self._transpose_and_gather_feature(tl_tag, tl_inds)
        tl_tag = tl_tag.view(batch, self.K, 1)
        br_tag = self._transpose_and_gather_feature(br_tag, br_inds)
        br_tag = br_tag.view(batch, 1, self.K)
        dists = torch.abs(tl_tag - br_tag)

        tl_scores = tl_scores.view(batch, self.K, 1).expand(batch, self.K, self.K)
        br_scores = br_scores.view(batch, 1, self.K).expand(batch, self.K, self.K)
        scores = (tl_scores + br_scores) / 2

        # reject bboxes based on classes
        tl_clses = tl_clses.view(batch, self.K, 1).expand(batch, self.K, self.K)
        br_clses = br_clses.view(batch, 1, self.K).expand(batch, self.K, self.K)
        cls_inds = (tl_clses != br_clses)

        # reject bboxes based on distances
        dist_inds = (dists > self.ae_threshold)
        #reject bboses based on widths and heights
        width_inds = (br_xs < tl_xs)
        height_inds = (br_ys < tl_ys)

        scores[cls_inds] = -1
        scores[dist_inds] = -1
        scores[width_inds] = -1
        scores[height_inds] = -1

        scores = scores.view(batch, -1)
        scores, inds = torch.topk(scores, self.num_dets)

        score_mask = scores > 0
        inds = inds[score_mask].unsqueeze(0)
        scores = scores[score_mask].squeeze()

        bboxes = bboxes.view(batch, -1, 4)
        bboxes = self._gather_features(bboxes, inds)
        bboxes = bboxes.squeeze()

        clses = tl_clses.contiguous().view(batch, -1, 1)
        clses = self._gather_features(clses, inds).float()
        clses = clses.squeeze()

        return bboxes, scores, clses

    # Perform NMS on the final predictions
    def _post_process_bboxes(self, bboxes, scores, classes, image_sizes):
        """
            1. Reject the bboxes whose score smaller than 0
            2. NMS for detections
            3. Generate Instances structure predictions
        Args:
            detections:
        Returns:

        """
        #TODO change these config to file
        nms_thresh = 0.5
        soft_nms_enabled = True
        soft_nms_method = "linear"
        soft_nms_sigma = 0.5
        soft_nms_prune = 0.001

        image_shape = image_sizes[0] # only have one image

        if not soft_nms_enabled:
            keep = batched_nms(bboxes, scores, classes, nms_thresh)
        else:
            keep, soft_nms_scores = batched_soft_nms(
                bboxes,
                scores,
                classes,
                soft_nms_method,
                soft_nms_sigma,
                nms_thresh,
                soft_nms_prune,
            )
            scores[keep] = soft_nms_scores
        if self.num_dets > 0:
            keep = keep[:self.num_dets]
        bboxes, scores, classes = bboxes[keep], scores[keep], classes[keep]
        # TODO change the ratio to config file
        bboxes = bboxes * 4

        results = Instances(image_size=image_shape)
        results.pred_boxes = Boxes(bboxes)
        results.pred_boxes.clip(image_shape)
        results.scores = scores
        results.pred_classes = classes
        return results


@DET_ANCHORFREE_HEADS_REGISRY.register()
def cornernet_head_builder(cfg):
    corner_head = CornerNetHead(cfg)
    return corner_head


