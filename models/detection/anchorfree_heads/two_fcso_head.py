import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.nn.fcos_loss import IOULoss
from layers import ROIAlign
from utils.nn.smoothL1Loss import smooth_l1_loss


class TwoFCOSP3Head(nn.Module):
    def __init__(self, cfg):
        super(TwoFCOSP3Head, self).__init__()

        self.in_features = cfg.MODEL.FCOS_HEADS.TWO_IN_FEATURES

        self.out_shape = cfg.MODEL.FCOS_HEADS.TWO_OUT_SHAPE
        self.scale = cfg.MODEL.FCOS_HEADS.SCALES
        self.smapling_ratio = cfg.MODEL.FCOS_HEADS.TWO_SAMPLING_RATIO

        self.fc_dim = cfg.MODEL.FCOS_HEADS.TWO_FC_DIM
        self.num_classes = cfg.MODEL.FCOS_HEADS.NUM_CLASSES
        self.reg_iou_type = cfg.MODEL.FCOS_HEADS.TWO_REG_IOU_TYPE

        self.roi_aligns = nn.ModuleList([
            ROIAlign(self.out_shape, spatial_scale=self.scale[i], sampling_ratio=self.smapling_ratio, aligned=True)
            for i in range(len(self.in_features))
        ])
        self.two_cls_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, self.fc_dim),
            nn.ReLU(),
            nn.Linear(self.fc_dim, self.fc_dim),
            nn.ReLU(),
        )
        self.two_box_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, self.fc_dim),
            nn.ReLU(),
            nn.Linear(self.fc_dim, self.fc_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_dim, self.num_classes)
        )
        self.box_pred = nn.Sequential(
            nn.Linear(self.fc_dim, 4),
        )

    def forward(self, features, boxes):
        """
        Args:
            features: dict("p3": NCHW, ... )
            boxes: (N, 5) (index, xyxy)
        Returns:
        """
        feats = [features[f] for f in self.in_features]
        cls_feats, box_feats = [], []
        for i, feat in enumerate(feats):
            roi_results = torch.flatten(self.roi_aligns[i](feat, boxes), start_dim=1)
            cls_pred = self.classifier(self.two_cls_head(roi_results))
            box_pred = self.box_pred(self.two_box_head(roi_results))
            cls_feats.append(cls_pred)
            box_feats.append(box_pred)
        return cls_feats, box_feats

    def losses(self, preds, targets):
        cls_pred, box_pred, _ = preds
        num_images = len(cls_pred)
        cls_loss = 0
        reg_loss = 0
        for i in range(num_images):
            cls_prediction = cls_pred[i]
            box_prediction = box_pred[i]

            cls_target = targets['cls_targets'][i]
            box_target = targets['reg_targets'][i]

            reg_mask = cls_target != 80
            box_prediction = box_prediction[reg_mask, :]
            box_target = box_target[reg_mask, :]

            num_postive = reg_mask.float().sum()
            if num_postive == 0:
                cls_loss += 0 * cls_prediction.sum()
                reg_loss += 0 * box_prediction.sum()
            else:
                cls_loss += F.cross_entropy(cls_prediction, cls_target, ignore_index=80, reduction='sum') / cls_target.numel() / num_images
                reg_loss += smooth_l1_loss(box_prediction, box_target, beta=0, reduction='sum') / cls_target.numel() / num_images

        return {
            'two_cls_loss': cls_loss,
            'two_reg_loss': reg_loss
        }
