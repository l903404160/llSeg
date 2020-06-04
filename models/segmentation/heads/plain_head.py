import torch.nn as nn
import torch.nn.functional as F
from . import SEG_HEAD_REGISTRY

from models.losses import get_loss_from_cfg


class PlainHead(nn.Module):
    def __init__(self, cfg, norm_layer=None):
        super(PlainHead, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, cfg.MODEL.NUM_CLASSES, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.loss_fn = get_loss_from_cfg(cfg)

        self.aux_loss = False
        if cfg.MODEL.HEAD.AUX_LOSS:
            self.aux_classifier = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                norm_layer(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.05),
                nn.Conv2d(512, cfg.MODEL.NUM_CLASSES, kernel_size=1, stride=1, padding=0, bias=True)
            )
            self.aux_loss = True
            self.aux_weight = cfg.MODEL.HEAD.AUX_LOSS_WEIGHT

    def _compute_loss(self, pred, label):
        if self.aux_loss:
            pred, aux_pred = pred
            loss = self.loss_fn(pred, label)
            aux_loss = self.loss_fn(aux_pred, label)
            return loss + self.aux_weight * aux_loss
        else:
            loss = self.loss_fn(pred, label)
            return loss

    def forward(self, data_input, label=None):

        res4 = data_input['res4']
        pred = self.classifier(res4)

        if label is None:
            return pred
        size = label.size()[-2:]

        if self.aux_loss:
            aux_pred = self.aux_classifier(data_input['res3'])
            aux_pred = F.interpolate(aux_pred, size, mode='bilinear', align_corners=True)
            pred = F.interpolate(pred, size, mode='bilinear', align_corners=True)
            return self._compute_loss((pred, aux_pred), label)
        else:
            pred = F.interpolate(pred, size, mode='bilinear', align_corners=True)
            return self._compute_loss(pred, label)


@SEG_HEAD_REGISTRY.register()
def plainhead_builder(cfg):
    from layers import get_norm
    norm_layer = get_norm(cfg.MODEL.BN_LAYER)
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    return PlainHead(cfg, norm_layer)

