# Non-local Head for segmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import NonLocalLayer, EmbededNonLocalLayer
from . import SEG_HEAD_REGISTRY

from models.losses import get_loss_from_cfg


class NLHead(nn.Module):
    def __init__(self, cfg, norm_layer=None):
        super(NLHead, self).__init__()

        self.context = nn.Sequential(
            nn.Conv2d(2048, cfg.MODEL.HEAD.NL_INPUT, kernel_size=3, stride=1, padding=1),
            norm_layer(cfg.MODEL.HEAD.NL_INPUT),
            nn.ReLU(inplace=True),
        )
        # self.context = EmFusionNonLocalLayer(dim_in=512, dim_out=512, dim_inter=256, norm_layer=norm_layer)
        self.non_local = NonLocalLayer(dim_in=cfg.MODEL.HEAD.NL_INPUT, dim_out=cfg.MODEL.HEAD.NL_OUTPUT,
                                       dim_inter=cfg.MODEL.HEAD.NL_INTER, norm_layer=norm_layer)

        self.classifier = nn.Conv2d(cfg.MODEL.HEAD.NL_OUTPUT, cfg.MODEL.NUM_CLASSES, kernel_size=1, stride=1, padding=0, bias=False)

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
        x = self.context(data_input['res4'])
        x = self.non_local(x)

        pred = self.classifier(x)
        size = label.size()[-2:]
        if self.aux_loss:
            aux_pred = self.aux_classifier(data_input['res3'])
            aux_pred = F.interpolate(aux_pred, size, mode='bilinear', align_corners=True)
            pred = F.interpolate(pred, size, mode='bilinear', align_corners=True)
            return self._compute_loss((pred, aux_pred), label)
        else:
            pred = F.interpolate(pred, size, mode='bilinear', align_corners=True)
            return self._compute_loss(pred, label)

    def inference(self, data_input, size):
        x = self.context(data_input['res4'])
        x = self.non_local(x)
        pred = self.classifier(x)
        pred = F.interpolate(pred, size=size, mode='bilinear', align_corners=True)
        pred = F.softmax(pred, dim=1)
        pred = torch.max(pred, dim=1)[1]
        return pred

class NLCCNetHead(nn.Module):
    def __init__(self, cfg, norm_layer=None):
        super(NLCCNetHead, self).__init__()

        self.context = nn.Sequential(
            nn.Conv2d(2048, cfg.MODEL.HEAD.NL_INPUT, kernel_size=3, stride=1, padding=1),
            norm_layer(cfg.MODEL.HEAD.NL_INPUT),
        )
        self.non_local = NonLocalLayer(dim_in=cfg.MODEL.HEAD.NL_INPUT, dim_out=cfg.MODEL.HEAD.NL_OUTPUT,
                                       dim_inter=512, norm_layer=norm_layer)
        self.convb = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
                                   norm_layer(512))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2048 + 512, 512, kernel_size=3, padding=1, stride=1),
            norm_layer(512),
            nn.Dropout2d(0.1)
        )

        self.classifier = nn.Conv2d(cfg.MODEL.HEAD.NL_OUTPUT, cfg.MODEL.NUM_CLASSES, kernel_size=1, stride=1, padding=0, bias=False)

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
        x = self.context(data_input['res4'])
        x = self.non_local(x)
        x = self.convb(x)
        x = self.bottleneck(torch.cat([data_input['res4'], x], dim=1))
        pred = self.classifier(x)

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
def nlhead_builder(cfg):
    from layers import get_norm
    norm_layer = get_norm(cfg.MODEL.BN_LAYER)
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    return NLHead(cfg, norm_layer)

