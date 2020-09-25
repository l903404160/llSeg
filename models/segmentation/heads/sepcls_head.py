import torch
import torch.nn as nn
import torch.nn.functional as F
from . import SEG_HEAD_REGISTRY

from models.losses import get_loss_from_cfg


class SepClsHead(nn.Module):
    def __init__(self, cfg, norm_layer=None):
        super(SepClsHead, self).__init__()

        self.sep_small_classes = cfg.MODEL.HEAD.SEP_SMALL_CLASSES
        self.sep_big_classes = cfg.MODEL.HEAD.SEP_BIG_CLASSES
        self.num_classes = cfg.MODEL.NUM_CLASSES

        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, cfg.MODEL.NUM_CLASSES + 1, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.sep_class_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, cfg.MODEL.NUM_CLASSES + 1, kernel_size=1, stride=1, padding=0, bias=True)
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

    def _compute_losses(self, pred, sep_cls_pred, label, aux_pred=None):

        size = label.size()[-2:]
        pred = F.interpolate(pred, size, mode='bilinear', align_corners=True)
        small_pred = F.interpolate(sep_cls_pred, size, mode='bilinear', align_corners=True)

        small_mask = torch.zeros_like(label, device=label.device)
        for cls in self.sep_small_classes:
            small_mask[label == cls] = 1
        small_label = label * small_mask
        small_label[small_label == 0] = self.num_classes

        big_mask = torch.zeros_like(label, device=label.device)
        for cls in self.sep_big_classes:
            big_mask[label == cls] = 1
        big_label = label * big_mask
        big_label[big_label == 0] = self.num_classes

        if self.aux_loss:
            loss = self.loss_fn(pred, big_label.long())
            aux_pred = nn.functional.interpolate(aux_pred, size=size, mode='bilinear', align_corners=True)
            aux_loss = self.loss_fn(aux_pred, label)

            return {
                'loss': loss,
                'sep_cls_loss': self.loss_fn(small_pred, small_label.long()),
                'aux_loss': aux_loss * self.aux_weight,
            }
        else:
            return {
                'loss': self.loss_fn(pred, big_label.long()),
                'sep_cls_loss': self.loss_fn(small_pred, small_label.long())
            }

    def forward(self, data_input, label=None):

        res4 = data_input['res4']
        pred = self.classifier(res4)
        sep_cls_pred = self.sep_class_head(res4)

        if self.aux_loss:
            aux_pred = self.aux_classifier(data_input['res3'])
            return self._compute_losses(pred, sep_cls_pred, label, aux_pred)
        else:
            return self._compute_losses(pred, sep_cls_pred, label)

    def inference(self, data_input, size):
        res4 = data_input['res4']
        pred = self.classifier(res4)
        sep_cls_pred = self.sep_class_head(res4)

        pred = F.interpolate(pred, size=size, mode='bilinear', align_corners=True)
        pred = F.softmax(pred, dim=1)
        temp_pa = torch.max(pred, dim=1)[1]

        sep_cls_pred = F.interpolate(sep_cls_pred, size=size, mode='bilinear', align_corners=True)
        sep_cls_pred = F.softmax(sep_cls_pred, dim=1)
        temp_pb = torch.max(sep_cls_pred, dim=1)[1]
        return temp_pa

@SEG_HEAD_REGISTRY.register()
def sepclshead_builder(cfg):
    from layers.batch_norm import get_norm
    norm_layer = get_norm(cfg.MODEL.BN_LAYER)
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    return SepClsHead(cfg, norm_layer)

