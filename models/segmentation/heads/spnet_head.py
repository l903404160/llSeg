
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from . import SEG_HEAD_REGISTRY
from models.segmentation.segmods.spnetmods import PyramidPooling, StripPooling
from models.losses import get_loss_from_cfg

up_kwargs = {'mode': 'bilinear', 'align_corners': True}
class SPNet(nn.Module):
    def __init__(self, cfg, norm_layer=nn.BatchNorm2d):
        super(SPNet, self).__init__()
        self._up_kwargs = up_kwargs
        self.head = SPHead(2048, cfg.MODEL.NUM_CLASSES, norm_layer,self._up_kwargs)
        self.criterion = get_loss_from_cfg(cfg)
        self.aux_loss = cfg.MODEL.HEAD.AUX_LOSS
        if self.aux_loss:
            self.auxlayer = FCNHead(1024, cfg.MODEL.NUM_CLASSES, norm_layer,cfg.MODEL.SPNET.WITH_GLOBAL)

    def forward(self, x, y=None):
        c3 = x['res3']
        c4 = x['res4']

        x = self.head(c4)
        if y is None:
            return x
        h, w = y.size()[-2:]
        x = interpolate(x, (h, w),**self._up_kwargs)

        if self.aux_loss:
            aux = self.auxlayer(c3)
            aux = interpolate(aux, (h, w))
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            #loss = main_loss + aux_loss
            return {
                'loss': main_loss,
                'aux_loss': aux_loss
            }
        else:
            loss = self.criterion(x, y)
            return {
                'loss':loss
            }


class SPHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(SPHead, self).__init__()
        inter_channels = in_channels // 2
        self.trans_layer = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False),
                                         norm_layer(inter_channels),
                                         nn.ReLU(True)
                                         )
        self.strip_pool1 = StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)
        self.strip_pool2 = StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)
        self.score_layer = nn.Sequential(nn.Conv2d(inter_channels, inter_channels // 2, 3, 1, 1, bias=False),
                                         norm_layer(inter_channels // 2),
                                         nn.ReLU(True),
                                         nn.Dropout2d(0.1, False),
                                         nn.Conv2d(inter_channels // 2, out_channels, 1))

    def forward(self, x):
        x = self.trans_layer(x)
        x = self.strip_pool1(x)
        x = self.strip_pool2(x)
        x = self.score_layer(x)
        return x


class GlobalPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(GlobalPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return interpolate(pool, (h,w), **self._up_kwargs)


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, with_global):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self._up_kwargs = up_kwargs
        self.with_global = with_global
        if self.with_global:
            self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                       norm_layer(inter_channels),
                                       nn.ReLU(),
                                       GlobalPooling(inter_channels, inter_channels,
                                                     norm_layer),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(inter_channels, out_channels, 1))
        else:
            self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                        norm_layer(inter_channels),
                                        nn.ReLU(),
                                        nn.Dropout2d(0.1, False),
                                        nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)

@SEG_HEAD_REGISTRY.register()
def spnet_builder(cfg):
    from layers.batch_norm import get_norm
    norm_layer = get_norm(cfg.MODEL.BN_LAYER)
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    return SPNet(cfg, norm_layer)