import torch
import torch.nn as nn
import torch.nn.functional as F
from . import SEG_HEAD_REGISTRY

class RtHighFeatureFusion(nn.Module):
    def __init__(self, dim_high, dim_low, dim_out, expand_ratio=3):
        super(RtHighFeatureFusion, self).__init__()
        self.expand_ratio = expand_ratio
        mid_high_hidden = dim_high * self.expand_ratio
        mid_low_hidden = dim_low * self.expand_ratio

        self.prog_high = nn.Sequential(
            nn.Conv2d(dim_high, mid_high_hidden, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_high_hidden),
            nn.ReLU(inplace=True),
        )
        self.prog_low = nn.Sequential(
            nn.Conv2d(dim_low, mid_low_hidden, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_low_hidden),
            nn.ReLU(inplace=True),
        )

        self.low_conv = nn.Sequential(
            nn.Conv2d(mid_low_hidden, mid_low_hidden, kernel_size=3, padding=1, stride=1, bias=False, groups=mid_low_hidden),
            nn.BatchNorm2d(mid_low_hidden),
            nn.ReLU(inplace=True)
        )
        self.high_conv = nn.Sequential(
            nn.Conv2d(mid_high_hidden, mid_high_hidden, kernel_size=3, padding=1, stride=1, bias=False,
                      groups=mid_high_hidden),
            nn.BatchNorm2d(mid_high_hidden),
            nn.ReLU(inplace=True)
        )

        self.gather = nn.Sequential(
            nn.Conv2d(mid_low_hidden+mid_high_hidden, dim_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_out)
        )

    def forward(self, low, high):
        low = self.prog_low(low)
        low = self.low_conv(low)

        high = self.prog_high(high)
        high = self.high_conv(high)
        high = F.interpolate(high, size=low.size()[-2:], mode='bilinear', align_corners=True)

        fea = torch.cat([low, high], dim=1)
        fea = self.gather(fea)
        return fea


class RtClassifer_decoder(nn.Module):
    def __init__(self, num_classes, dim_low=64, dim_high=128, dim_out=128):
        super(RtClassifer_decoder, self).__init__()
        self.low_conv = nn.Sequential(
            nn.Conv2d(dim_low, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.last_conv = nn.Sequential(
            DepthWiseConv(dim_in=dim_high + 48, dim_out=dim_out, kernel_size=3, pad=1, stride=1),
            nn.Dropout(0.5),
            DepthWiseConv(dim_in=dim_out, dim_out=dim_out, kernel_size=3, pad=1, stride=1),
            nn.Dropout(0.1),
            nn.Conv2d(dim_out, num_classes, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, low, high):
        b, _, h, w = low.size()
        size = low.size()[-2:]
        low = self.low_conv(low)
        high = F.interpolate(high, size, mode='bilinear', align_corners=True)
        fea = torch.cat((low, high), dim=1)
        fea = self.last_conv(fea)
        return fea


class DepthWiseConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=1, pad=0, stride=1, d=1):
        super(DepthWiseConv, self).__init__()
        self.conv =  nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, stride=stride, padding=pad, groups=dim_in, bias=False, dilation=d),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(True),
            nn.Conv2d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class PCHead(nn.Module):
    def __init__(self, cfg):
        super(PCHead, self).__init__()
        self.feature_fusion = RtHighFeatureFusion(dim_high=128, dim_low=64, dim_out=128)
        self.classifier = RtClassifer_decoder(cfg.MODEL.NUM_CLASSES, dim_low=64, dim_high=128)

    def forward(self, feats):
        """
        :param feats: {'spatial', 'fea_16x', 'fea_32x'}
        :return: pred
        """
        spatial = feats['spatial']
        fea_16x = feats['fea_16x']
        fea_32x = feats['fea_32x']
        pred_high = self.feature_fusion(fea_16x, fea_32x)

        pred = self.classifier(spatial, pred_high)
        return pred


@SEG_HEAD_REGISTRY.register()
def pchead_builder(cfg):
    return PCHead(cfg)

