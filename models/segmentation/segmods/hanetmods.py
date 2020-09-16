# This file is created for implementing the extra modules for HANet
import math

import torch
import torch.nn as nn

import torch.nn.functional as F
from .PosEmbedding import PosEmbedding1D, PosEncoding1D


# refer to `https://github.com/shachoi/HANet/blob/master/network/HANet.py`
# @Mingyang Li
class HANet_Conv(nn.Module):
    # def __init__(self, cfg, norm_layer=nn.BatchNorm2d):
    def __init__(self, in_channel, out_channel, kernel_size=3, r_factor=64, layer=3, pos_injection=2, is_encoding=1,
                         pos_rfactor=8, pooling='mean', dropout_prob=0.0, pos_noise=0.0, norm_layer=nn.BatchNorm2d):
        super(HANet_Conv, self).__init__()

        self.pooling = pooling
        self.pos_injection = pos_injection
        self.layer = layer
        self.dropout_prob = dropout_prob
        self.sigmoid = nn.Sigmoid()

        if r_factor > 0:
            mid_1_channel = math.ceil(in_channel / r_factor)
        elif r_factor < 0:
            r_factor = r_factor * -1
            mid_1_channel = in_channel * r_factor

        if self.dropout_prob > 0:
            self.dropout = nn.Dropout2d(self.dropout_prob)

        self.attention_first = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=mid_1_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(mid_1_channel),
            nn.ReLU(inplace=True))

        if layer == 2:
            self.attention_second = nn.Sequential(
                nn.Conv1d(in_channels=mid_1_channel, out_channels=out_channel,
                          kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True))
        elif layer == 3:
            mid_2_channel = (mid_1_channel * 2)
            self.attention_second = nn.Sequential(
                nn.Conv1d(in_channels=mid_1_channel, out_channels=mid_2_channel,
                          kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(mid_2_channel),
                nn.ReLU(inplace=True))
            self.attention_third = nn.Sequential(
                nn.Conv1d(in_channels=mid_2_channel, out_channels=out_channel,
                          kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True))

        if self.pooling == 'mean':
            # print("##### average pooling")
            self.rowpool = nn.AdaptiveAvgPool2d((128 // pos_rfactor, 1))
        else:
            # print("##### max pooling")
            self.rowpool = nn.AdaptiveMaxPool2d((128 // pos_rfactor, 1))

        if pos_rfactor > 0:
            if is_encoding == 0:
                if self.pos_injection == 1:
                    self.pos_emb1d_1st = PosEmbedding1D(pos_rfactor, dim=in_channel, pos_noise=pos_noise)
                elif self.pos_injection == 2:
                    self.pos_emb1d_2nd = PosEmbedding1D(pos_rfactor, dim=mid_1_channel, pos_noise=pos_noise)
            elif is_encoding == 1:
                if self.pos_injection == 1:
                    self.pos_emb1d_1st = PosEncoding1D(pos_rfactor, dim=in_channel, pos_noise=pos_noise)
                elif self.pos_injection == 2:
                    self.pos_emb1d_2nd = PosEncoding1D(pos_rfactor, dim=mid_1_channel, pos_noise=pos_noise)
            else:
                print("Not supported position encoding")
                exit()

    def forward(self, x, out, pos=None, return_attention=False, return_posmap=False, attention_loss=False):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        H = out.size(2)
        x1d = self.rowpool(x).squeeze(3)

        if pos is not None and self.pos_injection == 1:
            if return_posmap:
                x1d, pos_map1 = self.pos_emb1d_1st(x1d, pos, True)
            else:
                x1d = self.pos_emb1d_1st(x1d, pos)

        if self.dropout_prob > 0:
            x1d = self.dropout(x1d)
        x1d = self.attention_first(x1d)

        if pos is not None and self.pos_injection == 2:
            if return_posmap:
                x1d, pos_map2 = self.pos_emb1d_2nd(x1d, pos, True)
            else:
                x1d = self.pos_emb1d_2nd(x1d, pos)

        x1d = self.attention_second(x1d)

        if self.layer == 3:
            x1d = self.attention_third(x1d)
            if attention_loss:
                last_attention = x1d
            x1d = self.sigmoid(x1d)
        else:
            if attention_loss:
                last_attention = x1d
            x1d = self.sigmoid(x1d)

        x1d = F.interpolate(x1d, size=H, mode='linear')
        out = torch.mul(out, x1d.unsqueeze(3))

        if return_attention:
            if return_posmap:
                if self.pos_injection == 1:
                    pos_map = (pos_map1)
                elif self.pos_injection == 2:
                    pos_map = (pos_map2)
                return out, x1d, pos_map
            else:
                return out, x1d
        else:
            if attention_loss:
                return out, last_attention
            else:
                return out


# refer to `https://github.com/shachoi/HANet/blob/master/network/deepv3.py`
# @Mingyang Li
class ASPP(nn.Module):
    def __init__(self, cfg, norm_layer=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn
        # put this configs to config files
        output_stride = cfg.MODEL.STRIDE
        rates = cfg.MODEL.HANET.ASPP_RATES
        in_dim = cfg.MODEL.HANET.IN_DIM
        reduction_dim = cfg.MODEL.HANET.REDUCTION_DIM

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 4:
            rates = [4 * r for r in rates]
        elif output_stride == 16:
            pass
        elif output_stride == 32:
            rates = [r // 2 for r in rates]
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          norm_layer(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                norm_layer(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1, bias=False),
            norm_layer(256), nn.ReLU(inplace=True))

    def forrward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = nn.functional.interpolate(img_features, size=x_size[2:], mode='bilinear',
                                                 align_corners=True)

        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out
