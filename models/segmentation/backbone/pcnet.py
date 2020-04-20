import torch
import torch.nn as nn
import torch.nn.functional as F
from . import SEG_BACKBONE_REGISTRY

class RtHead_block(nn.Module):
    def __init__(self):
        super(RtHead_block, self).__init__()
        channels = [32, 48]

        self.conv1 = self._make_conv_block(dim_in=3, dim_out=channels[0], kernel_size=3, pad=1, stride=2)
        self.conv2 = self._make_conv_block(dim_in=channels[0], dim_out=channels[1], kernel_size=3, pad=1, stride=2)
        self.conv3 = self._make_conv_block(dim_in=channels[1], dim_out=64, kernel_size=3, pad=1, stride=2)

    @staticmethod
    def _make_conv_block(dim_in, dim_out, kernel_size, pad, stride):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU()
        )

    def forward(self, fea):
        fea = self.conv3(self.conv2(self.conv1(fea)))
        return fea

class RtContext_expr_new(nn.Module):
    def __init__(self):
        super(RtContext_expr_new, self).__init__()
        group = 2
        dilations = [4, 4]
        self.conv16s_top = nn.Sequential(
            InvertedResidual_parallel(inp=64, oup=64, stride=2, expand_ratio=6, dilation=dilations[0], group=group),
            InvertedResidual_parallel(inp=64, oup=64, stride=1, expand_ratio=6, dilation=dilations[0], group=group),
            InvertedResidual_parallel(inp=64, oup=64, stride=1, expand_ratio=6, dilation=dilations[0], group=group),
            InvertedResidual_parallel(inp=64, oup=64, stride=1, expand_ratio=6, dilation=dilations[0], group=group),
            InvertedResidual_parallel(inp=64, oup=64, stride=1, expand_ratio=6, dilation=dilations[0], group=group),
            InvertedResidual_parallel(inp=64, oup=64, stride=1, expand_ratio=6, dilation=dilations[0], group=group),
        )
        self.conv16s_down = nn.Sequential(
            InvertedResidual_parallel(inp=64, oup=64, stride=1, expand_ratio=6, dilation=dilations[0], group=group),
            InvertedResidual_parallel(inp=64, oup=64, stride=1, expand_ratio=6, dilation=dilations[0], group=group),
            InvertedResidual_parallel(inp=64, oup=64, stride=1, expand_ratio=6, dilation=dilations[0], group=group),
            InvertedResidual_parallel(inp=64, oup=64, stride=1, expand_ratio=6, dilation=dilations[0], group=group),
        )
        self.conv32s_top = nn.Sequential(
            InvertedResidual_PKernel(inp=128, oup=96, stride=2, expand_ratio=6, dilation=3, group=group),
            InvertedResidual_PKernel(inp=96, oup=96, stride=1, expand_ratio=6, dilation=3, group=group),
            InvertedResidual_PKernel(inp=96, oup=128, stride=1, expand_ratio=6, dilation=3, group=group),
        )
        self.conv32s_down = nn.Sequential(
            InvertedResidual_PKernel(inp=128, oup=128, stride=1, expand_ratio=6, dilation=3, group=group),
            InvertedResidual_PKernel(inp=128, oup=128, stride=1, expand_ratio=6, dilation=3, group=group),
            InvertedResidual_PKernel(inp=128, oup=128, stride=1, expand_ratio=6, dilation=3, group=group),
        )

    def forward(self, fea):
        fea_16s = self.conv16s_top(fea)
        fea_16s = self.conv16s_down(fea_16s)

        fea_sp = F.interpolate(fea, size=fea_16s.size()[-2:], mode='bilinear', align_corners=True)
        fea_16s_with_sp = torch.cat([fea_16s, fea_sp], dim=1)

        fea_32s = self.conv32s_top(fea_16s_with_sp)
        fea_32s = self.conv32s_down(fea_32s)
        return fea_16s, fea_32s

class InvertedResidual_parallel(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, group=2, dilation=3):
        super(InvertedResidual_parallel, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.group = group
        self.dilations = [1,3]
        self.dilations[-1] = dilation
        hidden_dim = round(inp * expand_ratio)
        mid_hidden = hidden_dim // group

        self.use_res_connect = self.stride == 1 and inp == oup
        self.pw = nn.Sequential(
            nn.Conv2d(inp, mid_hidden, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_hidden),
            nn.ReLU(inplace=True),
        )

        self.dw = nn.ModuleList()
        for i in range(group):
            self.dw.append(
                nn.Sequential(
                    nn.Conv2d(mid_hidden, mid_hidden, 3, stride, self.dilations[i], groups=mid_hidden, bias=False,
                              dilation=self.dilations[i]),
                    nn.BatchNorm2d(mid_hidden),
                    nn.ReLU(inplace=True),
                )
            )

        self.pw_linear = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            residual = x
            x = self.pw(x)
            x = torch.cat([self.dw[j](x) for j in range(self.group)], dim=1)
            x = self.pw_linear(x)
            return residual + x
        else:
            x = self.pw(x)
            x = torch.cat([self.dw[j](x) for j in range(self.group)], dim=1)
            x = self.pw_linear(x)
            return x


class InvertedResidual_PKernel(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, group=2, dilation=2):
        super(InvertedResidual_PKernel, self).__init__()
        assert stride in [1, 2]
        self.group = group
        hidden_dim = round(inp * expand_ratio)
        mid_hidden = hidden_dim // group

        self.use_res_connect = stride == 1 and inp == oup
        self.pw = nn.Sequential(
            nn.Conv2d(inp, mid_hidden, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_hidden),
            nn.ReLU(inplace=True),
        )

        kernel_sizes = [5, 3]
        self.dilations = [1, 3]
        self.dilations[-1] = dilation
        padding = [2, 3]
        padding[-1] = dilation

        self.dw = nn.ModuleList()
        for i in range(group):
            self.dw.append(
                nn.Sequential(
                    nn.Conv2d(mid_hidden, mid_hidden, kernel_sizes[i], stride, padding=padding[i], groups=mid_hidden, bias=False,
                              dilation=self.dilations[i]),
                    nn.BatchNorm2d(mid_hidden),
                    nn.ReLU(inplace=True),
                )
            )

        self.pw_linear = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            residual = x
            x = self.pw(x)
            x = torch.cat([self.dw[j](x) for j in range(self.group)], dim=1)
            x = self.pw_linear(x)
            return residual + x
        else:
            x = self.pw(x)
            x = torch.cat([self.dw[j](x) for j in range(self.group)], dim=1)
            x = self.pw_linear(x)
            return x


class PCBackbone(nn.Module):
    def __init__(self):
        super(PCBackbone, self).__init__()
        self.head = RtHead_block()
        self.context = RtContext_expr_new()

    def forward(self, data_input):
        spatial = self.head(data_input)
        fea_16x, fea_32x = self.context(spatial)

        out = {
            'spatial': spatial,
            'fea_16x': fea_16x,
            'fea_32x': fea_32x
        }
        return out

@SEG_BACKBONE_REGISTRY.register()
def pcnet_builder(cfg):
    return PCBackbone()
