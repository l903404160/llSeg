import torch
import torch.nn as nn
import torch.nn.functional as F

operation_sets = {}
operation_sets['skip'] = lambda C_in, C_out, stride: Identity(dim_in=C_in, dim_out=C_out, stride=stride)
operation_sets['conv_k1'] = lambda C_in, C_out, stride: ConvGnReLU(dim_in=C_in, dim_out=C_out, kernel_size=1, stride=stride, bias=False)
operation_sets['sep_k3'] = lambda C_in, C_out, stride: SepConv(dim_in=C_in, dim_out=C_out, kernel_size=3, stride=stride)
operation_sets['sep_k5'] = lambda C_in, C_out, stride: SepConv(dim_in=C_in, dim_out=C_out, kernel_size=5, stride=stride)

operation_sets['mbconv_k3_e1'] = lambda C_in, C_out, stride: MBBlock(dim_in=C_in, dim_out=C_out, expansion=1, stride=stride, kernel_size=3, dilation=1)
operation_sets['mbconv_k3_e3'] = lambda C_in, C_out, stride: MBBlock(dim_in=C_in, dim_out=C_out, expansion=3, stride=stride, kernel_size=3, dilation=1)
operation_sets['mbconv_k3_e1_d2'] = lambda C_in, C_out, stride: MBBlock(dim_in=C_in, dim_out=C_out, expansion=1, stride=stride, kernel_size=3, dilation=2)
operation_sets['mbconv_k3_e3_d2'] = lambda C_in, C_out, stride: MBBlock(dim_in=C_in, dim_out=C_out, expansion=3, stride=stride, kernel_size=3, dilation=2)

operation_sets['mbconv_k5_e1'] = lambda C_in, C_out, stride: MBBlock(dim_in=C_in, dim_out=C_out, expansion=1, stride=stride, kernel_size=5, dilation=1)
operation_sets['mbconv_k5_e3'] = lambda C_in, C_out, stride: MBBlock(dim_in=C_in, dim_out=C_out, expansion=3, stride=stride, kernel_size=3, dilation=1)
operation_sets['mbconv_k5_e1_d2'] = lambda C_in, C_out, stride: MBBlock(dim_in=C_in, dim_out=C_out, expansion=1, stride=stride, kernel_size=3, dilation=2)
operation_sets['mbconv_k5_e3_d2'] = lambda C_in, C_out, stride: MBBlock(dim_in=C_in, dim_out=C_out, expansion=3, stride=stride, kernel_size=3, dilation=2)


# Operation Definition
class SepConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, groups=1):
        super(SepConv, self).__init__()
        self.conv1 = ConvGnReLU(dim_in, dim_in, kernel_size=kernel_size, stride=stride,
                                bias=False, groups=dim_in)
        self.conv2 = ConvGnReLU(dim_in, dim_out, kernel_size=1, stride=1,
                                bias=False, groups=groups, relu='none')

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out


class Identity(nn.Module):
    def __init__(self,dim_in, dim_out, stride):
        super(Identity, self).__init__()
        if dim_in != dim_out or stride != 1:
            self.conv = ConvGnReLU(dim_in, dim_out, kernel_size=1, stride=stride, bias=False, relu='relu')
        else:
            self.conv = nn.Sequential()

    def forward(self, x):
        return self.conv(x)


class ConvGnReLU(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, dilation=1, bias=False, groups=1, relu='relu'):
        super(ConvGnReLU, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding=padding, dilation=dilation, bias=bias, groups=groups)
        self.gn = nn.GroupNorm(dim_out // 8, dim_out)
        self.relu = nn.ReLU() if relu == 'relu' else nn.Sequential()
        # initialize
        torch.nn.init.normal_(self.conv.weight, std=0.01)
        if self.conv.bias is not None:
            torch.nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        out = self.relu(self.gn(self.conv(x)))
        return out


class SE(nn.Module):
    def __init__(self, dim_in, se_ratio):
        super(SE, self).__init__()
        self._dim_in, self._se_ratio = dim_in, se_ratio
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Conv2d(dim_in, max(1, int(dim_in * se_ratio)), 1, bias=False)
        self.fc2 = nn.Conv2d(max(1, int(dim_in * se_ratio)), dim_in, 1, bias=False)

    def forward(self, x):
        out = self.pooling(x)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out


class ChannelShuffle(nn.Module):
    def __init__(self, groups=1):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        if self.groups == 1:
            return x
        N, C, H, W = x.size()
        cpg = C // self.groups  # channels per group
        out = x.view(N, self.groups, cpg, H, W)
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        out = out.view(N, C, H, W)
        return out


class MBBlock(nn.Module):
    def __init__(self, dim_in, dim_out, expansion, stride, kernel_size, dilation=1, groups=1, with_se=False):
        super(MBBlock, self).__init__()
        self._dim_in = dim_in
        self._dim_out = dim_out
        self._with_se = with_se
        self._stride = stride
        self._groups = groups
        mid_channels = dim_in * expansion

        self.conv1 = ConvGnReLU(dim_in, mid_channels, kernel_size=1, stride=1, dilation=1, bias=False, groups=groups)
        self.conv2 = ConvGnReLU(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=False, groups=mid_channels)
        self.conv3 = ConvGnReLU(mid_channels, dim_out, kernel_size=1, stride=1, dilation=1, bias=False, groups=groups, relu='none')

        if self._with_se:
            self.se = SE(mid_channels, se_ratio=0.05)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self._with_se:
            out = out * self.se(out)
        out = self.conv3(out)
        if self._dim_in == self._dim_out and self._stride == 1:
            out += x
        return out
