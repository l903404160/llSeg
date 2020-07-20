import torch
import torch.nn as nn

from ._cpools import TopPool, LeftPool, BottomPool, RightPool

from layers import CNNBlockBase
from layers import get_norm_with_channels as get_norm


class AnchorFreeConvBlock(CNNBlockBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, norm="BN"):
        super(AnchorFreeConvBlock, self).__init__(in_channels=in_channels, out_channels=out_channels, stride=stride)

        # Parameter initialization
        pad = (kernel_size - 1) // 2
        self.operation = nn.Sequential(
            # TODO check the influence of bias
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, bias=False),
            get_norm(norm, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, in_feature):
        out = self.operation(in_feature)
        return out


class CornerNetResidual(CNNBlockBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, norm="BN"):
        super(CornerNetResidual, self).__init__(in_channels=in_channels, out_channels=out_channels, stride=stride)

        # Operation initialization
        self.operation_conva = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, stride=stride, bias=False),
            get_norm(norm, out_channels),
            nn.ReLU(inplace=True)
        )

        self.operation_convb = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            get_norm(norm, out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            get_norm(norm, out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Sequential()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, in_feature):
        res = self.operation_conva(in_feature)
        res = self.operation_convb(res)

        shortcut = self.shortcut(in_feature)
        return self.relu(res + shortcut)


class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2


# ------------------------------------- Corner Pooling
class CornerPooling(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(CornerPooling, self).__init__()

        self.pool1_conv1 = AnchorFreeConvBlock(in_channels=dim, out_channels=128, kernel_size=3)
        self.pool2_conv1 = AnchorFreeConvBlock(in_channels=dim, out_channels=128, kernel_size=3)

        self.pool_conv1 = nn.Conv2d(128, dim, kernel_size=3, padding=1, bias=False)
        # TODO change it to get_norm
        self.pool_bn1 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = AnchorFreeConvBlock(in_channels=dim, out_channels=dim, kernel_size=3)

        self.pool1 = pool1()
        self.pool2 = pool2()

    def forward(self, in_feature):
        # Pooling 1
        pool1_conv1 = self.pool1_conv1(in_feature)
        pool1 = self.pool1(pool1_conv1)

        # Pooling 2
        pool2_conv1 = self.pool2_conv1(in_feature)
        pool2 = self.pool2(pool2_conv1)

        # Pooling1 + Pooling2
        pool_conv1 = self.pool_conv1(pool1 + pool2)
        pool_bn1 = self.pool_bn1(pool_conv1)

        conv1 = self.conv1(in_feature)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(pool_bn1 + bn1)
        conv2 = self.conv2(relu1)
        return conv2


class TopLeftCornerPooling(CornerPooling):
    def __init__(self, dim):
        super(TopLeftCornerPooling, self).__init__(dim, TopPool, LeftPool)


class BottomRightCornerPooling(CornerPooling):
    def __init__(self, dim):
        super(BottomRightCornerPooling, self).__init__(dim, BottomPool, RightPool)


