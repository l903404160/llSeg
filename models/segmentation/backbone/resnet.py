import torch.nn as nn
import math
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from . import SEG_BACKBONE_REGISTRY


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None, norm_mom=3e-4):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes, momentum=norm_mom)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, dilation, dilation,
                               bias=False)
        self.bn2 = norm_layer(planes, momentum=norm_mom)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = norm_layer(planes * 4, momentum=norm_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, stride=8, norm_layer=None, norm_mom=3e-4, grids=None):
        self.inplanes = 128
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64, momentum=norm_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(64, momentum=norm_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False))

        self.bn1 = norm_layer(self.inplanes, momentum=norm_mom)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, norm_mom=norm_mom)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer, norm_mom=norm_mom)

        if stride == 16:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer, norm_mom=norm_mom)
            self.layer4 = self._make_layer(
                block, 512, layers[3], stride=1, dilation=2, grids=grids, norm_layer=norm_layer, norm_mom=norm_mom)
        elif stride == 8:
            self.layer3 = self._make_layer(
                block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer, norm_mom=norm_mom)
            self.layer4 = self._make_layer(
                block, 512, layers[3], stride=1, dilation=4, grids=grids, norm_layer=norm_layer, norm_mom=norm_mom)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    grids=None, norm_layer=None, norm_mom=3e-4):
        downsample = None
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion, momentum=norm_mom))

        layers = []
        if grids is None:
            grids = [1] * blocks

        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer, norm_mom=norm_mom))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer, norm_mom=norm_mom))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                dilation=dilation * grids[i],
                                previous_dilation=dilation, norm_layer=norm_layer, norm_mom=norm_mom))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class DilatedResNetWrapper(nn.Module):
    def __init__(self, ori_resnet):
        super(DilatedResNetWrapper, self).__init__()
        self.conv1 = ori_resnet.conv1
        self.bn1 = ori_resnet.bn1
        self.relu = ori_resnet.relu
        self.maxpool = ori_resnet.maxpool
        self.layer1 = ori_resnet.layer1
        self.layer2 = ori_resnet.layer2
        self.layer3 = ori_resnet.layer3
        self.layer4 = ori_resnet.layer4

    def forward(self, data_input):
        fea = self.maxpool(self.relu(self.bn1(self.conv1(data_input))))
        res1 = self.layer1(fea)
        res2 = self.layer2(res1)
        res3 = self.layer3(res2)
        res4 = self.layer4(res3)

        out = {
            'res1': res1,
            'res2': res2,
            'res3': res3,
            'res4': res4,
        }
        return out


@SEG_BACKBONE_REGISTRY.register()
def resnet_builder(cfg):
    from layers.batch_norm import get_norm
    norm_layer = get_norm(cfg.MODEL.BN_LAYER)
    norm_mom = cfg.MODEL.BN_MOM
    n_layers, stride = cfg.MODEL.N_LAYERS, cfg.MODEL.STRIDE
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    layers = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[n_layers]

    net = ResNet(Bottleneck, layers=layers, stride=stride,
                 norm_layer=norm_layer, norm_mom=norm_mom, grids=cfg.MODEL.BACKBONE.MULTI_GRIDS)
    resnet = DilatedResNetWrapper(net)
    del net
    return resnet
