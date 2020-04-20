import torch.nn as nn
from models.modules.base import conv_bn_relu
from models.backbone import BACKBONE_REGISTRY


# model from https://github.com/MegviiDetection/video_analyst/blob/master/videoanalyst/model/backbone/backbone_impl/alexnet_bn.py
@BACKBONE_REGISTRY.register()
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = conv_bn_relu(3, 96, stride=2, kszie=11, pad=0)
        self.pool1 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)
        self.conv2 = conv_bn_relu(96, 256, 1, 5, 0)
        self.pool2 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)
        self.conv3 = conv_bn_relu(256, 384, 1, 3, 0)
        self.conv4 = conv_bn_relu(384, 384, 1, 3, 0)
        self.conv5 = conv_bn_relu(384, 256, 1, 3, 0, has_relu=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x



