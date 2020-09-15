# This file is created for implementing the extra modules for HANet
import torch.nn as nn


# refer to `https://github.com/shachoi/HANet/blob/master/network/HANet.py`
# @Mingyang Li
class HANet_Conv(nn.Module):
    def __init__(self):
        super(HANet_Conv, self).__init__()

    def forward(self, x):
        pass


# refer to `https://github.com/shachoi/HANet/blob/master/network/deepv3.py`
# @Mingyang Li
class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()

    def forrward(self, x):
        pass