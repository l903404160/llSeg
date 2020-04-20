import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from . import SEG_HEAD_REGISTRY

from models.losses import cross_entropy_loss


from utils.comm import all_gather


class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation, norm_layer=None, norm_mom=3e-4):
        super(ConvBNReLU, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(c_out, momentum=norm_mom)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''

    def __init__(self, c, k, stage_num=3, norm_layer=None, norm_mom=3e-4):
        super(EMAU, self).__init__()
        self.stage_num = stage_num

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        mu = torch.Tensor(1, c, k)
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            norm_layer(c, momentum=norm_mom))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)  # b * n * c
                z = torch.bmm(x_t, mu)  # b * n * k
                z = F.softmax(z, dim=2)  # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)  # b * c * k
                mu = self._l2norm(mu, dim=1)

        # !!! The moving averaging operation is writtern in train.py, which is significant.

        z_t = z.permute(0, 2, 1)  # b * k * n
        x = mu.matmul(z_t)  # b * c * n
        x = x.view(b, c, h, w)  # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x, mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.
        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.
        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))


class EMAHead(nn.Module):
    def __init__(self, cfg, norm_layer=None, norm_mom=3e-4):
        super(EMAHead, self).__init__()
        self.fc0 = ConvBNReLU(2048, 512, 3, 1, 1, 1, norm_layer=norm_layer, norm_mom=norm_mom)
        # TODO rm settings
        self.emau = EMAU(512, 64, cfg.MODEL.STAGE_NUM, norm_layer=norm_layer, norm_mom=norm_mom)
        self.fc1 = nn.Sequential(
            ConvBNReLU(512, 256, 3, 1, 1, 1, norm_layer=norm_layer, norm_mom=norm_mom),
            nn.Dropout2d(p=0.1))
        self.fc2 = nn.Conv2d(256, cfg.MODEL.NUM_CLASSES, 1)

    def _compute_loss(self, pred, label):
        return cross_entropy_loss(pred, label)

    def forward(self, x, label=None):
        x = self.fc0(x['res4'])
        x, mu = self.emau(x)
        x = self.fc1(x)
        pred = self.fc2(x)

        if label is None:
            return pred

        size = label.size()[-2:]
        pred = F.interpolate(pred, size=size, mode='bilinear', align_corners=True)
        with torch.no_grad():
            temp = all_gather(mu)
            total_mu = torch.cat(temp, dim=0)
            mean_mu = total_mu.mean(dim=0, keepdim=True).to(self.device)
            momentum = self._cfg.MODEL.EM_MOM
            self.head.emau.mu *= momentum
            self.head.emau.mu += mean_mu * (1 - momentum)

        return self._compute_loss(pred, label)


@SEG_HEAD_REGISTRY.register()
def emahead_builder(cfg):
    from layers.batch_norm import get_norm
    norm_layer = get_norm(cfg.MODEL.BN_LAYER)
    norm_mom = cfg.MODEL.BN_MOM
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    return EMAHead(cfg, norm_layer, norm_mom)