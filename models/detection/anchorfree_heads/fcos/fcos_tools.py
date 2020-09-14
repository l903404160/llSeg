import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.nn.weight_init import c2_msra_fill


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


class SearchFPN(nn.Module):
    def __init__(self,dim_in, dim_mid, dim_out):
        super(SearchFPN, self).__init__()
        self.p3_conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_mid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(dim_mid // 8, dim_mid),
            nn.ReLU()
        )
        self.p4_conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_mid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(dim_mid // 8, dim_mid),
            nn.ReLU()
        )
        self.p5_conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_mid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(dim_mid // 8, dim_mid),
            nn.ReLU()
        )
        self.p6_conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_mid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(dim_mid // 8, dim_mid),
            nn.ReLU()
        )
        self.p7_conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_mid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(dim_mid // 8, dim_mid),
            nn.ReLU()
        )




    def forward(self, features):
        """
        Args:
            features: {'p3','p4',...}
        Returns:
        """
        p4 = self.p4_conv(features['p4'])
        p5 = self.p5_conv(features['p5'])
        p6 = self.p6_conv(features['p6'])

        size = p5.size()[2:]
        feat = F.interpolate(p4, size=size, mode='nearest') + p5 + F.interpolate(p6, size=size, mode='nearest')
        feat = feat / 3

        feat = self.depth_conv(feat)

        for k, v in features.items():
            features[k] = v + F.interpolate(feat, size=v.size()[2:], mode='nearest')
        return features


class BalancedFeaturePyramids(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out):
        super(BalancedFeaturePyramids, self).__init__()
        self._dim_in = dim_in
        self._dim_mid = dim_mid
        self._dim_out = dim_out
        self._num_levels = 5
        self._refine_level = 2

        self.refine_conv = NonLocal2D(dim_in=self._dim_in, dim_mid=self._dim_mid, dim_out=self._dim_out)

    def forward(self, features):
        # step 1: gather multi-level features by resize and average
        feats = []
        gather_size = features[self._refine_level].size()[2:]
        for i in range(self._num_levels):
            if i < self._refine_level:
                gathered = F.adaptive_max_pool2d(features[i], output_size=gather_size)
            else:
                gathered = F.interpolate(features[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        bsf = sum(feats) / len(feats)
        # step 2: refine gathered features
        bsf = self.refine_conv(bsf)

        outs = []
        for i in range(self._num_levels):
            out_size = features[i].size()[2:]
            if i < self._refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + features[i])
        return outs


class NonLocal2D(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out, mode='dot_product'):
        super(NonLocal2D, self).__init__()
        self._dim_in = dim_in
        self._dim_mid =dim_mid
        self._dim_out = dim_out
        self.mode = mode

        self.g = nn.Conv2d(self._dim_in, self._dim_mid, kernel_size=1, padding=0, stride=1)
        self.theta = nn.Conv2d(self._dim_in, self._dim_mid, kernel_size=1, padding=0, stride=1)
        self.phi = nn.Conv2d(self._dim_in, self._dim_mid, kernel_size=1, padding=0, stride=1)

        self.conv_out = nn.Sequential(
            nn.Conv2d(self._dim_mid, self._dim_out, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(self._dim_out // 8, self._dim_out)
        )

        self.init_weights()

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def init_weights(self, std=0.01, zero_init=True):
        for m in [self.g, self.phi, self.theta]:
            nn.init.normal_(m.weight, std=std)
        if zero_init:
            nn.init.constant_(self.conv_out[0].weight, 0)
        else:
            nn.init.normal_(self.conv_out[0].weight, std=std)

    def forward(self, x):
        n, _, h, w = x.shape

        g_x = self.g(x).view(n, self._dim_mid, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(n, self._dim_mid, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(n, self._dim_mid, -1)

        pairwise_weight = self.dot_product(theta_x, phi_x)

        y = torch.matmul(pairwise_weight, g_x)
        y = y.permute(0,2,1).reshape(n, self._dim_mid, h, w).contiguous()

        output = x + self.conv_out(y)
        return output


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    channels = 256
    # m = SearchFPN(dim_in=channels, dim_mid=channels, dim_out=channels).cuda()
    #
    m = BalancedFeaturePyramids(dim_in=channels, dim_mid=channels, dim_out=channels).cuda()

    x = torch.randn(2, channels, 32, 32)
    features = {
        'p3': torch.randn(2, channels, 64, 64).cuda(),
        'p4': torch.randn(2, channels, 32, 32).cuda(),
        'p5': torch.randn(2, channels, 16, 16).cuda(),
        'p6': torch.randn(2, channels, 8, 8).cuda(),
        'p7': torch.randn(2, channels, 4, 4).cuda(),
    }
    features = [features[f] for f in features.keys()]

    for i in range(100):
        out = m(features)
        del out

    torch.save(m, './test.pth')
    # print(out[1].size())
