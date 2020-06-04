# implementation of ASPP module
import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class ASPPModule(nn.Module):
    def __init__(self, dim_in, dim_inter, dim_out, dilations=None, norm_layer=None, with_gp=True):
        super(ASPPModule, self).__init__()

        self.with_gp = with_gp
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_inter, kernel_size=1, padding=0),
            norm_layer(dim_inter)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_inter, kernel_size=3, padding=dilations[0], dilation=dilations[0]),
            norm_layer(dim_inter)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_inter, kernel_size=3, padding=dilations[1], dilation=dilations[1]),
            norm_layer(dim_inter)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_inter, kernel_size=3, padding=dilations[2], dilation=dilations[2]),
            norm_layer(dim_inter)
        )
        if self.with_gp:
            self.gp_conv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim_in, dim_inter, kernel_size=1, padding=0),
                norm_layer(dim_inter)
            )
            self.bottleneck = nn.Sequential(
                nn.Conv2d(dim_inter * 5, dim_out, kernel_size=1, padding=0),
                norm_layer(dim_out)
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(dim_inter * 4, dim_out, kernel_size=1, padding=0),
                norm_layer(dim_out)
            )

    def forward(self, data_input):
        h,w = data_input.size()[-2:]
        x1 = self.conv1(data_input)
        x2 = self.conv2(data_input)
        x3 = self.conv3(data_input)
        x4 = self.conv4(data_input)
        if self.with_gp:
            gp_x = self.gp_conv(data_input)
            gp_x = interpolate(gp_x, (h,w), mode='bilinear', align_corners=True)
            x = torch.cat([gp_x, x1, x2, x3, x4], dim=1)
        else:
            x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.bottleneck(x)
        return x

if __name__ == '__main__':
    m = ASPPModule(2048, 256, 512, dilations=[12, 24, 36], with_gp=False, norm_layer=nn.BatchNorm2d).cuda(0)

    x = torch.randn(2, 2048, 64, 64).cuda(0)

    import time
    st = time.time()
    for i in range(100):
        y = m(x)
        # y = m(y)
    end = time.time()

    print(" 100 iter using %f" % (end - st))
    print(y.size())