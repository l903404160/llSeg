import torch
import torch.nn as nn

operation_sets = {}
# TODO check 'none' is necessary or not
operation_sets['conv_3x3'] = lambda C: ReLUConvGN(C, C, 3)
operation_sets['conv_5x5'] = lambda C: ReLUConvGN(C, C, 5)
operation_sets['skip_connect'] = lambda C: Identity()
operation_sets['sep_conv_3x3'] = lambda C: SepConv(C, C, 3, 1, 1)
operation_sets['sep_conv_5x5'] = lambda C: SepConv(C, C, 5, 1, 2)
operation_sets['dil_2_conv_3x3'] = lambda C: DilatedConv(C, C, 3, padding=2, dilation=2)
operation_sets['dil_4_conv_3x3'] = lambda C: DilatedConv(C, C, 3, padding=4, dilation=4)
# operation_sets['side_conv_1x3'] = lambda C: SideConv(C, C, (1, 3), padding=(0, 1))
# operation_sets['side_conv_3x1'] = lambda C:SideConv(C, C, (3, 1), padding=(1, 0))

# Operation Definition
# Pass
class ReLUConvGN(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size):
        super(ReLUConvGN, self).__init__()

        padding = (kernel_size - 1) // 2

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(dim_in, dim_out, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(dim_out // 8, dim_out)
        )

    def forward(self, x):
        return self.op(x)

# Pass
class DilatedConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, dilation):
        super(DilatedConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(dim_in, dim_out, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(dim_out // 8, dim_out)
        )

    def forward(self, x):
        return self.op(x)

# Pass
class SepConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(dim_in, dim_in, kernel_size, stride, padding, bias=False, groups=dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(dim_out // 8, dim_out)
        )

    def forward(self, x):
        return self.op(x)

# Pass
class SideConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding):
        super(SideConv, self).__init__()
        assert isinstance(kernel_size, tuple)
        assert isinstance(padding, tuple)
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(dim_in, dim_out, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(dim_out // 8, dim_out)
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
