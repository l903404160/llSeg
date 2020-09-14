import torch
import torch.nn as nn
from models.detection.anchorfree_heads.search_head.genotype import PRIMITIVES, PRIMITIVES_box
from models.detection.anchorfree_heads.search_head.operations import *


primitives_dict = {
    'cls': PRIMITIVES,
    'box': PRIMITIVES_box
}


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, stride=1, primitives=None):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in primitives:
            op = operation_sets[primitive](C_in, C_out, stride)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class PCMixedOp(nn.Module):
    def __init__(self, C_in, C_out, stride=1, primitives=None):
        super(PCMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.k = 4
        for primitive in primitives:
            op = operation_sets[primitive](C_in // self.k, C_out // self.k, stride)
            self._ops += [op]

    def forward(self, x, weights):
        dim_2 = x.size(1)
        xtemp = x[:, :dim_2 // self.k, :, :]
        xtemp2 = x[:, dim_2 // self.k:, :, :]
        temp1 = sum(w * op(xtemp) for w,op in zip(weights, self._ops))
        ans = torch.cat([temp1, xtemp2], dim=1)
        ans = channel_shuffle(ans, self.k)
        return ans


class DirectCell(nn.Module):
    def __init__(self, C_in, C_out, stride, depth=4, pc=False, primitives=None):
        super(DirectCell, self).__init__()
        self._depth = depth
        self._C_in = C_in

        self._ops = nn.ModuleList()
        for i in range(self._depth):
            if pc:
                op = PCMixedOp(self._C_in, C_out, stride=stride, primitives=primitives)
            else:
                op = MixedOp(self._C_in, self._C_in, stride=stride, primitives=primitives)
            self._ops.append(op)

    def forward(self, s0, weights):
        for i in range(self._depth):
            s0 = self._ops[i](s0, weights[i])
        return s0


def build_direct_cells(layers, C_in, C_mid, C_out, stride, depth=4, pc=False, search_part='cls'):
    cells = nn.ModuleList()

    primitives = primitives_dict[search_part]

    for i in range(layers):
        if i == 0:
            cell = DirectCell(C_in, C_out=C_mid, stride=stride, depth=depth, pc=pc, primitives=primitives)
        elif i == layers - 1:
            cell = DirectCell(C_mid, C_out=C_out, stride=stride, depth=depth, pc=pc, primitives=primitives)
        else:
            cell = DirectCell(C_mid, C_mid, stride=stride, depth=depth, pc=pc, primitives=primitives)
        cells += [cell]
    k = sum(1 for i in range(depth))

    return cells, k


class DenseCell(nn.Module):
    def __init__(self, C_in, C_out, stride, depth=4, pc=False):
        super(DenseCell, self).__init__()
        self._depth = depth
        self._C_in = C_in
        self._C_out = C_out
        self._stride = stride

        self._ops = nn.ModuleList()
        for i in range(self._depth):
            for j in range(i + 1):
                if pc:
                    op = PCMixedOp(self._C_in, self._C_out, self._stride)
                else:
                    op = MixedOp(self._C_in, self._C_out, self._stride)
                self._ops.append(op)

    def forward(self, s0, weights, weights2=None):
        states = [s0]
        offset = 0
        for i in range(self._depth):
            if weights2 is not None:
                s = sum(weights2[offset + j] * self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            else:
                s = sum(self._ops[offset + j](h, weights[offset + j]) for j,h in enumerate(states))
            offset += len(states)
            states.append(s)
        out = states[-1]
        return out


def build_dense_cells(layers, C_in, C_out, stride, steps=4, pc=False):
    cells = nn.ModuleList()

    for i in range(layers):
        if i == layers - 1:
            cell = DenseCell(C_in, C_out=C_out, stride=stride, depth=steps, pc=pc)
        else:
            cell = DenseCell(C_in, C_in, stride=stride, depth=steps, pc=pc)
        cells += [cell]
    k = sum(i + 1 for i in range(steps))
    return cells, k


class FPNCell(nn.Module):
    def __init__(self, C_in, C_out, depth=2, pc=False, primitives=None):
        super(FPNCell, self).__init__()
        self._fpn_strides = [8, 16, 32, 64, 128]
        self._ops = nn.ModuleList()
        if pc:
            cell_func = PCMixedOp
        else:
            cell_func = MixedOp

        for i in range(3):
            self._ops += [cell_func(C_in=C_in, C_out=C_out, stride=1, primitives=primitives)]

        for j in range(depth):
            if (j+1) != depth:
                self._ops += [cell_func(C_in=C_out, C_out=C_out, stride=1, primitives=primitives)]
            else:
                self._ops += [cell_func(C_in=C_out, C_out=C_in, stride=1, primitives=primitives)]

    def forward(self, features, weights):
        pass


if __name__ == '__main__':
    m, k = build_direct_cells(layers=1, C_in=64, C_out=64, stride=1, steps=4, pc=True)
    # m, k = build_dense_cells(layers=1, C_in=64, C_out=64, stride=1, steps=4, pc=True)
    # m = PCMixedOp(C_in=64, C_out=64, stride=1)
    # m = DirectCell(C_in=64, C_out=64, stride=1, depth=4, pc=True)
    print(m)

    x = torch.randn(2, 64, 64, 64)
    # weights = torch.randn(4, 7)
    weights = torch.randn(k, 8)

    y = m[0](x, weights)
    print(y.size())

