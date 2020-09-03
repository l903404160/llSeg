import torch
import torch.nn as nn
from models.detection.anchorfree_heads.search_head.genotype import PRIMITIVES
from models.detection.anchorfree_heads.search_head.operations import *


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
    def __init__(self, C):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = operation_sets[primitive](C)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class PCMixedOp(nn.Module):
    def __init__(self, C):
        super(PCMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.k = 4
        for primitive in PRIMITIVES:
            op = operation_sets[primitive](C // self.k)
            self._ops += [op]

    def forward(self, x, weights):
        dim_2 = x.size(1)
        xtemp = x[:, :dim_2 // self.k, :, :]
        xtemp2 = x[:, dim_2 // self.k:, :, :]
        temp1 = sum(w * op(xtemp) for w,op in zip(weights, self._ops))
        ans = torch.cat([temp1, xtemp2], dim=1)
        ans = channel_shuffle(ans, self.k)
        return ans


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, pc=False):
        """
        Args:
            steps: (4 in DARTS)
            multiplier: (multiplier=4, stem_multiplier=3 in DARTS)
            C_prev_prev: s0 channel
            C_prev: s1 channel
            C: current channel
        """
        super(Cell, self).__init__()
        self.preprocess0 = ReLUConvGN(C_prev_prev, C, kernel_size=3)
        self.preprocess1 = ReLUConvGN(C_prev, C, kernel_size=3)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        # TODO check the _bns
        # self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                if pc:
                    op = PCMixedOp(C)
                else:
                    op = MixedOp(C)
                self._ops.append(op)

    def forward(self, s0, s1, weights, weights2=None):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            if weights2 is not None:
                s = sum(
                    weights2[offset + j] * self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            else:
                s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)


def build_darts_cells(layers, C_prev_prev, C_prev, C_curr, multiplier=2, steps=4, pc=False):
    cells = nn.ModuleList()
    for i in range(layers):
        cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, pc=pc)
        cells += [cell]
        C_prev_prev, C_prev = C_prev, multiplier * C_curr
    k = sum(1 for i in range(steps) for n in range(2+i))
    return cells, k


class DirectCell(nn.Module):
    def __init__(self, C, C_out, depth=4, pc=False):
        super(DirectCell, self).__init__()
        self._depth = depth
        self._C = C

        self._ops = nn.ModuleList()
        for i in range(self._depth):
            if pc:
                op = PCMixedOp(self._C)
            else:
                op = MixedOp(self._C)
            self._ops.append(op)

        self.align_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self._C, C_out, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(C_out // 8, C_out),
            nn.ReLU()
        )

    def forward(self, s0, weights):
        for i in range(self._depth):
            s0 = self._ops[i](s0, weights[i])
        s0 = self.align_conv(s0)
        return s0


def build_direct_cells(layers, C_in, C_out, steps=4, pc=False):
    cells = nn.ModuleList()
    for i in range(layers):
        if i == layers - 1:
            cell = DirectCell(C_in, C_out=C_out, depth=steps, pc=pc)
        else:
            cell = DirectCell(C_in, C_in, depth=steps, pc=pc)
        cells += [cell]
    k = sum(1 for i in range(steps))

    return cells, k


class DenseCell(nn.Module):
    def __init__(self, C, C_out, depth=4, pc=False):
        super(DenseCell, self).__init__()
        self._depth = depth
        self._C = C
        self._C_out = C_out

        self._ops = nn.ModuleList()
        for i in range(self._depth):
            for j in range(i + 1):
                if pc:
                    op = PCMixedOp(self._C)
                else:
                    op = MixedOp(self._C)
                self._ops.append(op)
        self.align_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self._C, C_out, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(C_out // 8, C_out),
            nn.ReLU()
        )

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
        out = self.align_conv(states[-1])
        return out


def build_dense_cells(layers, C_in, C_out, steps=4, pc=False):
    cells = nn.ModuleList()

    for i in range(layers):
        if i == layers - 1:
            cell = DenseCell(C_in, C_out=C_out, depth=steps, pc=pc)
        else:
            cell = DenseCell(C_in, C_in, depth=steps, pc=pc)
        cells += [cell]
    k = sum(i + 1 for i in range(steps))
    return cells, k


if __name__ == '__main__':
    m, k = build_direct_cells(layers=1, C_in=128, C_out=128, steps=4, pc=True)

    print(m)

    x = torch.randn(2, 128, 64, 64)
    weights = torch.randn(4, 7)

    y = m[0](x, weights)
    print(y.size())

