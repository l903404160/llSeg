import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.detection.anchorfree_heads.fcos.fcos_tools import Scale
from models.detection.anchorfree_heads.search_head.genotype import Genotype, DARTS_FCOS_HEAD
from models.detection.anchorfree_heads.search_head.operations import ReLUConvGN, operation_sets

"""
    Generate Head model according to the Genotype
"""

genotype = Genotype(normal=[('dil_4_conv_3x3', 0), ('skip_connect', 1), ('dil_4_conv_3x3', 0), ('conv_3x3', 2),
                           ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3)], normal_concat=range(4, 6))


class Cell(nn.Module):
    def __init__(self, genotype:Genotype, C_prev_prev, C_prev, C):
        super(Cell, self).__init__()
        self.preprocess0 = ReLUConvGN(C_prev_prev, C, kernel_size=3)
        self.preprocess1 = ReLUConvGN(C_prev, C, kernel_size=3)
        op_names, indices = zip(*genotype.normal)
        self._concat = genotype.normal_concat

        # parse the genotype
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._multiplier = len(self._concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = operation_sets[name](C)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i + 1]]

            op1 = self._ops[2*i]
            op2 = self._ops[2*i + 1]

            h1 = op1(h1)
            h2 = op2(h2)

            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class SearchFCOSHead(nn.Module):
    def __init__(self):
        super(SearchFCOSHead, self).__init__()
        # TODO change these configs to config file
        self._num_classes = 80
        multiplier = 2
        C_in = 256
        C = 128
        C_curr = 128
        layers = 1
        self._in_features = ['p3','p4','p5','p6','p7']
        self.in_channels_to_top_module = C_in

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(C // 8, C)
        )

        # Cells
        self.cells = nn.ModuleList()
        C_prev_prev, C_prev = C, C
        for i in range(layers):
            cell = Cell(genotype=genotype, C_prev_prev=C_prev_prev, C_prev=C_prev, C=C)
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C

        # Headers
        self.cls_logits = nn.Conv2d(
            multiplier * C_curr, self._num_classes,
            kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            multiplier * C_curr, 4, kernel_size=3,
            stride=1, padding=1
        )
        self.ctrness = nn.Conv2d(
            multiplier * C_curr, 1, kernel_size=3,
            stride=1, padding=1
        )
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(self._in_features))])

        for modules in [self.cls_logits,self.bbox_pred, self.ctrness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, features):

        cls_preds = []
        box_preds = []
        ctr_preds = []
        top_feat = []
        bbox_towers = []

        for i, feat in enumerate(features):
            s0 = s1 = self.stem(feat)

            for j, cell in enumerate(self.cells):
                s0, s1 = s1, cell(s0, s1)
            cls_preds.append(self.cls_logits(s1))
            ctr_preds.append(self.ctrness(s1))
            box_pred = F.relu(self.scales[i](self.bbox_pred(s1)))
            box_preds.append(box_pred)
        return cls_preds, box_preds, ctr_preds, top_feat, bbox_towers


class SimFCOSHead(nn.Module):
    def __init__(self):
        super(SimFCOSHead, self).__init__()
        self._in_features = ['p3','p4','p5','p6','p7']
        channels = 256
        self._num_classes = 80
        self.m = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(channels//8, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(channels//8, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(channels//8, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(channels//8, channels),
            nn.ReLU(),
        )

        self.cls_logits = nn.Conv2d(
            256, self._num_classes,
            kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            256, 4, kernel_size=3,
            stride=1, padding=1
        )
        self.ctrness = nn.Conv2d(
            256, 1, kernel_size=3,
            stride=1, padding=1
        )
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(self._in_features))])

    def forward(self, features):
        feats = [features[f] for f in self._in_features]
        cls_preds = []
        box_preds = []
        ctr_preds = []

        for i, feat in enumerate(feats):
            feat_a = self.m(feat)
            cls_preds.append(self.cls_logits(feat_a))
            ctr_preds.append(self.ctrness(feat_a))
            box_pred = F.relu(self.scales[i](self.bbox_pred(feat_a)))
            box_preds.append(box_pred)
        return cls_preds, box_preds, ctr_preds


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    # m = Cell(genotype=DARTS_FCOS_HEAD, C_in=128, C=128).cuda()
    m = SearchFCOSHead().cuda()
    print(m)
    # m = SimFCOSHead().cuda()
    # print(m)

    # channels = 256
    # m = nn.Sequential(
    #     nn.Conv2d(channels, channels, 3),
    #     nn.GroupNorm(channels//8, channels),
    #     nn.ReLU(),
    #     nn.Conv2d(channels, channels, 3),
    #     nn.GroupNorm(channels//8, channels),
    #     nn.ReLU(),
    #     nn.Conv2d(channels, channels, 3),
    #     nn.GroupNorm(channels//8, channels),
    #     nn.ReLU(),
    #     nn.Conv2d(channels, channels, 3),
    #     nn.GroupNorm(channels//8, channels),
    #     nn.ReLU(),
    # ).cuda()
    # print(m)
    # torch.save(m, "./test.pth")

    import time

    x = torch.randn(1, 256, 64, 64)
    # x = torch.randn(1, 256, 64, 64).cuda()
    features = {
        'p3': torch.randn(1, 256, 100, 122).cuda(),
        'p4': torch.randn(1, 256, 50, 62).cuda(),
        'p5': torch.randn(1, 256, 25, 32).cuda(),
        'p6': torch.randn(1, 256, 13, 16).cuda(),
        'p7': torch.randn(1, 256, 7, 8).cuda(),
    }

    st = time.time()
    for i in range(100):
        y = m(features)
        # y = m(x)

    print("using %f s" % (time.time() - st))

    print(y[0][0].size())




