import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.detection.anchorfree_heads.fcos.fcos_tools import Scale
from models.detection.anchorfree_heads.search_head.genotype import Genotype, DARTS_FCOS_HEAD, Genotype_fcos
from models.detection.anchorfree_heads.search_head.operations import operation_sets

"""
    Generate Head model according to the Genotype
"""

# genotype = Genotype(normal=[('dil_4_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_4_conv_3x3', 2), ('dil_4_conv_3x3', 0),
#                             ('sep_conv_3x3', 1), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1)], normal_concat=range(4, 6))

genotype = Genotype_fcos(normal_cls=[('mbconv_k3_e3', 0), ('mbconv_k3_e3', 1), ('mbconv_k3_e1', 2), ('mbconv_k3_e3', 3)],
                    normal_box=[('mbconv_k3_e3_d2', 0), ('mbconv_k3_e3', 1), ('mbconv_k3_e3_d2', 2), ('mbconv_k3_e3_d2', 3)])


class DirectCell(nn.Module):
    def __init__(self, genotype:Genotype, C_in, C_mid, C_out, part='cls'):
        super(DirectCell, self).__init__()
        self._C_in = C_in
        self._C_out = C_out
        self._C_mid = C_mid

        if part == 'cls':
            op_names, indices = zip(*genotype.normal_cls)
        elif part == 'box':
            op_names, indices = zip(*genotype.normal_box)
        else:
            op_names, indices = [], []

        assert len(op_names) == len(indices)

        self._ops = nn.ModuleList()

        for i, (name, ind) in enumerate(zip(op_names, indices)):
            if i == 0:
                op = operation_sets[name](self._C_in, self._C_mid, 1)
            elif i == (len(op_names) - 1):
                op = operation_sets[name](self._C_mid, self._C_out, 1)
            else:
                op = operation_sets[name](self._C_mid, self._C_mid, 1)
            self._ops += [op]

    def forward(self, s0):
        for i in range(len(self._ops)):
            s0 = self._ops[i](s0)
        return s0


class SearchFCOSHead(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out):
        super(SearchFCOSHead, self).__init__()
        # TODO change these configs to config file
        self._dim_in = dim_in
        self._dim_mid = dim_mid
        self._dim_out = dim_out
        self._num_classes = 80
        layers = 1
        self._in_features = ['p3','p4','p5','p6','p7']
        self.in_channels_to_top_module = self._dim_in

        # Cells
        self.cls_cells = nn.ModuleList()
        self.box_cells = nn.ModuleList()
        for i in range(layers):
            cell = DirectCell(genotype, C_in=self._dim_in, C_mid=self._dim_mid, C_out=self._dim_out, part='cls')
            self.cls_cells += [cell]
            cell = DirectCell(genotype, C_in=self._dim_in, C_mid=self._dim_mid, C_out=self._dim_out, part='box')
            self.box_cells += [cell]

        # Headers
        self.cls_logits = nn.Conv2d(
            self._dim_out, self._num_classes,
            kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            self._dim_out, 4, kernel_size=3,
            stride=1, padding=1
        )
        self.ctrness = nn.Conv2d(
            self._dim_out, 1, kernel_size=3,
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
            for j, cell in enumerate(self.cls_cells):
                if j == 0:
                    cls_feat = cell(feat)
                    box_feat = self.box_cells[j](feat)
                else:
                    cls_feat = cell(cls_feat)
                    box_feat = self.box_cells[j](box_feat)

            cls_preds.append(self.cls_logits(cls_feat))
            ctr_preds.append(self.ctrness(box_feat))
            box_pred = F.relu(self.scales[i](self.bbox_pred(box_feat)))
            box_preds.append(box_pred)
        return cls_preds, box_preds, ctr_preds, top_feat, bbox_towers


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    # m = Cell(genotype=DARTS_FCOS_HEAD, C_in=128, C=128).cuda()
    m = SearchFCOSHead(dim_in=256, dim_mid=256, dim_out=256).cuda()
    print(m)
    # m = SimFCOSHead().cuda()
    # print(m)


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
    features = [features[k] for k in features.keys()]

    st = time.time()
    for i in range(100):
        y = m(features)
        # y = m(x)

    print("using %f s" % (time.time() - st))

    print(y[0][0].size())




