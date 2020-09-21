import torch
import torch.nn as nn
import torch.nn.functional as F
from models.detection.anchorfree_heads.search_head.cells import PCMixedOp, MixedOp
from models.detection.anchorfree_heads.search_head.genotype import PRIMITIVES_fpn

primitives_dict = {
    'fpn': PRIMITIVES_fpn,
}


class FPNCell(nn.Module):
    def __init__(self, C_in, C_out, pc=False, primitives=None):
        super(FPNCell, self).__init__()
        self._ops = nn.ModuleList()
        self.fuse_type = 'sum'

        if pc:
            cell_func = PCMixedOp
        else:
            cell_func = MixedOp

        self.lateral_conv = cell_func(C_in, C_out, stride=1, primitives=primitives)
        self.output_conv = cell_func(C_in, C_out, stride=1, primitives=primitives)

    def forward(self, s0, s1, weights, weight_invert=False):
        size = s0.size()[-2:]
        if weight_invert:
            s0 = self.lateral_conv(s0, weights[1])
        else:
            s0 = self.lateral_conv(s0, weights[0])

        s1 = F.interpolate(s1, size=size, mode='nearest')
        s0 = s0 + s1
        if self.fuse_type == 'avg':
            s0 /= 2
        if weight_invert:
            s0 = self.output_conv(s0, weights[0])
        else:
            s0 = self.output_conv(s0, weights[1])
        return s0


class SearchFPN(nn.Module):
    def __init__(self, C_in, C_out, pc=False, primitives=None):
        super(SearchFPN, self).__init__()
        self.in_features = ['p3','p4','p5','p6','p7']
        self.cell_nums = [1,2,2,2,1]

        total_num = sum(self.cell_nums)
        self.fpn_cells = nn.ModuleList()
        for i in range(total_num):
            cell = FPNCell(C_in, C_out, pc, primitives=primitives)
            self.fpn_cells += [cell]

    def forward(self, features, weights_op, weights_edge):
        feats = [features[f] for f in self.in_features]
        # p3
        p3 = self.fpn_cells[0](feats[0], feats[1], weights_op[:2])
        # p4
        p4_1 = self.fpn_cells[1](feats[1], feats[0], weights_op[2:4], weight_invert=True)
        p4_2 = self.fpn_cells[2](feats[1], feats[2], weights_op[3:5])
        p4 = weights_edge[0][0] * p4_1 + weights_edge[0][1] * p4_2
        # p5
        p5_1 = self.fpn_cells[3](feats[2], feats[1], weights_op[5:7], weight_invert=True)
        p5_2 = self.fpn_cells[4](feats[2], feats[3], weights_op[6:8])
        p5 = weights_edge[1][0] * p5_1 + weights_edge[1][1] * p5_2
        # p6
        p6_1 = self.fpn_cells[5](feats[3], feats[2], weights_op[8:10], weight_invert=True)
        p6_2 = self.fpn_cells[6](feats[3], feats[4], weights_op[9:11])
        p6 = weights_edge[2][0] * p6_1 + weights_edge[2][1] * p6_2
        # p7
        p7 = self.fpn_cells[-1](feats[-1], feats[-2], weights_op[-2:])
        return {
            'p3': p3,
            'p4': p4,
            'p5': p5,
            'p6': p6,
            'p7': p7,
        }


def build_fpn_cell(layers, C_in, C_out, pc=False, search_part='fpn'):
    primitives = primitives_dict[search_part]

    cells = nn.ModuleList()
    for i in range(layers):
        cell = SearchFPN(C_in, C_out, pc, primitives)
        cells += [cell]

    k = sum([2,3,3,3,2])
    edge = 6
    return cells, k, edge


def parse_fpn(weights_op, weights_edge, search_part='fpn'):
    gene = []
    edge_gene = []

    primitives = primitives_dict[search_part]

    op_inds = torch.max(weights_op, dim=1)[1]
    edge_inds = torch.max(weights_edge, dim=1)[1]
    for i in range(weights_edge.size()[0]):
        edge_gene.append(('p' + str(i+4), edge_inds[i].item()))

    # p3
    ops = op_inds[:2]
    gene.append((primitives[ops[0]], 0))
    gene.append((primitives[ops[1]], 1))
    # p4
    ops = op_inds[2:5]
    if edge_inds[0] == 0:
        gene.append((primitives[ops[1]], 0))
        gene.append((primitives[ops[0]], 1))
    else:
        gene.append((primitives[ops[1]], 0))
        gene.append((primitives[ops[2]], 1))
    # p5
    ops = op_inds[5:8]
    if edge_inds[1] == 0:
        gene.append((primitives[ops[1]], 0))
        gene.append((primitives[ops[0]], 1))
    else:
        gene.append((primitives[ops[1]], 0))
        gene.append((primitives[ops[2]], 1))
    # p6
    ops = op_inds[8:11]
    if edge_inds[2] == 0:
        gene.append((primitives[ops[1]], 0))
        gene.append((primitives[ops[0]], 1))
    else:
        gene.append((primitives[ops[1]], 0))
        gene.append((primitives[ops[2]], 1))
    # p7
    ops = op_inds[-2:]
    gene.append((primitives[ops[0]], 0))
    gene.append((primitives[ops[1]], 1))

    return gene, edge_gene

if __name__ == '__main__':
    weights_op = torch.randn(13, len(PRIMITIVES_fpn))
    weights_op = torch.softmax(weights_op, dim=-1)
    weights_edge = torch.randn(6)
    weights_edge = weights_edge.view(-1, 2)
    weights_edge = torch.softmax(weights_edge, dim=-1)

    gene, edge_gene = parse_fpn(weights_op, weights_edge)
    print(gene)
    print(edge_gene)