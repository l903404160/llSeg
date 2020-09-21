# Genotype of FCOS Head
import torch

from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal, normal_concat')
Genotype_fcos = namedtuple('Genotype', 'normal_cls, normal_box')
Genotype_fpn_fcos = namedtuple('Genotype', 'normal_cls, normal_box, normal_fpn, normal_edge')

PRIMITIVES = [
    'conv_k3',
    'conv_k3_d2',
    'conv_k3_d4',
    'conv_k5',
    'conv_k5_d2',
    'sep_k3',
    'sep_k3_d2',
    'sep_k5',
    'sep_k5_d2',
]

PRIMITIVES_box = [
    'conv_k1',
    'conv_k3',
    'conv_k3_d2',
    # 'conv_k3_d4',
    'conv_k5',
    # 'conv_k5_d2',
    'sep_k3',
    # 'sep_k3_d2',
    'sep_k5',
    # 'sep_k5_d2',
]

PRIMITIVES_fpn = [
    'conv_k1',
    'conv_k3',
    'conv_k3_d2',
    'conv_k3_d4',
    'conv_k5',
    # 'conv_k5_d2',
    'sep_k3',
    # 'sep_k3_d2',
    'sep_k5',
    # 'sep_k5_d2',
]


DARTS_FCOS_HEAD = Genotype(normal=[('dil_4_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_4_conv_3x3', 2), ('dil_4_conv_3x3', 0),
                            ('sep_conv_3x3', 1), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1)], normal_concat=range(4, 6))

    # Genotype(normal=[('dil_4_conv_3x3', 0), ('skip_connect', 1), ('dil_4_conv_3x3', 0), ('conv_3x3', 2),
    #                                ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3)], normal_concat=range(4, 6))

    # Genotype(normal=[('dil_4_conv_3x3', 0), ('sep_conv_3x3', 1), ('conv_3x3', 2),('sep_conv_3x3', 0),
    #                                ('sep_conv_3x3', 2), ('side_conv_1x3', 0), ('conv_3x3', 4), ('side_conv_1x3', 2)], normal_concat=range(4, 6))

# Genotype(normal=[('dil_4_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_4_conv_3x3', 0), ('conv_3x3', 2),
#                                    ('side_conv_1x3', 0), ('sep_conv_3x3', 2), ('conv_3x3', 4), ('side_conv_3x1', 3)], normal_concat=range(4, 6))


def parse_darts(weights, steps):
    gene = []
    n = 2
    start = 0
    for i in range(steps):
        end = start + n
        W = weights[start:end].copy()
        # TODO check the none
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))
        edges = edges[:2]
        for j in edges:
            k_best = None
            for k in range(len(W[j])):
                if k_best is None or W[j][k] > W[j][k_best]:
                    k_best = k
            gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
    return gene


def parse_direct(weights, search_part='cls'):
    gene = []

    if search_part == 'cls':
        primitives = PRIMITIVES
    elif search_part == 'box':
        primitives = PRIMITIVES_box

    op_inds = torch.max(weights, dim=1)[1]
    for i in range(len(op_inds)):
        gene.append((primitives[op_inds[i]], i))
    return gene


