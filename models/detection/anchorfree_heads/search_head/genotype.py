# Genotype of FCOS Head

from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal, normal_concat')

PRIMITIVES = [
    'conv_3x3',
    # 'conv_5x5',
    'skip_connect',
    'sep_conv_3x3',
    # 'sep_conv_5x5',
    'dil_2_conv_3x3',
    'dil_4_conv_3x3',
    'side_conv_1x3',
    'side_conv_3x1'
]

DARTS_FCOS_HEAD = Genotype(normal=[('dil_4_conv_3x3', 0), ('skip_connect', 1), ('dil_4_conv_3x3', 0), ('conv_3x3', 2),
                                   ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3)], normal_concat=range(4, 6))


    # Genotype(normal=[('dil_4_conv_3x3', 0), ('sep_conv_3x3', 1), ('conv_3x3', 2),('sep_conv_3x3', 0),
    #                                ('sep_conv_3x3', 2), ('side_conv_1x3', 0), ('conv_3x3', 4), ('side_conv_1x3', 2)], normal_concat=range(4, 6))



# Genotype(normal=[('dil_4_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_4_conv_3x3', 0), ('conv_3x3', 2),
#                                    ('side_conv_1x3', 0), ('sep_conv_3x3', 2), ('conv_3x3', 4), ('side_conv_3x1', 3)], normal_concat=range(4, 6))