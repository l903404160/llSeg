import torch.nn as nn
from utils.registry import Registry

DET_ANCHORFREE_HEADS_REGISRY = Registry("DET_ANCHORFREE_HEAD")

from .corner_head import cornernet_head_builder
from .borderdet import borderdet_head_builder
from .fcos.fcos_head import fcos_head_builder


def det_onestage_anchorfree_builder(cfg, input_shape=None):
    builder = DET_ANCHORFREE_HEADS_REGISRY.get(cfg.MODEL.ANCHORFREE_HEADS.NAME)
    anchorfreehead = builder(cfg, input_shape)
    return anchorfreehead

