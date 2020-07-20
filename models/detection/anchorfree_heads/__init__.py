import torch.nn as nn
from utils.registry import Registry

DET_ANCHORFREE_HEADS_REGISRY = Registry("DET_ANCHORFREE_HEAD")

from .anchorfree_heads import AnchorFreeHead
from .corner_head import cornernet_head_builder

