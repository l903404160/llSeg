from utils.registry import Registry

DET_BACKBONE_REGISTRY = Registry("DET_BACKBONE")

from .backbone import Backbone
from .resnet import resnet_builder
from .fpn import retinanet_resnet_fpn_builder, resnet_fpn_builder
from .hourglass import hourglass_builder

def backbone_builder(cfg) -> Backbone:
    builder = DET_BACKBONE_REGISTRY.get(cfg.MODEL.BACKBONE.NAME)
    backbone = builder(cfg)
    return backbone