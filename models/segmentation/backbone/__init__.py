from utils.registry import Registry
SEG_BACKBONE_REGISTRY = Registry("SEG_BACKBONE")

from .resnet import resnet_builder
from .pcnet import pcnet_builder


def backbone_builder(cfg):
    builder = SEG_BACKBONE_REGISTRY.get(cfg.MODEL.BACKBONE.NAME)
    backbone = builder(cfg)
    return backbone


