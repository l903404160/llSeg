"""
    entrance of backbones
"""
from utils.registry import Registry

BACKBONE_REGISTRY = Registry("BACKBONE")

from .alexnet import AlexNet
from .googlenet import Inception3


def backbone_builder(cfg):
    """
    Backbone builder
    :param cfg:
    :return:  backbone
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)()
    return backbone

