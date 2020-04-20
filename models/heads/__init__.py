"""
    entrance of heads
"""
from utils.registry import Registry

HEADS_REGISTRY = Registry("HEADS")

from .track_heads.siamfcpp_head import SiamFCPPHead

def heads_builder(cfg):
    """
    Backbone builder
    :param cfg:
    :return:  backbone
    """
    heads_name = cfg.MODEL.TRACK_HEAD.NAME
    head = HEADS_REGISTRY.get(heads_name)(cfg)
    return head