from utils.registry import Registry
SEG_HEAD_REGISTRY = Registry("SEG_HEAD")

from .emanet import emahead_builder
from .pcnet import pchead_builder
from .plain_head import plainhead_builder
from .nl_head import nlhead_builder
from .hanet_head import hanet_builder


def head_builder(cfg):
    builder = SEG_HEAD_REGISTRY.get(cfg.MODEL.HEAD.NAME)
    head = builder(cfg)
    return head