"""
    entrance of tracking
"""
from utils.registry import Registry
from models import MODEL_BUILDER_REGISTRY

TRACKING_REGISTRY = Registry("TRACKING")

from .siamfcpp import SiamFCPP
from models.backbone import backbone_builder
from models.heads import heads_builder

@MODEL_BUILDER_REGISTRY.register()
def tracking_builder(cfg):
    """
    :param cfg:
    :return:  tracker model
    """
    backbone = backbone_builder(cfg)
    head = heads_builder(cfg)
    tracker = TRACKER_REGISTRY.get(cfg.MODEL.NAME)(cfg, backbone, head)
    return tracker