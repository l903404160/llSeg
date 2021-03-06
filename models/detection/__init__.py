"""
    entrance of detection
"""
import torch
from models import MODEL_BUILDER_REGISTRY
from .base import GeneralizedRCNN
from .retina_base import RetinaNet


@MODEL_BUILDER_REGISTRY.register()
def base_rcnn_builder(cfg):
    """
    :param cfg:
    :return:  detection model
    """
    detection_model = GeneralizedRCNN(cfg)
    detection_model.to(torch.device(cfg.MODEL.DEVICE))
    return detection_model

@MODEL_BUILDER_REGISTRY.register()
def base_retina_builder(cfg):
    """
    :param cfg:
    :return:  detection model
    """
    detection_model = RetinaNet(cfg)
    detection_model.to(torch.device(cfg.MODEL.DEVICE))
    return detection_model
