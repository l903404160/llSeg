"""
    entrance of detection
"""
import torch
from models import MODEL_BUILDER_REGISTRY
from .base import GeneralizedRCNN


@MODEL_BUILDER_REGISTRY.register()
def detection_builder(cfg):
    """
    :param cfg:
    :return:  detection model
    """
    detection_model = GeneralizedRCNN(cfg)
    detection_model.to(torch.device(cfg.MODEL.DEVICE))
    return detection_model
