"""
    entrance of detection
"""
from models import MODEL_BUILDER_REGISTRY
from .base import GeneralizedRCNN


@MODEL_BUILDER_REGISTRY.register()
def detection_builder(cfg):
    """
    :param cfg:
    :return:  detection model
    """
    detection_model = GeneralizedRCNN(cfg)
    return detection_model
