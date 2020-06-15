"""
    entrance of models
"""
import torch.nn as nn
from utils.registry import Registry

MODEL_BUILDER_REGISTRY = Registry("MODEL_BUILDER")

from .tracking import tracking_builder
from .segmentation import segmentation_builder
from .detection import base_rcnn_builder, base_retina_builder


def model_builder(cfg) -> nn.Module:
    builder = MODEL_BUILDER_REGISTRY.get(cfg.MODEL.BUILDER)
    model = builder(cfg)
    return model

