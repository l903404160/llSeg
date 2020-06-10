"""
    entrance of segmentation
"""
from utils.registry import Registry
from models import MODEL_BUILDER_REGISTRY


from .base import GeneralSemanticSegmentationModel


@MODEL_BUILDER_REGISTRY.register()
def segmentation_builder(cfg):
    """
    :param cfg:
    :return:  segmentation model
    """
    # TODO set a general model builder for segmentation
    segmentation_model = GeneralSemanticSegmentationModel(cfg)
    return segmentation_model
