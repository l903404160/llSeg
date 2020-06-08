"""
    entrance of detection
"""
from utils.registry import Registry
from models import MODEL_BUILDER_REGISTRY

SEGMENTATION_REGISTRY = Registry("DETECTION")


@MODEL_BUILDER_REGISTRY.register()
def detection_builder(cfg):
    """
    :param cfg:
    :return:  segmentation model
    """
    pass
    # TODO complete the detection code.
    # segmentation_model = GeneralSemanticSegmentationModel(cfg)
    # return segmentation_model
