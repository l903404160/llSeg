from abc import ABCMeta, abstractmethod

import torch.nn as nn

from layers import ShapeSpec

__all__ = ["Backbone"]

class Backbone(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network backbone
    """
    def __init__(self):
        """
        The '__init__' method of any subclass can specify its own set of arguments
        """
        super(Backbone, self).__init__()

    @abstractmethod
    def forward(self):
        """
        Subclasses must override this method,
        but adhere to the same return type
        :return:
        dict[str->Tensor]: mapping from feature name
        """
        pass

    @property
    def size_divisibility(self):
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        :return:
        """
        return 0

    def output_shape(self):
        """
        :return: dict[str -> ShpaeSpec]
        """
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }