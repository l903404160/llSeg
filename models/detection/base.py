"""
    General Object Detection Models
"""
import torch
import torch.nn as nn


class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that ontains the following three components:
    1. Per-image feature extraction (backbone network)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """
    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone