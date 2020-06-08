"""
    General Object Detection Models
"""
import torch
import torch.nn as nn

from models.detection.backbone import backbone_builder


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
        self.backbone = backbone_builder(cfg)
        # TODO 1. proposal_generator  RoI Head
        self.proposal_generator = proposal_generator_builder(cfg, self.backbone.output_shape())

        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))


if __name__ == '__main__':
    from configs.configs_files.detection.detection_defaults import _C as cfg
    cfg.MODEL.BACKBONE.NAME = "resnet_fpn_builder"
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]


    m = GeneralizedRCNN(cfg)

    import torch
    x = torch.randn(2,3,960, 960)
    y = m.backbone(x)
    print(m)