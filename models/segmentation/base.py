"""
    General Semantic Segmentation Models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.segmentation.backbone import backbone_builder
from models.segmentation.heads import head_builder


class GeneralSemanticSegmentationModel(nn.Module):
    def __init__(self, cfg):
        super(GeneralSemanticSegmentationModel, self).__init__()
        self._cfg = cfg
        self.device = cfg.MODEL.DEVICE
        self.pos_information = cfg.MODEL.HANET.POS_INFORMATION
        self.norm = self.preprocess_image()

        self.backbone = backbone_builder(cfg)
        self.head = head_builder(cfg)
        self.to(self.device)

    def preprocess_image(self):
        mean = torch.Tensor(self._cfg.MODEL.PIXEL_MEAN).view(1, 3, 1, 1).to(self.device)
        std = torch.Tensor(self._cfg.MODEL.PIXEL_STD).view(1, 3, 1, 1).to(self.device)
        norm = lambda x: (x-mean) / std
        return norm

    def forward(self, data_dict):
        data_input = data_dict['image'].to(self.device)
        data_input = self.norm(data_input)
        size = data_input.size()[-2:]
        if 'sem_seg' not in data_dict.keys():
            label = None
        else:
            label = data_dict['sem_seg'].to(self.device)
        if 'scale' in data_dict.keys():
            scale = data_dict['scale']
            h, w = data_dict['height'], data_dict['width']
            newh, neww = scale * h, scale * w
            neww = int(neww + 0.5)
            newh = int(newh + 0.5)
            data_input = F.interpolate(data_input, size=(newh, neww), mode='bilinear', align_corners=True)
            if data_dict['flip']:
                # TODO change the flip process
                data_input = torch.flip(data_input, dims=[3])

        feats = self.backbone(data_input)
        if self.training:
            assert label is not None, "Label should have correct value during training"
            if self.pos_information:
                pos = (data_dict['pos_h'], data_dict['pos_w'])
                loss_dict = self.head(feats, label, pos)
            else:
                loss_dict = self.head(feats, label)
            return loss_dict
        else:
            pred = self.head(feats)
            pred = F.interpolate(pred, size=size, mode='bilinear', align_corners=True)
            prediction = F.softmax(pred, dim=1)
            return {'sem_seg': prediction}
