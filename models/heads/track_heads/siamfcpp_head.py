import torch
import numpy as np
import torch.nn as nn
from models.modules.base import conv_bn_relu

# registry
from models.heads import HEADS_REGISTRY


@HEADS_REGISTRY.register()
class SiamFCPPHead(nn.Module):
    def __init__(self, cfg):
        super(SiamFCPPHead, self).__init__()
        cls_channel = cfg.MODEL.TRACK_HEAD.CLS_CHANNEL
        bbox_channel = cfg.MODEL.TRACK_HEAD.BBOX_CHANNEL
        score_size = 17
        x_size = 303
        self.total_strides = 8

        self.cls_conv3x3 = nn.Sequential(
            conv_bn_relu(cls_channel, cls_channel, stride=1, kszie=3, pad=0, has_bn=False),
            conv_bn_relu(cls_channel, cls_channel, stride=1, kszie=3, pad=0, has_bn=False),
            conv_bn_relu(cls_channel, cls_channel, stride=1, kszie=3, pad=0, has_bn=True),
        )

        self.bbox_conv3x3 = nn.Sequential(
            conv_bn_relu(bbox_channel, bbox_channel, stride=1, kszie=3, pad=0, has_bn=False),
            conv_bn_relu(bbox_channel, bbox_channel, stride=1, kszie=3, pad=0, has_bn=False),
            conv_bn_relu(bbox_channel, bbox_channel, stride=1, kszie=3, pad=0, has_bn=True),
        )

        # Output Score output
        self.cls_score_p5 = conv_bn_relu(cls_channel, 1, stride=1, kszie=1, pad=0, has_relu=False)
        self.ctr_score_p5 = conv_bn_relu(cls_channel, 1, stride=1, kszie=1, pad=0, has_relu=False)
        self.bbox_offsets_p5 = conv_bn_relu(bbox_channel, 4, stride=1, kszie=1, pad=0, has_relu=False)

        self.bi = torch.nn.Parameter(torch.tensor(0.).type(torch.Tensor))
        self.si = torch.nn.Parameter(torch.tensor(1.).type(torch.Tensor))

        self.fm_ctr = self.get_fm_ctr(score_size, x_size, self.total_strides)

        self.initialize_weights()

    def initialize_weights(self):
        conv_weight_std = 0.0001
        pi = 0.01
        bv = -np.log((1-pi) / pi)
        for m in self.modules():
            if isinstance(m, conv_bn_relu):
                torch.nn.init.normal_(m.conv.weight, std=conv_weight_std)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.conv.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(m.conv.bias, -bound, bound)

        torch.nn.init.constant_(self.cls_score_p5.conv.bias, torch.tensor(bv))

    def get_fm_ctr(self, score_size, x_size, total_stride):
        score_offsets = (x_size - 1 - (score_size - 1) * total_stride) // 2
        ctr = self.get_xy_ctr(score_size=score_size, score_offset=score_offsets, total_stride=total_stride)
        return ctr

    def get_xy_ctr(self, score_size, score_offset, total_stride):
        batch, fm_height, fm_width = 1, score_size, score_size
        y_list = torch.linspace(0, fm_height-1, fm_height).reshape(1, fm_height, 1, 1).repeat(1, 1, fm_width, 1)
        x_list = torch.linspace(0, fm_width-1, fm_width).reshape(1, 1, fm_width, 1).repeat(1, fm_height, 1, 1)
        xy_list = score_offset + torch.cat([x_list, y_list], 3) * total_stride
        xy_ctr = xy_list.repeat(batch, 1, 1, 1).reshape(batch, -1, 2)
        xy_ctr = xy_ctr.type(torch.Tensor)
        return xy_ctr

    def get_bbox(self, xy_ctr, offsets):
        offsets = offsets.permute(0, 2, 3, 1)
        offsets = offsets.reshape(offsets.size(0), -1, 4)

        xy0 = (xy_ctr[:,:,:] - offsets[:,:,:2])
        xy1 = (xy_ctr[:,:,:] + offsets[:,:,2:])
        bboxes_pred = torch.cat([xy0, xy1], 2)
        return bboxes_pred

    def forward(self, c_out, r_out):
        cls = c_out
        bbox = r_out

        cls = self.cls_conv3x3(cls)
        bbox = self.bbox_conv3x3(bbox)

        # classification score
        cls_score = self.cls_score_p5(cls)
        cls_score = cls_score.permute(0, 2, 3, 1)
        cls_score = cls_score.reshape(cls_score.size(0), -1, 1)
        # center-ness score
        ctr_score = self.ctr_score_p5(cls)
        ctr_score = ctr_score.permute(0, 2, 3, 1)
        ctr_score = ctr_score.reshape(ctr_score.size(0), -1, 1)
        # regssion
        offsets = self.bbox_offsets_p5(bbox)
        offsets = torch.exp(self.si * offsets + self.bi) + self.total_strides
        # bbox decoding
        self.fm_ctr = self.fm_ctr.to(offsets.device)
        bbox = self.get_bbox(self.fm_ctr, offsets)
        return [cls_score, ctr_score, bbox]


if __name__ == '__main__':
    model = SiamFCPPHead()
    print(model)

    cls = torch.randn(2, 256, 17+2*3, 17+2*3)
    bbox = torch.randn(2, 256, 17+2*3, 17+2*3)

    result = model(cls, bbox)
    print(len(result))