import torch
import torch.nn as nn
from models.modules.base import conv_bn_relu, xcorr_depthwise
from models.tracking import TRACKING_REGISTRY


@TRACKING_REGISTRY.register()
class SiamFCPP(nn.Module):
    def __init__(self, cfg, backbone, head, loss=None):
        super(SiamFCPP, self).__init__()
        cls_channel = cfg.MODEL.BACKBONE.CLS_CHANNEL
        bbox_channel = cfg.MODEL.BACKBONE.BBOX_CHANNEL

        self.backbone = backbone
        self.head = head
        self.loss = loss

        self.r_z_k = conv_bn_relu(bbox_channel, bbox_channel, stride=1, kszie=3, pad=0, has_relu=False)
        self.c_z_k = conv_bn_relu(cls_channel, cls_channel, stride=1, kszie=3, pad=0, has_relu=False)
        self.r_x = conv_bn_relu(bbox_channel, bbox_channel, stride=1, kszie=3, pad=0, has_relu=False)
        self.c_x = conv_bn_relu(cls_channel, cls_channel, stride=1, kszie=3, pad=0, has_relu=False)

        self._initialize_conv()

    def _initialize_conv(self):
        conv_weight_std = 0.01
        conv_list = [
            self.c_z_k, self.r_z_k, self.c_x, self.r_x
        ]
        for item in conv_list:
            torch.nn.init.normal_(item.conv.weight, std=conv_weight_std)

    def train_phase(self, target_img, search_img):
        f_z = self.backbone(target_img)
        f_x = self.backbone(search_img)

        # feature adjustment
        c_z_k = self.c_z_k(f_z)
        r_z_k = self.r_z_k(f_z)
        c_x = self.c_x(f_x)
        r_x = self.r_x(f_x)

        # feature matching
        c_out = xcorr_depthwise(c_x, c_z_k)
        r_out = xcorr_depthwise(r_x, r_z_k)

        # head
        fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final = self.head(c_out, r_out)
        predict_data = dict(
            cls_pred=fcos_cls_score_final,
            ctr_pred=fcos_ctr_score_final,
            box_pred=fcos_bbox_final
        )
        return predict_data

    def extract_template(self, target_img):
        with torch.no_grad():
            f_z = self.backbone(target_img)
            c_z_k = self.c_z_k(f_z)
            r_z_k = self.r_z_k(f_z)
        return c_z_k, r_z_k

    def track(self, search_img, c_z_k, r_z_k):
        with torch.no_grad():
            f_x = self.backbone(search_img)
            c_x = self.c_x(f_x)
            r_x = self.r_x(f_x)

            c_out = xcorr_depthwise(c_x, c_z_k)
            r_out = xcorr_depthwise(r_x, r_z_k)
            # track head
            fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final = self.head(c_out, r_out)
            # apply sigmoid
            fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
            fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)
            fcos_score_final = fcos_cls_score_final * fcos_ctr_score_final
            output = fcos_score_final, fcos_bbox_final, fcos_cls_prob_final, fcos_ctr_prob_final
        return output

    def forward(self, target_img, search_img):
        return self.train_phase(target_img, search_img)
