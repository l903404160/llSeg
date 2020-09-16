import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ROIAlign

import utils.nn.weight_init as weight_init
from utils.nn.smoothL1Loss import smooth_l1_loss
from structures import pairwise_iou


class TwoStageFCOSHead(nn.Module):
    def __init__(self, cfg):
        super(TwoStageFCOSHead, self).__init__()

        self.num_classes = cfg.MODEL.FCOS_HEADS.NUM_CLASSES

        self.align_convs = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.GroupNorm(32, 256),
            nn.ReLU()
        )
        # pooler
        self.roi_poolors = nn.ModuleList([
            ROIAlign(output_size=(7,7), spatial_scale=0.0625, sampling_ratio=0, aligned=True)
        ])

        self.box_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.Linear(1024, 1024)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 80)
        )
        self.box_predictor = nn.Sequential(
            nn.Linear(1024, 4)
        )

        # initialization
        for layer in self.box_head:
            weight_init.c2_xavier_fill(layer)
        nn.init.normal_(self.classifier[0].weight, std=0.01)
        nn.init.normal_(self.box_predictor[0].weight, std=0.001)

    def forward(self, x, proposals, gt_instances):
        # p3
        x = self.align_convs(x['p4'])
        with torch.no_grad():
            num_instances = [len(proposals[i]) for i in range(len(proposals))]
            image_indexes = [torch.ones(len(proposals[i])) * i for i in range(len(proposals))]
            image_indexes = torch.cat(image_indexes, dim=0)[:, None].cuda()
            boxes = torch.cat([proposals[i].pred_boxes.tensor for i in range(len(proposals))], dim=0)
            boxes = torch.cat([image_indexes, boxes], dim=1)

        roi_feats = self.roi_poolors[0](x, boxes)
        roi_feats = torch.flatten(roi_feats, start_dim=1)
        roi_feats = self.box_head(roi_feats)
        cls_pred = self.classifier(roi_feats)
        box_pred = self.box_predictor(roi_feats)
        # cls_pred = torch.split(cls_pred, num_instances, dim=0)
        # box_pred = torch.split(box_pred, num_instances, dim=0)
        preds = cls_pred, box_pred, proposals

        if self.training:
            losses = self.losses(preds, gt_instances)
            return losses
        else:
            return self.inference(preds)

    def inference(self, preds):
        # TODO compute the boxes
        pass

    def losses(self, preds, gt_instances):
        cls_pred, box_pred, proposals = preds

        proposals_targets = self._get_proposal_targets(proposals, gt_instances)

        num_images = len(proposals)

        cls_target = torch.cat(proposals_targets['cls_targets'], dim=0)
        box_target = torch.cat(proposals_targets['reg_targets'], dim=0)
        reg_mask = cls_target != 80

        box_prediction = box_pred[reg_mask, :]
        box_target = box_target[reg_mask, :]
        num_postive = reg_mask.float().sum()

        if num_postive == 0:
            cls_loss = 0.0 * cls_pred.sum()
            reg_loss = 0.0 * box_prediction.sum()
        else:
            cls_loss = F.cross_entropy(cls_pred, cls_target, ignore_index=80,
                                        reduction='mean') / num_images
            reg_loss = smooth_l1_loss(box_prediction, box_target, beta=0,
                                       reduction='mean') / num_images
        return {
            'two_cls_loss': cls_loss,
            'two_reg_loss': reg_loss
        }

    def _get_proposal_targets(self, proposals, gt_instances):
        num_images = len(gt_instances)
        proposal_cls_targets = []
        proposal_reg_targets = []

        for im_ind in range(num_images):
            proposal = proposals[im_ind]
            if len(proposal) == 0:
                reg_targets = torch.zeros_like(proposal.pred_boxes.tensor)
                two_gt_classes = torch.ones_like(proposal.pred_classes) * self.num_classes
                proposal_reg_targets.append(reg_targets)
                proposal_cls_targets.append(two_gt_classes)
                continue
            gt_target = gt_instances[im_ind]
            gt_boxes = gt_target.gt_boxes
            gt_classes = gt_target.gt_classes
            candidate_boxes = proposal.pred_boxes
            iou = pairwise_iou(candidate_boxes, gt_boxes)
            (max_iou, argmax_iou) = iou.max(dim=1)
            invalid = max_iou < 0.6
            two_gt_classes = gt_classes[argmax_iou]
            two_gt_classes[invalid] = self.num_classes

            gt_boxes = gt_boxes[argmax_iou].tensor
            candidate_boxes = candidate_boxes.tensor
            whs = candidate_boxes[:, 2:4] - candidate_boxes[:, 0:2]
            whs = torch.cat([whs, whs], dim=1)

            reg_targets = (gt_boxes - candidate_boxes) / (whs * 0.5)

            proposal_reg_targets.append(reg_targets)
            proposal_cls_targets.append(two_gt_classes)

        return {
            'cls_targets': proposal_cls_targets,
            'reg_targets': proposal_reg_targets
        }

if __name__ == '__main__':
    cfg = None
    m = TwoStageFCOSHead(cfg)
    print(m)
