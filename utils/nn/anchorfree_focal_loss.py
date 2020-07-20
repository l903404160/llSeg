# AnchorFree Focal Loss

import torch
import torch.nn as nn


class CornerNetFocalLoss(nn.Module):
    def __init__(self, cfg):
        super(CornerNetFocalLoss, self).__init__()
        # config
        self.pull_weight = cfg.MODEL.ANCHORFREE_HEADS.PULL_WEIGHT
        self.push_weight = cfg.MODEL.ANCHORFREE_HEADS.PUSH_WEIGHT
        self.regr_weight = cfg.MODEL.ANCHORFREE_HEADS.REGR_WEIGHT

    @staticmethod
    def _ae_loss(tag0, tag1, mask):
        num = mask.sum(dim=1, keepdim=True).float()
        # TODO check the bool()
        pull_mask = mask.bool()

        tag0 = tag0.squeeze()
        tag1 = tag1.squeeze()

        tag_mean = (tag0 + tag1) / 2

        tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
        tag0  = tag0[pull_mask].sum()
        tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
        tag1 = tag1[pull_mask].sum()
        pull = tag0 + tag1

        mask = mask.unsqueeze(1) + mask.unsqueeze(2)
        mask = mask.eq(2)
        num = num.unsqueeze(2)
        num2 = (num - 1) * num
        dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)

        dist = 1 - torch.abs(dist)
        dist = nn.functional.relu(dist, inplace=True)
        dist = dist - 1 / (num + 1e-4)
        dist = dist / (num2 + 1e-4)
        dist = dist[mask]
        push = dist.sum()
        return pull, push

    @staticmethod
    def _neg_loss(preds, targets):
        pos_inds = targets.eq(1)
        neg_inds = targets.lt(1)

        neg_weights = torch.pow(1 - targets[neg_inds], 4)

        loss = 0

        pos_pred = preds[pos_inds]
        neg_pred = preds[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    @staticmethod
    def _regr_loss(regr, target_regr, mask):
        num = mask.float().sum()
        mask = mask.unsqueeze(2).expand_as(target_regr)
        mask = mask.bool()
        regr = regr[mask]
        target_regr = target_regr[mask]

        regr_loss = nn.functional.smooth_l1_loss(regr, target_regr, reduction='sum')

        regr_loss = regr_loss / (num + 1e-4)
        return regr_loss

    def forward(self, preds, targets: dict):
        """
        Args:
            preds:
            targets: {
                dataset_dict["tl_heatmap"] = tl_heatmap
                dataset_dict["br_heatmap"] = br_heatmap
                dataset_dict["tl_regr"] = tl_regr
                dataset_dict["br_regr"] = br_regr
                dataset_dict["tag_mask"] = tag_mask
                dataset_dict["tl_tag"] = tl_tag
                dataset_dict["br_tag"] = br_tag
            }
        Returns:

        """
        # Loss forward need improvement
        tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr = preds

        gt_tl_heat = torch.stack([targets[i]['tl_heatmap'] for i in range(len(targets))], dim=0).cuda()
        gt_br_heat = torch.stack([targets[i]['br_heatmap'] for i in range(len(targets))], dim=0).cuda()
        gt_mask= torch.stack([targets[i]['tag_mask'] for i in range(len(targets))], dim=0).cuda()
        gt_tl_regr= torch.stack([targets[i]['tl_regr'] for i in range(len(targets))], dim=0).cuda()
        gt_br_regr = torch.stack([targets[i]['br_regr'] for i in range(len(targets))], dim=0).cuda()

        focal_loss = 0
        focal_loss += self._neg_loss(tl_heat, gt_tl_heat)
        focal_loss += self._neg_loss(br_heat, gt_br_heat)

        # tag loss
        pull_loss = 0
        push_loss = 0
        pull, push = self._ae_loss(tl_tag, br_tag, gt_mask)
        pull_loss += pull
        push_loss += push

        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        regr_loss = 0
        regr_loss += self._regr_loss(tl_regr, gt_tl_regr, gt_mask)
        regr_loss += self._regr_loss(br_regr, gt_br_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss

        print("focal_loss + pull_loss + push_loss + regr_loss: " , focal_loss, pull_loss, push_loss, regr_loss)
        loss = focal_loss + pull_loss + push_loss + regr_loss
        return loss.unsqueeze(0)
