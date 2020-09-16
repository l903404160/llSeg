import torch
import torch.nn as nn
import torch.nn.functional as F
from structures import Instances

from utils.nn.focal_loss import sigmoid_focal_loss_jit
from utils.comm import get_world_size, reduce_sum


class IOULoss(nn.Module):
    """
    Intersetion Over Union (IoU) loss which supports three
    different IoU computations:
    * IoU
    * Linear IoU
    * gIoU
    """
    def __init__(self, iou_type='iou'):
        super(IOULoss, self).__init__()
        self.loc_loss_type = iou_type

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: Nx4 predicted bounding boxes
            target: Nx4 target bounding boxes
            weight: N loss weight for each instance
        """
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]
        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        g_w_intersect = torch.max(pred_left, target_left) + \
                        torch.max(pred_right, target_right)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                        torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None:
            return (losses * weight).sum()
        else:
            return losses.sum()


class FCOSLoss(nn.Module):
    def __init__(self, cfg):
        super(FCOSLoss, self).__init__()
        self.focal_loss_alpha = cfg.MODEL.FCOS_HEADS.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS_HEADS.LOSS_GAMMA
        self.num_classes = cfg.MODEL.FCOS_HEADS.NUM_CLASSES
        self.loc_loss_func = IOULoss(cfg.MODEL.FCOS_HEADS.LOC_LOSS_TYPE)

    def forward(self, instances:Instances):
        num_classes = instances.logits_pred.size(1)
        assert num_classes == self.num_classes

        labels = instances.labels.flatten()

        pos_inds = torch.nonzero(labels != num_classes).squeeze(1)
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        # prepare one_hot
        class_target = torch.zeros_like(instances.logits_pred)
        class_target[pos_inds, labels[pos_inds]] = 1

        #TODO pred_logits 突然变得无穷小

        class_loss = sigmoid_focal_loss_jit(
            instances.logits_pred,
            class_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_pos_avg

        instances = instances[pos_inds]
        instances.pos_inds = pos_inds

        ctrness_targets = self._compute_ctrness_targets(instances.reg_targets)
        ctrness_targets_sum = ctrness_targets.sum()
        loss_denorm = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)
        instances.gt_ctrs = ctrness_targets

        if pos_inds.numel() > 0:
            reg_loss = self.loc_loss_func(
                instances.reg_pred,
                instances.reg_targets,
                ctrness_targets
            ) / loss_denorm

            ctrness_loss = F.binary_cross_entropy_with_logits(
                instances.ctrness_pred,
                ctrness_targets,
                reduction="sum"
            ) / num_pos_avg

        else:
            reg_loss = instances.reg_pred.sum() * 0.0
            ctrness_loss = instances.ctrness_pred.sum() * 0.0
        losses = {
            "loss_fcos_cls": class_loss,
            "loss_fcos_loc": reg_loss,
            "loss_fcos_ctr": ctrness_loss
        }
        # extras = {
        #     "instances": instances,
        #     "loss_denorm": loss_denorm
        # }
        return losses

    @staticmethod
    def _compute_ctrness_targets(reg_targets):
        if len(reg_targets) == 0:
            return reg_targets.new_zeros(len(reg_targets))
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(ctrness)


def iou_loss(inputs, targets, weight=None, box_mode="xyxy", loss_type="iou", reduction="none"):
    """
    Compute iou loss of type ['iou', 'giou', 'linear_iou']
    Args:
        inputs (tensor): pred values
        targets (tensor): target values
        weight (tensor): loss weight
        box_mode (str): 'xyxy' or 'ltrb', 'ltrb' is currently supported.
        loss_type (str): 'giou' or 'iou' or 'linear_iou'
        reduction (str): reduction manner
    Returns:
        loss (tensor): computed iou loss.
    """
    if box_mode == "ltrb":
        inputs = torch.cat((-inputs[..., :2], inputs[..., 2:]), dim=-1)
        targets = torch.cat((-targets[..., :2], targets[..., 2:]), dim=-1)
    elif box_mode != "xyxy":
        raise NotImplementedError

    eps = torch.finfo(torch.float32).eps

    inputs_area = (inputs[..., 2] - inputs[..., 0]).clamp_(min=0) \
        * (inputs[..., 3] - inputs[..., 1]).clamp_(min=0)
    targets_area = (targets[..., 2] - targets[..., 0]).clamp_(min=0) \
        * (targets[..., 3] - targets[..., 1]).clamp_(min=0)

    w_intersect = (torch.min(inputs[..., 2], targets[..., 2])
                   - torch.max(inputs[..., 0], targets[..., 0])).clamp_(min=0)
    h_intersect = (torch.min(inputs[..., 3], targets[..., 3])
                   - torch.max(inputs[..., 1], targets[..., 1])).clamp_(min=0)

    area_intersect = w_intersect * h_intersect
    area_union = targets_area + inputs_area - area_intersect
    ious = area_intersect / area_union.clamp(min=eps)

    if loss_type == "iou":
        loss = -ious.clamp(min=eps).log()
    elif loss_type == "linear_iou":
        loss = 1 - ious
    elif loss_type == "giou":
        g_w_intersect = torch.max(inputs[..., 2], targets[..., 2]) \
            - torch.min(inputs[..., 0], targets[..., 0])
        g_h_intersect = torch.max(inputs[..., 3], targets[..., 3]) \
            - torch.min(inputs[..., 1], targets[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss = 1 - gious
    else:
        raise NotImplementedError
    if weight is not None:
        loss = loss * weight.view(loss.size())
        if reduction == "mean":
            loss = loss.sum() / max(weight.sum().item(), eps)
    else:
        if reduction == "mean":
            loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()

    return loss