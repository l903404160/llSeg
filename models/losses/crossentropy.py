import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class PlainCrossEntropyLoss(nn.Module):
    def __init__(self, cfg):
        super(PlainCrossEntropyLoss, self).__init__()
        self.loss_func = CrossEntropyLoss(ignore_index=cfg.MODEL.IGNORE_LABEL)

    def forward(self, pred, label):
        assert pred.dim() == 4
        assert label.dim() == 3
        assert pred.size(0) == label.size(0), "Batch size should be same, got {} in prediction and {} in label ".format(
            pred.size(0), label.size(0))
        assert pred.size(2) == label.size(1), "H should be same, got {} in prediction and {} in label ".format(
            pred.size(0), label.size(0))
        assert pred.size(3) == label.size(2), "W should be same, got {} in prediction and {} in label ".format(
            pred.size(0), label.size(0))
        return self.loss_func(pred, label)


class OHEMCrossEntropyLoss(nn.Module):
    def __init__(self, cfg):
        super(OHEMCrossEntropyLoss, self).__init__()

        self.ignore_label = cfg.MODEL.IGNORE_LABEL
        self.thresh = float(cfg.MODEL.HEAD.LOSS_THRESH)
        self.min_kept = int(cfg.MODEL.HEAD.LOSS_MIN_KEPT)
        self.weights = cfg.MODEL.HEAD.LOSS_WEIGHTS
        self.reduction = cfg.MODEL.HEAD.LOSS_REDUCTION

    def forward(self, pred, label):
        assert pred.dim() == 4
        assert label.dim() == 3
        assert pred.size(0) == label.size(0), "Batch size should be same, got {} in prediction and {} in label ".format(pred.size(0), label.size(0))
        assert pred.size(2) == label.size(1), "H should be same, got {} in prediction and {} in label ".format(pred.size(0), label.size(0))
        assert pred.size(3) == label.size(2), "W should be same, got {} in prediction and {} in label ".format(pred.size(0), label.size(0))
        prob_out = F.softmax(pred, dim=1)
        tmp_label = label.clone()
        tmp_label[tmp_label == self.ignore_label] = 0
        prob = prob_out.gather(1, tmp_label.unsqueeze(1))
        mask = label.contiguous().view(-1, ) != self.ignore_label
        sort_prob, sort_indices = prob.contiguous().view(-1, )[mask].contiguous().sort()
        min_threshold = sort_prob[min(self.min_kept, sort_prob.numel() - 1)] if sort_prob.numel() > 0 else 0.0
        threshold = max(min_threshold, self.thresh)
        loss_matrix = F.cross_entropy(pred, label, ignore_index=self.ignore_label, reduction='none').contiguous().view(-1, )
        sort_loss_matrix = loss_matrix[mask][sort_indices]
        select_loss_matrix = sort_loss_matrix[sort_prob < threshold]

        if self.reduction == 'sum' or select_loss_matrix.numel() == 0:
            return select_loss_matrix.sum()
        elif self.reduction == 'mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError

