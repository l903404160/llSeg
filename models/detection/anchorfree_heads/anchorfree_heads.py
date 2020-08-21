# Anchor Free Head
# For CornerNet CenterNet ..
import torch.nn as nn

from models.detection.anchorfree_heads import DET_ANCHORFREE_HEADS_REGISRY


class AnchorFreeHeadBase(nn.Module):
    def __init__(self):
        super(AnchorFreeHeadBase, self).__init__()

    def forward(self, in_features):
        raise NotImplementedError

    def forward_bbox(self, features, targets):
        raise NotImplementedError

    def losses(self, pred, targets):
        raise NotImplementedError

    def inference(self, features, image_sizes):
        raise NotImplementedError


class AnchorFreeHead(nn.Module):
    """
        AnchorFreeHead for wrapping the AnchorFree methods.
    """
    def __init__(self, cfg):
        super(AnchorFreeHead, self).__init__()
        self.anchorfree_head = self._init_anchor_free_head(cfg)

    def _init_anchor_free_head(self, cfg):
        name = cfg.MODEL.ANCHORFREE_HEADS.NAME
        anchorfree_head = DET_ANCHORFREE_HEADS_REGISRY.get(name)(cfg)
        return anchorfree_head

    def forward(self, in_features, targets=None, image_sizes=None):
        """
        Args:
            in_features: {'hg1': tensor, 'hg2': tensor}
            targets: { TBD }
            image_sizes: [tuple]
        Returns:
            Loss(training) / Predicton(testing)
        """
        # compute the predictions
        feats = self.anchorfree_head(in_features, image_sizes)
        if self.training:
            loss = self.anchorfree_head.forward_bbox(feats, targets=targets) #, image_sizes=image_sizes, in_features=in_features
            return loss
        else:
            predictions = self.anchorfree_head.inference(feats, image_sizes=image_sizes, in_features=feats)
            return predictions


