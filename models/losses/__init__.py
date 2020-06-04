from .crossentropy import OHEMCrossEntropyLoss, PlainCrossEntropyLoss

loss_dict = {
    'CELoss': PlainCrossEntropyLoss,
    'OHEMCELoss': OHEMCrossEntropyLoss
}


def get_loss_from_cfg(cfg):
    return loss_dict[cfg.MODEL.HEAD.LOSS](cfg)

