"""
    The implementation of HANet Head
"""
import torch.nn as nn
from models.segmentation.heads import SEG_HEAD_REGISTRY



# refer to the implementation of `plain_head.py`
# refer to `https://github.com/shachoi/HANet/blob/master/network/deepv3.py`
# @Zhiqiang Song
class HANetHead(nn.Module):
    def __init__(self, cfg):
        super(HANetHead, self).__init__()

    def forward(self, data_input, label=None):
        pass


# register the hanet head into `SEG_HEAD_REGISTRY`
# @Jingfei Sun
@SEG_HEAD_REGISTRY.register()
def hanet_builder(cfg):
    from layers.batch_norm import get_norm
    norm_layer = get_norm(cfg.MODEL.BN_LAYER)
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    return HANetHead(cfg, norm_layer)

