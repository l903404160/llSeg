import torch
# from utils.torch_stat import stat
from torchstat import stat

# from models.modules.non_local import EmNonLocalLayer
from models.modules.ann import BNReLU
from models.segmentation.heads.emanet import EMAU

# m = EmNonLocalLayer(dim_in=512, dim_inter=256, dim_out=512, norm_layer=torch.nn.BatchNorm2d).eval()
# m = APNB(in_channels=512, out_channels=512, key_channels=256, value_channels=256, norm_type='SyncBN', dropout=0.05).eval()

# ---------------------------------------------------------------


# m = APNB(2048, 2048, 256, 256, dropout=0.05, norm_type='SyncBN')
# m = EmNonLocalLayer(dim_in=2048, dim_inter=256, dim_out=512, norm_layer=torch.nn.BatchNorm2d)
m = EMAU(c=512, k=64, stage_num=3)


stat(m, (512, 96, 96))