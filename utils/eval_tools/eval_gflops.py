import torch
from utils.eval_tools.blocks.apnb import APNB
from utils.eval_tools.blocks.emnl import EmNonLocalLayer, AvgNonLocalLayer
from utils.eval_tools.blocks.null import NullBlock
import time
import sys
#from thop import profile
from torchstat import stat as profile

def test_multi(size):
    model = NullBlock()
    profile(model,size[1:])
    del model
    torch.cuda.empty_cache()
    model = APNB(2048, 2048, 256, 256, dropout=0.05,norm_type="SyncBN")
    profile(model,size[1:])
    del model
    torch.cuda.empty_cache()
    model = EmNonLocalLayer(dim_in=2048, dim_inter=256, dim_out=2048,norm_layer=torch.nn.BatchNorm2d)
    profile(model,size[1:])
    del model
    torch.cuda.empty_cache()


sizes = [[1, 2048, 96, 96], [1, 2048, 128, 128], [1, 2048, 128, 256]]

test_multi(sizes[0])