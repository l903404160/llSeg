from configs.config import get_detection_config

from engine.defaults import default_setup, default_argument_setup

from datasets.detection.builder import build_anchorfree_detection_train_loader
from datasets.detection.det_mapper import AnchorFree_DetDatasetMapper
from datasets.dataset_utils import generate_cornernet_heatmap_tag_regr

import torch

args = default_argument_setup().parse_args()

args.config_file = "/home/haida_sunxin/lqx/code/llseg/configs/configs_files/detection/base_CornerNet.yaml"

args.num_gpus = 1

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_detection_config()
    print(args.config_file)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

cfg = setup(args)

mapper = AnchorFree_DetDatasetMapper(cfg, is_train=True)
dataloader = build_anchorfree_detection_train_loader(cfg)

d = iter(dataloader)

data = next(d)

# print(data)
print(data[0])

from models.detection.anchorfree_base import GeneralizedAnchorFreeDetector

model = GeneralizedAnchorFreeDetector(cfg)
model = model.to(torch.device(cfg.MODEL.DEVICE))

loss = model(data)
print(loss)
lossa = loss['hg1_loss'] + loss['hg2_loss']
lossa.backward()

model = model.eval()

predictions = model(data)

print(predictions)



