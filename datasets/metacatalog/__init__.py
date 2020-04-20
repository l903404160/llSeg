from .bulitin import register_all_cityscapes
from configs.defaults import _C as cfg

register_all_cityscapes(root=cfg.DATASETS.ROOT)