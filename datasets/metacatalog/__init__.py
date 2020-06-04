from .bulitin import register_all_cityscapes, register_all_voc_context, register_all_suim
from configs.defaults import _C as cfg

register_all_cityscapes(root=cfg.DATASETS.ROOT)
register_all_voc_context(root=cfg.DATASETS.ROOT)
# register_all_suim(root=cfg.DATASETS.ROOT)

