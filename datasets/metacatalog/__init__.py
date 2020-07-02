import os
from .catalog import DatasetCatalog, MetadataCatalog

from datasets.segmentation.seg_builtin import register_cityscapes, register_voc_context
from datasets.detection.det_builtin import register_coco, register_visdrone

# Change the implementation like Detectron2. Use the env variable `OW_DATASETS` to indicate the root folder.
_root = os.getenv("OW_DATASETS", "datasets")

# Detection
register_coco(root=_root)
register_visdrone(root=_root)

# Segmentation
register_cityscapes(root=_root)
register_voc_context(root=_root)

