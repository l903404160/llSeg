import os
from datasets.metacatalog.catalog import DatasetCatalog, MetadataCatalog
from datasets.metacatalog.builtin_meta import get_builtin_metadata
from datasets.segmentation.cityscapes import load_cityscapes_sem_seg_dict

_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_seg_train": ("cityscapes/leftImg8bit/train", "cityscapes/gtFine/train"),
    "cityscapes_fine_seg_val": ("cityscapes/leftImg8bit/val", "cityscapes/gtFine/val"),
    "cityscapes_fine_seg_test": ("cityscapes/leftImg8bit/test", "cityscapes/gtFine/test"),
}


def register_all_cityscapes(root):
    for k, (img_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = get_builtin_metadata('cityscapes')
        image_dir = os.path.join(root, img_dir)
        gt_dir = os.path.join(root, gt_dir)
        DatasetCatalog.register(k, lambda x=image_dir,y=gt_dir:load_cityscapes_sem_seg_dict(x, y))
        MetadataCatalog.get(k).set(
            image_dir=image_dir, gt=gt_dir, evaluator_type="sem_seg", **meta
        )

