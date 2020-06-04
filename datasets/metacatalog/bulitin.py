import os
from datasets.metacatalog.catalog import DatasetCatalog, MetadataCatalog
from datasets.metacatalog.builtin_meta import get_builtin_metadata
from datasets.segmentation.cityscapes import load_cityscapes_sem_seg_dict
from datasets.segmentation.voccontext import load_pascal_context_sem_seg_dict
from datasets.segmentation.suim import load_suim_sem_seg_dict

_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_seg_train": ("city/leftImg8bit/train", "city/gtFine/train"),
    "cityscapes_fine_seg_val": ("city/leftImg8bit/val", "city/gtFine/val"),
    "cityscapes_fine_seg_test": ("city/leftImg8bit/test", "city/gtFine/test"),
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


_RAW_VOC_CONTEXT_SPLITS = {
    "voc_context_seg_train": ("Context/train", "Context/train_labels"),
    "voc_context_seg_val": ("Context/val", "Context/val_labels"),
}


def register_all_voc_context(root):
    for k, (img_dir, gt_dir) in _RAW_VOC_CONTEXT_SPLITS.items():
        meta = get_builtin_metadata('voc_context')
        image_dir = os.path.join(root, img_dir)
        gt_dir = os.path.join(root, gt_dir)
        DatasetCatalog.register(k, lambda x=image_dir,y=gt_dir:load_pascal_context_sem_seg_dict(x, y))
        MetadataCatalog.get(k).set(
            image_dir=image_dir, gt=gt_dir, evaluator_type="sem_seg", **meta
        )


_RAW_SUIM_SPLITS = {
    "suim_seg_trainval": ("suim/train_val/images", "suim/train_val/masks_cvt"),
    "suim_seg_test": ("suim/test/images", "suim/test/masks_cvt"),
}


def register_all_suim(root):
    for k, (img_dir, gt_dir) in _RAW_SUIM_SPLITS.items():
        meta = get_builtin_metadata('suim')
        image_dir = os.path.join(root, img_dir)
        gt_dir = os.path.join(root, gt_dir)
        DatasetCatalog.register(k, lambda x=image_dir,y=gt_dir:load_suim_sem_seg_dict(x, y))
        MetadataCatalog.get(k).set(
            image_dir=image_dir, gt=gt_dir, evaluator_type="sem_seg", **meta
        )
