import os

from .register import register_cityscapes_segmentation, register_voc_context_segmentation_dataset
from .seg_builtin_meta import get_segmentation_builtin_meta_data


_CITYSCAPES_SPLITS = {}
_CITYSCAPES_SPLITS['cityscapes'] = {
    'cityscapes_train': ('cityscapes/leftImg8bit/train', 'cityscapes/gtFine/train'),
    'cityscapes_val': ('cityscapes/leftImg8bit/val', 'cityscapes/gtFine/val'),
    'cityscapes_test': ('cityscapes/leftImg8bit/test', 'cityscapes/gtFine/test')
}


def register_cityscapes(root):
    for dataset_name, splits_per_dataset in _CITYSCAPES_SPLITS.items():
        for k, (image_root, label_root) in splits_per_dataset.items():
            meta_data = get_segmentation_builtin_meta_data(dataset_name)
            # Registe Dataset
            register_cityscapes_segmentation(k, meta_data, os.path.join(root, image_root), os.path.join(root, label_root))

_RAW_VOC_CONTEXT_SPLITS = {}
_RAW_VOC_CONTEXT_SPLITS['voc_context'] = {
    "voc_context_seg_train": ("Context/train", "Context/train_labels"),
    "voc_context_seg_val": ("Context/val", "Context/val_labels"),
}


def register_voc_context(root):
    for dataset_name, splits_per_dataset in _RAW_VOC_CONTEXT_SPLITS.items():
        for k, (image_root, label_root) in splits_per_dataset.items():
            metadata = get_segmentation_builtin_meta_data(dataset_name)
            # Registe Dataset

            register_voc_context_segmentation_dataset(k, metadata, os.path.join(root, image_root), os.path.join(root, label_root))




