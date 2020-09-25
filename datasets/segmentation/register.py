import os

from datasets.metacatalog.catalog import MetadataCatalog, DatasetCatalog

from .cityscapes import load_cityscapes_sem_seg_dict
from .voccontext import load_pascal_context_sem_seg_dict

"""
This file contains functions to register a Segmentation dataset to the DatasetCatalog.
"""


def register_cityscapes_segmentation(name, metadata, image_root, label_root):
    """
    Register a dataset with CityScapes' format
    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        image_root (str or path-like): directory which contains all the images.
        label_root (str or path-like): directory which contains all the labels.
    """
    assert isinstance(name, str), name
    assert isinstance(image_root, (str, os.PathLike)), image_root
    assert isinstance(label_root, (str, os.PathLike)), label_root

    DatasetCatalog.register(name, lambda img_root=image_root, lbl_root=label_root: load_cityscapes_sem_seg_dict(img_root, lbl_root))
    MetadataCatalog.get(name).set(
        image_root=image_root, label_root=label_root, evaluator_type="city_sem_seg", **metadata
    )


def register_voc_context_segmentation_dataset(name, metadata, image_root, label_root):
    assert isinstance(name, str), name
    assert isinstance(image_root, (str, os.PathLike)), image_root
    assert isinstance(label_root, (str, os.PathLike)), label_root

    DatasetCatalog.register(name, lambda img_root=image_root, lbl_root=label_root: load_pascal_context_sem_seg_dict(img_root, lbl_root))
    MetadataCatalog.get(name).set(
        image_root=image_root, label_root=label_root, evaluator_type="sem_seg", **metadata
    )