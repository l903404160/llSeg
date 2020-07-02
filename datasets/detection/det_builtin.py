import os

from .register import register_coco_instances, register_visdrone_instances
from .det_builtin_meta import _get_builtin_metadata

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
}


def register_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


_PREDEFINED_SPLITS_VISDRONE = {}
_PREDEFINED_SPLITS_VISDRONE["visdrone"] = {
    "visdrone_train": ("DronesDET/train/images", "DronesDET/train/annotations/train.json"),
    "visdrone_val": ("DronesDET/val/images", "DronesDET/val/annotations/val.json"),
    "visdrone_test": ("DronesDET/test/images", "DronesDET/test/annotations/test.json"),
    "visdrone_patch_train": ("DronesDET/train/patch_train/images", "DronesDET/patch_train.json"),
    "visdrone_patch_val": ("DronesDET/val/patch_val/images", "DronesDET/patch_val.json"),
}

_PREDEFINED_SPLITS_VISDRONE["visdrone_wcluster"] = {
    "visdrone_cluster_train": ("DronesDET/train/images", "DronesDET/Clusters/visdrone_cluster_train.json"),
    "visdrone_cluster_val": ("DronesDET/val/images", "DronesDET/Clusters/visdrone_cluster_val.json"),
    "visdrone_cluster_test": ("DronesDET/test/images", "DronesDET/Clusters/visdrone_cluster_test.json"),
}


def register_visdrone(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_VISDRONE.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_visdrone_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )