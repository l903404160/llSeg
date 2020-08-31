"""
This file contains the segmentation mapping that's applied to "dataset dicts".
"""
import copy
import torch
import logging
import numpy as np
from PIL import Image

import datasets.dataset_utils as utils
import datasets.transforms.transforms_gen as T

from datasets.dataset_utils import generate_cornernet_heatmap_tag_regr


class DetDatasetMapper:
    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger("OUCWheel." + __name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = self.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        self.anchor_free_on = cfg.MODEL.ANCHORFREE.ANCHORFREE_ON
        self.anchor_free_arch = cfg.MODEL.ANCHORFREE.ARCH

        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    @staticmethod
    def build_transform_gen(cfg, is_train):
        """
        Create a list of :class:`TransformGen` from config.
        Now it includes resizing and flipping.
        Returns:
            list[TransformGen]
        """
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
        if sample_style == "range":
            assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
                len(min_size)
            )

        logger = logging.getLogger("OUCWheel." + __name__)
        tfm_gens = []
        tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
        if is_train:
            # TODO open the tfm
            # tfm_gens.append(T.RandomFlip())
            logger.info("TransformGens used in training: " + str(tfm_gens))
        return tfm_gens

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                self.proposal_min_box_size,
                self.proposal_topk,
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict


class AnchorFree_DetDatasetMapper:
    def __init__(self, cfg, is_train=True):
        self.crop_gen = None
        self.tfm_gens = self.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        self.anchor_free_on = cfg.MODEL.ANCHORFREE.ANCHORFREE_ON
        self.anchor_free_arch = cfg.MODEL.ANCHORFREE.ARCH

        self.is_train = is_train

    @staticmethod
    def build_transform_gen(cfg, is_train):
        """
        Create a list of :class:`TransformGen` from config.
        Now it includes resizing and flipping.
        Returns:
            list[TransformGen]
        """
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
        if sample_style == "range":
            assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
                len(min_size)
            )

        # configs -----------------------------------------------------
        scales = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        short_edge_length = 512
        crop_type = cfg.INPUT.CROP.TYPE
        crop_size = cfg.INPUT.CROP.SIZE
        color_jitter_var = 0.4
        lighting_var = 0.1

        logger = logging.getLogger("OUCWheel." + __name__)
        tfm_gens = []
        if is_train:
            tfm_gens.append(T.ResizeFromScales(scales=scales,
                                               short_edge_length=short_edge_length))
            tfm_gens.append(T.RandomCrop(crop_type=crop_type, crop_size=crop_size))
            tfm_gens.append(T.RandomFlip())
            tfm_gens.append(T.RandomContrast(intensity_min=1.0 - color_jitter_var, intensity_max=1.0 + color_jitter_var))
            tfm_gens.append(T.RandomSaturation(intensity_min=1.0 - color_jitter_var, intensity_max=1.0 + color_jitter_var))
            tfm_gens.append(T.RandomBrightness(intensity_min=1.0 - color_jitter_var, intensity_max=1.0 + color_jitter_var))
            tfm_gens.append(T.RandomLighting(scale=lighting_var))
            # TODO: Draw gaussian and generate heatmap tag and regr
            logger.info("TransformGens used in training: " + str(tfm_gens))
        else:
            # TODO testing transform
            tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
        return tfm_gens

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # Crop around an instance if there are instances in the image.
        # USER: Remove if you don't use cropping
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # TODO change the test mapper
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            instances = utils.filter_empty_instances(instances)
            dataset_dict["instances"] = instances

            outs = generate_cornernet_heatmap_tag_regr(outout_size=(128, 128), max_tag_length=128, num_classes=80,
                                                       instances=instances)
            tl_heatmap, br_heatmap, tl_regr, br_regr, tag_mask, tl_tag, br_tag = outs
            dataset_dict["tl_heatmap"] = tl_heatmap
            dataset_dict["br_heatmap"] = br_heatmap
            dataset_dict["tl_regr"] = tl_regr
            dataset_dict["br_regr"] = br_regr
            dataset_dict["tag_mask"] = tag_mask
            dataset_dict["tl_tag"] = tl_tag
            dataset_dict["br_tag"] = br_tag

        return dataset_dict
