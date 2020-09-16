"""
This file contains the segmentation mapping that's applied to "dataset dicts".
"""
import copy
import torch
import logging
import numpy as np
import datasets.dataset_utils as utils
import datasets.transforms.transforms_gen as T


class SegDatasetMapper(object):
    def __init__(self, cfg, is_train=True):
        self.img_format = cfg.INPUT.IMG_FORMAT
        self.lbl_format = cfg.INPUT.LBL_FORMAT
        self.tfm_gens = self.build_transform_gen(cfg, is_train)

    def __call__(self, dataset_dict):
        """
        :param dataset_dict: Metadata of one image
        :return: dict contain image and label data
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict['file_name'], format=self.img_format)
        label = utils.read_image(dataset_dict['sem_seg_file_name'], format=self.lbl_format)

        image, tfm = T.apply_transform_gens(self.tfm_gens, image)
        label = tfm.apply_segmentation(label)

        h, w = image.shape[:2]

        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        label = torch.as_tensor(label.astype('long'))

        dataset_dict['image'] = image
        dataset_dict['sem_seg'] = label
        dataset_dict['height'] = h
        dataset_dict['width'] = w
        return dataset_dict

    @staticmethod
    def build_transform_gen(cfg, is_train):
        """
        Create a list of :class:`TransformGen` from config.
        Now it includes resizing and flipping.
        Returns:
            list[TransformGen]
        """
        if is_train:
            height = cfg.INPUT.HEIGHT_TRAIN
            width = cfg.INPUT.WIDTH_TRAIN
            scales = cfg.INPUT.SCALES_TRAIN
        else:
            scales = cfg.INPUT.SCALES_TEST

        logger = logging.getLogger("OUCWheel."+__name__)
        tfm_gens = []
        if is_train:
            tfm_gens.append(T.RandomBrightness(intensity_min=0.5, intensity_max=1.5))
            tfm_gens.append(T.RandomContrast(intensity_min=0.5, intensity_max=1.5))
            tfm_gens.append(T.RandomSaturation(intensity_min=0.5, intensity_max=1.5))
            tfm_gens.append(T.RandomFlip())
            tfm_gens.append(T.ResizeFromScales(scales=scales)) #
            tfm_gens.append(T.RandomCrop(crop_type='absolute', crop_size=(height, width)))
            logger.info("TransformGens used in training: " + str(tfm_gens))
        else:
            tfm_gens.append(T.ResizeFromScales(scales=scales))
        return tfm_gens
