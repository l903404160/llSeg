import os
import logging
from collections import OrderedDict

from engine.defaults import DefaultTrainer
from datasets.metacatalog.catalog import MetadataCatalog

from evaluation import DatasetEvaluators, COCOEvaluator, VisDroneEvaluator
from datasets.detection.builder import build_anchorfree_detection_train_loader, build_detection_test_loader

# TODO Test Aug
from models.detection.test_time_augmentation import GeneralizedRCNNWithTTA


# AnchorFreeTrainer
class AnchorFreeTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        return build_anchorfree_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type in ["visdrone"]:
            evaluator_list.append(VisDroneEvaluator(dataset_name, cfg, True, output_folder))
        elif evaluator_type in ["coco"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("OUCWheel.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


