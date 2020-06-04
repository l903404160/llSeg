# Test file for defaultTrainer
import os
import sys
# sys.path.append(os.environ['OW_ROOT'])
# sys.path.append('/root/codes/vot')

from configs.sem_seg_configs.pcnet_config import PCNet_config as config

from engine.defaults import DefaultTrainer, default_setup
from datasets.segmentation.builder import build_segmentation_train_loader
from datasets.segmentation.builder import build_segmentation_test_loader
from datasets.metacatalog.catalog import MetadataCatalog
from evaluation.sem_seg_evaluator import SemSegEvaluator
from evaluation.evaluator import DatasetEvaluators
# TODO write the SegCheckpoint
from utils.checkpointers.tracking import TrackingCheckpoint
import utils.comm as comm
import logging
from collections import OrderedDict
from evaluation.testing import verify_results
import engine.hooks as hooks


class SegTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(SegTrainer, self).__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_segmentation_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_segmentation_test_loader(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator for a given dataset
        This uses the specoal metadata "evaluator_type" associated with each builtin dataset
        For your own dataset, you can simple create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here
        :param cfg:
        :param dataset_name:
        :return:
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        evaluator_list = []
        # TODO MetadataCatalog
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.NUM_CLASSES,
                    ignore_label=cfg.MODEL.IGNORE_LABEL,
                    output_dir=output_folder
                )
            )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "No Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_multi_scale(cls, cfg, model):
        logger = logging.getLogger("OUCWheel.Multi_scale_eval")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_MultiScale")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test_multi_scale(config, model, evaluators)
        res = OrderedDict({k + "_MultiScale": v for k, v in res.items()})
        return res


def main(args):
    default_setup(config, args)

    # Only perform evaluation
    if args.eval_only:
        model = SegTrainer.build_model(config)
        TrackingCheckpoint(model, save_dir=config.OUTPUT_DIR).resume_or_load(
            config.MODEL.WEIGHTS, resume=args.resume
        )
        res = SegTrainer.test(config, model)
        if config.TEST.AUG:
            res.update(SegTrainer.test_with_multi_scale(config, model))
        if comm.is_main_process():
            verify_results(config, res)
        return res

    # Normal training
    trainer = SegTrainer(config)

    trainer.resume_or_load(config.MODEL.WEIGHTS)
    if config.TEST.AUG:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_multi_scale(config, trainer.model))]
        )

    return trainer.train()


if __name__ == '__main__':
    from engine.defaults import default_setup, default_argument_setup
    from engine.launch import launch

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    args = default_argument_setup().parse_args()
    args.num_gpus = 2
    print('Command Line Args: ', args)

    # args.eval_only = True

    launch(
        main_func=main, num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, )
    )
