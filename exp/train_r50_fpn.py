"""
Detection Training Script.
This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
# TODO 重写训练程序文件，目前从Detectron2中抄过来的训练程序，思考是否有地方可以进行优化

import sys
sys.path.append("/home/haida_sunxin/lqx/code/llseg")

import logging
import os
from collections import OrderedDict

import  utils.comm as comm
from utils.checkpointers.detection_checkpoint import DetectionCheckpointer
from configs.config import get_detection_config
from datasets.metacatalog.catalog import MetadataCatalog
from engine.defaults import DefaultTrainer, default_argument_setup, default_setup, hooks
from engine.launch import launch
from evaluation import DatasetEvaluators, verify_results, VisDroneEvaluator
from datasets.detection.builder import build_detection_train_loader, build_detection_test_loader

# TODO Test Aug
from models.detection.test_time_augmentation import GeneralizedRCNNWithTTA


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type in ["visdrone"]:
            evaluator_list.append(VisDroneEvaluator(dataset_name, cfg, True, output_folder))
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


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_detection_config()
    print(args.config_file)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

    args = default_argument_setup().parse_args()

    # Config files  x101_fpn_cascade.yaml
    # args.config_file = "/home/haida_sunxin/lqx/code/llseg/configs/configs_files/detection/models/r50_fpn_cascade.yaml"
    # args.config_file = "/home/haida_sunxin/lqx/code/llseg/configs/configs_files/detection/models/x101_fpn_cascade.yaml"
    args.config_file = "/home/haida_sunxin/lqx/code/llseg/exp/Cascade_cluster_101/config.yaml"
    # args.config_file = "/home/haida_sunxin/lqx/code/llseg/configs/configs_files/detection/models/X101-64x4d-dcnv2-fpn.yaml"
    # args.config_file = "/home/haida_sunxin/lqx/code/llseg/configs/configs_files/detection/models/r50_fpn_retina.yaml"
    # args.config_file = "/home/haida_sunxin/lqx/code/llseg/configs/configs_files/detection/models/eff_b6_bifpn.yaml"

    args.num_gpus = 8

    # Weights
    # args.opts = ["MODEL.WEIGHTS", "/home/haida_sunxin/lqx/model_weight/R-50.pkl"]
    args.opts = ["MODEL.WEIGHTS", "/home/haida_sunxin/lqx/model_weight/X-101-32x8d.pkl"]
    args.opts = ["MODEL.WEIGHTS", "/home/haida_sunxin/lqx/code/llseg/exp/Cascade_cluster_101/model_0004999.pth"]
    # args.opts = ["MODEL.WEIGHTS", "/home/haida_sunxin/lqx/code/llseg/exp/Cascade_efficient/model_final.pth"]
    args.eval_only = True
    args.resume = True

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )