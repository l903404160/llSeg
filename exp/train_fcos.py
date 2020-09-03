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

from trainers.detection.det_trainer import DetTrainer

import  utils.comm as comm
from utils.checkpointers.detection_checkpoint import DetectionCheckpointer
from configs.config import get_detection_config
from engine.defaults import default_argument_setup, default_setup, hooks
from engine.launch import launch
from evaluation import verify_results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_detection_config()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = DetTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = DetTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(DetTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = DetTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    args = default_argument_setup().parse_args()

    # Config files  x101_fpn_cascade.yaml
    args.config_file = "/home/haida_sunxin/lqx/code/llseg/configs/configs_files/detection/base_fcos.yaml"
    # args.config_file = "/home/haida_sunxin/lqx/code/llseg/configs/configs_files/detection/models/borderdet_r50.yaml"
    # args.config_file = "/home/haida_sunxin/lqx/code/llseg/configs/configs_files/detection/models/x101_fpn_cascade.yaml"

    args.num_gpus = 4

    # Weights
    args.opts = ["MODEL.WEIGHTS", "/home/haida_sunxin/lqx/model_weight/R-50.pkl"]
    # args.opts = ["MODEL.WEIGHTS", "/home/haida_sunxin/lqx/code/llseg/exp/fcos_r50_new/model_0089999.pth"]

    # args.opts = ["MODEL.WEIGHTS", "/home/haida_sunxin/lqx/model_weight/X-101-32x8d.pkl"]
    args.eval_only = False
    args.resume = False

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )