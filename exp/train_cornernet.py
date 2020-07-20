import sys
sys.path.append("/home/haida_sunxin/lqx/code/llseg")

from trainers.detection.anchorfree_trainer import AnchorFreeTrainer

import utils.comm as comm
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
    print(args.config_file)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = AnchorFreeTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = AnchorFreeTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(AnchorFreeTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = AnchorFreeTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    args = default_argument_setup().parse_args()
    args.config_file = "/home/haida_sunxin/lqx/code/llseg/configs/configs_files/detection/base_CornerNet.yaml"

    args.num_gpus = 1

    # Weights
    # args.opts = ["MODEL.WEIGHTS", "/home/haida_sunxin/lqx/model_weight/X-101-32x8d.pkl"]

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