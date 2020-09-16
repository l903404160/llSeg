import os
import sys
sys.path.append('/home/haida_sunxin/lqx/code/llseg')
from configs import get_sem_seg_config

import engine.hooks as hooks
from engine.launch import launch
from engine.defaults import default_setup, default_argument_setup

from trainers.sem_seg.sem_seg_trainer import SegTrainer

from evaluation.testing import verify_results

import utils.comm as comm
from utils.checkpointers.generic import GenericCheckpoint


def setup_config(args):
    """
    Args:
        args: Create configs and perform basic setup
    Returns: custom configs
    """
    cfg = get_sem_seg_config()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    config = setup_config(args)
    # Only perform evaluation
    if args.eval_only:
        model = SegTrainer.build_model(config)
        GenericCheckpoint(model, save_dir=config.OUTPUT_DIR).resume_or_load(
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
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    args = default_argument_setup().parse_args()
    args.num_gpus = 8
    # args.config_file = "/home/haida_sunxin/lqx/code/llseg/configs/configs_files/sem_seg/models/baseline.yaml"
    args.config_file = "/home/haida_sunxin/lqx/code/llseg/configs/configs_files/sem_seg/models/hanet.yaml"


    launch(
        main_func=main, num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, )
    )
