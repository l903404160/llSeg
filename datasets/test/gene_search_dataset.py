"""
    1. 构建验证集数据集，并且能够正确加载 ground truth
    2. 构建模型能够加载预训练权重
    3. 进行一个小测试，完成搜索过程
"""

import bisect
import copy
import itertools
import logging
import numpy as np
import operator
import pickle
import torch.utils.data
from tabulate import tabulate
from termcolor import colored

from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import log_first_n

from datasets.metacatalog.catalog import DatasetCatalog, MetadataCatalog
from datasets.common import DatasetFromList, MapDataset, AspectRatioGroupedDataset
from datasets.detection.det_mapper import DetDatasetMapper, AnchorFree_DetDatasetMapper
from datasets.dataset_utils import check_metadata_consistency
from datasets.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from datasets.detection.builder import load_proposals_into_dataset, filter_images_with_few_keypoints, filter_images_with_only_crowd_annotations, print_instances_class_histogram


import  utils.comm as comm
from utils.checkpointers.detection_checkpoint import DetectionCheckpointer

from trainers.detection.det_trainer import DetTrainer

def build_detection_test_loader(cfg, dataset_name, mapper=None):
    """
    Similar to `build_detection_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.
    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.
    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DetDatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

def get_detection_dataset_dicts(
    dataset_names, filter_empty=True, min_keypoints=0, proposal_files=None
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.
    Args:
        dataset_names (list[str]): a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `dataset_names`.
    """
    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(dataset_names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    # Keep images without instance-level GT if the dataset has semantic labels.
    if filter_empty and has_instances and "sem_seg_file_name" not in dataset_dicts[0]:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)

    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

    if has_instances:
        try:
            class_names = MetadataCatalog.get(dataset_names[0]).thing_classes
            check_metadata_consistency("thing_classes", dataset_names)
            print_instances_class_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass
    return dataset_dicts


# train scripts
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


if __name__ == '__main__':
    from trainers.detection.gene_trainer import DetTrainer
    # build model
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    from configs.config import get_detection_config
    from engine.defaults import default_argument_setup, default_setup, hooks
    from engine.launch import launch
    from evaluation import verify_results

    args = default_argument_setup().parse_args()
    args.config_file = "/home/haida_sunxin/lqx/code/llseg/configs/configs_files/detection/base_fcos.yaml"
    args.num_gpus = 1
    args.opts = ["MODEL.WEIGHTS", "/home/haida_sunxin/lqx/code/llseg/exp/fcos_r50_new/model_0089999.pth"]
    args.eval_only = True
    args.resume = True
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

    """
        # print('debug ------------------', __name__)
        # for k,v in features.items():
        #     features[k] = features[k].detach().cpu()
        # features['labels'] = labels
        # features['reg_targets'] = reg_targets
        # features['ctr_targets'] = ctrness_targets
        # out_dir = '/home/haida_sunxin/lqx/data/search'
        # import os
        # im_name = os.path.basename(name)
        # torch.save(features, os.path.join(out_dir, im_name[:-4] + '.pth'))
        # print('debug ------------------', __name__)
        #
        # return None, None
    """