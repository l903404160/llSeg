import torch
import logging
import itertools
import numpy as np
from utils.env import seed_all_rng
from utils.comm import get_world_size


from datasets.metacatalog.catalog import DatasetCatalog
from datasets.common import MapDataset, DatasetFromList
from datasets.segmentation.seg_mapper import SegDatasetMapper
from datasets.samplers import TrainingSampler, InferenceSampler


def build_segmentation_train_loader(cfg, mapper=None):
    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    assert (
        images_per_batch % num_workers == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
        images_per_batch >= num_workers
    ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers
    dataset_dicts = get_semantic_segmentation_dataset_dicts(cfg.DATASETS.TRAIN)

    dataset = DatasetFromList(dataset_dicts, copy=False)
    if mapper is None:
        mapper = SegDatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger("OUCWheel."+__name__)
    logger.info('Using traning sampleer {}'.format(sampler_name))

    if sampler_name == 'TrainingSampler':
        sampler = TrainingSampler(len(dataset))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_worker, drop_last=True
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trainval_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )
    return data_loader


def build_segmentation_test_loader(cfg, mapper=None):

    dataset_dicts = get_semantic_segmentation_dataset_dicts(cfg.DATASETS.TEST)
    dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = SegDatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporing inference time in papers
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trainval_batch_collator
    )
    return data_loader


def get_semantic_segmentation_dataset_dicts(dataset_names):
    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
    return dataset_dicts


def trainval_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    batch_data_dict = dict()
    if len(batch) != 1:
        batch_data_dict['image'] = torch.stack([item_data['image'] for item_data in batch], dim=0)
        batch_data_dict['sem_seg'] = torch.stack([item_data['sem_seg'] for item_data in batch], dim=0)
        batch_data_dict['file_name'] = [item_data['file_name'] for item_data in batch]
        batch_data_dict['sem_seg_file_name'] = [item_data['sem_seg_file_name'] for item_data in batch]
        batch_data_dict['height'] = [item_data['height'] for item_data in batch]
        batch_data_dict['width'] = [item_data['width'] for item_data in batch]
        return batch_data_dict
    else:
        item = batch[0]
        batch_data_dict['image'] = item['image'].unsqueeze(0)
        batch_data_dict['sem_seg'] = item['sem_seg'].unsqueeze(0)
        batch_data_dict['file_name'] = item['file_name']
        batch_data_dict['sem_seg_file_name'] = item['sem_seg_file_name']
        batch_data_dict['height'] = item['height']
        batch_data_dict['width'] = item['width']
        return batch_data_dict

def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2**31) + worker_id)


if __name__ == '__main__':
    from configs.sem_seg_configs.baseline_config import BS_config as cfg

    cfg.DATASETS.TRAIN = ['voc_context_seg_train']
    cfg.DATASETS.TEST = ['voc_context_seg_val']
    cfg.DATASETS.ROOT = '//home/haida_sunxin/lqx/data/'
    cfg.RESIZE_SHORT_EDGE = True
    cfg.SHORTEST_EDGE = 480
    cfg.INPUT.WIDTH_TRAIN = 480
    cfg.INPUT.HEIGHT_TRAIN = 480
    cfg.INPUT.WIDTH_TEST = 480
    cfg.INPUT.HEIGHT_TEST = 480
    # loader = build_segmentation_train_loader(cfg)
    loader = build_segmentation_train_loader(cfg)
    print('done')

    import matplotlib.pyplot as plt

    for data in loader:
        print(data['image'].size())
        print(data['sem_seg'].size())
        temp_im = data['image'][0].numpy()
        temp_lb = data['sem_seg'][0].numpy()
        plt.subplot(1,2,1)
        plt.imshow(temp_im.transpose(1,2,0))
        plt.subplot(1,2,2)
        plt.imshow(temp_lb)
        plt.savefig('test.png')
        plt.show()
        print('ok')