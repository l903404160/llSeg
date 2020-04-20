import copy
import pickle
import logging
import numpy as np
import torch.utils.data as data


class MapDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.
    Args:
        dataset: a dataset where map function is applied.
        map_func: a callable which maps the element in dataset. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the dataset.
    """
    def __init__(self, dataset, map_func):
        self._dataset = dataset
        self._map_func = map_func

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        # TODO there is some different with detectron2
        retry_count = 0
        cur_idx = int(idx)
        data = self._map_func(self._dataset[idx])
        return data


class DatasetFromList(data.Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data
    """
    def __init__(self, lst:list, copy:bool=True, serialize:bool=True):
        """
        :param lst: a list which contains elements to produce
        :param copy:  whether to deepcopy the element when producing it,
        :param serialize: whether to hold memory using serialized objects, when
                enabled, data loader workers can use shared RAM from master process instead of making a copy
        """
        self._lst = lst
        self._copy = copy
        self._serialize = serialize

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            logger = logging.getLogger('OUCWheel.'+__name__)
            logger.info(
                "Serialize {} elements to byte tensors and concatenating them all ...".format(
                    len(self._lst)
                )
            )
            self._lst = [_serialize(x) for x in self._lst]
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.uint64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)
            logger.info("Serialized dataset takse {:.2f} Mib".format(len(self._lst) / 1024 **2))

    def __len__(self):
        if self._serialize:
            return len(self._addr)
        else:
            return len(self._lst)

    def __getitem__(self, idx):
        if self._serialize:
            start_addr = 0 if idx == 0 else self._addr[idx -1 ].item()
            end_addr = self._addr[idx].item()
            bytes = memoryview(self._lst[start_addr:end_addr])
            return pickle.loads(bytes)
        elif self._copy:
            return copy.deepcopy(self._lst[idx])
        else:
            return self._lst[idx]

if __name__ == '__main__':
    from torchvision.transforms.transforms import ToTensor
    from datasets import DATASET_REGISTRY
    from datasets.segmentation.seg_mapper import SegDatasetMapper
    import numpy as np
    info = {
        'root': '/root/cityscapes',
        'flag': 'train'
    }

    from configs.defaults import _C as cfg

    cs = DATASET_REGISTRY.get('CityScapes')(cfg, info)
    mapper = SegDatasetMapper(cfg, is_train=True)

    mapD = MapDataset(cs, mapper)
    print(len(mapD))
    for d in mapD:
        print(np.unique(d['anno']))
        print('hah')