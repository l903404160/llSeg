from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch


class COCOValSearch(Dataset):
    def __init__(self, root_dir, data_list):
        super(COCOValSearch, self).__init__()
        self._root_dir = root_dir
        self._data_list = data_list

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self._root_dir, self._data_list[idx])
        data = torch.load(file_name)
        features = {
            'p3': data['p3'],
            'p4': data['p4'],
            'p5': data['p5'],
            'p6': data['p6'],
            'p7': data['p7'],
        }
        targets = {
            'labels':data['labels'],
            'reg_targets': data['reg_targets'],
            'ctr_targets': data['ctr_targets']
        }
        return {
            'file_name': file_name,
            'features': features,
            'targets': targets
        }


def get_dataloader(dataset, batch_size):
    sampler = torch.utils.data.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1, sampler=sampler)
    return dataloader

if __name__ == '__main__':
    root_dir = '/home/haida_sunxin/lqx/data/search'
    list = os.listdir(root_dir)
    dst = COCOValSearch(root_dir, list)
    loader = get_dataloader(dst, 1)

