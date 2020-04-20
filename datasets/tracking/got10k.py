import os
import cv2
import glob
import logging
import numpy as np
from datasets.dataset_base import BaseDataset
from datasets import DATASET_REGISTRY



@DATASET_REGISTRY.register()
class GOT10K(BaseDataset):
    def initialize_dataset(self, cfg) -> list:
        """
        :param cfg: contains general dataset configs
        :return:
            dict{
                'seq_name': 'name'
                'image_list': 'path/to/img'
                'anno': 'box annotations' or 'path/to/label'
                'info': {
                    'image_size': [w, h]
                    'frames': number of frame
                    ...
                }
                'dataset': dataset name
            }
        """
        self.root = '/path/to/root'
        self.subset = 'train' # 'val', 'test'
        seq_list_file = 'list.txt'
        with open(seq_list_file, 'r') as f:
            seq_names = f.read().strip().split('\n')
        return self._build_seq_dict(seq_names)

    def _build_seq_dict(self, seqs):

        data_set = []

        # load a single seq into a dict
        def _load_single_seq(seq_dir):
            img_names = sorted(glob.glob(os.pardir.join(seq_dir, '*.jpg')))
            anno = np.loadtxt(os.pardir.join(seq_dir, 'groundtruth.txt'), delimiter=',')
            if self.subset == 'test' and anno.ndim == 1:
                assert len(anno) == 4
                anno = anno[np.newaxis, :]
            else:
                assert len(img_names) == len(anno)
            return img_names, anno

        logger = logging.getLogger(__name__)
        logger.info(" == > find {} seqs in the GOT10K {} set".format(len(seqs), self.subset))
        for seq_name in seqs:
            temp_dict = {}
            seq_dir = os.path.join(self.root, self.subset, seq_name)
            image_names, anno = _load_single_seq(seq_dir)
            with cv2.imread(image_names[0]) as im:
                h, w, c = im.shape

            temp_dict['seq_name'] = seq_name
            temp_dict['image_list'] = image_names
            temp_dict['anno'] = anno
            temp_dict['info'] = {
                'img_width': w,
                'img_height': h,
                'frames': len(image_names)
            }
            temp_dict['dataset'] = 'got10k'
            data_set.append(temp_dict)
        return data_set

    # TODO test getitem function
    def __getitem__(self, idx):
        seq_desc = self.data_set[idx]
        img_list = seq_desc['image_list']
        anno = seq_desc['anno']

        assert len(img_list) == len(anno)










