import os
import json
import glob
import PIL.Image as I
from datasets.dataset_base import BaseDataset
from datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CityScapes(BaseDataset):
    def initialize_dataset(self, cfg, infos) -> list:
        """
        :param cfg: contains general dataset configs
        :param infos: contains dataset infos
        :return:
            List[dict]: contains data description
        """
        img_root = os.path.join(infos['root'], 'leftImg8bit', infos['flag'])
        lbl_root = os.path.join(infos['root'], 'gtFine', infos['flag'])
        img_files = glob.glob(os.path.join(img_root, '*/*.png'))
        lbl_files = [os.path.join(lbl_root, os.path.basename(img_files[i]).split('_')[0], os.path.basename(img_files[i]).replace('leftImg8bit', 'gtFine_labelTrainIds')) for i in range(len(img_files))]
        data_set = []
        for item in range(len(img_files)):
            temp = dict()
            temp['image'] = img_files[item]
            temp['anno'] = lbl_files[item]
            w, h = I.open(img_files[item]).size
            temp['info'] = {
                'width': w,
                'height': h,
                'dataset_name': 'CityScapes',
                'type': 'segmentation'
            }
            data_set.append(temp)
        return data_set

    def __getitem__(self, idx):
        idx = 10
        return self.data_set[idx]


def load_cityscapes_sem_seg_dict(image_dir, gt_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    for image_file in glob.glob(os.path.join(image_dir, "**/*.png")):
        suffix = "leftImg8bit.png"
        assert image_file.endswith(suffix)
        prefix = image_dir

        label_file = gt_dir + image_file[len(prefix): -len(suffix)] + "gtFine_labelTrainIds.png"
        assert os.path.isfile(
            label_file
        ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"

        json_file = gt_dir + image_file[len(prefix) : -len(suffix)] + "gtFine_polygons.json"
        with open(json_file, 'r') as f:
            jsonobj = json.load(f)
            ret.append(
                {
                    'file_name': image_file,
                    'sem_seg_file_name': label_file,
                    'height': jsonobj['imgHeight'],
                    'width': jsonobj['imgWidth']
                }
            )
    return ret


