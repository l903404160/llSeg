from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, cfg, infos: dict):
        self.data_set = self.initialize_dataset(cfg, infos)

    def initialize_dataset(self, cfg, infos: dict) -> list:
        """
        :param cfg: contains general dataset configs
        :param infos contains dataset infos
        :return:
            dict{
                'image': 'path/to/img'
                'anno': 'box annotations' or 'path/to/label'
                'info': {
                    'image_size': [w, h]
                    'frames': number of frame
                    ...
                }
                'dataset': dataset name
            }
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        raise NotImplementedError
