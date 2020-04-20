from datasets import DATASET_REGISTRY

_DATASET_MAP = {
    'cityscapes_fine_seg_train': {
        'name': 'CityScapes',
        'root': '/root/cityscapes',
        'flag': 'train'
    },
    'cityscapes_fine_seg_val': {
        'name': 'CityScapes',
        'root': '/root/cityscapes',
        'flag': 'val'
    },
    'cityscapes_fine_seg_test': {
        'name': 'CityScapes',
        'root': '/root/cityscapes',
        'flag': 'test'
    },
    'got10k': {
        'root': 'root/dataset/got10k'
    }
}

def dataset_manager(name):
    assert name in _DATASET_MAP.keys(), "No Implementation for the '{}' dataset".format(name)
    return _DATASET_MAP[name]