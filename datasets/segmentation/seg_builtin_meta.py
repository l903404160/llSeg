# CityScapes Meta Dataset Information

CITYSCAPES_STUFF_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]

CITYSCAPES_STUFF_CLASSES_COLORMAP = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]
]

CITYSCAPES_THING_CLASSES = [
    "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]


def _get_cityscapes_segmentation_meta():
    return {
        "thing_classes": CITYSCAPES_THING_CLASSES,
        "stuff_classes_colormap": CITYSCAPES_STUFF_CLASSES_COLORMAP,
        "stuff_classes": CITYSCAPES_STUFF_CLASSES,
    }

# VOC Context Metadata

VOC_CONTEXT_STUFF_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'table',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'bag',
    'bed', 'bench', 'book', 'building', 'cabinet', 'ceiling', 'cloth', 'computer', 'cup', 'door', 'fence',
    'floor', 'flower', 'food', 'grass', 'ground', 'keyboard', 'light', 'mountain', 'mouse', 'curtain', 'platform',
    'sign', 'plate', 'road', 'rock', 'shelves', 'sidewalk', 'sky', 'snow', 'bedclothes', 'track', 'tree', 'truck',
    'wall', 'water', 'window', 'wood'
]

def _get_voc_context_segmentation_meta():
    return {
        "stuff_classes": VOC_CONTEXT_STUFF_CLASSES,
    }


def get_segmentation_builtin_meta_data(dataset_name):
    if dataset_name == 'cityscapes':
        return _get_cityscapes_segmentation_meta()
    elif dataset_name == 'voc_context':
        return _get_voc_context_segmentation_meta()

    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))