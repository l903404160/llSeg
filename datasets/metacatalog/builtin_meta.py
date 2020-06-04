

def get_builtin_metadata(dataset_name):
    if dataset_name == 'cityscapes':
        # fmt: off
        CITYSCAPES_THING_CLASSES = [
            "person", "rider", "car", "truck",
            "bus", "train", "motorcycle", "bicycle",
        ]
        CITYSCAPES_STUFF_CLASSES = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
            "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle",
        ]
        # fmt: on
        return {
            "thing_classes": CITYSCAPES_THING_CLASSES,
            "stuff_classes": CITYSCAPES_STUFF_CLASSES,
        }

    if dataset_name == 'voc_context':
        # fmt: off
        VOC_CONTEXT_STUFF_CLASSES = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'table',
            'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'bag',
            'bed', 'bench', 'book', 'building', 'cabinet', 'ceiling', 'cloth', 'computer', 'cup', 'door', 'fence',
            'floor', 'flower', 'food', 'grass', 'ground', 'keyboard', 'light', 'mountain', 'mouse', 'curtain', 'platform',
            'sign', 'plate', 'road', 'rock', 'shelves', 'sidewalk', 'sky', 'snow', 'bedclothes', 'track', 'tree', 'truck',
            'wall', 'water', 'window', 'wood'
        ]
        # fmt: on
        return {
            "stuff_classes": VOC_CONTEXT_STUFF_CLASSES,
        }
    if dataset_name == 'suim':
        SUIM_STUFF_CLASSES = [
            'Background waterbody', 'Human divers', 'Plants/sea-grass', 'Wrecks/ruins', 'Robots/instruments',
            'Reefs and invertebrates', 'Fish and vertebrates', 'Sand/sea-floor (& rocks)'
        ]
        return {
            "stuff_classes": SUIM_STUFF_CLASSES
        }
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
