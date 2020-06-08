def get_config():
    from .defaults import _C
    return _C.clone()


# Semantic Segmentation configs
def get_sem_seg_config():
    from .configs_files.sem_seg.sem_seg_defaults import _C
    return _C.clone()


# Detection configs
def get_detection_config():
    from .configs_files.detection.detection_defaults import _C
    return _C.clone()