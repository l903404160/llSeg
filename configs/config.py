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


# def confugurable(init_func):
#     """
#     Decorate a class's __init__ method so that it can be called with a CfgNode
#     object using the class's from_config classmethod.
#     Examples:
#     .. code-block:: python
#         class A:
#             @configurable
#             def __init__(self, a, b=2, c=3):
#                 pass
#             @classmethod
#             def from_config(cls, cfg):
#                 # Returns kwargs to be passed to __init__
#                 return {"a": cfg.A, "b": cfg.B}
#         a1 = A(a=1, b=2)  # regular construction
#         a2 = A(cfg)       # construct with a cfg
#         a3 = A(cfg, b=3, c=4)  # construct with extra overwrite
#     """
#     assert init_func.__name__ == "__init__", "@configurable should only be used for __init__!"
#     if init_func.__module__.startswith("")