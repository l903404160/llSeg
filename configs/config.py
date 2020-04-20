def get_config():
    from .defaults import _C
    return _C.clone()