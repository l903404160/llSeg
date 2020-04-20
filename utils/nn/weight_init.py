import torch.nn as nn


def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2
    Also initializes `module.bias` to 0
    :param module:
    :return:
    """
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def c2_msra_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSTAFill" implemented in Caffe2
    Also initializes `module.bias` to 0
    :param module:
    :return:
    """
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)