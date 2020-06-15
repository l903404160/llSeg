import torch

from structures import Boxes


def giou_loss(input:Boxes, target:Boxes, reduction:str="None") -> torch.Tensor:
    """
        GIoU Loss
            It can be defined by:
                                   Ac - U
             GIoU Loss = IoU   -  - - - - -
                                     Ac
    Args:
        input: [R, 4]
        target: [R, 4]
        reduction:
    Returns:
    """
    giou = pairwise_giou(input, target)
    loss = 1 - giou
    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()

    return loss


def pairwise_giou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
        Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.
    Returns:
        Tensor: IoU, sized [N,M].
    Args:
        boxes1:
        boxes2:

    Returns:

    """
    area1 = boxes1.area()
    area2 = boxes2.area()
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor

    # [N, M, 2]
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])
    width_height.clamp_(min=0)
    inter = width_height.prod(dim=2)
    del width_height

    enclosing_width_height = torch.max(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.min(boxes1[:, None, :2], boxes2[:, :2])
    enclosing_width_height.clamp_(min=0)
    enclosing = enclosing_width_height.prod(dim=2)# + 1e-7
    del enclosing_width_height

    # compute the iou
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device)
    )

    # compute the enclosing term
    # 1. find top_left and bottom right

    enclosing_term = (enclosing - (area1[: None] + area2 - inter) + 1e-7) / (enclosing + 1e-7)
    giou = iou - enclosing_term
    giou = torch.diag(giou)
    return giou

if __name__ == '__main__':
    import numpy as np
    boxes = np.array([
        [100, 100, 200, 200],
        [80, 80, 180, 180],
        [90, 90, 190, 190],
        [110, 110, 210, 210],
        [120, 120, 220, 220],
        [300, 300, 400, 400],
        [350, 350, 450, 450],
        [370, 370, 470, 470],
        [230, 230, 330, 330],
        [280, 280, 380, 380],
        [30, 30, 40, 40],
        [31, 30, 41, 40],
        [37, 37, 47, 47],
        [23, 23, 33, 33],
        [24, 24, 38, 38],
    ])

    boxes_o = np.array([
        [100, 100, 200, 200],
        [80, 80, 180, 180],
        [90, 90, 190, 190],
        [110, 110, 210, 210],
        [120, 120, 220, 220],
        [300, 300, 400, 400],
        [350, 350, 450, 450],
        [370, 370, 470, 470],
        [230, 230, 330, 330],
        [280, 280, 380, 380],
        [30, 28, 40, 38],
        [31, 30, 41, 40],
        [37, 37, 47, 47],
        [23, 23, 33, 33],
        [28, 28, 42, 42],
    ])
    boxes1 = torch.from_numpy(boxes)
    boxes2 = torch.from_numpy(boxes_o)

    boxes1 = Boxes(boxes1)
    boxes2 = Boxes(boxes2)

    temp = boxes1[:2]
    print(temp)

    ious = pairwise_giou(boxes1, boxes2)
    loss = 1-ious
    print(loss)

    # tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    #         0.0000, 0.3333, 0.0000, 0.0000, 0.0000, 0.7563])

    # |  AP   |  AP50  |  AP75  |
    # |:-----:|:------:|:------:|
    # | 0.239 | 0.442  | 0.224  |

    # |  AP   |  AP50  |  AP75  |
    # |:-----:|:------:|:------:|
    # | 0.245 | 0.458  | 0.229  |

    # |  AP   |  AP50  |  AP75  |
    # |:-----:|:------:|:------:|
    # | 0.250 | 0.460  | 0.236  |