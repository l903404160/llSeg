from __future__ import absolute_import, division, print_function, unicode_literals

# TODO make it custom
from detectron2 import _C


def pairwise_iou_rotated(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes1:
        boxes2:

    Returns:

    """

    return _C.box_iou_rotated(boxes1, boxes2)
