from utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")

from .segmentation.cityscapes import CityScapes
from .tracking.got10k import GOT10K
