from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from utils.nn import smooth_l1_loss

import torch.nn as nn

from layers import ShapeSpec, cat
# TODO structures

from utils.events import get_event_storage
