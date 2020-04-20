from .base import CfgNode as CN

_C = CN()

# Output dir
_C.OUTPUT_DIR = './output_dir'
_C.CUDNN_BENCHMARK = False
_C.SEED = -1

_C.DATASETS = CN()
_C.DATASETS.ROOT = '/root'
_C.DATASETS.TRAIN = ['cityscapes_fine_seg_train']
_C.DATASETS.TEST = ['cityscapes_fine_seg_val']

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.SAMPLER_TRAIN = 'TrainingSampler'

_C.INPUT = CN()
_C.INPUT.WIDTH_TRAIN = 2048
_C.INPUT.HEIGHT_TRAIN = 1024
_C.INPUT.SCALES_TRAIN = [1.0, 1.25, 1.5]
_C.INPUT.WIDTH_TEST = 2048
_C.INPUT.HEIGHT_TEST = 1024
_C.INPUT.SCALES_TEST = [1.0]
_C.INPUT.IMG_FORMAT = 'RGB'
_C.INPUT.LBL_FORMAT = 'L'

# Model settings
# Need implemented by model

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
_C.SOLVER.IMS_PER_BATCH = 2

_C.SOLVER.MAX_ITER = 30  # 90000
_C.SOLVER.STEPS = (10,)  # 60000
_C.SOLVER.GAMMA = 0.1

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = 'linear'

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 5000

_C.SOLVER.CLIP_GRADIENTS = 0.1
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# Test config
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 2
_C.TEST.EVAL_PERIOD = 0

_C.TEST.PRECISE_BN = CN({"ENABLED": False})
_C.TEST.PRECISE_BN.NUM_ITER = 200

# For end-to-end tests to verify the expected accuracy.
# Each item is [task, metric, value, tolerance]
# e.g.: [['bbox', 'AP', 38.5, 0.2]]
_C.TEST.EXPECTED_RESULTS = []