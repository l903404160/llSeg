from configs.base import CfgNode as CN

_C = CN()

# Output dir
_C.OUTPUT_DIR = './output_dir'
_C.CUDNN_BENCHMARK = False
_C.SEED = -1

_C.DATASETS = CN()
_C.DATASETS.ROOT = '/home/haida_sunxin/lqx/data/'
_C.DATASETS.TRAIN = ['cityscapes_train']
_C.DATASETS.TEST = ['cityscapes_val']
_C.DATASETS.RESIZE_SHORT_EDGE = False
_C.DATASETS.SHORTEST_EDGE = 480

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.SAMPLER_TRAIN = 'TrainingSampler'

_C.INPUT = CN()
_C.INPUT.WIDTH_TRAIN = 2048
_C.INPUT.HEIGHT_TRAIN = 1024
_C.INPUT.SCALES_TRAIN = [1.0, 1.25, 1.5]
_C.INPUT.SCALES_TEST = [1.0] # This is always set to be [1.0] # TODO change the test loader
_C.INPUT.IMG_FORMAT = 'RGB'
_C.INPUT.LBL_FORMAT = 'L'

# Model settings
_C.MODEL = CN()
_C.MODEL.NAME = 'BaselineNet'
_C.MODEL.BUILDER = 'segmentation_builder'
_C.MODEL.WEIGHTS = '/root/models/baseline_res_101.pth'
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NUM_CLASSES = 19
_C.MODEL.N_LAYERS = 101
_C.MODEL.STRIDE = 16

_C.MODEL.BN_LAYER = "SyncBN"
_C.MODEL.BN_MOM = 0.1

# EM stages
_C.MODEL.IGNORE_LABEL = 255

# Values to be used for image normalization (BGR order, since INPUT.FORMAT defaults to BGR).
# To train on images of different number of channels, just set different mean & std.
# Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
#TODO now is RGB
_C.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
# When using pre-trained models in Detectron1 or any MSRA models,
# std has been absorbed into its conv1 weights, so the std needs to be set 1.
# Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
# TODO now is RGB
_C.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]

# BACKBONE setting
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "resnet_builder"
_C.MODEL.BACKBONE.MULTI_GRIDS = [1, 1, 1]

# HEAD Setting
_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.NAME = "plainhead_builder"
_C.MODEL.HEAD.LOSS = 'OHEMCELoss'
_C.MODEL.HEAD.LOSS_THRESH = 0.7
_C.MODEL.HEAD.LOSS_MIN_KEPT = 100000
_C.MODEL.HEAD.LOSS_WEIGHTS = []
_C.MODEL.HEAD.LOSS_REDUCTION = 'mean'
_C.MODEL.HEAD.AUX_LOSS = True
_C.MODEL.HEAD.AUX_LOSS_WEIGHT = 0.4

# Non-Local Block Layer
_C.MODEL.HEAD.NL_INPUT = 512
_C.MODEL.HEAD.NL_INTER = 256
_C.MODEL.HEAD.NL_OUTPUT = 512

# HANet Head
_C.MODEL.HANET = CN()
_C.MODEL.HANET.ASPP_RATES = [6, 12, 18]
_C.MODEL.HANET.IN_DIM = 2048 #ResNet final channel
_C.MODEL.HANET.REDUCTION_DIM = 256
_C.MODEL.HANET.HANET_CONV_FLAGS = [1,1,1,1,0]
_C.MODEL.HANET.POS_INFORMATION = False
_C.MODEL.HANET.KERNEL_SIZE = 3
_C.MODEL.HANET.R_FACTOR = 64
_C.MODEL.HANET.LAYER = 3
_C.MODEL.HANET.POS_INJECTION = 2
_C.MODEL.HANET.IS_ENCODING = 1
_C.MODEL.HANET.POS_RFACTOR = 8
_C.MODEL.HANET.POOLING = 'mean'
_C.MODEL.HANET.DROPOUT_PROB = 0.1
_C.MODEL.HANET.POS_NOISE = 0.5


##Spnet
_C.MODEL.SPNET = CN()
_C.MODEL.SPNET.WITH_GLOBAL = True
# Need implemented by model

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
_C.SOLVER.IMS_PER_BATCH = 2

_C.SOLVER.MAX_ITER = 90000
_C.SOLVER.STEPS = (60000, 80000)
_C.SOLVER.GAMMA = 0.1

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.MOMENTUM = 0.9

# TODO This is only used in segmentation for adjusting the learning rate (Poly).
_C.SOLVER.POWER = 0.9

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
_C.TEST.IMS_PER_BATCH = 1
_C.TEST.EVAL_PERIOD = 0

_C.TEST.PRECISE_BN = CN({"ENABLED": False})
_C.TEST.PRECISE_BN.NUM_ITER = 200

# Testing augmentation
_C.TEST.AUG = False
_C.TEST.SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
_C.TEST.FLIP = False
_C.TEST.SAVE_PREDICTION = False

# For end-to-end tests to verify the expected accuracy.
# Each item is [task, metric, value, tolerance]
# e.g.: [['bbox', 'AP', 38.5, 0.2]]
_C.TEST.EXPECTED_RESULTS = []