from configs import get_config, CfgNode

NL_config = get_config()

NL_config.OUTPUT_DIR = './output_dir'

NL_config.MODEL = CfgNode()
NL_config.MODEL.NAME = 'NonlocalNet'
NL_config.MODEL.BUILDER = 'segmentation_builder'
NL_config.MODEL.WEIGHTS = '/root/models/baseline_res_101.pth'
NL_config.MODEL.DEVICE = "cuda"
NL_config.MODEL.NUM_CLASSES = 19
NL_config.MODEL.N_LAYERS = 101
NL_config.MODEL.STRIDE = 16

NL_config.MODEL.BN_LAYER = "SyncBN"
NL_config.MODEL.BN_MOM = 0.1

NL_config.MODEL.IGNORE_LABEL = 255

# Values to be used for image normalization (BGR order, since INPUT.FORMAT defaults to BGR).
# To train on images of different number of channels, just set different mean & std.
# Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
#TODO now is RGB
NL_config.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
# When using pre-trained models in Detectron1 or any MSRA models,
# std has been absorbed into its conv1 weights, so the std needs to be set 1.
# Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
# TODO now is RGB
NL_config.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]

# BACKBONE setting
NL_config.MODEL.BACKBONE = CfgNode()
NL_config.MODEL.BACKBONE.NAME = "resnet_builder"
NL_config.MODEL.BACKBONE.MULTI_GRIDS = [1, 1, 1]

NL_config.MODEL.HEAD = CfgNode()
NL_config.MODEL.HEAD.NAME = "nlhead_builder"
NL_config.MODEL.HEAD.AUX_LOSS = True
NL_config.MODEL.HEAD.AUX_LOSS_WEIGHT = 0.4
# NL settings
NL_config.MODEL.HEAD.NL_INPUT = 512
NL_config.MODEL.HEAD.NL_INTER = 256
NL_config.MODEL.HEAD.NL_OUTPUT = 512

# Input
NL_config.INPUT.WIDTH_TRAIN = 768
NL_config.INPUT.HEIGHT_TRAIN = 768
NL_config.INPUT.SCALES_TRAIN = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
NL_config.INPUT.WIDTH_TEST = 2048
NL_config.INPUT.HEIGHT_TEST = 1024
NL_config.INPUT.SCALES_TEST = [1.0]


# SOLVER
NL_config.SOLVER.BASE_LR = 0.01
NL_config.SOLVER.LR_SCHEDULER_NAME = "WarmupPolyLR"
NL_config.SOLVER.POWER = 0.9
NL_config.SOLVER.IMS_PER_BATCH = 4
NL_config.SOLVER.MAX_ITER = 60000  # 90000
NL_config.SOLVER.STEPS = (10,)  # 60000
NL_config.SOLVER.GAMMA = 0.1
