from configs import get_config, CfgNode

EMANet_config = get_config()

EMANet_config.MODEL = CfgNode()
EMANet_config.MODEL.NAME = 'EMANet'
EMANet_config.MODEL.BUILDER = 'segmentation_builder'
EMANet_config.MODEL.WEIGHTS = '/root/models/ema_res_50.pth'
EMANet_config.MODEL.DEVICE = "cuda"
EMANet_config.MODEL.NUM_CLASSES = 19
EMANet_config.MODEL.N_LAYERS = 50
EMANet_config.MODEL.STRIDE = 16
EMANet_config.MODEL.EM_MOM = 0.9
EMANet_config.MODEL.BN_MOM = 3e-4
EMANet_config.MODEL.BN_LAYER = "SyncBN"

# EM stages
EMANet_config.MODEL.STAGE_NUM = 3
EMANet_config.MODEL.IGNORE_LABEL = 255

# Values to be used for image normalization (BGR order, since INPUT.FORMAT defaults to BGR).
# To train on images of different number of channels, just set different mean & std.
# Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
#TODO now is RGB
EMANet_config.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
# When using pre-trained models in Detectron1 or any MSRA models,
# std has been absorbed into its conv1 weights, so the std needs to be set 1.
# Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
# TODO now is RGB
EMANet_config.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]

# BACKBONE setting
EMANet_config.MODEL.BACKBONE = CfgNode()
EMANet_config.MODEL.BACKBONE.NAME = "resnet_builder"
EMANet_config.MODEL.BACKBONE.MULTI_GRIDS = [1, 1, 1]

EMANet_config.MODEL.HEAD = CfgNode()
EMANet_config.MODEL.HEAD.NAME = "emahead_builder"
EMANet_config.MODEL.HEAD.AUX_LOSS = False

# Input
EMANet_config.INPUT.WIDTH_TRAIN = 768
EMANet_config.INPUT.HEIGHT_TRAIN = 768
EMANet_config.INPUT.SCALES_TRAIN = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
EMANet_config.INPUT.WIDTH_TEST = 2048
EMANet_config.INPUT.HEIGHT_TEST = 1024
EMANet_config.INPUT.SCALES_TEST = [1.0]


# SOLVER
EMANet_config.SOLVER.BASE_LR = 0.009
EMANet_config.SOLVER.LR_SCHEDULER_NAME = "WarmupPolyLR"
EMANet_config.SOLVER.POWER = 0.9
EMANet_config.SOLVER.IMS_PER_BATCH = 4
EMANet_config.SOLVER.MAX_ITER = 30000  # 90000
EMANet_config.SOLVER.STEPS = (10,)  # 60000
EMANet_config.SOLVER.GAMMA = 0.1
