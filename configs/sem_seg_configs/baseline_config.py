from configs import get_config, CfgNode

BS_config = get_config()

BS_config.MODEL = CfgNode()
BS_config.MODEL.NAME = 'BaselineNet'
BS_config.MODEL.BUILDER = 'segmentation_builder'
BS_config.MODEL.WEIGHTS = '/root/models/baseline_res_101.pth'
BS_config.MODEL.DEVICE = "cuda"
BS_config.MODEL.NUM_CLASSES = 19
BS_config.MODEL.N_LAYERS = 101
BS_config.MODEL.STRIDE = 16

BS_config.MODEL.BN_LAYER = "SyncBN"
BS_config.MODEL.BN_MOM = 0.1

# EM stages
BS_config.MODEL.IGNORE_LABEL = 255

# Values to be used for image normalization (BGR order, since INPUT.FORMAT defaults to BGR).
# To train on images of different number of channels, just set different mean & std.
# Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
#TODO now is RGB
BS_config.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
# When using pre-trained models in Detectron1 or any MSRA models,
# std has been absorbed into its conv1 weights, so the std needs to be set 1.
# Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
# TODO now is RGB
BS_config.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]

# BACKBONE setting
BS_config.MODEL.BACKBONE = CfgNode()
BS_config.MODEL.BACKBONE.NAME = "resnet_builder"
BS_config.MODEL.BACKBONE.MULTI_GRIDS = [1, 1, 1]

BS_config.MODEL.HEAD = CfgNode()
BS_config.MODEL.HEAD.NAME = "plainhead_builder"
BS_config.MODEL.HEAD.AUX_LOSS = True
BS_config.MODEL.HEAD.AUX_LOSS_WEIGHT = 0.4
# Input
BS_config.INPUT.WIDTH_TRAIN = 768
BS_config.INPUT.HEIGHT_TRAIN = 768
BS_config.INPUT.SCALES_TRAIN = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
BS_config.INPUT.WIDTH_TEST = 2048
BS_config.INPUT.HEIGHT_TEST = 1024
BS_config.INPUT.SCALES_TEST = [1.0]


# SOLVER
BS_config.SOLVER.BASE_LR = 0.01
BS_config.SOLVER.LR_SCHEDULER_NAME = "WarmupPolyLR"
BS_config.SOLVER.POWER = 0.9
BS_config.SOLVER.IMS_PER_BATCH = 4
BS_config.SOLVER.MAX_ITER = 60000  # 90000
BS_config.SOLVER.STEPS = (10,)  # 60000
BS_config.SOLVER.GAMMA = 0.1
