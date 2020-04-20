from configs import get_config, CfgNode


PCNet_config = get_config()

PCNet_config.MODEL = CfgNode()
PCNet_config.MODEL.NAME = 'PCNet'
PCNet_config.MODEL.BUILDER = 'segmentation_builder'
PCNet_config.MODEL.WEIGHTS = ''
PCNet_config.MODEL.DEVICE = "cuda"
PCNet_config.MODEL.NUM_CLASSES = 19
PCNet_config.MODEL.IGNORE_LABEL = 255

# Values to be used for image normalization (BGR order, since INPUT.FORMAT defaults to BGR).
# To train on images of different number of channels, just set different mean & std.
# Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
PCNet_config.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
# When using pre-trained models in Detectron1 or any MSRA models,
# std has been absorbed into its conv1 weights, so the std needs to be set 1.
# Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
PCNet_config.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]

PCNet_config.MODEL.BACKBONE = CfgNode()
PCNet_config.MODEL.BACKBONE.NAME = "pcnet_builder"

PCNet_config.MODEL.HEAD = CfgNode()
PCNet_config.MODEL.HEAD.NAME = "pchead_builder"
PCNet_config.MODEL.HEAD.AUX_LOSS = False
# Input
PCNet_config.INPUT.WIDTH_TRAIN = 1024
PCNet_config.INPUT.HEIGHT_TRAIN = 512
PCNet_config.INPUT.SCALES_TRAIN = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] #
PCNet_config.INPUT.WIDTH_TEST = 2048
PCNet_config.INPUT.HEIGHT_TEST = 1024
PCNet_config.INPUT.SCALES_TEST = [1.0]


# SOLVER
PCNet_config.SOLVER.BASE_LR = 0.01
PCNet_config.SOLVER.LR_SCHEDULER_NAME = "WarmupPolyLR"
PCNet_config.SOLVER.POWER = 0.9
PCNet_config.SOLVER.IMS_PER_BATCH = 16
PCNet_config.SOLVER.MAX_ITER = 180000  # 90000
PCNet_config.SOLVER.STEPS = (10,)  # 60000
PCNet_config.SOLVER.GAMMA = 0.1
PCNet_config.SOLVER.WEIGHT_DECAY = 5e-4
