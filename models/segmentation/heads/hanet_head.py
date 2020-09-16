"""
    The implementation of HANet Head
"""
import torch
import torch.nn as nn
from models.losses import get_loss_from_cfg
from models.segmentation.heads import SEG_HEAD_REGISTRY
from models.segmentation.segmods.hanetmods import ASPP, HANet_Conv


# refer to the implementation of `plain_head.py`
# refer to `https://github.com/shachoi/HANet/blob/master/network/deepv3.py`
# @Zhiqiang Song
class HANetHead(nn.Module):
    def __init__(self, cfg, norm_layer=None):
        super(HANetHead, self).__init__()

        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.hanet_conv_flags = cfg.MODEL.HANET.HANET_CONV_FLAGS
        self.aug_loss = cfg.MODEL.HEAD.AUX_LOSS
        self.aux_weight = cfg.MODEL.HEAD.AUX_LOSS_WEIGHT

        channel_3rd = 256
        prev_final_channel = 1024
        final_channel = 2048

        self.aspp = ASPP(cfg)

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            norm_layer(48),
            nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True))

        self.final1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(256, cfg.num_classes, kernel_size=1, bias=True))
        if self.cfg.aux_loss is True:
            self.dsn = nn.Sequential(
                nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
                norm_layer(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(512, cfg.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
            initialize_weights(self.dsn)

        if self.hanet_conv_flags[0] == 1:
            self.hanet0 = HANet_Conv(prev_final_channel, final_channel,
                                     self.cfg.hanet_set[0], self.cfg.hanet_set[1], self.cfg.hanet_set[2],
                                     self.cfg.hanet_pos[0], self.cfg.hanet_pos[1],
                                     pos_rfactor=self.cfg.pos_rfactor, pooling=self.cfg.pooling,
                                     dropout_prob=self.cfg.dropout, pos_noise=self.cfg.pos_noise)
            initialize_weights(self.hanet0)

        if self.hanet_conv_flags[1] == 1:
            self.hanet1 = HANet_Conv(final_channel, 1280,
                                     self.cfg.hanet_set[0], self.cfg.hanet_set[1], self.cfg.hanet_set[2],
                                     self.cfg.hanet_pos[0], self.cfg.hanet_pos[1],
                                     pos_rfactor=self.cfg.pos_rfactor, pooling=self.cfg.pooling,
                                     dropout_prob=self.cfg.dropout, pos_noise=self.cfg.pos_noise)
            initialize_weights(self.hanet1)

        if self.hanet_conv_flags[2] == 1:
            self.hanet2 = HANet_Conv(1280, 256,
                                     self.cfg.hanet_set[0], self.cfg.hanet_set[1], self.cfg.hanet_set[2],
                                     self.cfg.hanet_pos[0], self.cfg.hanet_pos[1],
                                     pos_rfactor=self.cfg.pos_rfactor, pooling=self.cfg.pooling,
                                     dropout_prob=self.cfg.dropout, pos_noise=self.cfg.pos_noise)
            initialize_weights(self.hanet2)

        if self.hanet_conv_flags[3] == 1:
            self.hanet3 = HANet_Conv(304, 256,
                                     self.cfg.hanet_set[0], self.cfg.hanet_set[1], self.cfg.hanet_set[2],
                                     self.cfg.hanet_pos[0], self.cfg.hanet_pos[1],
                                     pos_rfactor=self.cfg.pos_rfactor, pooling=self.cfg.pooling,
                                     dropout_prob=self.cfg.dropout, pos_noise=self.cfg.pos_noise)
            initialize_weights(self.hanet3)

        if self.hanet_conv_flags[4] == 1:
            self.hanet4 = HANet_Conv(256, self.num_classes,
                                     self.cfg.hanet_set[0], self.cfg.hanet_set[1], self.cfg.hanet_set[2],
                                     self.cfg.hanet_pos[0], self.cfg.hanet_pos[1],
                                     pos_rfactor=self.cfg.pos_rfactor, pooling='max',
                                     dropout_prob=self.cfg.dropout, pos_noise=self.cfg.pos_noise)
        initialize_weights(self.hanet4)
        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        self.loss_fn = get_loss_from_cfg(cfg)

    def forward(self, data_input, label=None):
        low_level = data_input['fea']
        aux_out = data_input['res3']
        x = data_input['res4']
        x_size = label.size()[-2:]

        # hanet 0
        if self.hanet_conv_flags[0] == 1:
            x = self.hanet0(aux_out, x, self.cfg.pos)

        represent = x

        x = self.aspp(x)

        # hanet 1
        if self.hanet_conv_flags[1] == 1:
            x = self.hanet1(represent, x, self.cfg.pos)

        dec0_up = self.bot_aspp(x)

        # hanet 2
        if self.hanet_conv_flags[2] == 1:
            dec0_up = self.hanet2(x, dec0_up, self.cfg.pos)

        dec0_fine = self.bot_fine(low_level)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final1(dec0)

        # hanet 3
        if self.hanet_conv_flags[3] == 1:
            dec1 = self.hanet3(dec0, dec1, self.cfg.pos)

        dec2 = self.final2(dec1)

        if self.hanet_conv_flags[4] == 1:
            dec2 = self.hanet4(dec1, dec2, self.cfg.pos)

        main_out = Upsample(dec2, x_size[2:])

        if label is None:
            return main_out

        if self.aug_loss:
            aux_out = self.dsn(aux_out)
            pred = (main_out, aux_out)
            return self._compute_loss(pred, label)

        else:
            return self._compute_loss(main_out, label)

    def _compute_loss(self, pred, label):
        if self.aux_loss:
            pred, aux_pred = pred
            loss = self.loss_fn(pred, label)
            label = nn.functional.interpolate(label, size=aux_pred.shape[2:], mode='nearest')
            aux_loss = self.loss_fn(aux_pred, label)
            return {
                'loss': loss,
                'aux_loss': aux_loss
            }
        else:
            loss = self.loss_fn(pred, label)
            return {
                'loss': loss
            }


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or \
                isinstance(module, nn.GroupNorm) or isinstance(module, nn.SyncBatchNorm):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


# register the hanet head into `SEG_HEAD_REGISTRY`
# @Jingfei Sun
@SEG_HEAD_REGISTRY.register()
def hanet_builder(cfg):
    from layers.batch_norm import get_norm
    norm_layer = get_norm(cfg.MODEL.BN_LAYER)
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    return HANetHead(cfg, norm_layer)

