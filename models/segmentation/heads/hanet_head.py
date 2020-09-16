"""
    The implementation of HANet Head
"""
import torch
import torch.nn as nn
from models.segmentation.segmods.hanetmods import ASPP, HANet_Conv
from models.segmentation.heads import SEG_HEAD_REGISTRY


# refer to the implementation of `plain_head.py`
# refer to `https://github.com/shachoi/HANet/blob/master/network/deepv3.py`
# @Zhiqiang Song
class HANetHead(nn.Module):
    def __init__(self, cfg):
        super(HANetHead, self).__init__()
        self.cfg= cfg
        self.criterion = cfg.criterion
        self.criterion_aux = cfg.criterion_aux
        self.variant = cfg.variant
        self.num_attention_layer = 0
        self.trunk = cfg.trunk
        channel_1st = 3
        channel_2nd = 64
        channel_3rd = 256
        channel_4th = 512
        prev_final_channel = 1024
        final_channel = 2048
        for i in range(5):
            if cfg.hanet[i] > 0:
                self.num_attention_layer += 1
        if self.variant == 'D':
            os = 8
        elif self.variant == 'D4':
            os = 4
        elif self.variant == 'D16':
            os = 16
        else:
            os = 32
        self.aspp = ASPP(final_channel, 256, output_stride=os)

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.final1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(256, cfg.num_classes, kernel_size=1, bias=True))
        if self.cfg.aux_loss is True:
            self.dsn = nn.Sequential(
                nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(512, cfg.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
            initialize_weights(self.dsn)

        if self.cfg.hanet[0] == 1:
            self.hanet0 = HANet_Conv(prev_final_channel, final_channel,
                                     self.cfg.hanet_set[0], self.cfg.hanet_set[1], self.cfg.hanet_set[2],
                                     self.cfg.hanet_pos[0], self.cfg.hanet_pos[1],
                                     pos_rfactor=self.cfg.pos_rfactor, pooling=self.cfg.pooling,
                                     dropout_prob=self.cfg.dropout, pos_noise=self.cfg.pos_noise)
            initialize_weights(self.hanet0)

        if self.cfg.hanet[1] == 1:
            self.hanet1 = HANet_Conv(final_channel, 1280,
                                     self.cfg.hanet_set[0], self.cfg.hanet_set[1], self.cfg.hanet_set[2],
                                     self.cfg.hanet_pos[0], self.cfg.hanet_pos[1],
                                     pos_rfactor=self.cfg.pos_rfactor, pooling=self.cfg.pooling,
                                     dropout_prob=self.cfg.dropout, pos_noise=self.cfg.pos_noise)
            initialize_weights(self.hanet1)

        if self.cfg.hanet[2] == 1:
            self.hanet2 = HANet_Conv(1280, 256,
                                     self.cfg.hanet_set[0], self.cfg.hanet_set[1], self.cfg.hanet_set[2],
                                     self.cfg.hanet_pos[0], self.cfg.hanet_pos[1],
                                     pos_rfactor=self.cfg.pos_rfactor, pooling=self.cfg.pooling,
                                     dropout_prob=self.cfg.dropout, pos_noise=self.cfg.pos_noise)
            initialize_weights(self.hanet2)

        if self.cfg.hanet[3] == 1:
            self.hanet3 = HANet_Conv(304, 256,
                                     self.cfg.hanet_set[0], self.cfg.hanet_set[1], self.cfg.hanet_set[2],
                                     self.cfg.hanet_pos[0], self.cfg.hanet_pos[1],
                                     pos_rfactor=self.cfg.pos_rfactor, pooling=self.cfg.pooling,
                                     dropout_prob=self.cfg.dropout, pos_noise=self.cfg.pos_noise)
            initialize_weights(self.hanet3)

        if self.cfg.hanet[4] == 1:
            self.hanet4 = HANet_Conv(256, num_classes,
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


    def forward(self, data_input,label = None):


        low_level = data_input['fea']
        aux_out = data_input['res3']
        x = data_input['res4']
        x_size = label.size()[-2:]
        if self.num_attention_layer > 0:
            if self.cfg.attention_map:
                attention_maps = [torch.Tensor() for i in range(self.num_attention_layer)]
                pos_maps = [torch.Tensor() for i in range(self.num_attention_layer)]
                map_index = 0

        if set.cfg.hanet[0] == 1:
            if self.cfg.attention_map:
                x, attention_maps[map_index], pos_maps[map_index] = self.hanet0(aux_out, x, self.cfg.pos, return_attention=True,
                                                                                return_posmap=True)
                map_index += 1
            else:
                x = self.hanet0(aux_out, x, self.cfg.pos)

        represent = x

        x = self.aspp(x)

        if self.args.hanet[1] == 1:
            if self.cfg.attention_map:
                x, attention_maps[map_index], pos_maps[map_index] = self.hanet1(represent, x, self.cfg.pos,
                                                                                return_attention=True,
                                                                                return_posmap=True)
                map_index += 1
            else:
                x = self.hanet1(represent, x, self.cfg.pos)

        dec0_up = self.bot_aspp(x)

        if self.args.hanet[2] == 1:
            if self.cfg.attention_map:
                dec0_up, attention_maps[map_index], pos_maps[map_index] = self.hanet2(x, dec0_up, self.cfg.pos,
                                                                                      return_attention=True,
                                                                                      return_posmap=True)
                map_index += 1
            else:
                dec0_up = self.hanet2(x, dec0_up, self.cfg.pos)

        dec0_fine = self.bot_fine(low_level)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final1(dec0)

        if self.args.hanet[3] == 1:
            if self.cfg.attention_map:
                dec1, attention_maps[map_index], pos_maps[map_index] = self.hanet3(dec0, dec1, self.cfg.pos,
                                                                                   return_attention=True,
                                                                                   return_posmap=True)
                map_index += 1
            else:
                dec1 = self.hanet3(dec0, dec1, self.cfg.pos)

        dec2 = self.final2(dec1)

        if self.args.hanet[4] == 1:
            if self.cfg.attention_map:
                dec2, attention_maps[map_index], pos_maps[map_index] = self.hanet4(dec1, dec2, self.cfg.pos,
                                                                                   return_attention=True,
                                                                                   return_posmap=True)
                map_index += 1
            elif self.cfg.attention_loss:
                dec2, last_attention = self.hanet4(dec1, dec2, self.cfg.pos, return_attention=False, return_posmap=False,
                                                   attention_loss=True)
            else:
                dec2 = self.hanet4(dec1, dec2, self.cfg.pos)

        main_out = Upsample(dec2, x_size[2:])

        if self.training:
            loss1 = self.criterion(main_out, self.cfg.gts)

            if self.cfg.aux_loss is True:
                aux_out = self.dsn(aux_out)
                if self.cfg.aux_gts.dim() == 1:
                    aux_gts = self.cfg.gts
                aux_gts = aux_gts.unsqueeze(1).float()
                aux_gts = nn.functional.interpolate(aux_gts, size=aux_out.shape[2:], mode='nearest')
                aux_gts = aux_gts.squeeze(1).long()
                loss2 = self.criterion_aux(aux_out, aux_gts)
                if self.cfg.attention_loss:
                    return (loss1, loss2, last_attention)
                else:
                    return (loss1, loss2)
            else:
                if self.cfg.attention_loss:
                    return (loss1, last_attention)
                else:
                    return loss1
        else:
            if self.cfg.attention_map:
                return main_out, attention_maps, pos_maps
            else:
                return main_out


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

