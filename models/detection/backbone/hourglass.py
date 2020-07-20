# CenterNet
# CornerNet
import torch.nn as nn


from models.detection.backbone.backbone import Backbone
from layers import AnchorFreeConvBlock, CornerNetResidual, MergeUp, ShapeSpec
from . import DET_BACKBONE_REGISTRY


class HourGlassBackbone(Backbone):
    def __init__(self, n, dims, modules, nstack, cnv_dim, out_features, input_shape):
        super(HourGlassBackbone, self).__init__()
        self.n = n
        self.dims = dims
        self.hg_modules = modules
        self.nstack = nstack
        self.cnv_dim = cnv_dim
        self._out_features = out_features

        dims = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]

        curr_dim = dims[0]
        cnv_dim = 256

        # TODO align model names
        self.pre = nn.Sequential(
            AnchorFreeConvBlock(in_channels=input_shape.channels, out_channels=128, kernel_size=7, stride=2),
            CornerNetResidual(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        )

        # pre stride
        current_stride = 4
        self._out_feature_strides = {'stem': current_stride}
        self._out_feature_channels = {'stem': curr_dim}


        self.kps = nn.ModuleList(
            [HourGlassModule(n, dims, modules) for _ in range(self.nstack)]
        )

        self.cnvs = nn.ModuleList(
            [self._make_conv_layers(curr_dim, cnv_dim) for _ in range(self.nstack)]
        )

        self.inters = nn.ModuleList([
            self._make_inter_layers(curr_dim) for _ in range(self.nstack - 1)
        ])

        self.inter_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, kernel_size=1, bias=False),
                # TODO change the BN to `get_norm`
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(self.nstack - 1)
        ])
        self.cnvs_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, kernel_size=1, bias=False),
                # TODO change the BN to `get_norm`
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(self.nstack - 1)
        ])

        self.relu = nn.ReLU(inplace=True)

        for i in range(self.nstack):
            name = 'hg' + str(i+1)
            self._out_feature_strides[name] = current_stride
            self._out_feature_channels[name] = curr_dim

    def _make_conv_layers(self, in_channels, out_channels):
        return AnchorFreeConvBlock(in_channels, out_channels, kernel_size=3)

    def _make_inter_layers(self, in_channels):
        # TODO change get_norm
        return CornerNetResidual(in_channels, in_channels, kernel_size=3, stride=1)

    def forward(self, in_feature):
        outputs = {}
        inter = self.pre(in_feature)
        if 'stem' in self._out_features:
            outputs['stem'] = inter
        # stge - 1
        kp_1 = self.kps[0](inter)
        cnv_1 = self.cnvs[0](kp_1)


        if 'hg1' in self._out_features:
            outputs['hg1'] = cnv_1
        # inter stage
        inter = self.inter_[0](inter) + self.cnvs_[0](cnv_1)
        inter = self.relu(inter)
        inter = self.inters[0](inter)

        # stage - 2
        kp_2 = self.kps[1](inter)
        cnv_2 = self.cnvs[1](kp_2)

        if 'hg2' in self._out_features:
            outputs['hg2'] = cnv_2
        return outputs

    @property
    def size_divisibility(self):
        return 128


class HourGlassModule(nn.Module):
    def __init__(self, n, dims, modules):
        super(HourGlassModule, self).__init__()
        self.n = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1 = self._make_up_layer(in_channels=curr_dim, out_channels=curr_dim, kernel_size=3, stride=1, norm="BN", modules=curr_mod)
        self.max1 = self._make_pool_layer(curr_dim)
        self.low1 = self._make_hg_layer(in_channels=curr_dim, out_channels=next_dim, kernel_size=3, modules=curr_mod)

        self.low2 = HourGlassModule(n=n-1, dims=dims[1:], modules=modules[1:]) if self.n > 1 else self._make_low_layer(in_channels=next_dim, out_channels=next_dim, kernel_size=3, modules=next_mod)

        self.low3 = self._make_hg_layer_revr(in_channels=next_dim, out_channels=curr_dim, kernel_size=3, modules=curr_mod)

        self.up2 = self._make_unpool_layer(curr_dim)
        self.merge = self._make_merge_layer(curr_dim)

    def _make_up_layer(self, in_channels, out_channels, kernel_size, stride=1, norm="BN", modules=1):
        # TODO check the CornerNetResidual
        layers = [CornerNetResidual(in_channels, out_channels, kernel_size, stride, norm=norm)]
        for _ in range(1, modules):
            layers.append(CornerNetResidual(in_channels, out_channels, kernel_size, stride, norm=norm))
        return nn.Sequential(*layers)

    def _make_pool_layer(self, curr_dim):
        return nn.Sequential()

    def _make_hg_layer(self, in_channels, out_channels, kernel_size, stride=1, norm="BN", modules=1):
        # TODO check the AnchorFreeConvBlock
        layers = [CornerNetResidual(in_channels, out_channels, kernel_size, stride=2, norm=norm)]
        layers += [CornerNetResidual(out_channels, out_channels, kernel_size, norm=norm) for _ in range(modules - 1)]
        return nn.Sequential(*layers)

    def _make_low_layer(self, in_channels, out_channels, kernel_size, stride=1, norm="BN", modules=1):
        # TODO check the CornerNetResidual
        layers = [CornerNetResidual(in_channels, out_channels, kernel_size, stride, norm=norm)]
        for _ in range(1, modules):
            layers.append(CornerNetResidual(in_channels, out_channels, kernel_size, stride, norm=norm))
        return nn.Sequential(*layers)

    def _make_hg_layer_revr(self, in_channels, out_channels, kernel_size, stride=1, norm="BN", modules=1):
        # TODO check the CornerNetResidual
        layers = []

        for _ in range(1, modules):
            layers.append(CornerNetResidual(in_channels, in_channels, kernel_size, stride, norm))
        layers.append(CornerNetResidual(in_channels, out_channels, kernel_size, stride, norm))
        return nn.Sequential(*layers)

    def _make_unpool_layer(self, curr_dim):
        return nn.Upsample(scale_factor=2)

    def _make_merge_layer(self, curr_dim):
        return MergeUp()

    def forward(self, in_feature):
        up1 = self.up1(in_feature)
        max1 = self.max1(in_feature)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return self.merge(up1, up2)


def build_hourglass(cfg, input_shape):
    n = cfg.MODEL.HOURGLASS.N
    dims = cfg.MODEL.HOURGLASS.DIMS
    modules = cfg.MODEL.HOURGLASS.MODULES
    cnv_dim = cfg.MODEL.HOURGLASS.CNV_DIM
    nstack = cfg.MODEL.HOURGLASS.NSTACK
    out_features = cfg.MODEL.HOURGLASS.OUT_FEATURES

    model = HourGlassBackbone(n, dims, modules, nstack, cnv_dim, out_features, input_shape)
    return model

@DET_BACKBONE_REGISTRY.register()
def hourglass_builder(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    backbone = build_hourglass(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone

if __name__ == '__main__':
    import torch
    from configs.config import get_detection_config
    cfg = get_detection_config()

    model = hourglass_builder(cfg)

    x = torch.randn(2,3,128,128)

    # x = torch.randn(2, 256, 64, 64)
    # y = module(x)
    y = model(x)
    print(y['hg1'].size())
    print(y['hg2'].size())
    print(model.output_shape())