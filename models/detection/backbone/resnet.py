import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import DET_BACKBONE_REGISTRY
import utils.nn.weight_init as weight_init

from layers import (
    Conv2d,
    ShapeSpec,
    FrozenBatchNorm2d,
    DeformConv,
    ModulatedDeformConv
)

from layers import get_norm_with_channels as get_norm

from .backbone import Backbone


class ResNetBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        The '__init__' method of any subclass should alse contain these arguments.

        :param in_channels:
        :param out_channels:
        :param stride:
        """
        super(ResNetBlockBase, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self

class BasicBlock(ResNetBlockBase):
    def __init__(self, in_channels, out_channels, *, stride=1, norm="BN"):
        """
        The standard block type for ResNet18 and ResNet34
        :param in_channels:
        :param out_channels:
        :param stride:
        :param norm:
        """
        super(BasicBlock, self).__init__(in_channels, out_channels, stride)
        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride,
                bias=False, norm=get_norm(norm, out_channels)
            )
        else:
            self.shortcut = None

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        ot = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out += shortcut
        out = F.relu_(out)
        return out


class BottleneckBlock(ResNetBlockBase):
    def __init__(self, in_channels, out_channels, *, bottleneck_channels, stride=1, num_groups=1, norm="BN", stride_in_1x1=False, dilation=1):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
            stride_in_1x1 (bool): when stride==2, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
        """
        super(BottleneckBlock, self).__init__(in_channels, out_channels, stride)
        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride,
                bias=False, norm=get_norm(norm, out_channels)
            )
        else:
            self.shortcut = None
        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementation have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels, bottleneck_channels,
            kernel_size=1, stride=stride_1x1,
            bias=False, norm=get_norm(norm, bottleneck_channels)
        )
        self.conv2 = Conv2d(
            bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride_3x3, padding=1*dilation,
            bias=False, groups=num_groups, dilation=dilation,
            norm=get_norm(norm, bottleneck_channels)
        )
        self.conv3 = Conv2d(
            bottleneck_channels, out_channels,
            kernel_size=1, bias=False,
            norm=get_norm(norm, out_channels)
        )
        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class DeformBottleneckBlock(ResNetBlockBase):
    def __init__(self, in_channels, out_channels, *, bottleneck_channels, stride=1, num_groups=1, norm='BN',
                 stride_in_1x1=False, dilation=1, deform_modulated=False, deform_num_groups=1):
        """
        With deformable conv in the 3x3 convolution.
        :param in_channels:
        :param out_channels:
        :param bottleneck_channels:
        :param stride:
        :param num_groups:
        :param norm:
        :param stride_in_1x1:
        :param dilation:
        :param deform_modulated:
        :param deform_num_group:
        """
        super(DeformBottleneckBlock, self).__init__(in_channels, out_channels, stride)
        self.deform_modulated = deform_modulated
        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels)
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels, bottleneck_channels,
            kernel_size=1, stride=stride_1x1,
            bias=False, norm=get_norm(norm, bottleneck_channels)
        )

        if deform_modulated:
            # TODO ModulatedDeformConv
            deform_conv_op = ModulatedDeformConv
            offset_channels = 27
        else:
            # TODO DeformConv
            deform_conv_op = DeformConv
            offset_channels = 18

        self.conv2_offset = Conv2d(
            bottleneck_channels, offset_channels * deform_num_groups,
            kernel_size=3, stride=stride_3x3,
            padding=1*dilation, dilation=dilation,
        )
        self.conv2 = deform_conv_op(
            bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride_3x3, padding=1*dilation,
            bias=False, groups=num_groups, dilation=dilation,
            deformable_groups=deform_num_groups, norm=get_norm(norm, bottleneck_channels)
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1, bias=False,
            norm=get_norm(norm, out_channels)
        )
        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

        nn.init.constant_(self.conv2_offset.weight, 0)
        nn.init.constant_(self.conv2_offset.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        if self.deform_modulated:
            offset_mask = self.conv2_offset(out)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


def make_stage(block_class, num_blocks, first_stride, **kwargs):
    """
    Create a resnet stage by creating many blocks.
    :param block_class: a subclass of ResNetBlockBase
    :param num_blocks:
    :param first_stride:the stride of the first block. The other blocks will have stride=1.
            A `stride` argument will be passed to the block constructor.
    :param kwargs: other arguments passed to the block constructor.
    :return:
    """
    blocks = []
    for i in range(num_blocks):
        blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
        kwargs["in_channels"] = kwargs["out_channels"]
    return blocks


class BasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, norm="BN"):
        """
        :param in_channels:
        :param out_channels:
        :param norm:
        """
        super(BasicStem, self).__init__()
        self.conv1 = Conv2d(
            in_channels, out_channels, kernel_size=7, stride=2,
            padding=3, bias=False, norm=get_norm(norm, out_channels)
        )
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 4 # = stride 2 conv -> stride 2 max pool


class ResNet(Backbone):
    def __init__(self, stem, stages, num_classes=None, out_features=None):
        """
        :param stem: a stem module
        :param stages:  several stages each comtains multiple :class'ResNetBlockBase'
        :param num_classes: if None, will not perform classification
        :param out_features:  name of the layers whose outputs should be returned in forward.
                                if None, will return the output of the last layer
        """
        super(ResNet, self).__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, ResNetBlockBase), block
                curr_channels = block.out_channels
            stage = nn.Sequential(*blocks)
            name = "res" + str(i+2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(current_stride * np.prod([k.stride for k in blocks]))
            self._out_feature_channels[name] = blocks[-1].out_channels

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.linear = nn.Linear(curr_channels, num_classes)
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x

        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }



def build_resnet(cfg, input_shape):
    """
    Create a ResNet instance from config
    :param cfg:
    :param input_shape:
    :return: ResNet: a "class:'ResNet' instance.
    """
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm
    )
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    # fmt:off
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    depth = cfg.MODEL.RESNETS.DEPTH
    num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS

    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(
            deform_on_per_stage
        ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/34"

    stages = []
    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2":2, "res3":3, "res4":4, "res5":5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)

    for idx, stage_idx in enumerate(range(2, max_stage_idx+1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        # Use BasicBlock for R18 and R34
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"]: dilation
            stage_kargs["num_groups"]: num_groups
            if deform_on_per_stage[idx]:
                stage_kargs["block_class"] = DeformBottleneckBlock
                stage_kargs["deform_modulated"] = deform_modulated
                stage_kargs["deform_num_groups"] = deform_num_groups
            else:
                stage_kargs["block_class"] = BottleneckBlock
        blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at > stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features)


@DET_BACKBONE_REGISTRY.register()
def resnet_builder(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    backbone = build_resnet(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone
