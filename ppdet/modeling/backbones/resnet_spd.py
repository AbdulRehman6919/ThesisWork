# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numbers import Integral

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register, serializable
from .name_adapter import NameAdapter
from ..shape_spec import ShapeSpec
from .resnet import ConvNormLayer, BottleNeck, ResNet_cfg, Blocks, SELayer

__all__ = ['ResNetSPD', 'BasicBlockSPD']


class SpaceToDepth(nn.Layer):
    def forward(self, x):
        return paddle.concat(
            [
                x[:, :, 0::2, 0::2],
                x[:, :, 1::2, 0::2],
                x[:, :, 0::2, 1::2],
                x[:, :, 1::2, 1::2],
            ],
            axis=1,
        )


def _build_norm_layer(ch_out, norm_type, norm_decay, freeze_norm, lr):
    assert norm_type in ['bn', 'sync_bn']
    norm_lr = 0. if freeze_norm else lr
    param_attr = ParamAttr(
        learning_rate=norm_lr,
        regularizer=L2Decay(norm_decay),
        trainable=False if freeze_norm else True)
    bias_attr = ParamAttr(
        learning_rate=norm_lr,
        regularizer=L2Decay(norm_decay),
        trainable=False if freeze_norm else True)
    global_stats = True if freeze_norm else None
    return nn.BatchNorm2D(
        ch_out,
        weight_attr=param_attr,
        bias_attr=bias_attr,
        use_global_stats=global_stats)


class ConvLayer(nn.Layer):
    def __init__(self, ch_in, ch_out, filter_size, stride, groups=1, lr=1.0):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(learning_rate=lr),
            bias_attr=False)

    def forward(self, inputs):
        return self.conv(inputs)


class BasicBlockSPD(nn.Layer):

    expansion = 1

    def __init__(self,
                 ch_in,
                 ch_out,
                 stride,
                 shortcut,
                 variant='b',
                 groups=1,
                 base_width=64,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False,
                 std_senet=False):
        super().__init__()
        assert groups == 1 and base_width == 64, 'BasicBlock only supports groups=1 and base_width=64'
        self.shortcut = shortcut
        self.use_spd = stride == 2

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential()
                self.short.add_sublayer(
                    'pool',
                    nn.AvgPool2D(
                        kernel_size=2, stride=2, padding=0, ceil_mode=True))
                self.short.add_sublayer(
                    'conv',
                    ConvNormLayer(
                        ch_in=ch_in,
                        ch_out=ch_out,
                        filter_size=1,
                        stride=1,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        lr=lr))
            else:
                self.short = ConvNormLayer(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    filter_size=1,
                    stride=stride,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=lr)

        if self.use_spd:
            self.branch2a_conv = ConvLayer(
                ch_in=ch_in,
                ch_out=ch_out,
                filter_size=3,
                stride=1,
                groups=1,
                lr=lr)
            self.branch2a_spd = SpaceToDepth()
            self.branch2a_norm = _build_norm_layer(
                ch_out=4 * ch_out,
                norm_type=norm_type,
                norm_decay=norm_decay,
                freeze_norm=freeze_norm,
                lr=lr)
        else:
            self.branch2a = ConvNormLayer(
                ch_in=ch_in,
                ch_out=ch_out,
                filter_size=3,
                stride=stride,
                act='relu',
                norm_type=norm_type,
                norm_decay=norm_decay,
                freeze_norm=freeze_norm,
                lr=lr)

        self.branch2b = ConvNormLayer(
            ch_in=4 * ch_out if self.use_spd else ch_out,
            ch_out=ch_out,
            filter_size=3,
            stride=1,
            act=None,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
            dcn_v2=dcn_v2)

        self.std_senet = std_senet
        if self.std_senet:
            self.se = SELayer(ch_out)

    def forward(self, inputs):
        if self.use_spd:
            out = self.branch2a_conv(inputs)
            out = self.branch2a_spd(out)
            out = self.branch2a_norm(out)
            out = F.relu(out)
        else:
            out = self.branch2a(inputs)

        out = self.branch2b(out)
        if self.std_senet:
            out = self.se(out)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        out = paddle.add(x=out, y=short)
        out = F.relu(out)
        return out


@register
@serializable
class ResNetSPD(nn.Layer):
    __shared__ = ['norm_type']

    def __init__(self,
                 depth=50,
                 ch_in=64,
                 variant='b',
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0],
                 groups=1,
                 base_width=64,
                 norm_type='bn',
                 norm_decay=0,
                 freeze_norm=True,
                 freeze_at=0,
                 return_idx=[0, 1, 2, 3],
                 dcn_v2_stages=[-1],
                 num_stages=4,
                 std_senet=False,
                 freeze_stem_only=False):
        """
        ResNet backbone with SPD-Conv downsampling blocks for depth < 50.
        """
        super().__init__()
        self._model_type = 'ResNetSPD' if groups == 1 else 'ResNeXtSPD'
        assert num_stages >= 1 and num_stages <= 4
        self.depth = depth
        self.variant = variant
        self.groups = groups
        self.base_width = base_width
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm
        self.freeze_at = freeze_at
        if isinstance(return_idx, Integral):
            return_idx = [return_idx]
        assert max(return_idx) < num_stages, \
            'the maximum return index must smaller than num_stages, ' \
            'but received maximum return index is {} and num_stages ' \
            'is {}'.format(max(return_idx), num_stages)
        self.return_idx = return_idx
        self.num_stages = num_stages
        assert len(lr_mult_list) == 4, \
            "lr_mult_list length must be 4 but got {}".format(len(lr_mult_list))
        if isinstance(dcn_v2_stages, Integral):
            dcn_v2_stages = [dcn_v2_stages]
        assert max(dcn_v2_stages) < num_stages
        self.dcn_v2_stages = dcn_v2_stages

        block_nums = ResNet_cfg[depth]
        na = NameAdapter(self)

        conv1_name = na.fix_c1_stage_name()
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, conv1_name]]
        self.conv1 = nn.Sequential()
        for (c_in, c_out, k, s, _name) in conv_def:
            self.conv1.add_sublayer(
                _name,
                ConvNormLayer(
                    ch_in=c_in,
                    ch_out=c_out,
                    filter_size=k,
                    stride=s,
                    groups=1,
                    act='relu',
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=1.0))

        self.ch_in = ch_in
        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlockSPD

        self._out_channels = [block.expansion * v for v in ch_out_list]
        self._out_strides = [4, 8, 16, 32]

        self.res_layers = []
        for i in range(num_stages):
            lr_mult = lr_mult_list[i]
            stage_num = i + 2
            res_name = "res{}".format(stage_num)
            res_layer = self.add_sublayer(
                res_name,
                Blocks(
                    block,
                    self.ch_in,
                    ch_out_list[i],
                    count=block_nums[i],
                    name_adapter=na,
                    stage_num=stage_num,
                    variant=variant,
                    groups=groups,
                    base_width=base_width,
                    lr=lr_mult,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    dcn_v2=(i in self.dcn_v2_stages),
                    std_senet=std_senet))
            self.res_layers.append(res_layer)
            self.ch_in = self._out_channels[i]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, num_stages)):
                    self._freeze_parameters(self.res_layers[i])

    def _freeze_parameters(self, m):
        for p in m.parameters():
            p.stop_gradient = True

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]

    def forward(self, inputs):
        x = inputs['image']
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
