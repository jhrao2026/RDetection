"""练习 2.2b：FPN（特征金字塔）neck

================================================================================
为什么需要 FPN？
    backbone 的 C5（stride 32）语义强但分辨率低，小目标几乎没了；
    C2（stride 4）分辨率高但语义弱。FPN 用 "自顶向下 + 横向连接" 把高层语义传到低层：

        C5 ─1x1─► L5 ───────────────────────3x3─► P5 ──maxpool(stride 2)──► P6
                   │ 上采样 ×2
        C4 ─1x1─► L4 (+) ───────────────────3x3─► P4
                   │ 上采样
        C3 ─1x1─► L3 (+) ───────────────────3x3─► P3
                   │ 上采样
        C2 ─1x1─► L2 (+) ───────────────────3x3─► P2

    - lateral_convs（1x1）：把各层通道统一到 out_channels（256）
    - 自顶向下：上层上采样后 **加** 到下一层
    - fpn_convs（3x3）：消除上采样带来的混叠
    - 额外层：Faster R-CNN 用 P5 做 stride=2 的 max pool 得到 P6（add_extra_convs=False）；
      RetinaNet 用 stride=2 的 3x3 卷积生成 P6、P7（add_extra_convs='on_input'，2.6 会用到）

    ★ 命名与 mmdet 一致：lateral_convs.{i}.conv、fpn_convs.{i}.conv。
      mmdet 用 mmcv 的 ConvModule（conv + 可选 norm + 可选激活），所以参数名中间多了一层 ".conv"。
      下面先写一个极简的 ConvModule 来对齐命名。

参考源码
    mmdet/models/necks/fpn.py
    torchvision/ops/feature_pyramid_network.py（等价于 add_extra_convs=False 的版本，测试会和它对比）

完成标准
    pytest tests/test_fpn.py
================================================================================
"""

from typing import List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from rdet.registry import MODELS


class ConvModule(nn.Module):
    """mmcv ConvModule 的极简版（已完成）：只有 conv，可选 ReLU。命名为 .conv 以对齐 mmdet。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, act=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x):
        x = self.conv(x)
        return self.act(x) if self.act is not None else x


@MODELS.register_module()
class RFPN(nn.Module):
    """Args:
        in_channels: backbone 各层通道，例如 [256, 512, 1024, 2048]
        out_channels: 输出通道，256
        num_outs: 输出层数（Faster R-CNN 是 5：P2–P6）
        start_level: 从 backbone 的第几层开始用（RetinaNet 是 1，不用 C2）
        add_extra_convs: False | 'on_input' | 'on_lateral' | 'on_output'
            False：额外层用 max pool；否则用 stride=2 的 3x3 卷积，并说明第一个额外卷积的输入来源
        relu_before_extra_convs: 第二个及以后的额外卷积之前是否先过 ReLU
    """

    def __init__(self, in_channels: Sequence[int], out_channels: int, num_outs: int,
                 start_level: int = 0, add_extra_convs: Union[bool, str] = False,
                 relu_before_extra_convs: bool = False, init_cfg=None, **kwargs):
        super().__init__()
        assert add_extra_convs in (False, 'on_input', 'on_lateral', 'on_output')
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.start_level = start_level
        self.backbone_end_level = len(in_channels)
        self.add_extra_convs = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        # TODO(1) 对 i in range(start_level, backbone_end_level)：
        #         lateral_convs 追加 ConvModule(in_channels[i], out_channels, 1)
        #         fpn_convs     追加 ConvModule(out_channels, out_channels, 3, padding=1)

        # TODO(2) 额外卷积（只有 add_extra_convs 不为 False 时才有）：
        #         extra_levels = num_outs - (backbone_end_level - start_level)
        #         对第 i 个额外层（i 从 0 开始）：
        #             in_ch = in_channels[backbone_end_level - 1] 如果 i == 0 且 add_extra_convs == 'on_input'
        #                     否则 out_channels
        #             fpn_convs 追加 ConvModule(in_ch, out_channels, 3, stride=2, padding=1)
        #         注意：额外卷积是 **接在 fpn_convs 后面** 的，不是单独的 ModuleList（为了对齐 mmdet 权重）
        raise NotImplementedError('RFPN.__init__')

    def init_weights(self) -> None:
        """练习 4：与 mmdet 一致，所有卷积用 xavier_uniform 初始化，bias 置 0。

        提示：遍历 self.modules()，对 nn.Conv2d 调用 nn.init.xavier_uniform_(m.weight) 和
        nn.init.constant_(m.bias, 0)。
        思考：PyTorch 的 Conv2d 默认是什么初始化？为什么检测器的新层不直接用默认值？
        """
        # TODO(4)
        raise NotImplementedError('RFPN.init_weights')

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """练习 3。

        assert len(inputs) == len(self.in_channels)

        1. laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in ...]

        2. 自顶向下（从最高层往下，逐层加）：
               for i in range(len(laterals) - 1, 0, -1):
                   prev_shape = laterals[i - 1].shape[2:]
                   laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, mode='nearest')
           思考：为什么用 size= 而不是 scale_factor=2？（提示：输入图像边长不是 32 的倍数时）
           注意：不要写成 laterals[i - 1] += ...（in-place 操作在某些情况下会破坏反向传播）

        3. outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]

        4. 额外层（num_outs > len(outs) 时）：
           - add_extra_convs 为 False：
                 for _ in range(num_outs - len(outs)): outs.append(F.max_pool2d(outs[-1], 1, stride=2))
           - 否则：
                 第一个额外层的输入 extra_source：
                     'on_input'   → inputs[self.backbone_end_level - 1]
                     'on_lateral' → laterals[-1]
                     'on_output'  → outs[-1]
                 outs.append(self.fpn_convs[len(laterals)](extra_source))
                 之后的额外层 i：输入是 outs[-1]（relu_before_extra_convs 时先 F.relu）

        5. return tuple(outs)
        """
        # TODO(3)
        raise NotImplementedError('RFPN.forward')


_ = (F, List)  # 写完后删掉这一行
