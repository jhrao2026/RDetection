"""练习 2.2a：ResNet-50 backbone

================================================================================
目标
    实现检测里最经典的 backbone：ResNet-50（Bottleneck × [3, 4, 6, 3]），
    输出 4 个 stage 的特征图 C2–C5（stride 4 / 8 / 16 / 32，通道 256 / 512 / 1024 / 2048）。

和分类用的 ResNet 有什么不同？
    1. 不要最后的 avgpool 和 fc，而是输出 **多个 stage** 的特征给 FPN
    2. frozen_stages：冻结前几个 stage 的参数（ImageNet 预训练学到的边缘、纹理足够通用，
       冻结后省显存、更稳）
    3. norm_eval：训练时也让 BN 保持 eval 模式 —— 检测每卡只有 2 张图，
       batch 统计量非常不准，不如直接用 ImageNet 上统计好的 running_mean / running_var

    ★ 子模块命名必须和 mmdet / torchvision 一致（conv1, bn1, layer1.0.conv1, layer1.0.downsample.0 ...），
      这样才能直接加载 ImageNet 预训练权重和 mmdet 官方 checkpoint。测试会检查这一点。

你会学到
    - 残差结构、Bottleneck（1x1 降维 → 3x3 → 1x1 升维）为什么能把网络做深
    - "pytorch 风格" 与 "caffe 风格"：stride=2 放在 3x3 卷积上还是 1x1 卷积上
    - nn.Module.train() 可以被重写，这是实现 norm_eval 的关键

参考源码
    mmdet/models/backbones/resnet.py（Bottleneck、ResNet、make_res_layer、_freeze_stages、train）
    torchvision/models/resnet.py（结构完全一样，代码更短，可以先读这个）

完成标准
    pytest tests/test_resnet.py   （加载 torchvision resnet50 的权重，各 stage 输出必须一致）
================================================================================
"""

from typing import Sequence, Tuple

import torch
from torch import nn

from rdet.registry import MODELS


class Bottleneck(nn.Module):
    """1x1 (降维) → 3x3 (stride 在这里) → 1x1 (升维 ×4)，再加上 shortcut。

    Args:
        inplanes: 输入通道
        planes: 中间通道；输出通道 = planes * expansion
        stride: 3x3 卷积的步长（每个 stage 的第一个 block 为 2，layer1 除外）
        downsample: shortcut 分支；输入输出形状不一致时为 Sequential(1x1 conv, BN)，否则 None

    练习 1：
    - __init__：conv1 / bn1 / conv2 / bn2 / conv3 / bn3 / relu / downsample，
      所有 conv 都 bias=False（后面紧跟 BN，bias 是多余的 —— 思考为什么）
      conv2 是 3x3、padding=1、stride=stride
    - forward：
          identity = x
          out = relu(bn1(conv1(x))); out = relu(bn2(conv2(out))); out = bn3(conv3(out))
          if downsample is not None: identity = downsample(x)
          return relu(out + identity)
    """
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        # TODO(1.1)
        raise NotImplementedError('Bottleneck.__init__')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(1.2)
        raise NotImplementedError('Bottleneck.forward')


@MODELS.register_module()
class RResNet(nn.Module):
    """Args:
        depth: 50 或 101
        out_indices: 输出哪些 stage（0 = layer1 ... 3 = layer4）
        frozen_stages: -1 不冻结；0 冻结 stem（conv1 + bn1）；k 冻结 stem 和 layer1..layerk
        norm_eval: 训练时是否让所有 BN 保持 eval
        init_cfg: dict(type='Pretrained', checkpoint=...)，checkpoint 可以是
                  'torchvision://resnet50'（下载 torchvision 的 ImageNet 权重）或本地 .pth 路径；
                  None 表示随机初始化。真正加载发生在 init_weights()（练习 7）
    """

    arch_settings = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

    def __init__(self, depth: int = 50, out_indices: Sequence[int] = (0, 1, 2, 3),
                 frozen_stages: int = 1, norm_eval: bool = True, init_cfg=None, **kwargs):
        super().__init__()
        assert depth in self.arch_settings, f'invalid depth {depth}'
        self.out_indices = tuple(out_indices)
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.init_cfg = init_cfg

        # TODO(2.1) stem：conv1(3→64, 7x7, stride 2, padding 3, bias=False)、bn1、relu、
        #           maxpool(3x3, stride 2, padding 1)
        #           到这里特征图已经是原图的 1/4

        # TODO(2.2) 4 个 stage：self.layer1 ... self.layer4
        #           planes 依次 64 / 128 / 256 / 512；stride 依次 1 / 2 / 2 / 2；
        #           block 数来自 self.arch_settings[depth]
        #           用 self._make_layer 构建，并用 setattr(self, f'layer{i + 1}', layer) 注册
        #           提示：self.inplanes 记录当前输入通道，_make_layer 会更新它

        # TODO(2.3) 调用 self._freeze_stages()
        raise NotImplementedError('RResNet.__init__')

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        """练习 3：构建一个 stage。

        - 第一个 block：如果 stride != 1 或 self.inplanes != planes * 4，需要 downsample：
              nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, 1, stride, bias=False),
                            nn.BatchNorm2d(planes * 4))
        - 之后 self.inplanes = planes * 4，再追加 blocks - 1 个 stride=1 的 Bottleneck
        """
        # TODO(3)
        raise NotImplementedError('RResNet._make_layer')

    def _freeze_stages(self) -> None:
        """练习 4：冻结参数。

        - frozen_stages >= 0：bn1 设为 eval()，conv1 和 bn1 的所有参数 requires_grad = False
        - 对 i in 1..frozen_stages：layer{i} 设为 eval()，参数 requires_grad = False

        思考：只设 requires_grad=False 而不设 eval()，BN 的 running_mean 还会变吗？
        """
        # TODO(4)
        raise NotImplementedError('RResNet._freeze_stages')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """练习 5：stem → layer1..4，把 out_indices 里的 stage 输出收集成 tuple 返回。"""
        # TODO(5)
        raise NotImplementedError('RResNet.forward')

    def init_weights(self) -> None:
        """练习 7：加载 ImageNet 预训练权重。

        mmengine 的 Runner 在训练开始前会调用 model.init_weights()，mmdet 的 BaseModule 会递归
        调用每个子模块的 init_weights —— 所以只要实现这个方法，配置里的 init_cfg 就会生效。
        （测试 / 推理时不会调用它：那时权重来自 load_from 的检查点）

        步骤：
            if self.init_cfg is None or self.init_cfg.get('type') != 'Pretrained': return
            ckpt = self.init_cfg['checkpoint']
            if ckpt.startswith('torchvision://'):
                import torchvision
                url = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.url   # 只考虑 resnet50
                state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
            else:
                state_dict = torch.load(ckpt, map_location='cpu')
                state_dict = state_dict.get('state_dict', state_dict)      # mmdet 的 ckpt 外面还包了一层
            去掉 'fc.' 开头的 key，然后 self.load_state_dict(state_dict, strict=True)

        思考：为什么 strict=True？如果某个 key 拼错了，strict=False 会默默跳过，你会得到一个
              "看起来在训练、其实 backbone 是随机初始化" 的模型 —— 这类 bug 极难发现。
        """
        # TODO(7)
        raise NotImplementedError('RResNet.init_weights')

    def train(self, mode: bool = True):
        """练习 6：重写 train()。

        super().train(mode) 会把所有子模块设成训练模式，冻结的部分和 BN 也被 "解冻" 了，所以：
            super().train(mode)
            self._freeze_stages()
            if mode and self.norm_eval:
                把所有 nn.BatchNorm2d（用 self.modules() 遍历 + isinstance 判断）设为 eval()
            return self

        思考：Runner 每个 epoch 开始都会调用 model.train()，如果不重写这个方法会怎样？
        """
        # TODO(6)
        raise NotImplementedError('RResNet.train')
