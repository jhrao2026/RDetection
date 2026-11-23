"""练习 2.1c：AnchorGenerator —— 在每个特征图位置铺 anchor

================================================================================
目标
    为 FPN 的每一层特征图生成 anchor（先验框）。Faster R-CNN 的 RPN 配置是：

        anchor_generator=dict(type='AnchorGenerator',
                              scales=[8], ratios=[0.5, 1.0, 2.0],
                              strides=[4, 8, 16, 32, 64])

    含义：第 l 层特征图的每个位置放 len(scales) * len(ratios) = 3 个 anchor，
    基础边长 base_size = stride，实际边长约 base_size * scale（stride 4 那层就是 32 像素）。

两步走
    1. gen_base_anchors：生成"以原点为中心"的那几个基础 anchor（每层一组）
    2. grid_priors：把基础 anchor 平移到特征图每个格点对应的原图位置

你会学到
    - "特征图上的位置 (i, j)" ↔ "原图上的坐标 (j*stride, i*stride)" 的对应关系
    - 为什么 FPN 低层负责小目标、高层负责大目标（看各层 anchor 的尺寸）
    - anchor 的排列顺序（必须和 head 输出的通道排列一致，否则 loss 全错）

参考源码
    mmdet/models/task_modules/prior_generators/anchor_generator.py
    （真实版本还有 octave_base_scale、center_offset、valid_flags、单层生成等，这里只保留核心）

完成标准
    pytest tests/test_bbox_utils.py -k anchor
    （装了 mmdet 时还会和 mmdet 的 AnchorGenerator 逐元素对比）
================================================================================
"""

from typing import List, Sequence, Tuple

import math

import torch

from rdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class AnchorGenerator:
    """Args:
        strides: 每层特征图相对原图的下采样倍数，例如 [4, 8, 16, 32, 64]
        ratios: 高宽比 h / w 的列表
        scales: 相对 base_size 的缩放倍数列表
        base_sizes: 每层基础边长，默认等于 strides
        center_offset: anchor 中心相对格点左上角的偏移（0 表示中心就在格点上，mmdet 默认 0）
    """

    def __init__(self, strides: Sequence[int], ratios: Sequence[float], scales: Sequence[float],
                 base_sizes: Sequence[int] = None, center_offset: float = 0.):
        # 已完成：保存参数
        self.strides = [(s, s) if isinstance(s, int) else tuple(s) for s in strides]
        self.ratios = torch.tensor(ratios, dtype=torch.float32)
        self.scales = torch.tensor(scales, dtype=torch.float32)
        self.base_sizes = list(base_sizes) if base_sizes is not None else [s[0] for s in self.strides]
        assert len(self.base_sizes) == len(self.strides)
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_levels(self) -> int:
        return len(self.strides)

    @property
    def num_base_anchors(self) -> List[int]:
        """每层每个位置的 anchor 数，Faster R-CNN 是 [3, 3, 3, 3, 3]。"""
        return [len(self.ratios) * len(self.scales)] * self.num_levels

    def gen_single_level_base_anchors(self, base_size: int) -> torch.Tensor:
        """生成一层的基础 anchor，返回 [num_base_anchors, 4]（xyxy）。

        练习 1。公式（与 mmdet 一致，务必保持顺序）：
            w = h = base_size
            x_center = center_offset * w;  y_center = center_offset * h
            h_ratios = sqrt(ratios);  w_ratios = 1 / h_ratios     # 保证 面积不随 ratio 变化
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
            base_anchors = stack([x_center - 0.5 * ws, y_center - 0.5 * hs,
                                  x_center + 0.5 * ws, y_center + 0.5 * hs], dim=-1)

        注意 ``[:, None] * [None, :]`` 再 view(-1) 决定了排列顺序：ratio 在外层、scale 在内层。

        小例子：base_size=16, ratios=[1.], scales=[8.] → [[-64, -64, 64, 64]]
        思考：ratio=0.5 时 anchor 是"扁"的还是"高"的？算出来验证一下。
        """
        # TODO(1)
        raise NotImplementedError('AnchorGenerator.gen_single_level_base_anchors')

    def gen_base_anchors(self) -> List[torch.Tensor]:
        # 已完成：对每一层调用上面的函数
        return [self.gen_single_level_base_anchors(b) for b in self.base_sizes]

    def single_level_grid_priors(self, featmap_size: Tuple[int, int], level_idx: int,
                                 device='cpu') -> torch.Tensor:
        """把第 level_idx 层的基础 anchor 铺满整张特征图，返回 [H * W * A, 4]。

        练习 2。步骤：
            feat_h, feat_w = featmap_size
            stride_w, stride_h = self.strides[level_idx]
            shift_x = torch.arange(0, feat_w, device=device) * stride_w      # 每一列在原图上的 x
            shift_y = torch.arange(0, feat_h, device=device) * stride_h
            shift_yy, shift_xx = torch.meshgrid(shift_y, shift_x, indexing='ij')
            shifts = stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1).view(-1, 4)   # [H*W, 4]
            all_anchors = base_anchors[None, :, :] + shifts[:, None, :]                  # [H*W, A, 4]
            return all_anchors.view(-1, 4)

        排列顺序 = 位置在外层（先行后列），同一位置的 A 个 anchor 在内层。
        思考：RPN head 对每个位置输出 A 个分数，形状 [A, H, W]；要把它展平成和 anchor
             一一对应的 [H*W*A]，应该怎样 permute？（答案在 2.3 RPN 练习里会用到）
        """
        # TODO(2) 注意 base_anchors 要 .to(device)
        raise NotImplementedError('AnchorGenerator.single_level_grid_priors')

    def grid_priors(self, featmap_sizes: List[Tuple[int, int]], device='cpu') -> List[torch.Tensor]:
        """所有层的 anchor，返回 list，第 l 项形状 [H_l * W_l * A, 4]。"""
        assert len(featmap_sizes) == self.num_levels
        # TODO(3) 对每层调用 single_level_grid_priors
        raise NotImplementedError('AnchorGenerator.grid_priors')

    def valid_flags(self, featmap_sizes: List[Tuple[int, int]], pad_shape: Tuple[int, int],
                    device='cpu') -> List[torch.Tensor]:
        """（已完成，2.3 RPN 会用到）标记哪些 anchor 落在 "真实图像" 范围内。

        一个 batch 里的图片大小不同，DetDataPreprocessor 会把它们 pad 到同样大小。
        pad 出来的区域全是 0，在那里放的 anchor 没有意义，训练时要排除。
        pad_shape 是这张图自己（pad 到 32 的倍数后）的 (h, w)，不是整个 batch 的大小。

        返回 list，第 l 项是 [H_l * W_l * A] 的 bool，排列顺序与 grid_priors 一致。
        """
        flags = []
        for i, (feat_h, feat_w) in enumerate(featmap_sizes):
            stride_w, stride_h = self.strides[i]
            valid_h = min(int(math.ceil(pad_shape[0] / stride_h)), feat_h)
            valid_w = min(int(math.ceil(pad_shape[1] / stride_w)), feat_w)
            valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
            valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
            valid_y[:valid_h] = True
            valid_x[:valid_w] = True
            valid = (valid_y[:, None] & valid_x[None, :]).view(-1)
            flags.append(valid[:, None].expand(-1, self.num_base_anchors[i]).reshape(-1))
        return flags
