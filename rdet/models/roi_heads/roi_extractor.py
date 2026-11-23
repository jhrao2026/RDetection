"""练习 2.4a：RoI Extractor —— 从 FPN 上把每个 proposal 的特征 "抠" 成固定大小

================================================================================
为什么需要它？
    RPN 给出的 proposal 大小不一，但后面的全连接层要求输入固定大小（256 × 7 × 7）。
    RoIAlign 把任意大小的框区域双线性采样成 7 × 7。

    FPN 有 4 层（P2–P5），一个 proposal 该从哪一层抠？FPN 论文的规则：

        level = floor(log2(sqrt(w * h) / 56 + 1e-6))，再 clamp 到 [0, 3]

    即边长 < 112 用 P2，112–224 用 P3，224–448 用 P4，≥ 448 用 P5（finest_scale = 56）。
    大框用低分辨率、语义强的层；小框用高分辨率层。

rois 的格式（mmdet 约定）
    [K, 5]：每行 (batch_idx, x1, y1, x2, y2)，batch_idx 表示这个框属于 batch 里的第几张图。

参考源码
    mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py：map_roi_levels / forward
    mmcv.ops.RoIAlign（这里用 torchvision.ops.roi_align，aligned=True 时两者结果一致）

完成标准
    pytest tests/test_roi_head.py -k extractor
================================================================================
"""

from typing import Sequence

import torch
from torch import nn
from torchvision.ops import roi_align  # noqa: F401

from rdet.registry import MODELS


@MODELS.register_module()
class RSingleRoIExtractor(nn.Module):
    """Args:
        roi_layer: dict(type='RoIAlign', output_size=7, sampling_ratio=0, aligned=True)
            只读取 output_size / sampling_ratio / aligned 三个字段
        out_channels: 特征通道数（256）
        featmap_strides: 用到的 FPN 层的 stride，Faster R-CNN 是 [4, 8, 16, 32]（不用 P6）
        finest_scale: 见文件开头的公式
    """

    def __init__(self, roi_layer: dict, out_channels: int, featmap_strides: Sequence[int],
                 finest_scale: int = 56, init_cfg=None):
        super().__init__()
        # 已完成
        size = roi_layer['output_size']
        self.output_size = (size, size) if isinstance(size, int) else tuple(size)
        self.sampling_ratio = roi_layer.get('sampling_ratio', 0)
        self.aligned = roi_layer.get('aligned', True)
        self.out_channels = out_channels
        self.featmap_strides = list(featmap_strides)
        self.finest_scale = finest_scale

    @property
    def num_inputs(self) -> int:
        """用几层 FPN 特征（RoI head 会把 feats[:num_inputs] 传进来）。"""
        return len(self.featmap_strides)

    def map_roi_levels(self, rois: torch.Tensor, num_levels: int) -> torch.Tensor:
        """练习 1：按文件开头的公式，返回每个 roi 的层号 [K] long。

        注意 rois 第 0 列是 batch_idx，宽 = rois[:, 3] - rois[:, 1]，高 = rois[:, 4] - rois[:, 2]。
        思考：+1e-6 是为了什么？去掉它，边长恰好 112 的框会被分到哪一层？
        """
        # TODO(1)
        raise NotImplementedError('RSingleRoIExtractor.map_roi_levels')

    def forward(self, feats: Sequence[torch.Tensor], rois: torch.Tensor) -> torch.Tensor:
        """练习 2：返回 [K, out_channels, 7, 7]。

            roi_feats = feats[0].new_zeros(K, self.out_channels, *self.output_size)
            target_lvls = self.map_roi_levels(rois, len(feats))
            for i in range(len(feats)):
                inds = (target_lvls == i).nonzero().squeeze(1)
                if inds.numel() > 0:
                    roi_feats[inds] = roi_align(feats[i], rois[inds], self.output_size,
                                                spatial_scale=1. / self.featmap_strides[i],
                                                sampling_ratio=self.sampling_ratio, aligned=self.aligned)
            return roi_feats

        思考：
        - spatial_scale=1/stride 做了什么？（rois 是原图坐标，特征图是缩小了 stride 倍的）
        - aligned=True 修正了 RoIAlign 原始实现里的半像素偏移，具体是怎么偏的？
          （提示：像素 (0, 0) 的中心在坐标 (0.5, 0.5)）
        - mmdet 在某一层没有 roi 时，还会加一项 ``feats[i].sum() * 0``。这是为了解决多卡训练
          （DDP）里的什么问题？（提示：某些参数在这次迭代里没有梯度）
        """
        rois = rois.type_as(feats[0])
        # TODO(2)
        raise NotImplementedError('RSingleRoIExtractor.forward')
