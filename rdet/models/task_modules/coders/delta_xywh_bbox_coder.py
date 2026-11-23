"""练习 2.1d：DeltaXYWHBBoxCoder —— 框 ↔ 回归目标 的编码与解码

================================================================================
为什么不直接回归框的绝对坐标？
    网络直接输出 "x1=532.7" 这种大数很难学：尺度跨度大、和 anchor 大小无关。
    R-CNN 系列的做法是回归 **相对 anchor 的偏移量 (delta)**：

        dx = (gx - px) / pw      dy = (gy - py) / ph       # 中心偏移，用 anchor 宽高归一化
        dw = log(gw / pw)        dh = log(gh / ph)         # 尺度变化，取 log 让放大缩小对称

    其中 (px, py, pw, ph) 是 anchor 的中心和宽高，(gx, gy, gw, gh) 是 GT 的。
    最后再做标准化：delta = (delta - means) / stds
    （RPN 用 stds=(1,1,1,1)；RoI head 用 (0.1, 0.1, 0.2, 0.2)，让各分量的数值范围接近）

    - encode：训练时把 GT 编码成回归目标
    - decode：推理时把网络输出的 delta 还原成框

你会学到
    - 回归目标的设计如何影响训练稳定性
    - 为什么 decode 时要 clamp dw / dh（exp 很容易数值爆炸）

参考源码
    mmdet/models/task_modules/coders/delta_xywh_bbox_coder.py：bbox2delta / delta2bbox
    （真实版本还有 add_ctr_clamp、批量维度等）

完成标准
    pytest tests/test_bbox_utils.py -k coder
================================================================================
"""

import math
from typing import Optional, Sequence, Tuple

import torch

from rdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class RDeltaXYWHBBoxCoder:

    def __init__(self, target_means: Sequence[float] = (0., 0., 0., 0.),
                 target_stds: Sequence[float] = (1., 1., 1., 1.),
                 clip_border: bool = True):
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border
        self.encode_size = 4  # 每个框编码后的长度（mmdet 用这个属性决定 head 的输出通道数）

    def encode(self, bboxes: torch.Tensor, gt_bboxes: torch.Tensor) -> torch.Tensor:
        """Args:
            bboxes: [N, 4] anchor / proposal，xyxy
            gt_bboxes: [N, 4] 与 bboxes 一一对应的 GT，xyxy
        Returns:
            deltas: [N, 4]，(dx, dy, dw, dh) 经过 means / stds 标准化

        练习 1：
            px = (x1 + x2) * 0.5;  pw = x2 - x1   （y、h 同理；gt 同理）
            按文件开头的公式算 dx, dy, dw, dh，stack 成 [N, 4]
            deltas = (deltas - means) / stds     # means / stds 要先 new_tensor 再 unsqueeze(0)
        """
        assert bboxes.size(0) == gt_bboxes.size(0)
        # TODO(1)
        raise NotImplementedError('RDeltaXYWHBBoxCoder.encode')

    def decode(self, bboxes: torch.Tensor, pred_bboxes: torch.Tensor,
               max_shape: Optional[Tuple[int, int]] = None,
               wh_ratio_clip: float = 16 / 1000) -> torch.Tensor:
        """Args:
            bboxes: [N, 4] anchor / proposal
            pred_bboxes: [N, 4] 网络预测的 delta
            max_shape: (h, w)，图像尺寸；给了且 clip_border=True 时把框裁到图像内
            wh_ratio_clip: dw / dh 的最大绝对值是 |log(wh_ratio_clip)|，约 4.135，
                           即宽高最多放大 / 缩小 1000/16 = 62.5 倍
        Returns:
            [N, 4] xyxy

        练习 2：
            1. 反标准化：deltas = pred_bboxes * stds + means
            2. max_ratio = abs(math.log(wh_ratio_clip))；dw、dh clamp 到 [-max_ratio, max_ratio]
            3. gx = px + pw * dx;  gw = pw * dw.exp()    （y、h 同理）
            4. x1 = gx - gw * 0.5;  x2 = gx + gw * 0.5 ...
            5. clip：x 坐标 clamp 到 [0, w]，y 坐标 clamp 到 [0, h]
               注意 max_shape 是 (h, w)，不是 (w, h)！这是检测代码里最常见的 bug 之一

        思考：如果不 clamp dw，一个刚初始化的网络输出 dw=50 会发生什么？
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        # TODO(2)
        raise NotImplementedError('RDeltaXYWHBBoxCoder.decode')


_ = math  # 练习 2 会用到 math.log；写完后删掉这一行
