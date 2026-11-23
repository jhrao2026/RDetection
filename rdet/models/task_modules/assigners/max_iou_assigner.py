"""练习 2.1e：MaxIoUAssigner —— 决定每个 anchor 是正样本、负样本还是忽略

================================================================================
目标
    一张图有上万个 anchor、几个 GT。训练前必须回答：每个 anchor 该学什么？
        - 正样本：和某个 GT 的 IoU 足够大 → 学这个 GT 的类别和框
        - 负样本：和所有 GT 的 IoU 都很小 → 学 "背景"
        - 忽略：  介于两者之间 → 不参与 loss（模棱两可的样本会干扰训练）

    RPN 的配置：pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3
    RoI head：  pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5

输出的编码方式（与 mmdet 一致）
    gt_inds[i] = -1   忽略
    gt_inds[i] =  0   负样本
    gt_inds[i] =  k   正样本，匹配第 k-1 个 GT（从 1 开始，0 已经被负样本占用）

你会学到
    - 为什么需要 "low quality match"：一个很小或形状奇怪的 GT 可能没有任何 anchor 的 IoU ≥ 0.7，
      如果不做补救，这个 GT 就永远学不到
    - 正负样本定义是检测器设计的核心：后面的 ATSS、TaskAligned（YOLOv8）、
      匈牙利匹配（DETR）都是在回答同一个问题

参考源码
    mmdet/models/task_modules/assigners/max_iou_assigner.py：assign / assign_wrt_overlaps
    mmdet/models/task_modules/assigners/assign_result.py

完成标准
    pytest tests/test_bbox_utils.py -k assigner
================================================================================
"""

from dataclasses import dataclass

import torch

from rdet.registry import TASK_UTILS
from rdet.structures import bbox_overlaps  # noqa: F401  练习里会用到


@dataclass
class AssignResult:
    """分配结果（已完成）。

    Attributes:
        num_gts: GT 数量
        gt_inds: [N] long，编码见文件开头
        max_overlaps: [N] float，每个 anchor 与所有 GT 的最大 IoU
        labels: [N] long，正样本为对应 GT 的类别，其余为 -1
    """
    num_gts: int
    gt_inds: torch.Tensor
    max_overlaps: torch.Tensor
    labels: torch.Tensor

    def add_gt_(self, gt_labels: torch.Tensor) -> None:
        """把 GT 自己当成 proposal 加到最前面（RoI head 训练时用，见 RandomSampler）。

        GT 和自己的 IoU 是 1，当然是正样本，匹配的就是自己。
        """
        self_inds = torch.arange(1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_overlaps = torch.cat([self.max_overlaps.new_ones(len(gt_labels)), self.max_overlaps])
        self.labels = torch.cat([gt_labels, self.labels])


@TASK_UTILS.register_module()
class MaxIoUAssigner:
    """Args:
        pos_iou_thr: max IoU ≥ 该值 → 正样本
        neg_iou_thr: float 时 0 ≤ max IoU < 该值 → 负样本；
                     tuple (lo, hi) 时 lo ≤ max IoU < hi → 负样本
        min_pos_iou: low quality match 时，GT 的最佳 IoU 至少要达到这个值
        gt_max_assign_all: 如果多个 anchor 并列 GT 的最佳 IoU，是否全部设为正样本
        match_low_quality: 是否开启 low quality match（RPN 开，Cascade 的后几级关）
    """

    def __init__(self, pos_iou_thr: float, neg_iou_thr, min_pos_iou: float = .0,
                 gt_max_assign_all: bool = True, match_low_quality: bool = True):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.match_low_quality = match_low_quality

    def assign(self, priors: torch.Tensor, gt_bboxes: torch.Tensor,
               gt_labels: torch.Tensor) -> AssignResult:
        """Args:
            priors: [N, 4] anchor 或 proposal
            gt_bboxes: [K, 4]
            gt_labels: [K]

        练习。按顺序实现（顺序很重要，后面的步骤会覆盖前面的结果）：

        0. overlaps = bbox_overlaps(gt_bboxes, priors)      # 注意是 [K, N]，GT 在前
           assigned_gt_inds = overlaps.new_full((N,), -1, dtype=torch.long)   # 先全部设为忽略

        1. 特殊情况：K == 0 或 N == 0
           - max_overlaps 全 0；K == 0 时没有 GT，所有 anchor 都是负样本（gt_inds 全 0）
           - labels 全 -1，直接返回 AssignResult

        2. max_overlaps, argmax_overlaps = overlaps.max(dim=0)     # 每个 anchor 最匹配的 GT
           gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)  # 每个 GT 最匹配的 anchor

        3. 负样本：max_overlaps 落在 [0, neg_iou_thr) 内（或 tuple 的 [lo, hi)）→ 设为 0

        4. 正样本：max_overlaps >= pos_iou_thr → 设为 argmax_overlaps + 1

        5. low quality match（match_low_quality=True 时）：
           for i in range(K):
               if gt_max_overlaps[i] >= min_pos_iou:
                   if gt_max_assign_all:
                       把所有 overlaps[i, :] == gt_max_overlaps[i] 的 anchor 设为 i + 1
                   else:
                       只把 gt_argmax_overlaps[i] 设为 i + 1
           思考：第 5 步可能把一个 "本来已匹配给别的 GT" 的 anchor 改匹配给 GT i，这合理吗？
                 也可能把一个 IoU 只有 0.35 的 anchor 设为正样本 —— 这就是 "low quality" 的含义。

        6. labels：全 -1，正样本处填 gt_labels[assigned_gt_inds[pos] - 1]

        7. return AssignResult(K, assigned_gt_inds, max_overlaps, labels)
        """
        # TODO
        raise NotImplementedError('MaxIoUAssigner.assign')
