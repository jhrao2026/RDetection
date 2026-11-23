"""练习 2.1f：RandomSampler —— 从正负样本里随机抽一个固定大小的子集算 loss

================================================================================
为什么要采样？
    assigner 之后，一张图可能有 20 个正样本、15 万个负样本。直接全算 loss，
    分类器会被海量 "简单背景" 淹没，学成 "全部预测背景"。
    Faster R-CNN 的解决方法很朴素：每张图固定抽 num 个样本，正样本最多占 pos_fraction。

        RPN:      num=256, pos_fraction=0.5,  add_gt_as_proposals=False
        RoI head: num=512, pos_fraction=0.25, add_gt_as_proposals=True

    后来 RetinaNet 用 Focal Loss 换了一种思路（不采样，直接降低简单样本的权重），
    你会在 2.6 里实现它，对比两种方法。

参考源码
    mmdet/models/task_modules/samplers/base_sampler.py（sample 主流程）
    mmdet/models/task_modules/samplers/random_sampler.py（_sample_pos / _sample_neg）
    mmdet/models/task_modules/samplers/sampling_result.py

完成标准
    pytest tests/test_bbox_utils.py -k sampler
================================================================================
"""

from dataclasses import dataclass

import torch

from rdet.registry import TASK_UTILS

from ..assigners import AssignResult


@dataclass
class SamplingResult:
    """采样结果（已完成）。下游的 head 只需要这些信息就能构造 loss 的 target。"""
    pos_inds: torch.Tensor             # [P] 正样本在 priors 中的下标
    neg_inds: torch.Tensor             # [Q] 负样本在 priors 中的下标
    pos_priors: torch.Tensor           # [P, 4]
    neg_priors: torch.Tensor           # [Q, 4]
    pos_gt_bboxes: torch.Tensor        # [P, 4] 每个正样本对应的 GT 框
    pos_gt_labels: torch.Tensor        # [P]
    pos_assigned_gt_inds: torch.Tensor  # [P] 对应 GT 的下标（从 0 开始）
    pos_is_gt: torch.Tensor            # [P] bool，这个正样本是不是 "GT 自己当 proposal"

    @property
    def priors(self) -> torch.Tensor:
        """正样本在前、负样本在后拼在一起，RoI head 会对它们做 RoIAlign。"""
        return torch.cat([self.pos_priors, self.neg_priors])


@TASK_UTILS.register_module()
class RandomSampler:

    def __init__(self, num: int, pos_fraction: float, neg_pos_ub: int = -1,
                 add_gt_as_proposals: bool = True):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub  # 负样本数量上限 = neg_pos_ub * 正样本数（-1 表示不限）
        self.add_gt_as_proposals = add_gt_as_proposals

    @staticmethod
    def random_choice(inds: torch.Tensor, num: int) -> torch.Tensor:
        """从 inds 中随机不放回地选 num 个（已完成）。mmdet 用的也是 randperm。"""
        assert len(inds) >= num
        perm = torch.randperm(inds.numel(), device=inds.device)[:num]
        return inds[perm]

    def sample(self, assign_result: AssignResult, priors: torch.Tensor,
               gt_bboxes: torch.Tensor, gt_labels: torch.Tensor) -> SamplingResult:
        """练习。步骤：

        1. add_gt_as_proposals 且有 GT 时：
               priors = torch.cat([gt_bboxes, priors])
               assign_result.add_gt_(gt_labels)
               gt_flags = 前 K 个为 True，其余 False（bool tensor，长度 = 新的 priors 数）
           否则 gt_flags 全 False
           思考：RoI head 为什么要把 GT 加进 proposals？（提示：训练初期 RPN 很烂，几乎没有好的正样本）

        2. 正样本：
               num_expected_pos = int(self.num * self.pos_fraction)
               pos_inds = torch.nonzero(assign_result.gt_inds > 0).flatten()
               多于 num_expected_pos 就 random_choice
           （mmdet 还会先 unique 一下，这里 nonzero 结果本身不重复，可以省略）

        3. 负样本：
               num_expected_neg = self.num - len(pos_inds)      # 正样本不够时用负样本补满
               if self.neg_pos_ub >= 0:
                   num_expected_neg = min(num_expected_neg, max(1, len(pos_inds)) * self.neg_pos_ub)
               neg_inds = torch.nonzero(assign_result.gt_inds == 0).flatten()
               多于 num_expected_neg 就 random_choice

        4. 构造 SamplingResult：
               pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
               pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds]（没有 GT 时给一个 [0, 4] 的空 tensor）
               pos_gt_labels = assign_result.labels[pos_inds]
               pos_is_gt = gt_flags[pos_inds]
        """
        # TODO
        raise NotImplementedError('RandomSampler.sample')
