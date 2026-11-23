"""练习 2.4c：StandardRoIHead —— 把 assigner / sampler / extractor / bbox head 串起来

================================================================================
训练：
    proposals（RPN 输出）──► MaxIoUAssigner（IoU ≥ 0.5 为正）──► RandomSampler（512 个，正样本 ≤ 25%，并加入 GT）
        ──► bbox2roi ──► RoI Extractor ──► BBox Head ──► get_targets + loss

推理：
    proposals ──► bbox2roi ──► RoI Extractor ──► BBox Head ──► 按图拆开 ──► predict_single

    和 RPN 的 loss 对比一下：两阶段用的是同一套 "分配 → 采样 → 编码" 思路，
    区别只是候选框从 "anchor" 换成了 "proposal"。

参考源码
    mmdet/models/roi_heads/standard_roi_head.py：loss / bbox_loss / _bbox_forward
    mmdet/models/roi_heads/test_mixins.py：predict_bbox
    mmdet/structures/bbox/transforms.py：bbox2roi

完成标准
    pytest tests/test_roi_head.py
================================================================================
"""

from typing import List, Sequence, Tuple

import torch
from torch import nn

from rdet.registry import MODELS, TASK_UTILS


def bbox2roi(bbox_list: List[torch.Tensor]) -> torch.Tensor:
    """练习 1：把每张图的框 [n_i, 4] 拼成 rois [sum(n_i), 5]，第 0 列是图片下标。

    提示：对第 i 张图，torch.cat([boxes.new_full((len(boxes), 1), i), boxes[:, :4]], dim=1)
    思考：为什么不直接用 [B, n, 4] 的 batch 形式？（每张图的框数不一样）
    """
    # TODO(1)
    raise NotImplementedError('bbox2roi')


@MODELS.register_module()
class RStandardRoIHead(nn.Module):
    """★ 子模块名 bbox_roi_extractor / bbox_head 与 mmdet 一致（权重 key 是 roi_head.bbox_head.xxx）。"""

    def __init__(self, bbox_roi_extractor: dict, bbox_head: dict, train_cfg: dict = None,
                 test_cfg: dict = None, init_cfg=None, **kwargs):
        super().__init__()
        # 已完成
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if train_cfg is not None:
            self.bbox_assigner = TASK_UTILS.build(train_cfg['assigner'])
            self.bbox_sampler = TASK_UTILS.build(train_cfg['sampler'])

    def init_weights(self) -> None:
        # 已完成：mmdet 的 BaseModule 只会调用 "直接子模块" 的 init_weights，所以这里要往下传
        self.bbox_head.init_weights()

    def _bbox_forward(self, feats: Sequence[torch.Tensor], rois: torch.Tensor):
        """练习 2：feats[:num_inputs] 交给 extractor 得到 RoI 特征，再交给 bbox_head。

        返回 (cls_score, bbox_pred)。
        思考：为什么只用 feats[:4]？P6 去哪了？（它只给 RPN 的大 anchor 用）
        """
        # TODO(2)
        raise NotImplementedError('RStandardRoIHead._bbox_forward')

    def loss(self, feats: Sequence[torch.Tensor], batch_proposals: List[torch.Tensor],
             batch_gt_bboxes: List[torch.Tensor], batch_gt_labels: List[torch.Tensor]) -> dict:
        """练习 3。

        1. 对每张图 i：
               assign_result = self.bbox_assigner.assign(batch_proposals[i], batch_gt_bboxes[i], batch_gt_labels[i])
               sampling_result = self.bbox_sampler.sample(assign_result, batch_proposals[i],
                                                          batch_gt_bboxes[i], batch_gt_labels[i])
        2. rois = bbox2roi([res.priors for res in sampling_results])    # 每张图 [正..., 负...]
        3. cls_score, bbox_pred = self._bbox_forward(feats, rois)
        4. targets = self.bbox_head.get_targets(sampling_results)
           return self.bbox_head.loss(cls_score, bbox_pred, *targets)
        """
        # TODO(3)
        raise NotImplementedError('RStandardRoIHead.loss')

    def predict(self, feats: Sequence[torch.Tensor], batch_proposals: List[torch.Tensor],
                batch_img_shapes: List[Tuple[int, int]], batch_scale_factors: List[Tuple[float, float]],
                rescale: bool = True, cfg: dict = None):
        """练习 4：返回 list（每张图一项）：(bboxes, scores, labels)

        1. rois = bbox2roi(batch_proposals)；如果一个 roi 都没有，每张图返回空结果
        2. cls_score, bbox_pred = self._bbox_forward(feats, rois)
        3. 按每张图的 proposal 数拆开：num_per_img = [len(p) for p in batch_proposals]
               rois.split(num_per_img)、cls_score.split(num_per_img)、bbox_pred.split(num_per_img)
        4. 每张图调用 self.bbox_head.predict_single(...)
        """
        cfg = self.test_cfg if cfg is None else cfg
        # TODO(4)
        raise NotImplementedError('RStandardRoIHead.predict')
