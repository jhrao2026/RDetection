"""练习 2.4b：BBox Head（Shared2FCBBoxHead）—— 第二阶段的分类与回归

================================================================================
结构
    RoI 特征 [K, 256, 7, 7] ─flatten─► fc(12544→1024) ─ReLU─► fc(1024→1024) ─ReLU─┬─► fc_cls → [K, 81]
                                                                                   └─► fc_reg → [K, 80*4]

    - 分类是 81 类 softmax：80 个前景类 + 1 个背景类（背景放在 **最后**，下标 80）
    - 回归是 "每类一套" 的 delta（reg_class_agnostic=False）：训练时只用 GT 类别那 4 个数

    ★ 命名与 mmdet 一致：shared_fcs.0 / shared_fcs.1 / fc_cls / fc_reg

与 RPN 的对比（写之前先想清楚）
    |              | RPN                       | BBox Head                           |
    |--------------|---------------------------|-------------------------------------|
    | 输入         | anchor（几十万）           | proposal（每图采样 512 个）          |
    | 分类         | sigmoid 二分类            | softmax 81 类                       |
    | 回归 std     | (1, 1, 1, 1)              | (0.1, 0.1, 0.2, 0.2)                |
    | 回归 loss 归一 | 采样总数                 | 采样总数（不是正样本数！见练习 4）     |

参考源码
    mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py：Shared2FCBBoxHead / forward
    mmdet/models/roi_heads/bbox_heads/bbox_head.py：_get_targets_single / loss / _predict_by_feat_single
    mmdet/models/layers/bbox_nms.py：multiclass_nms
    mmdet/models/losses/accuracy.py

完成标准
    pytest tests/test_roi_head.py -k bbox_head
================================================================================
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F  # noqa: F401
from torch import nn
from torchvision.ops import batched_nms  # noqa: F401

from rdet.registry import MODELS, TASK_UTILS


@MODELS.register_module()
class RShared2FCBBoxHead(nn.Module):

    def __init__(self, in_channels: int = 256, fc_out_channels: int = 1024, roi_feat_size: int = 7,
                 num_classes: int = 80, bbox_coder: dict = None, reg_class_agnostic: bool = False,
                 init_cfg=None, **kwargs):
        super().__init__()
        assert not reg_class_agnostic, '本练习只实现 class-specific 回归'
        self.num_classes = num_classes
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        # TODO(1) shared_fcs = nn.ModuleList([Linear(in_channels * roi_feat_size ** 2, fc_out_channels),
        #                                     Linear(fc_out_channels, fc_out_channels)])
        #         fc_cls: fc_out_channels → num_classes + 1
        #         fc_reg: fc_out_channels → num_classes * 4
        #         relu
        raise NotImplementedError('RShared2FCBBoxHead.__init__')

    def init_weights(self) -> None:
        """练习 2（与 mmdet 一致）：
            shared_fcs：xavier_uniform_，bias 0
            fc_cls：normal(std=0.01)，bias 0
            fc_reg：normal(std=0.001)，bias 0
        思考：fc_reg 为什么用这么小的 std？（提示：训练刚开始时，希望 decode 出来的框 ≈ proposal 本身）
        """
        # TODO(2)
        raise NotImplementedError('RShared2FCBBoxHead.init_weights')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """练习 3：x [K, 256, 7, 7] → (cls_score [K, C+1], bbox_pred [K, C*4])"""
        # TODO(3) flatten(1) → 两个 fc（每个后面接 ReLU）→ fc_cls / fc_reg
        raise NotImplementedError('RShared2FCBBoxHead.forward')

    # ------------------------------------------------------------------ loss
    def get_targets(self, sampling_results: List) -> Tuple[torch.Tensor, ...]:
        """练习 4a：把每张图的 SamplingResult 变成训练目标，再把所有图 cat 起来。

        每张图里样本的顺序是 [正样本..., 负样本...]（和 SamplingResult.priors 一致）：
            labels:        [P+Q] long，正样本 = pos_gt_labels，负样本 = self.num_classes（背景）
            label_weights: [P+Q] 全 1
            bbox_targets:  [P+Q, 4]，前 P 个 = coder.encode(pos_priors, pos_gt_bboxes)，其余 0
            bbox_weights:  [P+Q, 4]，前 P 个为 1
        返回 (labels, label_weights, bbox_targets, bbox_weights)，都是 cat 后的结果
        """
        # TODO(4a)
        raise NotImplementedError('RShared2FCBBoxHead.get_targets')

    def loss(self, cls_score: torch.Tensor, bbox_pred: torch.Tensor, labels: torch.Tensor,
             label_weights: torch.Tensor, bbox_targets: torch.Tensor,
             bbox_weights: torch.Tensor) -> dict:
        """练习 4b：返回 dict(loss_cls=..., acc=..., loss_bbox=...)

        - loss_cls：交叉熵，按 label_weights 加权求和，再除以 max(label_weights > 0 的数量, 1)
              (F.cross_entropy(cls_score, labels, reduction='none') * label_weights).sum() / avg_factor
        - acc：top-1 准确率（百分比，0–100），只用于日志：
              (cls_score.argmax(1) == labels).float().mean() * 100
        - loss_bbox：只看正样本（labels < num_classes），并且只取 **GT 类别** 对应的那 4 个预测：
              pos = (labels >= 0) & (labels < self.num_classes)
              pos_pred = bbox_pred.view(K, self.num_classes, 4)[pos, labels[pos]]       # [P, 4]
              loss_bbox = ((pos_pred - bbox_targets[pos]).abs() * bbox_weights[pos]).sum() / bbox_targets.size(0)
          没有正样本时：loss_bbox = bbox_pred.sum() * 0（保持计算图，DDP 需要）

        思考：
        - ★ loss_bbox 除以的是 **全部** 采样数（512 × 图片数），不是正样本数。
          如果改成除以正样本数，回归 loss 会放大约几倍？这相当于改了哪个超参？
        - 为什么 acc 不参与反传也要放进 loss 字典？（回顾 mini_mm 的 parse_losses）
        """
        # TODO(4b)
        raise NotImplementedError('RShared2FCBBoxHead.loss')

    # ------------------------------------------------------------------ predict
    def predict_single(self, rois: torch.Tensor, cls_score: torch.Tensor, bbox_pred: torch.Tensor,
                       img_shape: Tuple[int, int], scale_factor: Tuple[float, float],
                       rescale: bool, cfg: dict):
        """练习 5：一张图的后处理，返回 (bboxes [n, 4], scores [n], labels [n])

        cfg 例如 dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5), max_per_img=100)

        1. scores = F.softmax(cls_score, dim=-1)                    # [K, C+1]
        2. 每个 roi 对每个类别都解码一个框：
               boxes = self.bbox_coder.decode(rois[:, 1:].repeat_interleave(C, dim=0),
                                              bbox_pred.view(-1, 4), max_shape=img_shape)   # [K*C, 4]
        3. rescale=True 时把框映射回 **原图** 尺寸：boxes / [w_scale, h_scale, w_scale, h_scale]
           （scale_factor 是 (w_scale, h_scale)：Resize 把原图放大了多少倍）
        4. multiclass NMS：
               scores = scores[:, :-1]（去掉背景列）；labels = 每列的类别 id，形状同 scores
               全部 flatten 后，只保留 score > score_thr 的
               keep = batched_nms(boxes, scores, labels, iou_threshold)[:max_per_img]
        5. 没有任何框时返回形状正确的空 tensor

        思考：
        - 同一个 roi 可能同时输出 "猫 0.6" 和 "狗 0.3" 两个检测结果，这合理吗？对 mAP 有什么影响？
        - 为什么不同类别之间不做 NMS？
        """
        # TODO(5)
        raise NotImplementedError('RShared2FCBBoxHead.predict_single')
