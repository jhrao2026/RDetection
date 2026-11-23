"""练习 2.6b：RetinaHead —— one-stage 检测头（衔接 Stage 3 的 YOLO）

================================================================================
和 RPN 比较（先把你写好的 rpn_head.py 打开放在旁边）：

    |              | RPN（2.3）                       | RetinaHead                                  |
    |--------------|----------------------------------|---------------------------------------------|
    | 输出         | 前景 / 背景（1 个 logit）         | 80 类（80 个 logit，sigmoid，没有背景通道）   |
    | 卷积塔       | 1 个 3x3                         | 分类、回归各 4 个 3x3（不共享）               |
    | anchor       | 每位置 3 个，scales=[8]           | 每位置 9 个：3 种 ratio × 3 种 scale          |
    | FPN 层       | P2–P6                            | P3–P7（FPN start_level=1，extra conv）       |
    | 正负样本     | MaxIoU(0.7, 0.3) + 采样 256 个    | MaxIoU(0.5, 0.4)，**不采样**，全部参与        |
    | 分类 loss    | BCE                              | Focal Loss                                  |
    | loss 归一化  | 正样本数 + 负样本数               | **只用正样本数**                              |
    | 输出         | proposals 给第二阶段              | 直接就是最终检测结果                          |

    loss / predict 的骨架和 RPN 几乎一样：建议先复制你的 RPN 代码，再按上表逐项修改。

★ 命名与 mmdet 一致：cls_convs.{i}.conv / reg_convs.{i}.conv / retina_cls / retina_reg

参考源码
    mmdet/models/dense_heads/retina_head.py：_init_layers / forward_single
    mmdet/models/dense_heads/anchor_head.py：_get_targets_single / loss_by_feat
    mmdet/models/dense_heads/base_dense_head.py：_predict_by_feat_single
    mmdet/models/utils/misc.py：filter_scores_and_topk

完成标准
    pytest tests/test_retinanet.py
================================================================================
"""

import math  # noqa: F401
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F  # noqa: F401
from torch import nn
from torchvision.ops import batched_nms  # noqa: F401

from rdet.registry import MODELS, TASK_UTILS

from ..losses import sigmoid_focal_loss  # noqa: F401
from ..necks.fpn import ConvModule  # noqa: F401  复用 FPN 里的极简 ConvModule（.conv 命名）


@MODELS.register_module()
class RRetinaHead(nn.Module):

    def __init__(self, num_classes: int, in_channels: int, stacked_convs: int = 4,
                 feat_channels: int = 256, anchor_generator: dict = None, bbox_coder: dict = None,
                 focal_gamma: float = 2.0, focal_alpha: float = 0.25,
                 train_cfg: dict = None, test_cfg: dict = None, init_cfg=None, **kwargs):
        super().__init__()
        # 已完成
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.prior_generator = TASK_UTILS.build(anchor_generator)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.num_base_priors = self.prior_generator.num_base_anchors[0]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if train_cfg is not None:
            self.assigner = TASK_UTILS.build(train_cfg['assigner'])

        # TODO(1) 卷积塔和输出层：
        #   cls_convs / reg_convs：各 stacked_convs 个 ConvModule(c_in, feat_channels, 3, padding=1, act=True)
        #                          第一个的 c_in = in_channels，其余 = feat_channels
        #   retina_cls: Conv2d(feat_channels, A * num_classes, 3, padding=1)
        #   retina_reg: Conv2d(feat_channels, A * 4, 3, padding=1)
        raise NotImplementedError('RRetinaHead.__init__')

    def init_weights(self) -> None:
        """练习 2：所有 Conv2d 用 normal(std=0.01)、bias 0；然后把 retina_cls 的 bias 设为

            -math.log((1 - 0.01) / 0.01)          # 约 -4.6

        思考：这样初始化后，sigmoid(bias) = 0.01，即一开始网络认为 "每个 anchor 有 1% 的概率是前景"。
              如果 bias 是 0（初始概率 0.5），第一次迭代 10 万个背景 anchor 的 loss 会有多大？
        """
        # TODO(2)
        raise NotImplementedError('RRetinaHead.init_weights')

    def forward(self, feats: Sequence[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """练习 3：每层：cls_feat 依次过 cls_convs → retina_cls；reg_feat 依次过 reg_convs → retina_reg。

        返回 (cls_scores [B, A*C, H, W] 列表, bbox_preds [B, A*4, H, W] 列表)
        """
        # TODO(3)
        raise NotImplementedError('RRetinaHead.forward')

    def loss(self, cls_scores: List[torch.Tensor], bbox_preds: List[torch.Tensor],
             batch_gt_bboxes: List[torch.Tensor], batch_gt_labels: List[torch.Tensor],
             batch_pad_shapes: List[Tuple[int, int]]) -> dict:
        """练习 4：返回 dict(loss_cls=..., loss_bbox=...)

        与 RPN.loss 的区别：
        1. 展平分类输出时，每个 anchor 有 C 个 logit：
               c.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)    → [B, N, C]
        2. 每张图的目标（只对 valid anchor 做 assign，同 RPN）：
               assign_result = self.assigner.assign(valid_anchors, gt_bboxes, gt_labels)   # 用真实类别
               正样本 = gt_inds > 0；负样本 = gt_inds == 0；**不采样**
               target  [N, C]：正样本在 assign_result.labels 那一列为 1，其余全 0
               weight  [N]：正样本和负样本为 1，忽略的（gt_inds == -1）和 invalid 的为 0
               bbox_targets / bbox_weights：同 RPN
               num_pos = max(正样本数, 1)
        3. avg_factor = 所有图 num_pos 之和
           loss_cls  = sigmoid_focal_loss(flat_cls, target, weight, gamma, alpha, avg_factor)
           loss_bbox = L1，除以同一个 avg_factor
        """
        # TODO(4)
        raise NotImplementedError('RRetinaHead.loss')

    def predict(self, cls_scores: List[torch.Tensor], bbox_preds: List[torch.Tensor],
                batch_img_shapes: List[Tuple[int, int]], batch_scale_factors: List[Tuple[float, float]],
                rescale: bool = True, cfg: dict = None):
        """练习 5：返回 list（每张图一项）：(bboxes, scores, labels)

        cfg 例如 dict(nms_pre=1000, min_bbox_size=0, score_thr=0.05,
                     nms=dict(type='nms', iou_threshold=0.5), max_per_img=100)

        对每张图、每一层：
            scores = cls.permute(1, 2, 0).reshape(-1, C).sigmoid()        # [n, C]
            deltas = reg.permute(1, 2, 0).reshape(-1, 4)
            ★ 先阈值、再 topk（与 mmdet 的 filter_scores_and_topk 一致）：
                valid_idxs = (scores > score_thr).nonzero()              # [m, 2]：(anchor 下标, 类别)
                s = scores[scores > score_thr]
                s, order = s.sort(descending=True); 保留前 min(nms_pre, m) 个
                anchor_idx, labels = valid_idxs[order[:k]].unbind(dim=1)
                deltas / priors 用 anchor_idx 取
        所有层 cat → decode(max_shape=img_shape) → rescale（同 bbox head）
            → 去掉宽或高 <= min_bbox_size 的框 → batched_nms(bboxes, scores, labels, iou)[:max_per_img]

        思考：和 RPN 的 predict 比，这里的 topk 是在 "(anchor, 类别) 对" 上做的，而不是在 anchor 上，为什么？
        """
        cfg = self.test_cfg if cfg is None else cfg
        # TODO(5)
        raise NotImplementedError('RRetinaHead.predict')
