"""练习 2.3：RPN Head —— 区域建议网络

================================================================================
RPN 做什么？
    在 FPN 每一层的每个位置、对每个 anchor 回答两个问题：
        1. 这里有没有物体？（前景 / 背景二分类，1 个 logit）
        2. 如果有，框应该怎么修正？（4 个 delta）
    然后挑出最可能的约 1000 个框（proposals）交给第二阶段的 RoI head 精修。

    网络本身极其简单（三个卷积），真正的工作量在 "怎么训练它"：

        anchors ──► assigner（谁是正/负样本） ──► sampler（抽 256 个） ──► coder（编码回归目标）
                                                                             │
        rpn_cls / rpn_reg 的输出 ──────────────── 展平、对齐 ─────────────────┴──► loss

    这条链路就是你在 2.1 写的那些工具组合起来。后面 RetinaNet（2.6）、YOLO（Stage 3）
    的 head 都是同样的骨架，只是换了 assigner 和 loss。

★ 命名与 mmdet 一致：rpn_conv / rpn_cls / rpn_reg，官方权重才能加载。

本练习的接口（比 mmdet 简单：不用 InstanceData / DetDataSample，全部是普通 tensor，
detector 负责把 mmdet 的数据结构转换成这些参数）：

    cls_scores, bbox_preds = head(feats)                    # forward
    losses  = head.loss(cls_scores, bbox_preds, batch_gt_bboxes, batch_pad_shapes)
    results = head.predict(cls_scores, bbox_preds, batch_img_shapes, cfg)   # [(bboxes, scores), ...]

参考源码
    mmdet/models/dense_heads/rpn_head.py：_init_layers / forward_single / _predict_by_feat_single / _bbox_post_process
    mmdet/models/dense_heads/anchor_head.py：get_anchors / _get_targets_single / loss_by_feat
    mmdet/models/losses/cross_entropy_loss.py：binary_cross_entropy（use_sigmoid=True 时）

完成标准
    pytest tests/test_rpn_head.py   （装了 mmdet 时会和 mmdet RPNHead 对比 forward / loss / proposals）
建议顺序：__init__ → init_weights → forward → get_anchors → _get_targets_single → loss → predict
================================================================================
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F  # noqa: F401
from torch import nn
from torchvision.ops import batched_nms  # noqa: F401  训练时用 CUDA 版本；可以换成你在 2.1 写的验证结果一致

from rdet.registry import MODELS, TASK_UTILS


@MODELS.register_module()
class RRPNHead(nn.Module):
    """Args:
        in_channels: 输入通道（FPN 输出，256）
        feat_channels: 中间 3x3 卷积的通道（256）
        anchor_generator / bbox_coder: 配置 dict，用 TASK_UTILS.build 构建
        train_cfg: dict(assigner=..., sampler=..., ...)；推理时可以为 None
        test_cfg: 推理配置，predict 时 cfg 为 None 就用它
    """

    def __init__(self, in_channels: int, feat_channels: int = 256, anchor_generator: dict = None,
                 bbox_coder: dict = None, train_cfg: dict = None, test_cfg: dict = None,
                 init_cfg=None, **kwargs):
        super().__init__()
        # 已完成：构建 2.1 写好的工具
        self.prior_generator = TASK_UTILS.build(anchor_generator)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.num_base_priors = self.prior_generator.num_base_anchors[0]  # 每个位置的 anchor 数 A
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if train_cfg is not None:
            self.assigner = TASK_UTILS.build(train_cfg['assigner'])
            self.sampler = TASK_UTILS.build(train_cfg['sampler'])

        # TODO(1) 三个卷积：
        #   rpn_conv: in_channels → feat_channels，3x3，padding=1
        #   rpn_cls:  feat_channels → A * 1，1x1        （sigmoid 二分类，每个 anchor 1 个 logit）
        #   rpn_reg:  feat_channels → A * 4，1x1        （每个 anchor 4 个 delta）
        # 思考：为什么用 1 个 logit + sigmoid，而不是 2 个 logit + softmax？两者等价吗？
        raise NotImplementedError('RRPNHead.__init__')

    def init_weights(self) -> None:
        """练习 2：三个卷积都用 normal(std=0.01) 初始化 weight，bias 置 0（与 mmdet 一致）。"""
        # TODO(2)
        raise NotImplementedError('RRPNHead.init_weights')

    # ------------------------------------------------------------------ forward
    def forward(self, feats: Sequence[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """练习 3：对每一层特征（所有层 **共享** 同一组卷积）：

            x = F.relu(self.rpn_conv(feat))
            cls_score = self.rpn_cls(x)     # [B, A, H, W]
            bbox_pred = self.rpn_reg(x)     # [B, A*4, H, W]

        返回 (cls_scores 列表, bbox_preds 列表)，每个列表长度 = FPN 层数。
        思考：不同层共享参数合理吗？（提示：anchor 的尺寸是随层变化的，网络只需要学 "相对" 的东西）
        """
        # TODO(3)
        raise NotImplementedError('RRPNHead.forward')

    # ------------------------------------------------------------------ loss
    def get_anchors(self, featmap_sizes, batch_pad_shapes, device):
        """练习 4：返回 (anchors, batch_valid_flags)

        - anchors: 把 self.prior_generator.grid_priors(featmap_sizes, device) 的各层结果 cat 成 [N, 4]
          （同一个 batch 的特征图大小相同，所以所有图片共用一套 anchor）
        - batch_valid_flags: 每张图一个 [N] 的 bool，用 prior_generator.valid_flags(featmap_sizes, pad_shape, device)
          得到各层结果再 cat
        """
        # TODO(4)
        raise NotImplementedError('RRPNHead.get_anchors')

    def _get_targets_single(self, anchors: torch.Tensor, valid_flags: torch.Tensor,
                            gt_bboxes: torch.Tensor):
        """练习 5：给一张图的所有 anchor 生成训练目标。

        Returns:
            labels:        [N] float，正样本 1，其余 0（RPN 只分前景 / 背景）
            label_weights: [N] float，被采样到的正、负样本为 1，其余 0（不参与分类 loss）
            bbox_targets:  [N, 4]，正样本处是 coder.encode(anchor, gt)，其余 0
            bbox_weights:  [N, 4]，正样本处为 1
            num_samples:   int，max(正样本数, 1) + max(负样本数, 1)，用于 loss 归一化

        步骤：
            1. inside_inds = valid_flags.nonzero().squeeze(1)；只对 valid 的 anchor 做 assign
               valid_anchors = anchors[inside_inds]
            2. RPN 不关心类别：gt_labels = 全 0 的 long tensor（长度 = GT 数）
               assign_result = self.assigner.assign(valid_anchors, gt_bboxes, gt_labels)
               sampling_result = self.sampler.sample(assign_result, valid_anchors, gt_bboxes, gt_labels)
            3. 先建全 0 的 4 个输出（长度是 **全部** anchor 数 N）
            4. sampling_result 里的 pos_inds / neg_inds 是 valid_anchors 中的下标，
               要映射回全部 anchor：pos = inside_inds[sampling_result.pos_inds]（neg 同理）
            5. 正样本：labels=1, label_weights=1, bbox_targets=encode(pos_priors, pos_gt_bboxes), bbox_weights=1
               负样本：label_weights=1
        思考：为什么 avg_factor 把负样本也算进去（mmdet 里 avg_factor_with_neg=True）？
              RetinaNet（2.6）只用正样本数归一化，原因是什么？
        """
        # TODO(5)
        raise NotImplementedError('RRPNHead._get_targets_single')

    def loss(self, cls_scores: List[torch.Tensor], bbox_preds: List[torch.Tensor],
             batch_gt_bboxes: List[torch.Tensor], batch_pad_shapes: List[Tuple[int, int]]) -> dict:
        """练习 6：计算 RPN loss。

        1. featmap_sizes = [c.shape[-2:] for c in cls_scores]；用 get_anchors 拿到 anchors 和 valid flags
        2. 把网络输出展平成和 anchor 一一对应的形状（★ 最容易错的一步）：
               B = cls_scores[0].size(0)
               flat_cls = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1) for c in cls_scores], dim=1)       # [B, N]
               flat_reg = torch.cat([r.permute(0, 2, 3, 1).reshape(B, -1, 4) for r in bbox_preds], dim=1)    # [B, N, 4]
           思考：为什么是 permute(0, 2, 3, 1)？对照 2.1 里 anchor 的排列顺序（位置在外、anchor 在内）
        3. 每张图调用 _get_targets_single，把结果 stack 成 [B, ...]；avg_factor = 所有图的 num_samples 之和
        4. loss_rpn_cls  = (F.binary_cross_entropy_with_logits(flat_cls, labels, reduction='none')
                            * label_weights).sum() / avg_factor
           loss_rpn_bbox = ((flat_reg - bbox_targets).abs() * bbox_weights).sum() / avg_factor   # L1 loss
        5. return dict(loss_rpn_cls=..., loss_rpn_bbox=...)

        思考：Faster R-CNN 原论文用的是 Smooth L1，mmdet 的默认配置换成了 L1，为什么？
        """
        # TODO(6)
        raise NotImplementedError('RRPNHead.loss')

    # ------------------------------------------------------------------ predict
    def predict(self, cls_scores: List[torch.Tensor], bbox_preds: List[torch.Tensor],
                batch_img_shapes: List[Tuple[int, int]], cfg: dict = None):
        """练习 7：生成 proposals。返回 list（每张图一项）：(bboxes [n, 4], scores [n])

        cfg（默认 self.test_cfg）例如：
            dict(nms_pre=1000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0)

        对每张图 i：
            1. 对每一层 lvl：
                   scores = cls_scores[lvl][i].detach().permute(1, 2, 0).reshape(-1).sigmoid()
                   deltas = bbox_preds[lvl][i].detach().permute(1, 2, 0).reshape(-1, 4)
                   priors = 这一层的 anchor
                   如果 0 < nms_pre < scores.numel()：
                       scores, order = scores.sort(descending=True)，保留前 nms_pre 个，deltas / priors 同步取
                   记录 level_ids：这一层的每个框都标上 lvl
            2. 所有层 cat 起来；bboxes = self.bbox_coder.decode(priors, deltas, max_shape=img_shape)
            3. min_bbox_size >= 0 时，去掉宽或高 <= min_bbox_size 的框
            4. keep = batched_nms(bboxes, scores, level_ids, cfg['nms']['iou_threshold'])
               —— 注意 idxs 传 level_ids：每一层 **分别** 做 NMS
               keep = keep[:cfg['max_per_img']]
            5. 结果为空时返回形状正确的空 tensor

        思考：
        - 为什么要 detach？（proposal 的坐标要不要对 RPN 求梯度？）
        - nms_pre 为什么是 "每层" 取 topk 而不是所有层合起来取？
        - 训练时 rpn_proposal 用 nms_pre=2000，测试时用 1000，为什么训练要更多？
        """
        cfg = self.test_cfg if cfg is None else cfg
        # TODO(7)
        raise NotImplementedError('RRPNHead.predict')
