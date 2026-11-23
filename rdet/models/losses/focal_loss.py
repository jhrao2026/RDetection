"""练习 2.6a：Focal Loss

================================================================================
问题
    one-stage 检测器（RetinaNet）没有 RPN 先过滤，也不做 256 个样本的采样，
    每张图约 10 万个 anchor 全部参与分类 loss，其中 99.9% 是 "很容易判断的背景"。
    即使每个简单背景只贡献很小的 loss，加起来也会淹没少数正样本。

Focal Loss（Lin et al. 2017）
    在 BCE 前乘一个调制因子，让 "已经分对的简单样本" 几乎不贡献 loss：

        p   = sigmoid(logit)
        p_t = p      若 target = 1
              1 - p  若 target = 0
        FL  = - alpha_t * (1 - p_t) ** gamma * log(p_t)

    其中 alpha_t = alpha（正样本）/ 1 - alpha（负样本），论文取 gamma=2, alpha=0.25。

    实现时用 BCE 的数值稳定版本代替 -log(p_t)：
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
        focal_weight = alpha_t * (1 - p_t) ** gamma

你会学到
    - 用 "重新加权" 代替 "采样" 来解决正负样本不平衡
    - 为什么 retina_cls 的 bias 要初始化成 -log((1 - 0.01) / 0.01)（见 2.6b）

参考源码
    mmdet/models/losses/focal_loss.py：py_sigmoid_focal_loss（纯 PyTorch 版本，正好对照）

完成标准
    pytest tests/test_retinanet.py -k focal
================================================================================
"""

import torch
import torch.nn.functional as F  # noqa: F401


def sigmoid_focal_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None,
                       gamma: float = 2.0, alpha: float = 0.25, avg_factor: float = 1.0) -> torch.Tensor:
    """Args:
        pred: [N, C] logits（没有背景通道：所有类别都是 0 就表示背景）
        target: [N, C] one-hot，float
        weight: [N] 或 None，每个样本的权重（0 表示忽略）
        avg_factor: 归一化因子，返回 sum(loss) / avg_factor

    练习：按文件开头的公式实现。weight 是 [N]，要扩成 [N, 1] 再乘。
    思考：gamma=0、alpha=0.5 时 Focal Loss 退化成什么？
    """
    # TODO
    raise NotImplementedError('sigmoid_focal_loss')
