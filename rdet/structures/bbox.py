"""rdet.structures.bbox —— 练习 2.1a：IoU 矩阵 bbox_overlaps

================================================================================
目标
    计算两组框两两之间的 IoU（或 IoF），这是检测里调用次数最多的函数之一：
    assigner 用它给 anchor 分正负样本，NMS 用它去重，评测用它判断 TP/FP。

框的格式
    全项目统一用 **xyxy**：``[x1, y1, x2, y2]``，左上角 + 右下角，单位是像素。
    mmdet 3.x 不再给宽高 +1（旧版 mmdet 1.x 用 ``x2 - x1 + 1``），这点和一些老代码不一样。

你会学到
    - 用广播（broadcast）代替双重 for 循环：``[N,1,2]`` 与 ``[1,M,2]`` → ``[N,M,2]``
    - 为什么要有 eps，以及没有交集时如何用 clamp(min=0) 处理

参考源码（写完后再对照）
    mmdet/structures/bbox/bbox_overlaps.py（真实版本还支持 batch 维和 GIoU，
    并且为了省显存做了很多 in-place 优化）

完成标准
    pytest tests/test_bbox_utils.py -k overlaps
================================================================================
"""

import torch


def bbox_overlaps(bboxes1: torch.Tensor, bboxes2: torch.Tensor, mode: str = 'iou',
                  is_aligned: bool = False, eps: float = 1e-6) -> torch.Tensor:
    """计算 IoU / IoF。

    Args:
        bboxes1: [N, 4]，xyxy
        bboxes2: [M, 4]，xyxy
        mode: 'iou' = 交集 / 并集；'iof' = 交集 / bboxes1 的面积（intersection over foreground）
        is_aligned: True 时要求 N == M，只算一一对应的 N 个值（返回 [N]）；
                    False 时算两两之间（返回 [N, M]）
        eps: 防止除以 0

    思考：
    - iof 用在哪里？（提示：判断一个框有多少落在 "忽略区域" 里，例如 crowd 标注）
    - 为什么 eps 加在分母上要用 ``torch.max(union, eps)`` 而不是 ``union + eps``？两者有什么区别？
    """
    assert mode in ('iou', 'iof'), f'unsupported mode {mode}'

    # TODO(1) 处理空输入：N == 0 或 M == 0 时，直接返回形状正确的空 tensor
    #         （is_aligned: [N]；否则 [N, M]），用 bboxes1.new_zeros(...)

    # TODO(2) 两组框各自的面积：area = (x2 - x1) * (y2 - y1)

    # TODO(3) 交集的左上角 lt 和右下角 rb
    #   is_aligned:  lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])        -> [N, 2]
    #   否则(广播):  lt = torch.max(bboxes1[:, None, :2], bboxes2[None, :, :2])  -> [N, M, 2]
    #   rb 同理用 torch.min 和 [2:]

    # TODO(4) 交集宽高 wh = (rb - lt).clamp(min=0)；overlap = wh[..., 0] * wh[..., 1]
    #         思考：不 clamp 的话，两个完全不相交的框会得到什么？

    # TODO(5) 分母：iou 模式是 area1 + area2 - overlap（注意广播：area1[:, None] + area2[None, :]）
    #               iof 模式是 area1（同样要广播成 [N, M]）
    #         union = torch.max(union, union.new_tensor([eps]))
    #         return overlap / union

    raise NotImplementedError('bbox_overlaps')
