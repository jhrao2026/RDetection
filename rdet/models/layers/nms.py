"""rdet.models.layers.nms —— 练习 2.1b：非极大值抑制（纯 Python / torch 版）

================================================================================
目标
    检测器对同一个物体通常会输出很多高度重叠的框。NMS 的规则很简单：
        按分数从高到低，保留当前最高分的框，删掉和它 IoU > 阈值的所有框，重复。

    真实训练用的是 mmcv.ops.nms / torchvision.ops.nms（CUDA 实现，快几十倍），
    这里手写一遍是为了彻底理解它 —— 后面 YOLOv10 / DETR 号称 "NMS-free"，
    你得先知道 NMS 到底做了什么、为什么它是部署时的瓶颈。

参考源码
    mmcv/ops/nms.py：nms / batched_nms（重点看 batched_nms 的 "坐标偏移" 技巧）

完成标准
    pytest tests/test_bbox_utils.py -k nms   （结果要和 torchvision.ops.nms 完全一致）
================================================================================
"""

import torch

from rdet.structures import bbox_overlaps  # noqa: F401  练习里会用到


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """单类别 NMS。

    Args:
        boxes: [N, 4] xyxy
        scores: [N]
        iou_threshold: IoU **严格大于** 该阈值的框会被抑制

    Returns:
        keep: [K] long，被保留框的下标，按分数从高到低排列

    提示（循环版就够了，不用追求速度）：
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]; keep.append(i)
            if order.numel() == 1: break
            ious = bbox_overlaps(boxes[i][None], boxes[order[1:]])[0]
            order = order[1:][ious <= iou_threshold]
        return torch.stack(keep) if keep else boxes.new_zeros(0, dtype=torch.long)

    写之前先在纸上画 4 个框模拟一遍。
    思考：这个循环最坏情况下的复杂度是多少？为什么 GPU 版本能并行？
    """
    # TODO(1) 按上面的思路实现；注意空输入
    raise NotImplementedError('nms')


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor,
                iou_threshold: float) -> torch.Tensor:
    """多类别 NMS：只在 **同一类别** 的框之间做抑制，不同类别互不影响。

    Args:
        idxs: [N] 每个框的类别 id

    朴素做法是 for 每个类别调用一次 nms。mmcv / torchvision 用了一个巧妙的技巧：

        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = nms(boxes_for_nms, scores, iou_threshold)

    思考：
    - 为什么加了偏移之后，不同类别的框之间 IoU 一定是 0？
    - 这个技巧把 C 次 nms 调用变成了 1 次，代价是什么？（提示：坐标很大时的浮点精度）
    - Faster R-CNN 里，RPN 的 proposals 用 batched_nms 时 idxs 传的是什么？
      （提示：mmdet RPNHead 传的是 FPN 层号 level_ids —— 每一层分开做 NMS）
    """
    # TODO(2) 空输入直接返回空的 long tensor
    # TODO(3) 按上面的偏移技巧实现，复用你写的 nms
    raise NotImplementedError('batched_nms')
