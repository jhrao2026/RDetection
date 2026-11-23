"""测试公用的小工具（不是练习）。"""
import math

import torch


def random_boxes(n, img_h=128., img_w=96., min_size=8., seed=0):
    """在 (img_h, img_w) 范围内随机生成 n 个 xyxy 框。"""
    g = torch.Generator().manual_seed(seed)
    wh = torch.rand(n, 2, generator=g) * torch.tensor([img_w, img_h]) / 2 + min_size
    xy = torch.rand(n, 2, generator=g) * (torch.tensor([img_w, img_h]) - wh)
    return torch.cat([xy, xy + wh], dim=1)


def fpn_feats(batch, img_h, img_w, strides=(4, 8, 16, 32, 64), channels=256, seed=0):
    """模拟 FPN 输出：第 l 层大小为 ceil(img / stride)。"""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(batch, channels, math.ceil(img_h / s), math.ceil(img_w / s), generator=g)
            for s in strides]


def sum_loss(v):
    """mmdet 的 loss 可能是 tensor，也可能是每层一个 tensor 的 list。"""
    return sum(v) if isinstance(v, (list, tuple)) else v


def mmdet_scope():
    """构建 mmdet 模块前设置 default scope（Runner 会自动做这件事，单独测试时要手动做）。"""
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet')


def mmdet_config(rel_path):
    """mmdet 包里自带的官方配置文件路径，例如 mmdet_config('faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py')。

    （配置文件 _base_ 里可以写 'mmdet::xxx'，但 Config.fromfile 的参数本身不支持这种写法）
    """
    import os.path as osp

    import mmdet
    return osp.join(osp.dirname(mmdet.__file__), '.mim', 'configs', rel_path)
