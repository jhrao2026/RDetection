"""Stage 2.1 基础工具的验收测试。

A 层：只依赖 torch / torchvision。
B 层（名字带 _vs_mmdet）：装了 mmdet 才会运行，和 mmdet 的实现逐元素对比。
"""
import math

import pytest

torch = pytest.importorskip('torch')
torchvision = pytest.importorskip('torchvision')

from rdet.models.layers import batched_nms, nms  # noqa: E402
from rdet.models.task_modules import (AssignResult, RAnchorGenerator,  # noqa: E402
                                      RDeltaXYWHBBoxCoder, RMaxIoUAssigner,
                                      RRandomSampler)
from rdet.structures import bbox_overlaps  # noqa: E402


def random_boxes(n, size=100., seed=0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g) * size
    wh = torch.rand(n, 2, generator=g) * size / 2 + 1
    return torch.cat([xy, xy + wh], dim=1)


# =============================================================== bbox_overlaps
def test_overlaps_iou_matches_torchvision():
    b1, b2 = random_boxes(30, seed=1), random_boxes(20, seed=2)
    assert torch.allclose(bbox_overlaps(b1, b2), torchvision.ops.box_iou(b1, b2), atol=1e-5)


def test_overlaps_hand_example_and_iof():
    b1 = torch.tensor([[0., 0., 10., 10.]])
    b2 = torch.tensor([[5., 5., 15., 15.], [20., 20., 30., 30.], [0., 0., 5., 10.]])
    iou = bbox_overlaps(b1, b2)
    assert torch.allclose(iou, torch.tensor([[25 / 175, 0., 0.5]]), atol=1e-6)
    iof = bbox_overlaps(b1, b2, mode='iof')
    assert torch.allclose(iof, torch.tensor([[0.25, 0., 0.5]]), atol=1e-6)


def test_overlaps_aligned_and_empty():
    b1, b2 = random_boxes(10, seed=3), random_boxes(10, seed=4)
    full = bbox_overlaps(b1, b2)
    assert torch.allclose(bbox_overlaps(b1, b2, is_aligned=True), full.diagonal(), atol=1e-6)
    assert bbox_overlaps(b1[:0], b2).shape == (0, 10)
    assert bbox_overlaps(b1, b2[:0]).shape == (10, 0)
    assert bbox_overlaps(b1[:0], b2[:0], is_aligned=True).shape == (0,)


# =============================================================== nms
def test_nms_matches_torchvision():
    boxes = random_boxes(200, seed=5)
    scores = torch.rand(200, generator=torch.Generator().manual_seed(6))
    for thr in (0.3, 0.5, 0.7):
        assert torch.equal(nms(boxes, scores, thr), torchvision.ops.nms(boxes, scores, thr))
    assert nms(boxes[:0], scores[:0], 0.5).numel() == 0


def test_batched_nms_matches_torchvision():
    boxes = random_boxes(200, seed=7)
    g = torch.Generator().manual_seed(8)
    scores = torch.rand(200, generator=g)
    idxs = torch.randint(0, 5, (200,), generator=g)
    expected = torchvision.ops.batched_nms(boxes, scores, idxs, 0.5)
    assert torch.equal(batched_nms(boxes, scores, idxs, 0.5), expected)
    assert batched_nms(boxes[:0], scores[:0], idxs[:0], 0.5).numel() == 0


# =============================================================== anchor generator
def _rpn_anchor_generator():
    return RAnchorGenerator(strides=[4, 8, 16, 32, 64], ratios=[0.5, 1.0, 2.0], scales=[8])


def test_base_anchors_hand_example():
    gen = RAnchorGenerator(strides=[16], ratios=[1.], scales=[8.])
    assert torch.allclose(gen.base_anchors[0], torch.tensor([[-64., -64., 64., 64.]]))
    gen = RAnchorGenerator(strides=[16], ratios=[0.5, 1., 2.], scales=[1., 2.])
    base = gen.base_anchors[0]
    assert base.shape == (6, 4)
    w, h = base[:, 2] - base[:, 0], base[:, 3] - base[:, 1]
    assert torch.allclose(h / w, torch.tensor([0.5, 0.5, 1., 1., 2., 2.]))   # ratio 在外层
    assert torch.allclose(w * h, torch.tensor([256., 1024.] * 3))             # scale 在内层，面积不随 ratio 变


def test_grid_priors_shape_and_order():
    gen = _rpn_anchor_generator()
    assert gen.num_base_anchors == [3] * 5
    sizes = [(8, 10), (4, 5), (2, 3), (1, 2), (1, 1)]
    priors = gen.grid_priors(sizes)
    assert [p.shape for p in priors] == [(h * w * 3, 4) for h, w in sizes]
    p0 = priors[0].view(8, 10, 3, 4)
    # 第 (row=1, col=2) 个位置 = 基础 anchor 平移 (x=2*4, y=1*4)
    shift = torch.tensor([8., 4., 8., 4.])
    assert torch.allclose(p0[1, 2], gen.base_anchors[0] + shift)


def test_anchor_generator_vs_mmdet():
    pytest.importorskip('mmdet.models')
    from mmdet.models.task_modules.prior_generators import AnchorGenerator
    ref = AnchorGenerator(strides=[4, 8, 16, 32, 64], ratios=[0.5, 1.0, 2.0], scales=[8])
    sizes = [(20, 25), (10, 13), (5, 7), (3, 4), (2, 2)]
    for a, b in zip(_rpn_anchor_generator().grid_priors(sizes), ref.grid_priors(sizes, device='cpu')):
        assert torch.allclose(a, b)


# =============================================================== bbox coder
def test_coder_known_values():
    coder = RDeltaXYWHBBoxCoder()
    anchor = torch.tensor([[0., 0., 10., 10.]])
    gt = torch.tensor([[5., 0., 15., 20.]])         # 中心右移半个宽度，高度变为 2 倍
    d = coder.encode(anchor, gt)
    assert torch.allclose(d, torch.tensor([[0.5, 0.5, 0., math.log(2.)]]), atol=1e-6)


def test_coder_roundtrip_with_stds():
    coder = RDeltaXYWHBBoxCoder(target_stds=(0.1, 0.1, 0.2, 0.2))
    anchors, gts = random_boxes(50, seed=9), random_boxes(50, seed=10)
    deltas = coder.encode(anchors, gts)
    assert torch.allclose(coder.decode(anchors, deltas), gts, atol=1e-3)


def test_coder_clip_and_ratio_clamp():
    coder = RDeltaXYWHBBoxCoder()
    anchor = torch.tensor([[40., 10., 60., 30.]])
    out = coder.decode(anchor, torch.tensor([[0., 0., 50., 50.]]), max_shape=(48, 64))  # (h, w)
    assert out[0, 0] == 0 and out[0, 1] == 0 and out[0, 2] == 64 and out[0, 3] == 48
    out = coder.decode(anchor, torch.tensor([[0., 0., 50., 0.]]))                       # 不裁剪
    max_w = 20 * 1000 / 16
    assert torch.isclose(out[0, 2] - out[0, 0], torch.tensor(max_w), rtol=1e-4)


def test_coder_vs_mmdet():
    pytest.importorskip('mmdet.models')
    from mmdet.models.task_modules.coders import DeltaXYWHBBoxCoder
    kw = dict(target_means=(0., 0., 0., 0.), target_stds=(0.1, 0.1, 0.2, 0.2))
    mine, ref = RDeltaXYWHBBoxCoder(**kw), DeltaXYWHBBoxCoder(**kw)
    anchors, gts = random_boxes(50, seed=11), random_boxes(50, seed=12)
    assert torch.allclose(mine.encode(anchors, gts), ref.encode(anchors, gts), atol=1e-5)
    deltas = torch.randn(50, 4, generator=torch.Generator().manual_seed(13))
    assert torch.allclose(mine.decode(anchors, deltas, max_shape=(80, 90)),
                          ref.decode(anchors, deltas, max_shape=(80, 90)), atol=1e-4)


# =============================================================== assigner
GT = torch.tensor([[0., 0., 10., 10.], [50., 50., 60., 60.]])
GT_LABELS = torch.tensor([3, 7])
PRIORS = torch.tensor([
    [0., 0., 10., 10.],     # IoU 1.0 with gt0      -> 正样本
    [0., 0., 10., 5.],      # IoU 0.5               -> 忽略
    [20., 20., 30., 30.],   # IoU 0                 -> 负样本
    [50., 50., 60., 54.5],  # IoU 0.45 with gt1，是 gt1 的最佳 anchor -> low quality 正样本
])


def test_assigner_basic():
    assigner = RMaxIoUAssigner(pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3)
    res = assigner.assign(PRIORS, GT, GT_LABELS)
    assert isinstance(res, AssignResult) and res.num_gts == 2
    assert res.gt_inds.tolist() == [1, -1, 0, 2]
    assert res.labels.tolist() == [3, -1, -1, 7]
    assert torch.allclose(res.max_overlaps, torch.tensor([1., 0.5, 0., 0.45]), atol=1e-5)


def test_assigner_options():
    res = RMaxIoUAssigner(0.7, 0.3, 0.3, match_low_quality=False).assign(PRIORS, GT, GT_LABELS)
    assert res.gt_inds.tolist() == [1, -1, 0, -1]
    res = RMaxIoUAssigner(0.7, (0.1, 0.3), 0.3).assign(PRIORS, GT, GT_LABELS)
    assert res.gt_inds.tolist() == [1, -1, -1, 2]          # IoU 0 不在 [0.1, 0.3) 内
    res = RMaxIoUAssigner(0.7, 0.3, 0.5).assign(PRIORS, GT, GT_LABELS)
    assert res.gt_inds.tolist() == [1, -1, 0, -1]          # 0.45 < min_pos_iou


def test_assigner_no_gt():
    res = RMaxIoUAssigner(0.7, 0.3).assign(PRIORS, GT[:0], GT_LABELS[:0])
    assert res.gt_inds.tolist() == [0, 0, 0, 0] and res.labels.tolist() == [-1] * 4


def test_assigner_vs_mmdet():
    pytest.importorskip('mmdet.models')
    from mmdet.models.task_modules.assigners import MaxIoUAssigner
    from mmengine.structures import InstanceData
    priors, gts = random_boxes(300, seed=14), random_boxes(6, seed=15)
    labels = torch.arange(6)
    ref = MaxIoUAssigner(0.7, 0.3, 0.3).assign(InstanceData(priors=priors),
                                               InstanceData(bboxes=gts, labels=labels))
    mine = RMaxIoUAssigner(0.7, 0.3, 0.3).assign(priors, gts, labels)
    assert torch.equal(mine.gt_inds, ref.gt_inds) and torch.equal(mine.labels, ref.labels)


# =============================================================== sampler
def _fake_assign(num_pos, num_neg, num_ignore=10, num_gts=3):
    gt_inds = torch.cat([torch.randint(1, num_gts + 1, (num_pos,)),
                         torch.zeros(num_neg, dtype=torch.long),
                         torch.full((num_ignore,), -1, dtype=torch.long)])
    labels = torch.where(gt_inds > 0, gt_inds + 10, torch.full_like(gt_inds, -1))
    return AssignResult(num_gts, gt_inds, torch.rand(len(gt_inds)), labels)


def test_sampler_counts():
    torch.manual_seed(0)
    n = 1110
    priors, gts, gt_labels = random_boxes(n), random_boxes(3), torch.tensor([11, 12, 13])
    sampler = RRandomSampler(num=256, pos_fraction=0.25, add_gt_as_proposals=False)
    res = sampler.sample(_fake_assign(100, 1000), priors, gts, gt_labels)
    assert len(res.pos_inds) == 64 and len(res.neg_inds) == 192
    assert (res.pos_inds < 100).all() and ((res.neg_inds >= 100) & (res.neg_inds < 1100)).all()
    assert torch.equal(res.pos_priors, priors[res.pos_inds])
    assert torch.equal(res.pos_gt_bboxes, gts[res.pos_assigned_gt_inds])
    assert torch.equal(res.pos_gt_labels, res.pos_assigned_gt_inds + 11)
    assert res.priors.shape == (256, 4)

    res = sampler.sample(_fake_assign(10, 1000), priors[:1020], gts, gt_labels)
    assert len(res.pos_inds) == 10 and len(res.neg_inds) == 246      # 正样本不够，用负样本补满

    ub = RRandomSampler(num=256, pos_fraction=0.25, neg_pos_ub=3, add_gt_as_proposals=False)
    res = ub.sample(_fake_assign(10, 1000), priors[:1020], gts, gt_labels)
    assert len(res.neg_inds) == 30


def test_sampler_add_gt():
    torch.manual_seed(0)
    priors, gts, gt_labels = random_boxes(30), random_boxes(3), torch.tensor([11, 12, 13])
    sampler = RRandomSampler(num=512, pos_fraction=0.25, add_gt_as_proposals=True)
    assign = _fake_assign(5, 15)
    res = sampler.sample(assign, priors, gts, gt_labels)
    assert len(res.pos_inds) == 8 and res.pos_is_gt.sum() == 3
    gt_rows = res.pos_inds[res.pos_is_gt]
    assert sorted(gt_rows.tolist()) == [0, 1, 2]
    assert torch.equal(res.pos_priors[res.pos_is_gt], res.pos_gt_bboxes[res.pos_is_gt])
    assert len(assign.gt_inds) == 33   # add_gt_ 已经修改了 assign_result
