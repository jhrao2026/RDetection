"""Stage 2.6 RetinaNet（Focal Loss + RetinaHead + SingleStageDetector）的验收测试。"""
import copy
import math
import os.path as osp

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
import torch.nn.functional as F  # noqa: E402

from helpers import build_from_config, fpn_feats, mmdet_config, random_boxes, scope, sum_loss  # noqa: E402

from rdet.models.dense_heads import RetinaHead  # noqa: E402
from rdet.models.losses import sigmoid_focal_loss  # noqa: E402

NUM_CLASSES = 4
SCALES = [4.0, 5.039684199579493, 6.3496042078727974]
STRIDES = (8, 16, 32, 64, 128)
ANCHOR = dict(type='AnchorGenerator', scales=SCALES, ratios=[0.5, 1.0, 2.0], strides=list(STRIDES))
CODER = dict(type='DeltaXYWHBBoxCoder', target_means=[0.] * 4, target_stds=[1.] * 4)
TRAIN = dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.4, min_pos_iou=0,
                           match_low_quality=True))
TEST = dict(nms_pre=1000, min_bbox_size=0, score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
IMG_SHAPES = [(256, 192), (200, 160)]
PAD_SHAPES = [(256, 192), (224, 160)]
SCALE_FACTORS = [(1.0, 1.0), (1.25, 1.25)]
GT = [random_boxes(3, 256, 192, min_size=32, seed=1), random_boxes(2, 200, 160, min_size=32, seed=2)]
LABELS = [torch.tensor([0, 2, 3]), torch.tensor([1, 1])]


# ------------------------------------------------------------------ focal loss
def test_focal_loss_formula():
    pred = torch.tensor([[2.0, -1.0], [0.5, 0.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    p = pred.sigmoid()
    pt = torch.where(target == 1, p, 1 - p)
    alpha_t = torch.where(target == 1, torch.tensor(0.25), torch.tensor(0.75))
    expected = (-alpha_t * (1 - pt) ** 2 * torch.log(pt)).sum() / 3
    assert torch.isclose(sigmoid_focal_loss(pred, target, avg_factor=3), expected, rtol=1e-5)
    w = torch.tensor([1.0, 0.0])
    assert torch.isclose(sigmoid_focal_loss(pred, target, w), (-alpha_t * (1 - pt) ** 2 * torch.log(pt))[0].sum(),
                         rtol=1e-5)
    # gamma=0, alpha=0.5 退化为 0.5 * BCE
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='sum')
    assert torch.isclose(sigmoid_focal_loss(pred, target, gamma=0., alpha=0.5), 0.5 * bce, rtol=1e-5)


def test_focal_loss_vs_mmdet():
    pytest.importorskip('mmdet.models')
    from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss
    pred = torch.randn(50, NUM_CLASSES)
    labels = torch.randint(0, NUM_CLASSES + 1, (50,))
    target = F.one_hot(labels, NUM_CLASSES + 1)[:, :NUM_CLASSES].float()
    w = (torch.rand(50) > 0.2).float()
    ref = py_sigmoid_focal_loss(pred, target, w[:, None], gamma=2.0, alpha=0.25, reduction='mean', avg_factor=7)
    assert torch.isclose(sigmoid_focal_loss(pred, target, w, avg_factor=7), ref, rtol=1e-5)


# ------------------------------------------------------------------ head
def make_head():
    torch.manual_seed(0)
    head = RetinaHead(NUM_CLASSES, 256, 4, 256, ANCHOR, CODER, train_cfg=TRAIN, test_cfg=TEST)
    head.init_weights()
    return head


def test_head_layers_init_forward():
    head = make_head()
    keys = set(head.state_dict())
    assert {'cls_convs.3.conv.weight', 'reg_convs.0.conv.bias', 'retina_cls.weight', 'retina_reg.bias'} <= keys
    assert len(keys) == 2 * 4 * 2 + 4
    assert torch.allclose(head.retina_cls.bias, torch.full_like(head.retina_cls.bias, -math.log(99)))
    cls_scores, bbox_preds = head(fpn_feats(2, 256, 192, STRIDES))
    assert cls_scores[0].shape == (2, 9 * NUM_CLASSES, 32, 24) and bbox_preds[4].shape == (2, 36, 2, 2)


def test_head_loss_and_predict():
    head = make_head()
    outs = head(fpn_feats(2, 256, 192, STRIDES))
    losses = head.loss(*outs, GT, LABELS, PAD_SHAPES)
    assert set(losses) == {'loss_cls', 'loss_bbox'} and losses['loss_cls'] > 0 and losses['loss_bbox'] > 0
    sum(losses.values()).backward()
    results = head.predict(*[[o.detach() for o in x] for x in outs], IMG_SHAPES, SCALE_FACTORS, True,
                           cfg=dict(TEST, score_thr=0.005))
    for (bboxes, scores, labels), (h, w), (sw, sh) in zip(results, IMG_SHAPES, SCALE_FACTORS):
        assert 0 < len(bboxes) <= 100 and labels.max() < NUM_CLASSES
        assert (bboxes[:, 0::2] <= w / sw + 1e-4).all() and (bboxes[:, 1::2] <= h / sh + 1e-4).all()


def test_head_vs_mmdet():
    pytest.importorskip('mmdet.models')
    from mmdet.models.dense_heads import RetinaHead as MMDetRetinaHead
    from mmengine.config import ConfigDict
    from mmengine.structures import InstanceData
    with scope('mmdet'):
        ref = MMDetRetinaHead(
            num_classes=NUM_CLASSES, in_channels=256, stacked_convs=4, feat_channels=256,
            anchor_generator=dict(type='AnchorGenerator', octave_base_scale=4, scales_per_octave=3,
                                  ratios=[0.5, 1.0, 2.0], strides=list(STRIDES)),
            bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.] * 4, target_stds=[1.] * 4),
            loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            train_cfg=ConfigDict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.4,
                                               min_pos_iou=0, ignore_iof_thr=-1),
                                 sampler=dict(type='PseudoSampler'), allowed_border=-1, pos_weight=-1, debug=False),
            test_cfg=ConfigDict(TEST))
    mine = make_head()
    mine.load_state_dict(ref.state_dict())
    with torch.no_grad():   # 分类输出调 "尖" 一点，保证 predict 有足够多超过阈值的框
        ref.retina_cls.weight.normal_(0, 0.05)
        mine.retina_cls.weight.copy_(ref.retina_cls.weight)
    feats = fpn_feats(2, 256, 192, STRIDES)
    outs, ref_outs = mine(feats), ref(feats)
    for a, b in zip(outs[0] + outs[1], ref_outs[0] + ref_outs[1]):
        assert torch.allclose(a, b, atol=1e-5)

    metas = [dict(img_shape=i, pad_shape=p, scale_factor=s) for i, p, s in zip(IMG_SHAPES, PAD_SHAPES, SCALE_FACTORS)]
    ref_losses = ref.loss_by_feat(*ref_outs, [InstanceData(bboxes=g, labels=lb) for g, lb in zip(GT, LABELS)], metas)
    losses = mine.loss(*outs, GT, LABELS, PAD_SHAPES)
    for k in ('loss_cls', 'loss_bbox'):
        assert torch.allclose(losses[k], sum_loss(ref_losses[k]), rtol=1e-4), k

    ref_res = ref.predict_by_feat(*ref_outs, batch_img_metas=metas, cfg=ConfigDict(TEST), rescale=True)
    for (bboxes, scores, labels), r in zip(mine.predict(*outs, IMG_SHAPES, SCALE_FACTORS, True), ref_res):
        assert len(bboxes) > 0 and bboxes.shape == r.bboxes.shape
        assert torch.allclose(bboxes, r.bboxes, atol=1e-3) and torch.allclose(scores, r.scores, atol=1e-5)
        assert torch.equal(labels, r.labels)


# ------------------------------------------------------------------ detector（需要 mmdet）
def test_detector_vs_mmdet():
    pytest.importorskip('mmdet.models')
    from test_faster_rcnn import _unpack, fake_batch

    torch.manual_seed(0)
    ref = build_from_config(mmdet_config('retinanet/retinanet_r50_fpn_1x_coco.py'))
    with torch.no_grad():   # 初始 bias=-4.6 时所有分数都 < 0.05，没有检测结果可比；调大让结果非空
        ref.bbox_head.retina_cls.bias.fill_(-2.0)
    mine = build_from_config(osp.join(osp.dirname(__file__), '..', 'configs', 'retinanet', 'retinanet_r50_fpn_1x_coco.py'))
    mine.load_state_dict(ref.state_dict(), strict=True)

    mine.train(), ref.train()
    data = fake_batch(ref, training=True)
    ref_losses = ref.loss(*_unpack(data))
    losses = mine.loss(*_unpack(data))
    for k in ('loss_cls', 'loss_bbox'):
        assert torch.allclose(losses[k], sum_loss(ref_losses[k]), rtol=1e-3), k

    mine.eval(), ref.eval()
    data = fake_batch(ref, training=False)
    with torch.no_grad():
        ref_out = ref.predict(*_unpack(data))
        out = mine.predict(*_unpack(data))
    for a, b in zip(out, ref_out):
        assert len(a.pred_instances) > 0 and len(a.pred_instances) == len(b.pred_instances)
        assert torch.allclose(a.pred_instances.bboxes, b.pred_instances.bboxes, atol=1e-3)
