"""Stage 2.3 RPN Head 的验收测试。"""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

from helpers import fpn_feats, random_boxes, scope, sum_loss  # noqa: E402

from rdet.models.dense_heads import RPNHead  # noqa: E402

ANCHOR = dict(type='AnchorGenerator', scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64])
CODER = dict(type='DeltaXYWHBBoxCoder', target_means=[0.] * 4, target_stds=[1.] * 4)
TRAIN = dict(
    assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3,
                  match_low_quality=True),
    sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1,
                 add_gt_as_proposals=False))
TEST = dict(nms_pre=1000, max_per_img=300, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0)

# 一个 batch 两张图：pad 后的 batch 大小是 128 x 96；第二张图本身只有 90 x 60（pad 到 96 x 64）
IMG_SHAPES = [(128, 96), (90, 60)]
PAD_SHAPES = [(128, 96), (96, 64)]
GT = [random_boxes(3, 128, 96, seed=1), random_boxes(2, 90, 60, seed=2)]


def make_head():
    torch.manual_seed(0)
    head = RPNHead(256, 256, ANCHOR, CODER, train_cfg=TRAIN, test_cfg=TEST)
    head.init_weights()
    return head


def test_layers_and_init():
    head = make_head()
    assert set(head.state_dict()) == {f'{n}.{p}' for n in ('rpn_conv', 'rpn_cls', 'rpn_reg')
                                      for p in ('weight', 'bias')}
    assert head.rpn_cls.out_channels == 3 and head.rpn_reg.out_channels == 12
    assert torch.all(head.rpn_conv.bias == 0)
    assert 0.005 < head.rpn_conv.weight.std() < 0.015


def test_forward_shapes():
    cls_scores, bbox_preds = make_head()(fpn_feats(2, 128, 96))
    assert [c.shape for c in cls_scores] == [(2, 3, 32, 24), (2, 3, 16, 12), (2, 3, 8, 6), (2, 3, 4, 3), (2, 3, 2, 2)]
    assert [b.shape[1] for b in bbox_preds] == [12] * 5


def test_targets_ignore_padding():
    head = make_head()
    sizes = [(32, 24), (16, 12), (8, 6), (4, 3), (2, 2)]
    anchors, flags = head.get_anchors(sizes, PAD_SHAPES, 'cpu')
    assert anchors.shape == (sum(h * w * 3 for h, w in sizes), 4) and len(flags) == 2
    assert flags[0].all() and not flags[1].all()
    labels, label_w, bbox_t, bbox_w, num = head._get_targets_single(anchors, flags[1], GT[1])
    assert (label_w[~flags[1]] == 0).all() and (bbox_w[~flags[1]] == 0).all()
    assert labels.sum() > 0 and (labels[labels > 0] == 1).all()
    assert torch.equal(labels > 0, bbox_w[:, 0] > 0)
    assert label_w.sum() == 256 and num == 256


def test_loss_and_backward():
    head = make_head()
    cls_scores, bbox_preds = head(fpn_feats(2, 128, 96))
    losses = head.loss(cls_scores, bbox_preds, GT, PAD_SHAPES)
    assert set(losses) == {'loss_rpn_cls', 'loss_rpn_bbox'}
    assert 0.5 < losses['loss_rpn_cls'] < 0.9          # 刚初始化：logit ≈ 0，BCE ≈ log 2 ≈ 0.69
    assert losses['loss_rpn_bbox'] > 0
    sum(losses.values()).backward()
    assert head.rpn_cls.weight.grad.abs().sum() > 0

    empty = [GT[0][:0], GT[1][:0]]
    losses = head.loss(*head(fpn_feats(2, 128, 96)), empty, PAD_SHAPES)
    assert losses['loss_rpn_bbox'] == 0 and losses['loss_rpn_cls'] > 0


def test_predict():
    head = make_head()
    results = head.predict(*head(fpn_feats(2, 128, 96)), IMG_SHAPES)
    assert len(results) == 2
    for (bboxes, scores), (h, w) in zip(results, IMG_SHAPES):
        assert 0 < len(bboxes) <= 300 and bboxes.shape[1] == 4 and scores.shape == (len(bboxes),)
        assert (bboxes[:, 0::2] <= w).all() and (bboxes[:, 1::2] <= h).all() and (bboxes >= 0).all()
        assert not scores.requires_grad


def _mmdet_rpn_head():
    from mmdet.models.dense_heads import RPNHead as MMDetRPNHead
    from mmengine.config import ConfigDict
    with scope('mmdet'):
        return _build_mmdet_rpn_head(MMDetRPNHead, ConfigDict)


def _build_mmdet_rpn_head(MMDetRPNHead, ConfigDict):
    return MMDetRPNHead(
        in_channels=256, feat_channels=256,
        anchor_generator=dict(type='AnchorGenerator', scales=[8], ratios=[0.5, 1.0, 2.0],
                              strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.] * 4, target_stds=[1.] * 4),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        train_cfg=ConfigDict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3,
                          match_low_quality=True, ignore_iof_thr=-1),
            sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1,
                         add_gt_as_proposals=False),
            allowed_border=-1, pos_weight=-1, debug=False),
        test_cfg=ConfigDict(TEST))


def test_rpn_head_vs_mmdet():
    pytest.importorskip('mmdet.models')
    from mmengine.config import ConfigDict
    from mmengine.structures import InstanceData
    ref = _mmdet_rpn_head()
    mine = make_head()
    mine.load_state_dict(ref.state_dict())
    feats = fpn_feats(2, 128, 96)
    metas = [dict(img_shape=i, pad_shape=p) for i, p in zip(IMG_SHAPES, PAD_SHAPES)]

    outs, ref_outs = mine(feats), ref(feats)
    for a, b in zip(outs[0] + outs[1], ref_outs[0] + ref_outs[1]):
        assert torch.allclose(a, b, atol=1e-6)

    torch.manual_seed(42)        # 两边用同一个随机种子，RandomSampler 会抽到同样的负样本
    ref_losses = ref.loss_by_feat(*ref_outs, [InstanceData(bboxes=g, labels=torch.zeros(len(g), dtype=torch.long))
                                              for g in GT], metas)
    torch.manual_seed(42)
    losses = mine.loss(*outs, GT, PAD_SHAPES)
    assert torch.allclose(losses['loss_rpn_cls'], sum_loss(ref_losses['loss_rpn_cls']), rtol=1e-4)
    assert torch.allclose(losses['loss_rpn_bbox'], sum_loss(ref_losses['loss_rpn_bbox']), rtol=1e-4)

    ref_res = ref.predict_by_feat(*ref_outs, batch_img_metas=metas, cfg=ConfigDict(TEST), rescale=False)
    for (bboxes, scores), r in zip(mine.predict(*outs, IMG_SHAPES), ref_res):
        assert bboxes.shape == r.bboxes.shape
        assert torch.allclose(bboxes, r.bboxes, atol=1e-4) and torch.allclose(scores, r.scores, atol=1e-6)
