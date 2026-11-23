"""Stage 2.4 RoI Head 的验收测试。"""
import pytest

torch = pytest.importorskip('torch')
torchvision = pytest.importorskip('torchvision')

from helpers import fpn_feats, random_boxes, scope  # noqa: E402

from rdet.models.roi_heads import (Shared2FCBBoxHead, SingleRoIExtractor,  # noqa: E402
                                   StandardRoIHead, bbox2roi)
from rdet.models.task_modules import SamplingResult  # noqa: E402

NUM_CLASSES = 5
EXTRACTOR = dict(type='SingleRoIExtractor', roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                 out_channels=256, featmap_strides=[4, 8, 16, 32])
CODER = dict(type='DeltaXYWHBBoxCoder', target_means=[0.] * 4, target_stds=[0.1, 0.1, 0.2, 0.2])
BBOX_HEAD = dict(type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024, roi_feat_size=7,
                 num_classes=NUM_CLASSES, bbox_coder=CODER, reg_class_agnostic=False)
TRAIN = dict(
    assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5,
                  match_low_quality=False),
    sampler=dict(type='RandomSampler', num=64, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True))
TEST = dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5), max_per_img=100)

IMG_SHAPES = [(256, 192), (200, 160)]
SCALE_FACTORS = [(2.0, 2.0), (1.6, 1.5)]
GT_BBOXES = [random_boxes(3, 256, 192, min_size=24, seed=1), random_boxes(2, 200, 160, min_size=24, seed=2)]
GT_LABELS = [torch.tensor([0, 3, 4]), torch.tensor([1, 1])]


def proposals(seed=0):
    """GT 附近抖动出来的框 + 随机框，保证既有正样本也有负样本。"""
    out = []
    for i, (gt, (h, w)) in enumerate(zip(GT_BBOXES, IMG_SHAPES)):
        g = torch.Generator().manual_seed(seed + i)
        jitter = gt.repeat(20, 1) + torch.randn(len(gt) * 20, 4, generator=g) * 6
        out.append(torch.cat([jitter, random_boxes(100, h, w, seed=seed + 10 + i)]).clamp(min=0))
    return out


def make_bbox_head(**kw):
    torch.manual_seed(0)
    head = Shared2FCBBoxHead(**{k: v for k, v in BBOX_HEAD.items() if k != 'type'}, **kw)
    head.init_weights()
    return head


# ------------------------------------------------------------------ bbox2roi / extractor
def test_bbox2roi():
    rois = bbox2roi([torch.ones(2, 4), torch.zeros(0, 4), torch.full((3, 4), 2.)])
    assert rois.shape == (5, 5)
    assert rois[:, 0].tolist() == [0, 0, 2, 2, 2] and torch.equal(rois[2:, 1:], torch.full((3, 4), 2.))


def test_extractor_map_levels():
    ext = SingleRoIExtractor(**{k: v for k, v in EXTRACTOR.items() if k != 'type'})
    sides = torch.tensor([50., 111.9, 112., 224., 448., 1000.])
    rois = torch.stack([torch.zeros(6), torch.zeros(6), torch.zeros(6), sides, sides], dim=1)
    assert ext.map_roi_levels(rois, 4).tolist() == [0, 0, 1, 2, 3, 3]


def test_extractor_forward_matches_roi_align():
    ext = SingleRoIExtractor(**{k: v for k, v in EXTRACTOR.items() if k != 'type'})
    feats = fpn_feats(2, 256, 192, strides=(4, 8, 16, 32))
    rois = torch.tensor([[0, 10., 10., 60., 50.], [1, 0., 0., 150., 180.]])   # 分别落在 P2 / P3
    out = ext(feats, rois)
    assert out.shape == (2, 256, 7, 7)
    ref0 = torchvision.ops.roi_align(feats[0], rois[:1], 7, 1 / 4, 0, aligned=True)
    ref1 = torchvision.ops.roi_align(feats[1], rois[1:], 7, 1 / 8, 0, aligned=True)
    assert torch.allclose(out[:1], ref0) and torch.allclose(out[1:], ref1)
    assert ext(feats, rois[:0]).shape == (0, 256, 7, 7)


def test_extractor_vs_mmdet():
    pytest.importorskip('mmdet.models')
    from mmdet.models.roi_heads.roi_extractors import SingleRoIExtractor as MMDetSingleRoIExtractor
    ref = MMDetSingleRoIExtractor(roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                             out_channels=256, featmap_strides=[4, 8, 16, 32])
    mine = SingleRoIExtractor(**{k: v for k, v in EXTRACTOR.items() if k != 'type'})
    feats = fpn_feats(2, 256, 192, strides=(4, 8, 16, 32))
    rois = bbox2roi(proposals())
    assert torch.allclose(mine(feats, rois), ref(feats, rois), atol=1e-5)


# ------------------------------------------------------------------ bbox head
def test_bbox_head_layers_and_forward():
    head = make_bbox_head()
    assert set(head.state_dict()) == {f'{n}.{p}' for n in ('shared_fcs.0', 'shared_fcs.1', 'fc_cls', 'fc_reg')
                                      for p in ('weight', 'bias')}
    cls_score, bbox_pred = head(torch.randn(7, 256, 7, 7))
    assert cls_score.shape == (7, NUM_CLASSES + 1) and bbox_pred.shape == (7, NUM_CLASSES * 4)
    assert torch.all(head.fc_reg.bias == 0) and head.fc_reg.weight.std() < 0.002


def _fake_sampling_result():
    pos_priors = torch.tensor([[0., 0., 10., 10.], [5., 5., 20., 20.]])
    neg_priors = torch.tensor([[50., 50., 60., 60.], [70., 70., 90., 90.], [0., 0., 3., 3.]])
    gt = torch.tensor([[1., 1., 11., 11.], [4., 4., 22., 22.]])
    return SamplingResult(pos_inds=torch.tensor([0, 1]), neg_inds=torch.tensor([2, 3, 4]),
                          pos_priors=pos_priors, neg_priors=neg_priors, pos_gt_bboxes=gt,
                          pos_gt_labels=torch.tensor([2, 4]), pos_assigned_gt_inds=torch.tensor([0, 1]),
                          pos_is_gt=torch.tensor([False, False]))


def test_bbox_head_targets_and_loss():
    head = make_bbox_head()
    sr = _fake_sampling_result()
    labels, label_w, bbox_t, bbox_w = head.get_targets([sr, sr])
    assert labels.tolist() == [2, 4, 5, 5, 5] * 2 and torch.all(label_w == 1)
    assert torch.allclose(bbox_t[:2], head.bbox_coder.encode(sr.pos_priors, sr.pos_gt_bboxes))
    assert torch.all(bbox_t[2:5] == 0) and bbox_w[:, 0].tolist() == [1, 1, 0, 0, 0] * 2

    cls_score = torch.zeros(10, NUM_CLASSES + 1)
    bbox_pred = torch.zeros(10, NUM_CLASSES * 4)
    bbox_pred.view(10, NUM_CLASSES, 4)[0, 2] = 1.0        # 只在 GT 类别那一组写值
    bbox_pred.view(10, NUM_CLASSES, 4)[0, 3] = 100.0      # 其他类别的值不应该影响 loss
    losses = head.loss(cls_score, bbox_pred, labels, label_w, bbox_t, bbox_w)
    assert torch.isclose(losses['loss_cls'], torch.log(torch.tensor(6.)))
    expected = ((bbox_pred.view(10, 5, 4)[[0, 1, 5, 6], [2, 4, 2, 4]] - bbox_t[[0, 1, 5, 6]]).abs().sum()) / 10
    assert torch.isclose(losses['loss_bbox'], expected)
    assert 0 <= float(losses['acc']) <= 100


def test_bbox_head_predict():
    head = make_bbox_head()
    rois = bbox2roi(proposals()[:1])
    cls_score = torch.randn(len(rois), NUM_CLASSES + 1) * 3
    bbox_pred = torch.randn(len(rois), NUM_CLASSES * 4) * 0.1
    bboxes, scores, labels = head.predict_single(rois, cls_score, bbox_pred, IMG_SHAPES[0],
                                                 SCALE_FACTORS[0], True, TEST)
    assert len(bboxes) <= 100 and (scores > 0.05).all() and labels.max() < NUM_CLASSES
    assert (bboxes[:, 0::2] <= IMG_SHAPES[0][1] / 2).all() and (bboxes[:, 1::2] <= IMG_SHAPES[0][0] / 2).all()
    empty = head.predict_single(rois[:0], cls_score[:0], bbox_pred[:0], IMG_SHAPES[0], SCALE_FACTORS[0], True, TEST)
    assert [e.shape[0] for e in empty] == [0, 0, 0]


def test_bbox_head_vs_mmdet():
    pytest.importorskip('mmdet.models')
    from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead as MMDetShared2FCBBoxHead
    from mmengine.config import ConfigDict
    with scope('mmdet'):
        ref = MMDetShared2FCBBoxHead(
            in_channels=256, fc_out_channels=1024, roi_feat_size=7, num_classes=NUM_CLASSES,
            bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.] * 4, target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))
    mine = make_bbox_head()
    mine.load_state_dict(ref.state_dict())
    x = torch.randn(10, 256, 7, 7)
    (c, b), (rc, rb) = mine(x), ref(x)
    assert torch.allclose(c, rc, atol=1e-5) and torch.allclose(b, rb, atol=1e-5)

    sr = _fake_sampling_result()
    rois = bbox2roi([sr.priors, sr.priors])
    ref_losses = ref.loss_and_target(rc, rb, rois, [sr, sr], ConfigDict(pos_weight=-1))['loss_bbox']
    losses = mine.loss(c, b, *mine.get_targets([sr, sr]))
    for k in ('loss_cls', 'loss_bbox', 'acc'):
        assert torch.allclose(losses[k].reshape(-1), ref_losses[k].reshape(-1), rtol=1e-4), k

    rois = bbox2roi(proposals()[:1])
    cls_score = torch.randn(len(rois), NUM_CLASSES + 1) * 3
    bbox_pred = torch.randn(len(rois), NUM_CLASSES * 4) * 0.1
    r = ref._predict_by_feat_single(rois, cls_score, bbox_pred,
                                    dict(img_shape=IMG_SHAPES[0], scale_factor=SCALE_FACTORS[0]),
                                    rescale=True, rcnn_test_cfg=ConfigDict(TEST))
    bboxes, scores, labels = mine.predict_single(rois, cls_score, bbox_pred, IMG_SHAPES[0],
                                                 SCALE_FACTORS[0], True, TEST)
    assert torch.allclose(bboxes, r.bboxes, atol=1e-4) and torch.allclose(scores, r.scores, atol=1e-6)
    assert torch.equal(labels, r.labels)


# ------------------------------------------------------------------ standard roi head
def make_roi_head():
    torch.manual_seed(0)
    head = StandardRoIHead(EXTRACTOR, BBOX_HEAD, train_cfg=TRAIN, test_cfg=TEST)
    head.init_weights()
    return head


def test_roi_head_loss_and_predict():
    head = make_roi_head()
    feats = fpn_feats(2, 256, 192)
    losses = head.loss(feats, proposals(), GT_BBOXES, GT_LABELS)
    assert set(losses) == {'loss_cls', 'acc', 'loss_bbox'}
    assert losses['loss_bbox'] > 0
    assert torch.isclose(losses['loss_cls'], torch.log(torch.tensor(6.)), atol=0.1)  # 刚初始化，近似均匀分布
    (losses['loss_cls'] + losses['loss_bbox']).backward()
    assert head.bbox_head.fc_cls.weight.grad is not None

    results = head.predict(feats, proposals(), IMG_SHAPES, SCALE_FACTORS, rescale=True,
                           cfg=dict(TEST, score_thr=0.0))
    assert len(results) == 2 and all(len(r[0]) == len(r[1]) == len(r[2]) for r in results)
    assert len(results[0][0]) > 0
    empty = head.predict(feats, [p[:0] for p in proposals()], IMG_SHAPES, SCALE_FACTORS)
    assert [len(r[0]) for r in empty] == [0, 0]


def test_roi_head_vs_mmdet():
    pytest.importorskip('mmdet.models')
    from mmdet.models.roi_heads import StandardRoIHead as MMDetStandardRoIHead
    from mmdet.structures import DetDataSample
    from mmengine.config import ConfigDict
    from mmengine.structures import InstanceData
    with scope('mmdet'):
        ref = MMDetStandardRoIHead(
            bbox_roi_extractor=dict(type='SingleRoIExtractor',
                                    roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                                    out_channels=256, featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024, roi_feat_size=7,
                           num_classes=NUM_CLASSES,
                           bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.] * 4,
                                           target_stds=[0.1, 0.1, 0.2, 0.2]),
                           reg_class_agnostic=False,
                           loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                           loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            train_cfg=ConfigDict(
                assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5,
                              match_low_quality=False, ignore_iof_thr=-1),
                sampler=dict(type='RandomSampler', num=64, pos_fraction=0.25, neg_pos_ub=-1,
                             add_gt_as_proposals=True),
                pos_weight=-1, debug=False),
            test_cfg=ConfigDict(TEST))
    mine = make_roi_head()
    mine.load_state_dict(ref.state_dict())
    with torch.no_grad():                    # 让分类更 "尖"，否则 softmax 后都低于 score_thr，没东西可比
        ref.bbox_head.fc_cls.weight.normal_(0, 0.05)
        mine.bbox_head.fc_cls.weight.copy_(ref.bbox_head.fc_cls.weight)
    feats = fpn_feats(2, 256, 192)
    samples = []
    for gt, lb, shape, sf in zip(GT_BBOXES, GT_LABELS, IMG_SHAPES, SCALE_FACTORS):
        ds = DetDataSample(metainfo=dict(img_shape=shape, scale_factor=sf))
        ds.gt_instances = InstanceData(bboxes=gt, labels=lb)
        samples.append(ds)

    torch.manual_seed(7)
    ref_losses = ref.loss(feats, [InstanceData(bboxes=p, scores=torch.ones(len(p))) for p in proposals()], samples)
    torch.manual_seed(7)
    losses = mine.loss(feats, proposals(), GT_BBOXES, GT_LABELS)
    for k in ('loss_cls', 'loss_bbox', 'acc'):
        assert torch.allclose(losses[k].reshape(-1), ref_losses[k].reshape(-1), rtol=1e-4), k

    ref_res = ref.predict(feats, [InstanceData(bboxes=p, scores=torch.ones(len(p))) for p in proposals()],
                          samples, rescale=True)
    for (bboxes, scores, labels), r in zip(mine.predict(feats, proposals(), IMG_SHAPES, SCALE_FACTORS), ref_res):
        assert len(bboxes) > 0 and bboxes.shape == r.bboxes.shape
        assert torch.allclose(bboxes, r.bboxes, atol=1e-3) and torch.equal(labels, r.labels)
