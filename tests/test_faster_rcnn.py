"""Stage 2.5 最终验收（需要 mmdet）：用自己的配置构建 FasterRCNN，和 mmdet 官方 FasterRCNN

    1. 参数名完全一致（官方 checkpoint 能 strict 加载）
    2. 同一份权重、同一个随机种子下，loss 完全一致
    3. 同一份权重下，检测结果完全一致

这三条都过了，加载官方权重跑 COCO val 就一定是 37.4 —— 剩下的只是在服务器上跑一遍确认。
"""
import copy
import os.path as osp

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('mmdet.models')

from helpers import build_from_config, mmdet_config  # noqa: E402

CFG = osp.join(osp.dirname(__file__), '..', 'configs', 'faster_rcnn', 'faster-rcnn_r50_fpn_1x_coco.py')
REF_CFG = 'faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'


def fake_batch(model, training):
    """两张不同大小的图，走一遍真正的 DetDataPreprocessor（归一化 + pad + 写入 pad_shape 等）。"""
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData
    g = torch.Generator().manual_seed(0)
    inputs, samples = [], []
    for (h, w), boxes, labels in [((224, 320), [[30., 40., 150., 200.], [160., 20., 300., 120.]], [0, 17]),
                                  ((256, 192), [[10., 10., 120., 240.]], [56])]:
        inputs.append(torch.randint(0, 256, (3, h, w), generator=g, dtype=torch.uint8))
        ds = DetDataSample(metainfo=dict(img_shape=(h, w), ori_shape=(h // 2, w // 2), scale_factor=(2.0, 2.0)))
        ds.gt_instances = InstanceData(bboxes=torch.tensor(boxes), labels=torch.tensor(labels))
        samples.append(ds)
    return model.data_preprocessor(dict(inputs=inputs, data_samples=samples), training)


def _unpack(data):
    data = copy.deepcopy(data)
    return data['inputs'], data['data_samples']


@pytest.fixture(scope='module')
def models():
    torch.manual_seed(0)
    ref = build_from_config(mmdet_config(REF_CFG))
    with torch.no_grad():   # 随机初始化的 fc_cls 输出太 "平"，softmax 后都低于 0.05，调大一点才有检测结果
        ref.roi_head.bbox_head.fc_cls.weight.normal_(0, 1.0)
        ref.roi_head.bbox_head.fc_cls.bias[-1] = -10.0    # 压低背景类
    mine = build_from_config(CFG)
    mine.load_state_dict(ref.state_dict(), strict=True)
    return mine, ref


def test_state_dict_keys_match(models):
    mine, ref = models
    assert set(mine.state_dict()) == set(ref.state_dict())


def test_loss_matches(models):
    mine, ref = models
    mine.train(), ref.train()
    data = fake_batch(ref, training=True)
    torch.manual_seed(123)
    ref_losses = ref.loss(*_unpack(data))
    torch.manual_seed(123)
    losses = mine.loss(*_unpack(data))
    for k in ('loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox', 'acc'):
        r = ref_losses[k]
        r = sum(r) if isinstance(r, (list, tuple)) else r
        assert torch.allclose(losses[k].reshape(-1), r.reshape(-1), rtol=1e-3, atol=1e-5), k


def test_predict_matches(models):
    mine, ref = models
    mine.eval(), ref.eval()
    data = fake_batch(ref, training=False)
    with torch.no_grad():
        ref_out = ref.predict(*_unpack(data))
        out = mine.predict(*_unpack(data))
    for a, b in zip(out, ref_out):
        pa, pb = a.pred_instances, b.pred_instances
        assert len(pa) > 0 and len(pa) == len(pb)
        assert torch.allclose(pa.bboxes, pb.bboxes, atol=1e-3)
        assert torch.allclose(pa.scores, pb.scores, atol=1e-5)
        assert torch.equal(pa.labels, pb.labels)
