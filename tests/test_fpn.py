"""Stage 2.2b FPN 的验收测试。"""
from collections import OrderedDict

import pytest

torch = pytest.importorskip('torch')
torchvision = pytest.importorskip('torchvision')
import torch.nn.functional as F  # noqa: E402

from rdet.models.necks import RFPN  # noqa: E402

IN_CHANNELS = [256, 512, 1024, 2048]


def _feats(batch=2, h=64, w=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(batch, c, h // 2 ** i, w // 2 ** i, generator=g)
                 for i, c in enumerate(IN_CHANNELS))


def test_faster_rcnn_fpn_shapes():
    outs = RFPN(IN_CHANNELS, 256, num_outs=5)(_feats())
    assert [o.shape for o in outs] == [(2, 256, 64 // 2 ** i, 48 // 2 ** i) for i in range(4)] + [(2, 256, 4, 3)]
    assert torch.equal(outs[4], F.max_pool2d(outs[3], 1, stride=2))


def test_matches_torchvision_fpn():
    from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
    tv = FeaturePyramidNetwork(IN_CHANNELS, 256, extra_blocks=LastLevelMaxPool())
    mine = RFPN(IN_CHANNELS, 256, num_outs=5)
    sd = {}
    for k, v in tv.state_dict().items():   # inner_blocks.i.0.weight -> lateral_convs.i.conv.weight
        name, i, _, p = k.split('.')
        sd[f"{dict(inner_blocks='lateral_convs', layer_blocks='fpn_convs')[name]}.{i}.conv.{p}"] = v
    mine.load_state_dict(sd)
    feats = _feats(h=60, w=44)  # 不是 2 的整数次幂，检验上采样用的是 size 而不是 scale_factor
    with torch.no_grad():
        ref = list(tv(OrderedDict((str(i), f) for i, f in enumerate(feats))).values())
        outs = mine(feats)
    assert len(outs) == len(ref) == 5
    for a, b in zip(outs, ref):
        assert torch.allclose(a, b, atol=1e-5)


def test_retinanet_style_extra_convs():
    fpn = RFPN(IN_CHANNELS, 256, num_outs=5, start_level=1, add_extra_convs='on_input')
    assert len(fpn.lateral_convs) == 3 and len(fpn.fpn_convs) == 5
    assert fpn.fpn_convs[3].conv.in_channels == 2048 and fpn.fpn_convs[4].conv.in_channels == 256
    outs = fpn(_feats())
    assert [o.shape[2:] for o in outs] == [(32, 24), (16, 12), (8, 6), (4, 3), (2, 2)]


@pytest.mark.parametrize('kwargs', [
    dict(num_outs=5),
    dict(num_outs=5, start_level=1, add_extra_convs='on_input'),
    dict(num_outs=5, start_level=1, add_extra_convs='on_output', relu_before_extra_convs=True),
])
def test_fpn_vs_mmdet(kwargs):
    pytest.importorskip('mmdet.models')
    from mmdet.models.necks import FPN
    ref = FPN(IN_CHANNELS, 256, **kwargs).eval()
    mine = RFPN(IN_CHANNELS, 256, **kwargs).eval()
    mine.load_state_dict(ref.state_dict())          # 命名必须完全一致（strict=True）
    feats = _feats(h=60, w=44)
    with torch.no_grad():
        for a, b in zip(mine(feats), ref(feats)):
            assert torch.allclose(a, b, atol=1e-5)
