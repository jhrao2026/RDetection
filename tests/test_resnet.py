"""Stage 2.2a ResNet 的验收测试：和 torchvision 的 resnet50 用同一份权重，输出必须一致。"""
import pytest

torch = pytest.importorskip('torch')
torchvision = pytest.importorskip('torchvision')
from torch import nn  # noqa: E402

from rdet.models.backbones import RResNet  # noqa: E402
from rdet.registry import MODELS  # noqa: E402


def _tv_stages(tv, x):
    x = tv.maxpool(tv.relu(tv.bn1(tv.conv1(x))))
    outs = []
    for name in ('layer1', 'layer2', 'layer3', 'layer4'):
        x = getattr(tv, name)(x)
        outs.append(x)
    return outs


def test_state_dict_keys_match_torchvision():
    tv_keys = {k for k in torchvision.models.resnet50().state_dict() if not k.startswith('fc.')}
    assert set(RResNet(depth=50).state_dict()) == tv_keys


def test_outputs_match_torchvision():
    torch.manual_seed(0)
    tv = torchvision.models.resnet50()
    for m in tv.modules():   # 随机化 BN 统计量，否则 eval 模式下 BN 近似恒等，测试不够严格
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.uniform_(-0.1, 0.1)
            m.running_var.uniform_(0.5, 1.5)
    mine = RResNet(depth=50, frozen_stages=-1)
    mine.load_state_dict({k: v for k, v in tv.state_dict().items() if not k.startswith('fc.')})
    tv.eval(), mine.eval()
    x = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        outs = mine(x)
        ref = _tv_stages(tv, x)
    assert [o.shape[1:] for o in outs] == [(256, 16, 24), (512, 8, 12), (1024, 4, 6), (2048, 2, 3)]
    for a, b in zip(outs, ref):
        assert torch.allclose(a, b, atol=1e-4)


def test_out_indices_and_depth101():
    outs = RResNet(depth=101, out_indices=(1, 3)).eval()(torch.randn(1, 3, 64, 64))
    assert len(outs) == 2 and outs[0].shape[1] == 512 and outs[1].shape[1] == 2048


def test_frozen_stages_and_norm_eval():
    model = RResNet(depth=50, frozen_stages=1, norm_eval=True)
    model.train()
    assert model.training
    assert not any(p.requires_grad for p in model.conv1.parameters())
    assert not any(p.requires_grad for p in model.layer1.parameters())
    assert all(p.requires_grad for p in model.layer2.parameters())
    assert all(not m.training for m in model.modules() if isinstance(m, nn.BatchNorm2d))

    model = RResNet(depth=50, frozen_stages=-1, norm_eval=False).train()
    assert all(m.training for m in model.modules() if isinstance(m, nn.BatchNorm2d))


def test_registered():
    assert isinstance(MODELS.build(dict(type='RResNet', depth=50)), RResNet)


def test_init_weights_from_local_checkpoint(tmp_path):
    tv = torchvision.models.resnet50()
    path = tmp_path / 'r50.pth'
    torch.save({'state_dict': tv.state_dict()}, path)       # 模拟 mmdet 风格的 ckpt（外面包了一层）
    model = RResNet(depth=50, init_cfg=dict(type='Pretrained', checkpoint=str(path)))
    model.init_weights()
    assert torch.equal(model.layer3[2].conv2.weight, tv.layer3[2].conv2.weight)

    untouched = RResNet(depth=50)
    before = untouched.conv1.weight.clone()
    untouched.init_weights()                                # init_cfg=None：什么都不做
    assert torch.equal(untouched.conv1.weight, before)
