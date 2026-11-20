import pytest

from mini_mm import Registry


def _make_registry():
    MODELS = Registry('model')

    @MODELS.register_module()
    class ResNet:
        def __init__(self, depth, num_classes=1000):
            self.depth = depth
            self.num_classes = num_classes

    @MODELS.register_module(name=['FPN', 'FeaturePyramid'])
    class FPN:
        def __init__(self, out_channels=256):
            self.out_channels = out_channels

    return MODELS, ResNet, FPN


def test_register_and_build():
    MODELS, ResNet, _ = _make_registry()
    assert 'ResNet' in MODELS and len(MODELS) == 3
    net = MODELS.build(dict(type='ResNet', depth=50))
    assert isinstance(net, ResNet) and net.depth == 50


def test_alias_names():
    MODELS, _, FPN = _make_registry()
    assert MODELS.get('FPN') is FPN and MODELS.get('FeaturePyramid') is FPN


def test_default_args_do_not_override_cfg():
    MODELS, _, _ = _make_registry()
    net = MODELS.build(dict(type='ResNet', depth=18, num_classes=10), num_classes=80)
    assert net.num_classes == 10
    net = MODELS.build(dict(type='ResNet', depth=18), num_classes=80)
    assert net.num_classes == 80


def test_build_does_not_mutate_cfg():
    MODELS, _, _ = _make_registry()
    cfg = dict(type='ResNet', depth=50)
    MODELS.build(cfg)
    assert cfg == dict(type='ResNet', depth=50)


def test_build_with_class_type():
    MODELS, ResNet, _ = _make_registry()
    assert isinstance(MODELS.build(dict(type=ResNet, depth=34)), ResNet)


def test_duplicate_and_force():
    MODELS, _, _ = _make_registry()
    with pytest.raises(KeyError):
        @MODELS.register_module(name='ResNet')
        class Other:
            pass

    @MODELS.register_module(name='ResNet', force=True)
    class Other2:
        pass

    assert MODELS.get('ResNet') is Other2


def test_direct_register_and_errors():
    MODELS = Registry('model')

    class SGD:
        def __init__(self, lr):
            self.lr = lr

    MODELS.register_module(module=SGD)
    assert MODELS.build(dict(type='SGD', lr=0.1)).lr == 0.1
    with pytest.raises(KeyError):
        MODELS.build(dict(type='NotExist'))
    with pytest.raises(KeyError):
        MODELS.build(dict(lr=0.1))
    with pytest.raises(TypeError):
        MODELS.build(dict(type='SGD'))  # 缺少必需参数
