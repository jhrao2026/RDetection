import pytest

from mini_mm.config import Config, ConfigDict, merge_dict


# ------------------------------------------------------------- 练习 1 ConfigDict
def test_config_dict_attr_access():
    cfg = ConfigDict(model=dict(backbone=dict(depth=50)),
                     pipeline=[dict(type='Resize'), dict(type='Flip')])
    assert cfg.model.backbone.depth == 50
    assert isinstance(cfg.model, ConfigDict)
    assert cfg.pipeline[1].type == 'Flip'

    cfg.optimizer = dict(lr=0.1)
    assert cfg['optimizer'].lr == 0.1

    assert not hasattr(cfg, 'not_exist')
    with pytest.raises(AttributeError):
        cfg.not_exist


# ------------------------------------------------------------- 练习 2 merge_dict
def test_merge_dict_recursive():
    base = dict(optimizer=dict(type='SGD', lr=0.01, momentum=0.9), epochs=12)
    child = dict(optimizer=dict(lr=0.02), work_dir='a')
    out = merge_dict(base, child)
    assert out == dict(optimizer=dict(type='SGD', lr=0.02, momentum=0.9),
                       epochs=12, work_dir='a')
    # 入参不能被修改
    assert base['optimizer']['lr'] == 0.01 and child == dict(optimizer=dict(lr=0.02), work_dir='a')


def test_merge_dict_delete():
    base = dict(optimizer=dict(type='SGD', lr=0.01, momentum=0.9))
    child = dict(optimizer=dict(_delete_=True, type='AdamW', lr=1e-4))
    out = merge_dict(base, child)
    assert out == dict(optimizer=dict(type='AdamW', lr=1e-4))
    assert '_delete_' in child['optimizer']  # 入参不能被修改


def test_merge_dict_non_dict_overrides_dict():
    assert merge_dict(dict(a=dict(b=1)), dict(a=3)) == dict(a=3)


# ------------------------------------------------------------- 练习 3/4 fromfile
def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path)


def test_fromfile_with_base(tmp_path):
    _write(tmp_path / '_base_' / 'model.py',
           "model = dict(type='RetinaNet', backbone=dict(type='ResNet', depth=50))\n")
    _write(tmp_path / '_base_' / 'schedule.py',
           "import math\noptimizer = dict(type='SGD', lr=0.01)\nmax_epochs = 12\n")
    path = _write(tmp_path / 'exp.py',
                  "_base_ = ['./_base_/model.py', './_base_/schedule.py']\n"
                  "model = dict(backbone=dict(depth=101))\n"
                  "optimizer = dict(lr=0.02)\n")
    cfg = Config.fromfile(path)
    assert cfg.model.type == 'RetinaNet'
    assert cfg.model.backbone.depth == 101 and cfg.model.backbone.type == 'ResNet'
    assert cfg.optimizer.lr == 0.02 and cfg.max_epochs == 12
    assert 'math' not in cfg and '_base_' not in cfg
    assert not any(k.startswith('__') for k in cfg.to_dict())


def test_fromfile_nested_base_and_str(tmp_path):
    _write(tmp_path / 'a.py', 'x = 1\ny = dict(z=1)\n')
    _write(tmp_path / 'sub' / 'b.py', "_base_ = '../a.py'\ny = dict(w=2)\n")
    cfg = Config.fromfile(_write(tmp_path / 'sub' / 'c.py', "_base_ = './b.py'\nx = 3\n"))
    assert cfg.x == 3 and cfg.y == dict(z=1, w=2)


def test_fromfile_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.fromfile(str(tmp_path / 'nope.py'))


# ------------------------------------------------------------- 练习 5 merge_from_dict
def test_merge_from_dict(tmp_path):
    cfg = Config.fromfile(_write(
        tmp_path / 'a.py', "model = dict(backbone=dict(depth=50, frozen=1))\noptimizer = dict(lr=0.01)\n"))
    cfg.merge_from_dict({'model.backbone.depth': 101, 'optimizer.lr': 0.1, 'seed': 0})
    assert cfg.model.backbone.depth == 101 and cfg.model.backbone.frozen == 1
    assert cfg.optimizer.lr == 0.1 and cfg.seed == 0
