import os.path as osp

import pytest

torch = pytest.importorskip('torch')
from torch import nn  # noqa: E402

from mini_mm.hooks import CheckpointHook, Hook, LoggerHook  # noqa: E402
from mini_mm.root import DATASETS, HOOKS, MODELS  # noqa: E402
from mini_mm.runner import Runner, parse_losses  # noqa: E402


# ------------------------------------------------------------- 测试用的玩具组件
class ToyDataset(torch.utils.data.Dataset):
    """y = x · w_true，用来验证训练真的让 loss 下降。"""

    def __init__(self, num=32, in_dim=4):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(num, in_dim, generator=g)
        self.y = self.x @ torch.arange(1., in_dim + 1).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class ToyRegressor(nn.Module):
    def __init__(self, in_dim=4):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, inputs, targets):
        pred = self.fc(inputs)
        return dict(loss_mse=((pred - targets) ** 2).mean(),
                    mae=(pred - targets).abs().mean())


class RecordHook(Hook):
    """记录所有位点的调用顺序。"""

    def __init__(self, priority='NORMAL', name='rec', record=None):
        self.priority = priority
        self.name = name
        self.record = record if record is not None else []

    def _log(self, what):
        self.record.append((self.name, what))

    def before_run(self, runner): self._log('before_run')
    def after_run(self, runner): self._log('after_run')
    def before_train_epoch(self, runner): self._log('before_train_epoch')
    def after_train_epoch(self, runner): self._log('after_train_epoch')
    def before_train_iter(self, runner, batch_idx, data_batch=None): self._log('before_train_iter')

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        assert 'loss' in outputs
        self._log('after_train_iter')


for _m, _r in ((ToyDataset, DATASETS), (ToyRegressor, MODELS), (RecordHook, HOOKS)):
    if _m.__name__ not in _r:
        _r.register_module(module=_m)


def _toy_runner(tmp_path, hooks=None, max_epochs=2, batch_size=8):
    model = ToyRegressor()
    loader = torch.utils.data.DataLoader(ToyDataset(), batch_size=batch_size)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    return Runner(model, loader, opt, max_epochs, str(tmp_path / 'work'), hooks)


# ------------------------------------------------------------- 练习 8
def test_parse_losses():
    losses = dict(loss_cls=torch.tensor([1., 3.]), loss_bbox=[torch.tensor(1.), torch.tensor(2.)],
                  acc=torch.tensor(0.5))
    loss, log_vars = parse_losses(losses)
    assert torch.isclose(loss, torch.tensor(5.))
    assert list(log_vars)[0] == 'loss'
    assert log_vars == dict(loss=5.0, loss_cls=2.0, loss_bbox=3.0, acc=0.5)
    assert all(isinstance(v, float) for v in log_vars.values())


# ------------------------------------------------------------- 练习 9 / 10
def test_register_hook_priority_and_stability(tmp_path):
    a = RecordHook('NORMAL', 'a')
    b = RecordHook('HIGH', 'b')
    c = RecordHook('NORMAL', 'c')
    d = RecordHook('LOWEST', 'd')
    runner = _toy_runner(tmp_path, hooks=[a, d, c, b])
    assert [h.name for h in runner.hooks] == ['b', 'a', 'c', 'd']

    runner.register_hook(dict(type='RecordHook', priority='VERY_HIGH', name='e'))
    assert [h.name for h in runner.hooks] == ['e', 'b', 'a', 'c', 'd']


def test_call_hook(tmp_path):
    rec = []
    runner = _toy_runner(tmp_path, hooks=[RecordHook(name='x', record=rec)])
    runner.call_hook('before_run')
    assert rec == [('x', 'before_run')]


# ------------------------------------------------------------- 练习 11
def test_train_hook_order_and_counters(tmp_path):
    rec = []
    runner = _toy_runner(tmp_path, hooks=[RecordHook(record=rec)], max_epochs=2, batch_size=16)
    runner.train()
    names = [w for _, w in rec]
    one_iter = ['before_train_iter', 'after_train_iter']
    one_epoch = ['before_train_epoch'] + one_iter * 2 + ['after_train_epoch']
    assert names == ['before_run'] + one_epoch * 2 + ['after_run']
    assert runner.epoch == 2 and runner.iter == 4


def test_train_reduces_loss(tmp_path):
    runner = _toy_runner(tmp_path, max_epochs=1)
    model = runner.model
    x, y = ToyDataset()[:]
    before = model(x, y)['loss_mse'].item()
    runner.max_epochs = 20
    runner.train()
    after = model(x, y)['loss_mse'].item()
    assert after < before * 0.1


# ------------------------------------------------------------- 练习 6 / 7
def test_logger_hook(tmp_path, capsys):
    runner = _toy_runner(tmp_path, hooks=[LoggerHook(interval=2)], max_epochs=1, batch_size=8)
    runner.train()
    assert len(runner.log_history) == 2
    assert runner.log_history[0].startswith('Epoch [1][2/4]')
    assert 'loss' in runner.log_history[0] and 'mae' in runner.log_history[0]
    assert 'Epoch [1][4/4]' in capsys.readouterr().out


def test_checkpoint_hook(tmp_path):
    runner = _toy_runner(tmp_path, hooks=[CheckpointHook(interval=2)], max_epochs=4)
    runner.train()
    assert not osp.exists(osp.join(runner.work_dir, 'epoch_1.pth'))
    path = osp.join(runner.work_dir, 'epoch_4.pth')
    ckpt = torch.load(path, weights_only=False)
    assert ckpt['epoch'] == 4 and ckpt['iter'] == 16
    assert set(ckpt) >= {'state_dict', 'optimizer'}
    fresh = ToyRegressor()
    fresh.load_state_dict(ckpt['state_dict'])
    assert torch.equal(fresh.fc.weight, runner.model.fc.weight)


# ------------------------------------------------------------- 练习 12
def test_from_cfg(tmp_path):
    cfg = dict(
        model=dict(type='ToyRegressor', in_dim=4),
        train_dataloader=dict(batch_size=8, shuffle=True, dataset=dict(type='ToyDataset', num=32)),
        optimizer=dict(type='SGD', lr=0.1),
        max_epochs=1,
        work_dir=str(tmp_path / 'from_cfg'),
        default_hooks=dict(logger=dict(type='LoggerHook', interval=1),
                           checkpoint=dict(type='CheckpointHook', interval=1)),
        custom_hooks=[dict(type='RecordHook', priority='HIGH', name='custom')],
    )
    runner = Runner.from_cfg(cfg)
    assert isinstance(runner.model, ToyRegressor)
    assert isinstance(runner.optimizer, torch.optim.SGD)
    assert [type(h).__name__ for h in runner.hooks] == ['RecordHook', 'LoggerHook', 'CheckpointHook']
    runner.train()
    assert osp.exists(osp.join(cfg['work_dir'], 'epoch_1.pth'))
    assert cfg['train_dataloader']['dataset'] == dict(type='ToyDataset', num=32)  # 配置不能被修改
