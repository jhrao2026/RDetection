"""mini_mm.hooks —— 练习：Hook 机制（Hook 基类已完成，两个具体 Hook 待实现）

================================================================================
为什么要有 Hook？
    训练循环本身只有几行：取数据 → 前向 → 反传 → 更新。但真实项目要在循环里插
    很多"杂事"：打印日志、存权重、调学习率、EMA、可视化、早停……
    如果全写进循环，Runner 会变成几千行的意大利面。

    Hook 的做法是：Runner 只在固定的"位点"广播一声（before_train_epoch、
    after_train_iter ...），所有注册的 Hook 按优先级依次响应。想加功能就写个
    Hook，Runner 一行都不改 —— 这就是"开闭原则"。

参考源码（写完后再对照）
    mmengine/hooks/hook.py            Hook 基类与全部位点
    mmengine/hooks/logger_hook.py     LoggerHook
    mmengine/hooks/checkpoint_hook.py CheckpointHook（真实版本还管理 max_keep_ckpts、save_best）

完成标准
    pytest tests/test_mini_mm_runner.py（Hook 和 Runner 一起测）
================================================================================
"""

from __future__ import annotations

import os.path as osp  # noqa: F401  练习 7 会用到

from .root import HOOKS

# 优先级：数字越小越先执行。mmengine 用字符串 'VERY_HIGH' / 'NORMAL' / 'LOW' 等映射到这些值。
PRIORITY = dict(HIGHEST=0, VERY_HIGH=10, HIGH=30, ABOVE_NORMAL=40, NORMAL=50,
                BELOW_NORMAL=60, LOW=70, VERY_LOW=90, LOWEST=100)


class Hook:
    """所有 Hook 的基类（已完成）。每个方法都是一个"位点"，默认什么都不做。

    子类只需覆盖关心的位点。``runner`` 参数让 Hook 能读写训练状态
    （runner.model / runner.optimizer / runner.epoch / runner.iter / runner.work_dir ...）。

    Runner.train() 中的调用顺序：

        before_run
        for epoch:
            before_train_epoch
            for batch_idx, batch in enumerate(dataloader):
                before_train_iter(batch_idx, batch)
                ... 前向 / 反传 / 更新 ...
                after_train_iter(batch_idx, batch, outputs)
            after_train_epoch
        after_run
    """

    priority = 'NORMAL'

    def before_run(self, runner) -> None:
        pass

    def after_run(self, runner) -> None:
        pass

    def before_train_epoch(self, runner) -> None:
        pass

    def after_train_epoch(self, runner) -> None:
        pass

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        pass

    def after_train_iter(self, runner, batch_idx: int, data_batch=None,
                         outputs: dict | None = None) -> None:
        pass


@HOOKS.register_module()
class LoggerHook(Hook):
    """每隔 ``interval`` 个 iter 打印一次 loss。

    练习 6。要求：
    - 在 ``after_train_iter`` 中，当 ``(batch_idx + 1) % self.interval == 0`` 时打印一行；
    - 格式 **必须** 以这个开头（测试会检查）：

          Epoch [{epoch}][{cur}/{total}]

      其中 epoch 从 1 开始（runner.epoch 从 0 开始！），cur = batch_idx + 1，
      total = len(runner.train_dataloader)；后面跟上 outputs 里每一项，例如：

          Epoch [1][2/4]  loss: 0.5321  loss_cls: 0.4000  loss_bbox: 0.1321

    - outputs 里的值可能是 tensor，用 ``float(v)`` 转成数字再格式化（``:.4f``）。
    - 同时把这一行 append 到 ``runner.log_history``（一个 list，Runner 负责创建），
      方便测试和之后画 loss 曲线。

    思考：为什么 LoggerHook 的优先级是 BELOW_NORMAL，而不是 NORMAL？
    （提示：如果有另一个 NORMAL 的 Hook 在 after_train_iter 里往 outputs 里塞了东西呢？）
    """

    priority = 'BELOW_NORMAL'

    def __init__(self, interval: int = 10):
        self.interval = interval

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        # TODO(6.1) 判断是否到了打印时机
        # TODO(6.2) 拼出日志字符串，print 出来，并 append 到 runner.log_history
        raise NotImplementedError('LoggerHook.after_train_iter')


@HOOKS.register_module()
class CheckpointHook(Hook):
    """每隔 ``interval`` 个 epoch 保存一次权重。

    练习 7。要求：
    - 在 ``after_train_epoch`` 中，当 ``(runner.epoch + 1) % self.interval == 0`` 时保存；
    - 文件路径：``osp.join(runner.work_dir, f'epoch_{runner.epoch + 1}.pth')``
    - 用 ``torch.save`` 保存一个 dict：

          dict(epoch=runner.epoch + 1,
               iter=runner.iter,
               state_dict=runner.model.state_dict(),
               optimizer=runner.optimizer.state_dict())

    思考：
    - 为什么要连 optimizer 的状态一起存？（提示：SGD momentum / Adam 的一阶二阶矩，
      以及 ``--resume`` 断点续训）
    - mmengine 的 CheckpointHook 优先级是 VERY_LOW，为什么要排在最后？
    """

    priority = 'VERY_LOW'

    def __init__(self, interval: int = 1):
        self.interval = interval

    def after_train_epoch(self, runner):
        # TODO(7.1) 判断是否到了保存时机
        # TODO(7.2) import torch，拼路径，torch.save(...)
        raise NotImplementedError('CheckpointHook.after_train_epoch')

