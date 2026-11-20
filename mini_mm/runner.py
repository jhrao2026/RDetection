"""mini_mm.runner —— 练习：最小训练执行器 Runner

================================================================================
Runner 是 OpenMMLab 的"总指挥"：它读配置、把各个组件 build 出来、跑训练循环、
在固定位点调用 Hook。理解了它，mmdet 的 tools/train.py 就只剩"解析命令行 +
Runner.from_cfg(cfg).train()"这两行。

本练习中的约定（mmengine 更复杂，但思想一致）：
    - dataloader 每次产出 ``(inputs, targets)``
    - 模型训练时调用 ``model(inputs, targets)``，返回 **loss 字典**，例如
      ``dict(loss_cls=tensor, loss_bbox=tensor, acc=tensor)``
      —— 这正是 mmdet 里 ``model.loss()`` 的返回形式：loss 由模型（head）自己算，
      Runner 不关心任务是分类还是检测。

参考源码（写完后再对照）
    mmengine/runner/runner.py      Runner.__init__ / from_cfg / register_hook / call_hook / train
    mmengine/runner/loops.py       EpochBasedTrainLoop.run / run_epoch / run_iter
    mmengine/model/base_model/base_model.py  BaseModel.parse_losses / train_step

完成标准
    pytest tests/test_mini_mm_runner.py 全部通过
建议顺序：parse_losses → register_hook → call_hook → train → from_cfg
================================================================================
"""

from __future__ import annotations

import os

from .hooks import PRIORITY, Hook  # noqa: F401
from .root import DATASETS, HOOKS, MODELS, OPTIMIZERS  # noqa: F401


def parse_losses(losses: dict):
    """把模型返回的 loss 字典变成 (总 loss, 用于打印的字典)。

    练习 8。规则（与 mmengine BaseModel.parse_losses 一致）：
    - 只有 **key 中包含 'loss'** 的项会加到总 loss 里参与反传；
      其他项（如 'acc'）只用于日志。
    - value 可能是 tensor，也可能是 tensor 的 list（多尺度 / 多层 decoder 的 loss），
      list 要先求和：``sum(v)``；tensor 统一 ``.mean()`` 成标量。
    - 返回：
        loss     : 所有 loss 项之和（tensor，可以 backward）
        log_vars : dict，第一个 key 是 'loss'（总和），后面依次是各项，值用 ``.item()`` 转成 float

    思考：为什么不让模型直接返回一个总 loss？（提示：日志里想分别看 loss_cls 和 loss_bbox）
    """
    # TODO(8.1) 遍历 losses，统一转成标量 tensor
    # TODO(8.2) 求和所有含 'loss' 的项
    # TODO(8.3) 构造 log_vars（'loss' 在最前）
    raise NotImplementedError('parse_losses')


class Runner:
    """最小 Runner。

    Args:
        model: nn.Module，满足上面的约定
        train_dataloader: 可迭代对象，产出 (inputs, targets)，需要支持 len()
        optimizer: torch 优化器
        max_epochs: 训练轮数
        work_dir: 日志和权重的保存目录，不存在要自动创建
        hooks: Hook 实例的 list
    """

    def __init__(self, model, train_dataloader, optimizer, max_epochs: int,
                 work_dir: str, hooks: list | None = None):
        # 已完成：保存组件与训练状态
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

        self.epoch = 0           # 当前 epoch（从 0 开始）
        self.iter = 0            # 全局 iter 计数（从 0 开始，每个 iter 结束后 +1）
        self.log_history: list[str] = []
        self._hooks: list[Hook] = []
        for hook in hooks or []:
            self.register_hook(hook)

    @property
    def hooks(self) -> list:
        return self._hooks

    # ------------------------------------------------------------- Hook 管理
    def register_hook(self, hook) -> None:
        """练习 9：按优先级把 hook 插入 self._hooks。

        - hook 可以是 Hook 实例，也可以是配置 dict（用 ``HOOKS.build`` 构建）；
        - 优先级取 ``PRIORITY[hook.priority]``，数字越小越靠前；
        - **优先级相同时，先注册的排在前面**（稳定）。
          提示：从后往前找第一个"优先级 <= 新 hook"的位置，插在它后面。

        思考：为什么要求"稳定"？如果两个 NORMAL 的 Hook 执行顺序随机，会有什么问题？
        """
        # TODO(9)
        raise NotImplementedError('Runner.register_hook')

    def call_hook(self, fn_name: str, **kwargs) -> None:
        """练习 10：依次调用每个 hook 的 ``fn_name`` 方法，传入 ``self`` 和 kwargs。

        提示：``getattr(hook, fn_name)(self, **kwargs)``
        """
        # TODO(10)
        raise NotImplementedError('Runner.call_hook')

    # ------------------------------------------------------------- 训练
    def train(self):
        """练习 11：训练主循环。严格按 hooks.Hook 文档里的调用顺序来。

        伪代码：

            model.train()
            call_hook('before_run')
            while self.epoch < self.max_epochs:
                call_hook('before_train_epoch')
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    call_hook('before_train_iter', batch_idx=..., data_batch=...)
                    losses = model(inputs, targets)
                    loss, log_vars = parse_losses(losses)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    call_hook('after_train_iter', batch_idx=..., data_batch=..., outputs=log_vars)
                    self.iter += 1
                call_hook('after_train_epoch')
                self.epoch += 1
            call_hook('after_run')
            return self.model

        思考：
        - mmengine 把"zero_grad / backward / step"封装进了 OptimWrapper。
          混合精度（AMP）、梯度累积、梯度裁剪都在那里实现，Runner 不用改。为什么这样设计更好？
        - 为什么 ``self.epoch += 1`` 放在 after_train_epoch **之后**？
          （提示：CheckpointHook 用 runner.epoch + 1 命名文件）
        """
        # TODO(11)
        raise NotImplementedError('Runner.train')

    # ------------------------------------------------------------- 从配置构建
    @classmethod
    def from_cfg(cls, cfg) -> 'Runner':
        """练习 12：从配置构建 Runner —— 这就是"配置驱动"最终落地的地方。

        配置的样子（见 tests/test_mini_mm_runner.py）：

            model = dict(type='ToyRegressor', in_dim=4)
            train_dataloader = dict(batch_size=8, shuffle=True,
                                    dataset=dict(type='ToyDataset', num=32))
            optimizer = dict(type='SGD', lr=0.1)
            max_epochs = 3
            work_dir = './work_dirs/toy'
            default_hooks = dict(logger=dict(type='LoggerHook', interval=1),
                                 checkpoint=dict(type='CheckpointHook', interval=1))
            custom_hooks = [dict(type='MyHook')]          # 可选

        步骤：
        1. ``model = MODELS.build(cfg['model'])``
        2. 复制 train_dataloader 配置，pop 出 dataset 配置用 DATASETS.build 构建，
           其余参数原样交给 ``torch.utils.data.DataLoader(dataset, **rest)``
        3. ``OPTIMIZERS.build(cfg['optimizer'], params=model.parameters())``
           —— params 不可能写在配置里，所以通过 default_args 注入
        4. hooks = default_hooks 的所有 value + custom_hooks（可能没有，用 cfg.get）
           交给 __init__，由 register_hook 负责 build 和排序
        5. 返回 ``cls(...)``

        提示：cfg 可能是 mini_mm.Config，也可能是普通 dict；只用 ``cfg[...]`` 和
        ``cfg.get(...)`` 就能同时兼容两者。
        """
        # TODO(12)
        raise NotImplementedError('Runner.from_cfg')
