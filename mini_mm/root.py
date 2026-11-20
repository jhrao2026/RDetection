"""mini_mm.root —— 全局 Registry 定义（已完成，供其他练习使用）。

对应 mmengine/registry/root.py。mmengine 在这里定义了 20 多个"根"注册表，
下游库（mmdet、mmyolo ...）再各自创建子注册表继承它们。我们只保留训练
一个最小模型所需的 4 个。

思考题：
- 为什么 torch 自带的优化器也要注册进 OPTIMIZERS？
  （提示：想想配置文件里 ``optimizer=dict(type='SGD', lr=0.01)`` 是怎么变成对象的）
- 为什么 params 不写在配置里，而是由 Runner 在 build 时通过 default_args 传入？
"""

from .registry import Registry

MODELS = Registry('model')
DATASETS = Registry('dataset')
HOOKS = Registry('hook')
OPTIMIZERS = Registry('optimizer')

try:  # 没装 torch 时 Registry / Config 练习依然可以做
    import torch

    for _name in ('SGD', 'Adam', 'AdamW'):
        OPTIMIZERS.register_module(module=getattr(torch.optim, _name))
except ImportError:  # pragma: no cover
    pass
