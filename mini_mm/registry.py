"""mini_mm.registry —— 仿写 mmengine/registry/registry.py 的最小版本。

为什么需要 Registry？
---------------------
在普通 PyTorch 项目里，换一个 backbone 往往要改训练脚本里的 import 和构造代码：

    from models.resnet import ResNet
    model = ResNet(depth=50)

模型一多，训练脚本就会充满 if/else。Registry 的思路是：
1. 每个模块在定义时把自己"登记"到一个全局表里（字符串名 -> 类）；
2. 训练脚本只拿到一个 dict 配置，例如 ``dict(type='ResNet', depth=50)``，
   再由 Registry 根据 ``type`` 查表并实例化。

这样"用哪个模块"完全由配置文件决定，训练代码一行都不用改，
这就是 OpenMMLab "配置驱动 + 模块解耦" 的核心。

与 mmengine 的差异（刻意简化，便于学习）：
- 不支持 scope（mmdet::xxx 这种跨库前缀）和父子 Registry；
- 不支持 lazy import / locations；
- build 只做递归最少的事情。
读懂本文件后，建议对照 mmengine 源码，看它多出来的部分分别解决什么问题。
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional, Type, Union


class Registry:
    """把字符串名字映射到类（或函数）的表。

    Example:
        >>> MODELS = Registry('model')
        >>> @MODELS.register_module()
        ... class ResNet:
        ...     def __init__(self, depth):
        ...         self.depth = depth
        >>> net = MODELS.build(dict(type='ResNet', depth=50))
        >>> net.depth
        50
    """

    def __init__(self, name: str, build_func: Optional[Callable] = None):
        self._name = name
        self._module_dict: Dict[str, Type] = {}
        # mmengine 允许为不同 Registry 定制构建逻辑（如 MODELS 会额外处理
        # 嵌套配置），这里保留同样的扩展点。
        self._build_func = build_func or build_from_cfg

    # ------------------------------------------------------------------ 查询
    @property
    def name(self) -> str:
        return self._name

    @property
    def module_dict(self) -> Dict[str, Type]:
        return self._module_dict

    def get(self, key: str) -> Optional[Type]:
        return self._module_dict.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict

    def __len__(self) -> int:
        return len(self._module_dict)

    def __repr__(self) -> str:
        items = ', '.join(sorted(self._module_dict))
        return f'Registry(name={self._name}, items=[{items}])'

    # ------------------------------------------------------------------ 注册
    def _register(self, module: Type, name: Optional[Union[str, list]] = None,
                  force: bool = False) -> None:
        if not (inspect.isclass(module) or inspect.isfunction(module)):
            raise TypeError(f'module must be a class or function, got {type(module)}')
        names = name if isinstance(name, (list, tuple)) else [name or module.__name__]
        for n in names:
            if not force and n in self._module_dict:
                # 防止两个同名模块悄悄互相覆盖 —— 这是真实项目里很常见的 bug 来源
                raise KeyError(f'{n} is already registered in {self._name}')
            self._module_dict[n] = module

    def register_module(self, name: Optional[Union[str, list]] = None,
                        force: bool = False, module: Optional[Type] = None):
        """既可以当装饰器用，也可以直接调用。

            @MODELS.register_module()          # 装饰器
            class A: ...

            MODELS.register_module(module=B)   # 直接注册（常用于第三方类，如 torch.optim.SGD）
        """
        if module is not None:
            self._register(module, name, force)
            return module

        def _decorator(cls):
            self._register(cls, name, force)
            return cls

        return _decorator

    # ------------------------------------------------------------------ 构建
    def build(self, cfg: dict, **default_args) -> Any:
        return self._build_func(cfg, self, default_args)


def build_from_cfg(cfg: dict, registry: Registry,
                   default_args: Optional[dict] = None) -> Any:
    """根据 ``cfg['type']`` 在 registry 中查找并实例化。

    ``default_args`` 用于由调用方注入、但不写在配置里的参数，
    例如 head 的 ``num_classes`` 往往由 detector 统一传入。
    配置里写了的值优先级更高（用 setdefault 实现）。
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, got {type(cfg)}')
    if 'type' not in cfg and not (default_args and 'type' in default_args):
        raise KeyError(f'cfg must contain the key "type", got {cfg}')

    args = dict(cfg)  # 复制一份，避免 pop 修改用户的配置
    for k, v in (default_args or {}).items():
        args.setdefault(k, v)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {registry.name} registry. '
                           f'Available: {sorted(registry.module_dict)}')
    elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
        # 也允许直接传类本身，mmengine 新版配置（纯 Python 风格）就是这么做的
        obj_cls = obj_type
    else:
        raise TypeError(f'type must be a str or class, got {type(obj_type)}')

    try:
        return obj_cls(**args)
    except Exception as e:  # 给出是哪个模块构建失败，便于排查配置错误
        raise type(e)(f'{obj_cls.__name__}: {e}') from e
