"""mini_mm.config —— 练习：仿写 mmengine 的 Config（Python 风格配置 + _base_ 继承）

================================================================================
目标
    让下面这种配置写法跑起来：

        # configs/retinanet_r50.py
        _base_ = ['./_base_/model.py', './_base_/schedule.py']
        model = dict(backbone=dict(depth=50))      # 只写"和 base 不一样"的部分
        optimizer = dict(lr=0.02)

        cfg = Config.fromfile('configs/retinanet_r50.py')
        cfg.model.backbone.depth   # -> 50，且 base 里的其他字段都还在

你会学到
    - 为什么配置要能"继承"：几十个实验只差几个超参，复制粘贴整份配置会失控
    - 字典递归合并的规则，以及 ``_delete_=True`` 为什么存在
    - 如何安全地执行一个 .py 文件并取出其中的变量

参考源码（写完后再对照）
    mmengine/config/config.py：Config.fromfile / Config._file2dict /
    Config._merge_a_into_b / Config.merge_from_dict；ConfigDict 在同一文件

完成标准
    pytest tests/test_mini_mm_config.py 全部通过（SKIPPED 表示还没写）

建议顺序：ConfigDict → merge_dict → _file2dict → Config.fromfile → merge_from_dict
================================================================================
"""

from __future__ import annotations

import os.path as osp  # noqa: F401  练习 3 会用到
from typing import Any


class ConfigDict(dict):
    """支持用 ``.`` 访问的 dict：``cfg.model.backbone`` 等价于 ``cfg['model']['backbone']``。

    练习 1：实现下面三个方法。

    提示：
    - ``__init__``：先 ``super().__init__(*args, **kwargs)``，然后遍历自己的每个
      value，如果是普通 dict 就递归转成 ConfigDict；如果是 list/tuple，
      里面的 dict 也要转（配置里经常有 ``pipeline=[dict(...), dict(...)]``）。
      可以写一个小工具函数 ``_to_cfg_dict(v)`` 处理这几种情况，后面还会复用。
    - ``__getattr__``：只有在正常属性查找失败时才会被调用。用 ``self[name]`` 取值；
      取不到时必须抛 ``AttributeError``（而不是 KeyError）——
      否则 ``hasattr``、``copy``、``pickle`` 都会出问题。想一想为什么。
    - ``__setattr__``：``cfg.a = dict(b=1)`` 应该等价于 ``cfg['a'] = ConfigDict(b=1)``。
    """

    def __init__(self, *args, **kwargs):
        # TODO(1.1) 调用父类初始化，并把嵌套的 dict / list 中的 dict 转成 ConfigDict
        raise NotImplementedError('ConfigDict.__init__')

    def __getattr__(self, name: str) -> Any:
        # TODO(1.2) 返回 self[name]；KeyError 要转成 AttributeError
        raise NotImplementedError('ConfigDict.__getattr__')

    def __setattr__(self, name: str, value: Any) -> None:
        # TODO(1.3) 写入 self[name]，dict 要先转成 ConfigDict
        raise NotImplementedError('ConfigDict.__setattr__')


def merge_dict(base: dict, child: dict) -> dict:
    """把 child 递归合并进 base 的一份副本，返回新 dict（不能修改入参）。

    练习 2。规则（和 mmengine 一致）：
    1. child 里的 key 覆盖 base 里的同名 key；
    2. 如果两边的值 **都是 dict**，不直接覆盖，而是递归合并；
    3. 如果 child 的这个 dict 里有 ``_delete_=True``，则 **不递归**，
       直接用 child 的值（去掉 ``_delete_`` 这个键）整个替换 base 的值。

    例子：
        base  = dict(optimizer=dict(type='SGD', lr=0.01, momentum=0.9))
        child = dict(optimizer=dict(lr=0.02))
        -> dict(optimizer=dict(type='SGD', lr=0.02, momentum=0.9))

        child = dict(optimizer=dict(_delete_=True, type='AdamW', lr=1e-4))
        -> dict(optimizer=dict(type='AdamW', lr=1e-4))   # momentum 没了

    思考：如果没有 _delete_，把 SGD 换成 AdamW 时会发生什么？
    （提示：AdamW 的 __init__ 不接受 momentum 参数）

    提示：用 ``copy.deepcopy(base)`` 得到副本再改，避免改动调用方的数据。
    """
    # TODO(2.1) 深拷贝 base
    # TODO(2.2) 遍历 child.items()：
    #           - v 是 dict 且 v.pop('_delete_', False) 为真 -> 直接替换（注意别改到 child 本身）
    #           - v 是 dict 且 base 中对应值也是 dict      -> 递归 merge_dict
    #           - 其他情况                                -> 直接覆盖
    raise NotImplementedError('merge_dict')


def _file2dict(filename: str) -> dict:
    """执行一个 .py 配置文件，返回其中的变量（已处理好 _base_ 继承）。

    练习 3。步骤：
    1. 读入文件内容，用 ``exec(compile(src, filename, 'exec'), namespace)`` 执行，
       namespace 是一个空 dict。
    2. 过滤掉不属于"配置"的东西：以 ``__`` 开头的名字（如 __builtins__），
       以及模块对象（配置里 ``import math`` 之后 math 不应该进配置）。
       提示：``import types; isinstance(v, types.ModuleType)``
    3. 如果有 ``_base_``（可能是 str，也可能是 list[str]）：
       - 路径是 **相对当前配置文件所在目录** 的：``osp.join(osp.dirname(filename), base)``
       - 对每个 base 递归调用 ``_file2dict``，依次合并成一个 base_dict
         （多个 base 之间如果有重复的顶层 key，mmengine 会报错；这里可以先简单地后者覆盖前者）
       - 最后 ``merge_dict(base_dict, 当前文件的变量去掉 _base_)``
    4. 返回结果 dict。

    思考：mmengine 真正的实现并不是直接 exec，而是先把文件复制到临时目录再
    import，还会解析 ``{{ fileDirname }}`` 这类占位符。它为什么要这么麻烦？
    """
    # TODO(3.1) 检查文件存在且以 .py 结尾，否则抛 FileNotFoundError / ValueError
    # TODO(3.2) exec 并过滤变量
    # TODO(3.3) 处理 _base_
    raise NotImplementedError('_file2dict')


class Config:
    """对外接口。真实 mmengine 的 Config 还负责 pretty print、dump、新式 lazy 配置等。

    这里只保存一个 ConfigDict，并把属性访问转发给它。
    """

    def __init__(self, cfg_dict: dict | None = None, filename: str | None = None):
        # 已完成：用 object.__setattr__ 绕开下面自定义的 __setattr__
        object.__setattr__(self, '_cfg_dict', ConfigDict(cfg_dict or {}))
        object.__setattr__(self, 'filename', filename)

    @staticmethod
    def fromfile(filename: str) -> 'Config':
        # TODO(4) 调用 _file2dict，包装成 Config 返回（记得传 filename）
        raise NotImplementedError('Config.fromfile')

    def merge_from_dict(self, options: dict) -> None:
        """命令行覆盖配置：``--cfg-options model.backbone.depth=101 optimizer.lr=0.1``

        tools/train.py 会把命令行解析成 ``{'model.backbone.depth': 101, 'optimizer.lr': 0.1}``
        传进来。

        练习 5：把每个 ``'a.b.c': v`` 展开成嵌套 dict ``{'a': {'b': {'c': v}}}``，
        全部展开后用 ``merge_dict`` 合并到当前配置，再写回 ``self._cfg_dict``。
        提示：``key.split('.')`` 然后从里往外包。
        """
        # TODO(5)
        raise NotImplementedError('Config.merge_from_dict')

    # ---------------- 以下已完成：把对 Config 的访问转发给内部的 ConfigDict
    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __setattr__(self, name, value):
        setattr(self._cfg_dict, name, value)

    def __getitem__(self, name):
        return self._cfg_dict[name]

    def __contains__(self, name):
        return name in self._cfg_dict

    def get(self, name, default=None):
        return self._cfg_dict.get(name, default)

    def to_dict(self) -> dict:
        import copy
        return copy.deepcopy(dict(self._cfg_dict))

    def __repr__(self):
        return f'Config(path: {self.filename}): {dict(self._cfg_dict)}'
