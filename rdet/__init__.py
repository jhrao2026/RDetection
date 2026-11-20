"""RDetection: 以 mmdet 插件形式实现的现代检测器集合。

在配置文件中加入下面这一行即可把 rdet 中注册的模块接入 mmdet：

    custom_imports = dict(imports=['rdet'], allow_failed_imports=False)

导入 rdet 时会依次导入各子包，触发其中 ``@MODELS.register_module()`` 装饰器，
从而把自定义模块登记到 mmdet 的全局 Registry 中。
"""

__version__ = '0.0.1'

from . import datasets, engine, models  # noqa: F401,E402
