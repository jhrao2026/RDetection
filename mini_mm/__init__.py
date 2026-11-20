"""mini_mm: 手写的最小 OpenMMLab 核心，仅用于学习架构设计（Stage 1）。

正式的检测器代码在 ``rdet/`` 中，依赖真正的 mmengine / mmdet。
"""

from .registry import Registry, build_from_cfg

__all__ = ['Registry', 'build_from_cfg']
