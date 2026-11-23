"""rdet.registry —— rdet 使用的注册表（已完成）。

对应 mmdet/registry.py。正常情况下直接复用 mmdet 的注册表，
这样 rdet 里 ``@MODELS.register_module()`` 的模块就能在 mmdet 配置里用 ``type='RResNet'`` 构建。

没装 mmdet 时（例如只在笔记本上做 Stage 2.1 / 2.2 的纯 torch 练习），
退回到 mini_mm 的 Registry，代码照样能 import、测试照样能跑。

思考：mmdet 里为什么要把 "模型" (MODELS) 和 "anchor 生成器 / assigner / coder" (TASK_UTILS)
      分成两个注册表？（提示：后者不是 nn.Module，没有参数，也不参与 state_dict）
"""

try:
    from mmdet.registry import MODELS, TASK_UTILS
    HAS_MMDET = True
except ImportError:
    from mini_mm import Registry

    MODELS = Registry('model')
    TASK_UTILS = Registry('task util')
    HAS_MMDET = False

__all__ = ['MODELS', 'TASK_UTILS', 'HAS_MMDET']
