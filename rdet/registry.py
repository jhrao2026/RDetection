"""rdet.registry —— rdet 自己的注册表（已完成）。

对应 mmdet/registry.py 和 mmyolo/registry.py。

为什么 rdet 里的模块可以和 mmdet 同名（ResNet、FPN、RPNHead ...）？
----------------------------------------------------------------
OpenMMLab 的注册表是一棵树：

    mmengine（根）
      └── mmdet（scope='mmdet'）
            └── rdet（scope='rdet'）    ← 本文件定义的就是这一层

每个下游库都有自己的 **子注册表**，名字只需要在自己这一层里唯一。
构建 ``dict(type='ResNet')`` 时，先看当前的 **default scope**（配置文件里的
``default_scope = 'rdet'``），从 rdet 这一层开始找：

    - rdet 里有 ResNet         → 用我们自己写的
    - rdet 里没有 CocoDataset  → 往父节点 mmdet 找，用 mmdet 的

所以配置里可以直接写 ``type='ResNet'``，而 CocoDataset、DetDataPreprocessor、
CocoMetric 这些我们没写的组件照样能用 mmdet 的。
如果某处确实想用 mmdet 的同名模块，可以加前缀显式指定：``type='mmdet.ResNet'``。

    ★ 因此 rdet 的配置文件里一定要有 ``default_scope = 'rdet'``，否则会悄悄用上 mmdet 的同名实现。

没装 mmdet 时（例如只在笔记本上做 Stage 2.1 / 2.2 的纯 torch 练习），
退回到 mini_mm 的 Registry，代码照样能 import、测试照样能跑。

思考：
- mmdet 里为什么要把 "模型" (MODELS) 和 "anchor 生成器 / assigner / coder" (TASK_UTILS)
  分成两个注册表？（提示：后者不是 nn.Module，没有参数，也不参与 state_dict）
- 为什么要给 DATASETS、METRICS 等 **所有** 注册表都建子节点，即使 rdet 现在一个数据集都没写？
  （提示：Runner 按 default_scope='rdet' 去找 "rdet 这一层的 DATASETS"，找不到这一层就无法回退到 mmdet）
"""

try:
    from mmdet import registry as _mmdet
    from mmengine.registry import Registry
    HAS_MMDET = True
except ImportError:
    HAS_MMDET = False

if HAS_MMDET:
    def _child(name: str, parent: Registry, *locations: str) -> Registry:
        return Registry(name, parent=parent, scope='rdet', locations=list(locations))

    RUNNERS = _child('runner', _mmdet.RUNNERS)
    RUNNER_CONSTRUCTORS = _child('runner constructor', _mmdet.RUNNER_CONSTRUCTORS)
    LOOPS = _child('loop', _mmdet.LOOPS)
    HOOKS = _child('hook', _mmdet.HOOKS, 'rdet.engine.hooks')
    DATASETS = _child('dataset', _mmdet.DATASETS, 'rdet.datasets')
    DATA_SAMPLERS = _child('data sampler', _mmdet.DATA_SAMPLERS)
    TRANSFORMS = _child('transform', _mmdet.TRANSFORMS, 'rdet.datasets.transforms')
    MODELS = _child('model', _mmdet.MODELS, 'rdet.models')
    MODEL_WRAPPERS = _child('model_wrapper', _mmdet.MODEL_WRAPPERS)
    WEIGHT_INITIALIZERS = _child('weight initializer', _mmdet.WEIGHT_INITIALIZERS)
    OPTIMIZERS = _child('optimizer', _mmdet.OPTIMIZERS)
    OPTIM_WRAPPERS = _child('optim_wrapper', _mmdet.OPTIM_WRAPPERS)
    OPTIM_WRAPPER_CONSTRUCTORS = _child('optimizer wrapper constructor', _mmdet.OPTIM_WRAPPER_CONSTRUCTORS)
    PARAM_SCHEDULERS = _child('parameter scheduler', _mmdet.PARAM_SCHEDULERS)
    METRICS = _child('metric', _mmdet.METRICS)
    EVALUATOR = _child('evaluator', _mmdet.EVALUATOR)
    TASK_UTILS = _child('task util', _mmdet.TASK_UTILS, 'rdet.models')
    VISUALIZERS = _child('visualizer', _mmdet.VISUALIZERS)
    VISBACKENDS = _child('vis_backend', _mmdet.VISBACKENDS)
    LOG_PROCESSORS = _child('log_processor', _mmdet.LOG_PROCESSORS)

    # 回退到父节点时，mmengine 只会在 "已经 import 过" 的模块里找，不会触发 mmdet 的懒加载。
    # 不提前导入的话，例如 DetVisualizationHook（在 mmdet.engine 里）会找不到，Runner 直接报错。
    from mmdet.utils import register_all_modules as _register_mmdet
    _register_mmdet(init_default_scope=False)
else:
    from mini_mm import Registry

    MODELS = Registry('model')
    TASK_UTILS = Registry('task util')

__all__ = ['MODELS', 'TASK_UTILS', 'HAS_MMDET']
