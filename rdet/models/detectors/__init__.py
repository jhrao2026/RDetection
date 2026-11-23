# detector 继承 mmdet 的 BaseDetector，只有装了 mmdet 才能导入
from rdet.registry import HAS_MMDET

if HAS_MMDET:
    from .faster_rcnn import FasterRCNN  # noqa: F401
    from .single_stage import SingleStageDetector  # noqa: F401
