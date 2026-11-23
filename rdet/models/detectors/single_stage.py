"""练习 2.6c：RSingleStageDetector —— one-stage 检测器（RetinaNet，Stage 3 的 YOLO 也会用）

================================================================================
写完 RFasterRCNN 之后，这个就很简单了：没有 RPN 和 RoI head，只有一个 bbox_head。

    backbone → neck → bbox_head → loss / predict

参考源码
    mmdet/models/detectors/single_stage.py

完成标准
    pytest tests/test_retinanet.py -k detector
================================================================================
"""

import copy  # noqa: F401

import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.structures.bbox import get_box_tensor  # noqa: F401
from mmengine.structures import InstanceData  # noqa: F401

from rdet.registry import MODELS


@MODELS.register_module()
class RSingleStageDetector(BaseDetector):

    def __init__(self, backbone: dict, neck: dict, bbox_head: dict, train_cfg: dict = None,
                 test_cfg: dict = None, data_preprocessor: dict = None, init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # TODO(1) 构建 backbone / neck / bbox_head；bbox_head 配置里加上 train_cfg 和 test_cfg（整个传进去）
        raise NotImplementedError('RSingleStageDetector.__init__')

    def extract_feat(self, batch_inputs: torch.Tensor):
        # TODO(2)
        raise NotImplementedError('RSingleStageDetector.extract_feat')

    def loss(self, batch_inputs: torch.Tensor, batch_data_samples) -> dict:
        """练习 3：x → bbox_head(x) → bbox_head.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels, pad_shapes)"""
        # TODO(3)
        raise NotImplementedError('RSingleStageDetector.loss')

    def predict(self, batch_inputs: torch.Tensor, batch_data_samples, rescale: bool = True):
        """练习 4：x → bbox_head(x) → bbox_head.predict(..., img_shapes, scale_factors, rescale)
        → InstanceData → add_pred_to_datasample"""
        # TODO(4)
        raise NotImplementedError('RSingleStageDetector.predict')

    def _forward(self, batch_inputs: torch.Tensor, batch_data_samples=None):
        return self.bbox_head(self.extract_feat(batch_inputs))
