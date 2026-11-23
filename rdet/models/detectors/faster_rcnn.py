"""练习 2.5：FasterRCNN —— 把所有组件组装成完整的两阶段检测器

================================================================================
这是 Stage 2 的终点：写完它，configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py 里
**每一个模块都是你自己写的**，加载官方权重后 mAP 应该和官方完全一样（37.4）。

继承 mmdet 的 BaseDetector（需要装 mmdet），它帮我们处理了：
    - data_preprocessor：归一化、BGR→RGB、pad 到 32 的倍数（这部分不是检测算法本身，直接复用）
    - forward(inputs, data_samples, mode)：根据 mode 分发到 loss / predict / _forward
    - add_pred_to_datasample：把结果塞回 DetDataSample，交给 evaluator 算 mAP

我们要做的是 **数据结构的转换**：mmdet 用 DetDataSample / InstanceData 打包数据，
而我们的 head 接口全是普通 tensor（见 2.3 / 2.4）。

    DetDataSample
      ├── gt_instances.bboxes   → batch_gt_bboxes   （可能是 HorizontalBoxes 对象，用 get_box_tensor 转 tensor）
      ├── gt_instances.labels   → batch_gt_labels
      └── metainfo
            ├── img_shape       → resize 之后、pad 之前的 (h, w)：decode 时裁剪框用
            ├── pad_shape       → pad 到 32 倍数之后的 (h, w)：anchor valid_flags 用
            └── scale_factor    → (w_scale, h_scale)：rescale 回原图用

参考源码
    mmdet/models/detectors/two_stage.py：__init__ / extract_feat / loss / predict / _forward
    mmdet/models/detectors/base.py：BaseDetector

完成标准
    pytest tests/test_faster_rcnn.py   （与 mmdet 官方 FasterRCNN 同权重对比 loss 和检测结果）
    最终：python tools/test.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py <官方权重> → mAP 37.4
================================================================================
"""

import copy

import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.structures.bbox import get_box_tensor  # noqa: F401
from mmengine.structures import InstanceData  # noqa: F401

from rdet.registry import MODELS


@MODELS.register_module()
class FasterRCNN(BaseDetector):

    def __init__(self, backbone: dict, neck: dict, rpn_head: dict, roi_head: dict,
                 train_cfg: dict = None, test_cfg: dict = None, data_preprocessor: dict = None,
                 init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # TODO(1) 构建四个组件，属性名必须是 backbone / neck / rpn_head / roi_head（对齐权重 key）
        #   - backbone、neck：MODELS.build
        #   - rpn_head：复制一份配置，加上 train_cfg=train_cfg['rpn']（train_cfg 为 None 时传 None）
        #               和 test_cfg=test_cfg['rpn']，再 build
        #   - roi_head：同理，用 train_cfg['rcnn'] 和 test_cfg['rcnn']
        #   提示：copy.deepcopy(rpn_head) 再 update，不要修改传入的配置
        # 思考：为什么 train_cfg / test_cfg 写在 detector 这一层，而不是直接写进各个 head 的配置里？
        raise NotImplementedError('FasterRCNN.__init__')

    def extract_feat(self, batch_inputs: torch.Tensor):
        """练习 2：backbone → neck，返回 FPN 的 5 层特征。"""
        # TODO(2)
        raise NotImplementedError('FasterRCNN.extract_feat')

    def loss(self, batch_inputs: torch.Tensor, batch_data_samples) -> dict:
        """练习 3：训练。

        1. x = self.extract_feat(batch_inputs)
        2. 从 batch_data_samples 取出 gt_bboxes（get_box_tensor）、gt_labels、pad_shape、img_shape
        3. RPN：
               cls_scores, bbox_preds = self.rpn_head(x)
               losses = self.rpn_head.loss(cls_scores, bbox_preds, gt_bboxes, pad_shapes)
               proposals = self.rpn_head.predict(cls_scores, bbox_preds, img_shapes,
                                                 cfg=self.train_cfg['rpn_proposal'])
        4. RoI head：
               losses.update(self.roi_head.loss(x, [p[0] for p in proposals], gt_bboxes, gt_labels))
        5. return losses

        思考：RPN 的 loss 和 RoI head 的 loss 直接相加一起反传（"end-to-end" 训练）。
              原始 Faster R-CNN 论文用的是 4 步交替训练，为什么后来都改成了端到端？
        """
        # TODO(3)
        raise NotImplementedError('FasterRCNN.loss')

    def predict(self, batch_inputs: torch.Tensor, batch_data_samples, rescale: bool = True):
        """练习 4：推理。

        1. x = self.extract_feat(batch_inputs)
        2. proposals = self.rpn_head.predict(*self.rpn_head(x), img_shapes)      # 用 test_cfg.rpn
        3. results = self.roi_head.predict(x, [p[0] for p in proposals], img_shapes, scale_factors, rescale)
        4. 每张图包成 InstanceData(bboxes=..., scores=..., labels=...)
        5. return self.add_pred_to_datasample(batch_data_samples, results_list)
        """
        # TODO(4)
        raise NotImplementedError('FasterRCNN.predict')

    def _forward(self, batch_inputs: torch.Tensor, batch_data_samples=None):
        """（已完成）mode='tensor' 时调用，只返回网络的原始输出，用于计算 FLOPs、导出模型等。"""
        return self.rpn_head(self.extract_feat(batch_inputs))


_ = copy  # 练习 1 会用到；写完后删掉这一行
