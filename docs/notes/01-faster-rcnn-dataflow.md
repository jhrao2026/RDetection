# 01 Faster R-CNN 数据流（Stage 0 作业，自己填）

目的：在 **官方 mmdet** 上用断点 / print 走一遍前向，记下每一步的输入输出。
这张表就是 Stage 2 的"地图"：你手写的每个模块，输入输出都必须和这里一致。

## 怎么做

```python
# 在 mmdet 源码里打断点，或者写一个小脚本：
from mmdet.apis import init_detector
model = init_detector('configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py',
                      'checkpoints/faster_rcnn_r50_fpn_1x_coco.pth', device='cuda:0')
# 建议断点位置：
#   mmdet/models/data_preprocessors/data_preprocessor.py  DetDataPreprocessor.forward
#   mmdet/models/detectors/two_stage.py                   TwoStageDetector.extract_feat / loss / predict
#   mmdet/models/dense_heads/rpn_head.py                  RPNHead.forward_single / _predict_by_feat_single
#   mmdet/models/roi_heads/standard_roi_head.py           StandardRoIHead.bbox_forward
```

训练（loss）和推理（predict）各走一遍，两条路径不一样。

## 推理路径

| 步骤 | 模块 / 函数 | 输入形状 | 输出形状 | 备注（填你观察到的） |
|---|---|---|---|---|
| 1 | DetDataPreprocessor | | | 做了哪些归一化？pad 到多少？ |
| 2 | backbone (ResNet-50) | | C2–C5: | |
| 3 | neck (FPN) | | P2–P6: | |
| 4 | rpn_head.forward | | cls_scores: / bbox_preds: | 每层通道数为什么是 3 和 12？ |
| 5 | rpn_head.predict_by_feat | | proposals: | 每层取了多少 topk？NMS 阈值？最终保留多少？ |
| 6 | roi_extractor | | | 每个 RoI 被分到了哪一层？ |
| 7 | bbox_head.forward | | cls_score: / bbox_pred: | 为什么分类是 81 维？ |
| 8 | bbox_head.predict_by_feat | | | score_thr、NMS、max_per_img 分别是多少？ |

## 训练路径（loss）

| 步骤 | 模块 / 函数 | 关键数字 | 备注 |
|---|---|---|---|
| 1 | RPN：anchor 总数 | | |
| 2 | RPN：MaxIoUAssigner 后正 / 负 / 忽略数量 | | |
| 3 | RPN：RandomSampler 后正 / 负数量 | | |
| 4 | RPN loss 项 | loss_rpn_cls: / loss_rpn_bbox: | |
| 5 | 训练时 RPN 输出多少 proposals 给 RoI head | | 和推理时一样吗？ |
| 6 | RoI head：assigner + sampler 后正 / 负数量 | | 为什么正样本里有 GT 本身？ |
| 7 | RoI head loss 项 | loss_cls: / acc: / loss_bbox: | |

## 我的疑问

-
