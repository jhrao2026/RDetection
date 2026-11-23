# 00 学习路线

## 总体策略：先"拆"，后"仿"，再"融"

1. **拆**：先跑通 mmdet 官方模型，用断点和打印看清一次训练的完整调用链。
2. **仿**：在 `mini_mm/` 里从零手写最小版 Registry / Config / Runner / Hook，弄明白每个设计解决什么问题。
3. **融**：正式代码放在 `rdet/`，依赖 pip 安装的 mmengine / mmcv / mmdet，通过 `custom_imports` 注册。这样可以复用数据管线、评测和分布式训练，结果能和官方对齐。

复制 mmdet 代码的方法：**整文件复制 → 逐行注释 → 删到最小 → 用自己的方式重写**，每个模块都写单元测试。

## 阶段

### Stage 0：环境 + 跑通官方（3–5 天，不写代码）
目的：拿到"标准答案"。后面手写的每个组件都要和它逐层对齐，没有它就无法判断自己写得对不对。
- [ ] 按 README 安装环境，`python -c "import mmdet, mmcv.ops"` 不报错
- [ ] 下载官方 `faster-rcnn_r50_fpn_1x_coco` checkpoint，用 **mmdet 自带** 的 `tools/test.py` 复现 COCO val bbox mAP 37.4
- [ ] 用 mmdet 自带的 `tools/train.py` 训 1 个 epoch（可以用 `--cfg-options train_cfg.max_epochs=1`）
- [ ] 用 mmdet 自带的 `tools/misc/browse_dataset.py` 看一遍训练数据
- [ ] 断点走一遍前向，把每一步的 tensor 形状填进 [01-faster-rcnn-dataflow.md](01-faster-rcnn-dataflow.md)

### Stage 1：框架层（约 1–2 周）
**mini_mm**（有单元测试）
- [x] `mini_mm/registry.py`：仿 `mmengine/registry/registry.py`（已完成的示范，先读懂它）
- [ ] `mini_mm/config.py`：练习 1–5，`_base_` 继承与字典合并（`pytest tests/test_mini_mm_config.py`）
- [ ] `mini_mm/hooks.py`：练习 6–7，LoggerHook / CheckpointHook
- [ ] `mini_mm/runner.py`：练习 8–12，parse_losses / Hook 排序 / 训练循环 / from_cfg（`pytest tests/test_mini_mm_runner.py`）
- [ ] （可选）用 mini_mm 训练 CIFAR-10 小分类网，顺便补齐 Dataset/DataLoader、autograd、优化器、AMP、DDP

**tools**（用自己写的版本复现 Stage 0 的结果即为完成）
- [ ] `tools/train.py`
- [ ] `tools/dist_train.sh`
- [ ] `tools/test.py`
- [ ] `tools/dist_test.sh`
- [ ] `tools/misc/browse_dataset.py`

思考题：
- 为什么用 Registry 而不是直接 import？
- 为什么 `data_preprocessor` 要从模型里拆出来？
- 为什么 loss 写在 head 里而不是训练脚本里？

### Stage 2：亲手实现 Faster R-CNN（约 3–4 周）
把官方配置里的每一个模块都换成自己写的版本，最后加载官方权重得到同样的 mAP。

设计约束：
- 注册名加 `R` 前缀（`RResNet`、`RFPN` ...），避免和 mmdet 自带模块重名；
- **子模块 / 参数名与 mmdet 完全一致**，官方 checkpoint 才能直接加载 —— 这是最强的正确性检验；
- 测试分两层：A 层只依赖 torch / torchvision，B 层（`*_vs_mmdet`）装了 mmdet 才运行，和 mmdet 的实现逐元素对比。

**2.1 基础工具**（`pytest tests/test_bbox_utils.py`）
- [ ] `rdet/structures/bbox.py`：IoU 矩阵 `bbox_overlaps`
- [ ] `rdet/models/layers/nms.py`：纯 Python 的 `nms` / `batched_nms`
- [ ] `rdet/models/task_modules/prior_generators/anchor_generator.py`：`RAnchorGenerator`
- [ ] `rdet/models/task_modules/coders/delta_xywh_bbox_coder.py`：`RDeltaXYWHBBoxCoder`
- [ ] `rdet/models/task_modules/assigners/max_iou_assigner.py`：`RMaxIoUAssigner`
- [ ] `rdet/models/task_modules/samplers/random_sampler.py`：`RRandomSampler`

**2.2 Backbone & Neck**（`pytest tests/test_resnet.py tests/test_fpn.py`）
- [ ] `rdet/models/backbones/resnet.py`：`RResNet`（和 torchvision resnet50 输出一致；练习 7 加载预训练权重）
- [ ] `rdet/models/necks/fpn.py`：`RFPN`（和 torchvision / mmdet FPN 输出一致）

**2.3 RPN Head**（`pytest tests/test_rpn_head.py`）
- [ ] `rdet/models/dense_heads/rpn_head.py`：`RRPNHead` —— forward / 训练目标 / loss / 生成 proposals

**2.4 RoI Head**（`pytest tests/test_roi_head.py`）
- [ ] `rdet/models/roi_heads/roi_extractor.py`：`RSingleRoIExtractor`（按面积把 RoI 分配到 FPN 层 + RoIAlign）
- [ ] `rdet/models/roi_heads/bbox_head.py`：`RShared2FCBBoxHead`（Shared2FC + 分类 / 回归 + multiclass NMS）
- [ ] `rdet/models/roi_heads/standard_roi_head.py`：`bbox2roi` + `RStandardRoIHead`

**2.5 Detector + 配置**（`pytest tests/test_faster_rcnn.py`，需要 mmdet）
- [ ] `rdet/models/detectors/faster_rcnn.py`：`RFasterRCNN`
- [ ] 测试全部通过后，在服务器上跑最终验收：
  - [ ] `python tools/test.py configs/faster_rcnn/rfaster-rcnn_r50_fpn_1x_coco.py <官方权重>`：bbox mAP = 37.4
  - [ ] `bash tools/dist_train.sh configs/faster_rcnn/rfaster-rcnn_r50_fpn_1x_coco.py 8`：mAP 与官方差距 ≤ 0.3

**2.6 RetinaNet（衔接 YOLO）**（`pytest tests/test_retinanet.py`）
- [ ] `rdet/models/losses/focal_loss.py`：`sigmoid_focal_loss`
- [ ] `rdet/models/dense_heads/retina_head.py`：`RRetinaHead`（建议复制自己的 RPN 代码再改）
- [ ] `rdet/models/detectors/single_stage.py`：`RSingleStageDetector`
- [ ] 服务器验收：`configs/retinanet/rretinanet_r50_fpn_1x_coco.py` 加载官方权重 mAP = 36.5

> 小技巧：从头训练时，mmengine 会在日志里逐个参数打印 "Initialized by user-defined `init_weights` in XXX"，
> 可以用它确认自己写的 init_weights 真的生效了。

思考题：
- 为什么 bbox 回归预测 delta，而不是绝对坐标？
- 为什么 RPN 正负样本 1:1，而 RoI head 是 1:3？
- 为什么 FPN 的 P6 直接 max pool，而 RetinaNet 的 P6 / P7 要用卷积？
- 为什么 RoI 要按面积分配到不同的 FPN 层？

### Stage 3：YOLOv8 → YOLO11（约 2–3 周）
- 参考 mmyolo 的 YOLOv8：CSPDarknet / C2f、PAFPN、解耦头、DFL、TaskAlignedAssigner。
- 顺序：backbone → neck → head + loss → Mosaic / MixUp → EMA hook。
- 增量：v10（NMS-free，一对一 + 一对多双分配）、11（C3k2、C2PSA）。

### Stage 4：DINO（约 2 周）
- 精读 `mmdet/models/detectors/{deformable_detr,dino}.py` 和 `mmdet/models/layers/transformer/`。
- 重点：匈牙利匹配、Deformable Attention、去噪训练（CDN）、Mixed Query Selection、look-forward-twice。
- 它是 RT-DETR 的直接前置知识。

### Stage 5：RT-DETR（约 3 周）
- HybridEncoder（AIFI + CCFM）、IoU-aware query selection、decoder（复用 DINO 的去噪和可变形注意力）、VFL loss。
- 先写 `tools/model_converters/` 加载官方权重，对齐推理精度，再从头训练。

### Stage 6：D-FINE（约 2 周）
- 在 RT-DETR 基础上加 FDR（细粒度分布回归）和 GO-LSD（自蒸馏），主要改 head 和 loss。

### Stage 7：Co-DETR（约 3 周，可选）
- 参考 mmdet `projects/CO-DETR`，学习多辅助头协同训练和 `projects/` 插件的组织方式。

> 顺序说明：Stage 3 的 YOLO 会直接复用 Stage 2 的 IoU / NMS / assigner 思路和 browse_dataset 调试方式。RT-DETR 的去噪训练和 query selection 都来自 DINO，所以先做 DINO 再做 RT-DETR。

## 每个模型的统一流程
1. 读论文，画结构图，写进 `docs/notes/`
2. 加载官方权重做推理对齐，逐层比对输出 tensor
3. 单元测试：forward 输出形状、loss 能反传、assigner 小样例
4. 在小数据集（COCO 子集 / VOC）上验证能收敛
5. 多卡完整训练 COCO，把日志和 mAP 写进 README 的 Model Zoo
6. 一个模型一个分支 / PR，提交前跑 `pytest`

## 验收标准
- 加载官方权重后，`tools/test.py` 的 mAP 与官方相差在 ±0.3 以内
- 从头训练的 mAP 与官方相差不超过 1 个点
- 每阶段一篇笔记，不看代码也能讲清每个模块的设计理由
