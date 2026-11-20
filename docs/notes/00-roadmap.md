# 00 学习路线

## 总体策略：先"拆"，后"仿"，再"融"

1. **拆**：先跑通 mmdet 官方模型，用断点和打印看清一次训练的完整调用链。
2. **仿**：在 `mini_mm/` 里从零手写最小版 Registry / Config / Runner / Hook，弄明白每个设计解决什么问题。
3. **融**：正式代码放在 `rdet/`，依赖 pip 安装的 mmengine / mmcv / mmdet，通过 `custom_imports` 注册。这样可以复用数据管线、评测和分布式训练，结果能和官方对齐。

复制 mmdet 代码的方法：**整文件复制 → 逐行注释 → 删到最小 → 用自己的方式重写**，每个模块都写单元测试。

## 阶段

### Stage 0：环境 + 跑通官方（约 1 周）
- 按 README 安装；跑通 `configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py` 的推理和 1 个 epoch 训练。
- 会用 `tools/train.py`、`tools/dist_train.sh`、`tools/misc/browse_dataset.py`、`tools/analysis_tools/analyze_logs.py`。
- 作业：画出 `runner.train()` 的调用链：Config → Runner → build model / dataloader → train loop → Hooks。

### Stage 1：手写 mini_mm（约 1–2 周）
- [x] `mini_mm/registry.py`：仿 `mmengine/registry/registry.py`
- [ ] `mini_mm/config.py`：支持 `_base_` 继承和字典合并（仿 `mmengine/config/config.py`）
- [ ] `mini_mm/runner.py` + `hooks.py`：最小训练循环 + LoggerHook / CheckpointHook
- [ ] 用 mini_mm 训练 CIFAR-10 小分类网，顺便补齐 Dataset/DataLoader、autograd、优化器、AMP、DDP
- 思考题：
  - 为什么用 Registry 而不是直接 import？
  - 为什么 `data_preprocessor` 要从模型里拆出来？
  - 为什么 loss 写在 head 里而不是训练脚本里？

### Stage 2：插件骨架（约 1 周）
- 把 RetinaNet 的 head 复制到 `rdet/models/dense_heads/` 改名注册，训练结果应和原版一致，以此证明插件链路是通的。
- 补齐检测基础：anchor、IoU、NMS、FPN、assigner/sampler、bbox coder、COCO mAP。

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

> 顺序说明：RT-DETR 的去噪训练和 query selection 都来自 DINO，所以先做 DINO 再做 RT-DETR。

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
