# RDetection

Learning for Object Detection —— 以 **mmdetection 插件** 的形式，把现代检测器（YOLO 系、DINO、RT-DETR、D-FINE、Co-DETR）实现进 OpenMMLab 生态，并借此系统学习深度学习与目标检测。

## 项目结构

```
rdet/        正式代码 + Stage 2 起的模型练习（骨架 + TODO）
mini_mm/     练习：手写最小版 Registry（示范）/ Config / Hook / Runner
configs/     各模型配置（继承 mmdet 的 _base_）
tools/       练习：仿 mmdet 的 train / test / dist_*.sh / misc/browse_dataset
tests/       单元测试（pytest），也是练习的验收标准
docs/notes/  每个阶段的学习笔记
```

## 安装

```bash
conda create -n rdet python=3.10 -y && conda activate rdet
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -U openmim
mim install "mmengine>=0.10.0" "mmcv>=2.0.0,<2.2.0" "mmdet>=3.3.0,<3.4.0"
pip install -e .
pytest
```

在任意配置中加入以下一行即可启用 rdet 的模块：

```python
custom_imports = dict(imports=['rdet'], allow_failed_imports=False)
```

## 练习方式

本仓库里的大部分文件是**带讲解的空骨架**，需要你亲手把代码敲出来：

- 文件开头写明：目标、你会学到什么、对照的官方源码、完成标准；
- 函数里用 `# TODO(n)` 按步骤写好逻辑和提示，末尾是 `raise NotImplementedError`；
- `# 思考：` 是需要写进 `docs/notes/` 的问题，答不上来就去读对照的源码。

```bash
pytest -rs        # SKIPPED = 练习还没写；FAILED = 写了但不对；PASSED = 完成
```

建议做题顺序：
1. Stage 0：先**不写代码**，用 mmdet 自带的 tools 跑通官方 Faster R-CNN，拿到对照基准
2. Stage 1：读 `mini_mm/registry.py`（示范）→ `mini_mm/config.py` → `hooks.py` + `runner.py` → `tools/` 下的脚本
3. Stage 2：`rdet/structures/bbox.py` → `rdet/models/layers/nms.py` → `rdet/models/task_modules/` → `backbones/resnet.py` → `necks/fpn.py` → …

具体清单见 [docs/notes/00-roadmap.md](docs/notes/00-roadmap.md)。

写完一个就提交一次，commit 信息写清楚是哪个练习，方便日后回看。

> 注意：mmengine 0.10.x 与 torch ≥ 2.6 不兼容（`torch.load` 的 `weights_only` 默认值变了，`--resume` 会报错），
> 请按 `requirements.txt` 使用 torch 2.1。

## 学习路线

详见 [docs/notes/00-roadmap.md](docs/notes/00-roadmap.md)。

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 环境搭建，用 mmdet 自带工具跑通官方 Faster R-CNN（不写代码） | ⬜ |
| 1 | mini_mm（Registry / Config / Hook / Runner）+ tools 练习 | ⬜ |
| 2 | 亲手实现 Faster R-CNN：基础工具 → ResNet / FPN → RPN → RoI Head → Detector；RetinaNet | ⬜ |
| 3 | YOLOv8 → YOLOv10 → YOLO11 | ⬜ |
| 4 | DINO（精读 mmdet 已有实现） | ⬜ |
| 5 | RT-DETR | ⬜ |
| 6 | D-FINE | ⬜ |
| 7 | Co-DETR（可选） | ⬜ |

## Model Zoo

| 模型 | Backbone | 输入 | Epochs | box mAP（本项目） | box mAP（官方） | 配置 | 权重 |
|---|---|---|---|---|---|---|---|
| Faster R-CNN（rdet 实现） | R-50-FPN | 1333×800 | 12 | — | 37.4 | [config](configs/faster_rcnn/rfaster-rcnn_r50_fpn_1x_coco.py) | — |
| RetinaNet（rdet 实现） | R-50-FPN | 1333×800 | 12 | — | 36.5 | [config](configs/retinanet/rretinanet_r50_fpn_1x_coco.py) | — |
