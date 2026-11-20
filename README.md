# RDetection

Learning for Object Detection —— 以 **mmdetection 插件** 的形式，把现代检测器（YOLO 系、DINO、RT-DETR、D-FINE、Co-DETR）实现进 OpenMMLab 生态，并借此系统学习深度学习与目标检测。

## 项目结构

```
rdet/        正式代码：自定义模块，通过 custom_imports 注册进 mmdet
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
1. 读 `mini_mm/registry.py`（已完成的示范，看懂注释和测试的写法）
2. `mini_mm/config.py` → `mini_mm/hooks.py` + `mini_mm/runner.py`（有单元测试）
3. `tools/train.py` → `tools/dist_train.sh` → `tools/test.py` → `tools/dist_test.sh` → `tools/misc/browse_dataset.py`
   （需要装好 mmdet，用 mmdet 官方配置跑通即为完成；写完后对照 mmdetection v3.3.0 的同名文件）

写完一个就提交一次，commit 信息写清楚是哪个练习，方便日后回看。

> 注意：mmengine 0.10.x 与 torch ≥ 2.6 不兼容（`torch.load` 的 `weights_only` 默认值变了，`--resume` 会报错），
> 请按 `requirements.txt` 使用 torch 2.1。

## 学习路线

详见 [docs/notes/00-roadmap.md](docs/notes/00-roadmap.md)。

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 环境搭建，跑通 mmdet 官方 Faster R-CNN / RetinaNet；完成 tools/ 练习 | ⬜ |
| 1 | mini_mm：Registry（示范 ✅）/ Config / Hook / Runner | ⬜ |
| 2 | rdet 插件骨架，复制 RetinaNet head 验证插件链路 | ⬜ |
| 3 | YOLOv8 → YOLOv10 → YOLO11 | ⬜ |
| 4 | DINO（精读 mmdet 已有实现） | ⬜ |
| 5 | RT-DETR | ⬜ |
| 6 | D-FINE | ⬜ |
| 7 | Co-DETR（可选） | ⬜ |

## Model Zoo

| 模型 | Backbone | 输入 | Epochs | box mAP（本项目） | box mAP（官方） | 配置 | 权重 |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — |
