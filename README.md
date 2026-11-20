# RDetection

Learning for Object Detection —— 以 **mmdetection 插件** 的形式，把现代检测器（YOLO 系、DINO、RT-DETR、D-FINE、Co-DETR）实现进 OpenMMLab 生态，并借此系统学习深度学习与目标检测。

## 项目结构

```
rdet/        正式代码：自定义模块，通过 custom_imports 注册进 mmdet
mini_mm/     学习用：手写最小版 Registry / Config / Runner，理解 mm 架构设计
configs/     各模型配置（继承 mmdet 的 _base_）
tools/       train / test / 权重转换等脚本
tests/       单元测试（pytest）
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

## 学习路线

详见 [docs/notes/00-roadmap.md](docs/notes/00-roadmap.md)。

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 环境搭建，跑通 mmdet 官方 Faster R-CNN / RetinaNet | ⬜ |
| 1 | 手写 mini_mm：Registry ✅ / Config / Runner / Hook | 🟡 |
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
