# Faster R-CNN R50-FPN 1x COCO —— 所有模型组件都用 rdet 里自己实现的版本
#
# 对照：mmdet 的 configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py
#       和 configs/_base_/models/faster-rcnn_r50_fpn.py（模型部分逐项对比，看哪些字段被去掉了、为什么）
#
# 用法：
#   # 加载官方权重测试（Stage 2 最终验收：bbox mAP 应为 37.4）
#   python tools/test.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
#       checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
#   # 8 卡从头训练（mAP 与官方差距应 ≤ 0.3）
#   bash tools/dist_train.sh configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py 8
#
# 数据集、训练策略、运行时配置直接继承 mmdet 的 _base_（"mmdet::" 前缀表示从已安装的 mmdet 包里找）。
# 模型部分不继承，完整写出来 —— 这样不需要 _delete_，也能一眼看清每个模块的参数。
_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py',
]

custom_imports = dict(imports=['rdet'], allow_failed_imports=False)
# ★ 必须设置：从 rdet 的注册表开始查找，type='ResNet' 才会用到自己写的 ResNet（见 rdet/registry.py）。
#   rdet 里没有的模块（CocoDataset、DetDataPreprocessor、CocoMetric ...）会自动回退到 mmdet。
#   mmdet 的 default_runtime.py 里写的是 default_scope = 'mmdet'，这里覆盖掉它。
default_scope = 'rdet'

model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(               # 预处理直接用 mmdet 的
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0])),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False)),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False)),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True))),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
