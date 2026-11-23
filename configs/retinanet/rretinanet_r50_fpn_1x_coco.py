# RetinaNet R50-FPN 1x COCO —— 全部模块替换为 rdet 中自己实现的版本
#
# 对照：mmdet 的 configs/retinanet/retinanet_r50_fpn_1x_coco.py 和 configs/_base_/models/retinanet_r50_fpn.py
# 官方 bbox mAP：36.5
#
# 和 Faster R-CNN 的配置逐项对比：neck 的 start_level / add_extra_convs、anchor 的 scales / strides、
# assigner 的阈值、没有 sampler、学习率 0.01（而不是 0.02）—— 每一处都想清楚为什么。
_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py',
]

custom_imports = dict(imports=['rdet'], allow_failed_imports=False)

model = dict(
    type='RSingleStageDetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='RResNet',
        depth=50,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='RFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RRetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='RAnchorGenerator',
            # mmdet 写法是 octave_base_scale=4, scales_per_octave=3，等价于 4 * 2 ** (i / 3), i = 0, 1, 2
            scales=[4.0, 5.039684199579493, 6.3496042078727974],
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='RDeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        focal_gamma=2.0,
        focal_alpha=0.25),
    train_cfg=dict(
        assigner=dict(
            type='RMaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            match_low_quality=True)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
