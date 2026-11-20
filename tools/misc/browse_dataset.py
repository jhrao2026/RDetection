"""tools/misc/browse_dataset.py —— 可视化"进入模型之前"的训练数据（练习版）

================================================================================
对照原版：mmdetection v3.3.0 的 tools/misc/browse_dataset.py

为什么这个工具很重要？
    检测项目里大量 bug 出在数据上：框坐标格式错（xywh / xyxy）、增强后框没跟着变、
    类别 id 错位、图片是 BGR 还是 RGB……模型不收敛时，**第一件事永远是看数据**。
    以后实现 YOLO 的 Mosaic / MixUp 时，你会反复用到这个脚本。

用法（写完后）：
    # 服务器上不弹窗，把画好框的图存下来
    python tools/misc/browse_dataset.py configs/xxx.py --output-dir vis_data --not-show
    # 本地弹窗逐张查看
    python tools/misc/browse_dataset.py configs/xxx.py --show-interval 1

你会学到
    - mmdet 3.x 的数据流：dataset[i] 返回 dict(inputs=Tensor[C,H,W], data_samples=DetDataSample)
    - DetDataSample / InstanceData 这两个"数据容器"长什么样
    - 不经过 Runner，如何单独 build 一个组件来调试
================================================================================
"""

import argparse
import os.path as osp

# TODO(0) 导入：
#   from mmengine.config import Config, DictAction
#   from mmengine.registry import init_default_scope
#   from mmengine.utils import ProgressBar
#   from mmdet.models.utils import mask2ndarray
#   from mmdet.registry import DATASETS, VISUALIZERS
#   from mmdet.structures.bbox import BaseBoxes
#
# 思考：train.py 里用的是 Runner，Runner 会自动设置 default scope；
#       这里没有 Runner，为什么必须手动调用 init_default_scope？不调会报什么错？


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    # TODO(1) 位置参数 'config'
    # TODO(2) '--output-dir'：default=None，图片保存目录
    # TODO(3) '--not-show'：store_true
    # TODO(4) '--show-interval'：type=float, default=2
    # TODO(5) '--cfg-options'：nargs='+', action=DictAction
    raise NotImplementedError('browse_dataset.py: parse_args')


def main():
    # TODO(6) args = parse_args()；cfg = Config.fromfile(...)；merge cfg_options

    # TODO(7) init_default_scope(cfg.get('default_scope', 'mmdet'))

    # TODO(8) 单独构建数据集和可视化器：
    #         dataset = DATASETS.build(cfg.train_dataloader.dataset)
    #         visualizer = VISUALIZERS.build(cfg.visualizer)
    #         visualizer.dataset_meta = dataset.metainfo     # 类别名和调色板
    #         progress_bar = ProgressBar(len(dataset))

    # TODO(9) 遍历数据集：for item in dataset:
    #   a. 先 print(item.keys()) 和 print(item['data_samples']) 看一眼结构（看懂后删掉）
    #   b. img = item['inputs'].permute(1, 2, 0).numpy()
    #      思考：为什么要 permute？（Tensor 是 CHW，画图要 HWC）
    #   c. data_sample = item['data_samples'].numpy()
    #      gt_instances = data_sample.gt_instances
    #   d. img_path = osp.basename(item['data_samples'].img_path)
    #      out_file = osp.join(args.output_dir, img_path) if args.output_dir is not None else None
    #   e. img = img[..., [2, 1, 0]]   # BGR -> RGB
    #      思考：mmcv 读图默认是 BGR（继承自 OpenCV），这个约定还会在哪里坑到你？
    #           （提示：data_preprocessor 里的 bgr_to_rgb，以及加载 torchvision 预训练权重时）
    #   f. 框：gt_instances.get('bboxes') 如果是 BaseBoxes 对象，换成 .tensor
    #      掩码：gt_instances.get('masks') 不为 None 时，mask2ndarray(...).astype(bool)
    #      再写回 data_sample.gt_instances = gt_instances
    #   g. visualizer.add_datasample(
    #          img_path, img, data_sample, draw_pred=False,
    #          show=not args.not_show, wait_time=args.show_interval, out_file=out_file)
    #   h. progress_bar.update()

    raise NotImplementedError('browse_dataset.py: main')


if __name__ == '__main__':
    main()


_ = osp  # 写完 TODO 后删掉这一行
