"""tools/test.py —— 测试 / 评估入口（练习版）

================================================================================
对照原版：mmdetection v3.3.0 的 tools/test.py。

用法（写完后应该都能跑）：
    # 在验证集上算 COCO mAP
    python tools/test.py configs/xxx.py work_dirs/xxx/epoch_12.pth
    # 同时把可视化结果存到 work_dir/<时间戳>/show_dir
    python tools/test.py configs/xxx.py ckpt.pth --show-dir vis
    # 把原始预测结果 dump 成 pkl，之后可离线分析（tools/analysis_tools/ 里很多脚本吃这个）
    python tools/test.py configs/xxx.py ckpt.pth --out results.pkl
    # 测试时增强
    python tools/test.py configs/xxx.py ckpt.pth --tta

和 train.py 对比着写：两者 80% 一样，区别在于
    - 必须给 checkpoint，并写到 cfg.load_from
    - 最后调用 runner.test() 而不是 runner.train()
    - 多了"可视化"和"导出结果"两个可选功能，都是通过 **修改配置里的 Hook / Evaluator** 实现的
================================================================================
"""

import argparse
import os
import os.path as osp
import warnings

# TODO(0) 导入：
#   from mmengine.config import Config, ConfigDict, DictAction
#   from mmengine.registry import RUNNERS
#   from mmengine.runner import Runner
#   下面两个来自 mmdet，放到 main() 里需要时再 import，便于理解它们各自的作用：
#   from mmdet.engine.hooks.utils import trigger_visualization_hook
#   from mmdet.evaluation import DumpDetResults


def parse_args():
    parser = argparse.ArgumentParser(description='Test (and eval) a model')

    # TODO(1) 位置参数 'config' 和 'checkpoint'
    # TODO(2) '--work-dir'：保存评估指标的目录
    # TODO(3) '--out'：type=str，把预测结果 dump 成 .pkl 文件
    # TODO(4) '--show'：store_true，弹窗显示预测结果（服务器上一般不用）
    # TODO(5) '--show-dir'：可视化图片保存目录（相对于 work_dir/时间戳/）
    # TODO(6) '--wait-time'：type=float, default=2，--show 时每张图停留秒数
    # TODO(7) '--cfg-options'：同 train.py
    # TODO(8) '--launcher'：同 train.py
    # TODO(9) '--tta'：store_true
    # TODO(10) '--local_rank' / '--local-rank'：同 train.py，并设置 os.environ['LOCAL_RANK']

    raise NotImplementedError('tools/test.py: parse_args')


def main():
    # TODO(11) parse_args；Config.fromfile；cfg.launcher；merge cfg_options
    #          （和 train.py 完全一样 —— 思考：要不要抽成公共函数？mmdet 为什么没有抽？）

    # TODO(12) work_dir 优先级：同 train.py

    # TODO(13) cfg.load_from = args.checkpoint
    #          思考：为什么测试也用 load_from，而不是自己 torch.load？
    #          提示：Runner 会处理 state_dict 前缀（module.）、EMA 权重、map_location 等细节

    # TODO(14) 可视化：if args.show or args.show_dir:
    #              cfg = trigger_visualization_hook(cfg, args)
    #          读一下这个函数：它只是找到 default_hooks.visualization，把 draw=True 并填上参数。
    #          又一次说明：功能 = 配置里打开某个 Hook

    # TODO(15) TTA：if args.tta:
    #          - 先只支持配置里已经写好 tta_model 和 tta_pipeline 的情况，缺了就 raise 有意义的错误
    #            （原版会 warnings.warn 后补一个默认的翻转 TTA；写完后对照原版把这段补上）
    #          - cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)   # 用 TTA 模型包住原模型
    #          - cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
    #          思考：TTA 为什么被设计成"包住原模型的另一个模型"？（DetTTAModel 负责合并多次预测）

    # TODO(16) 构建 Runner：同 train.py

    # TODO(17) 导出结果：if args.out is not None:
    #          - 要求以 .pkl / .pickle 结尾
    #          - runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=args.out))
    #          思考：为什么"导出结果"被设计成一个 Metric，而不是写在 test 循环里？

    # TODO(18) runner.test()

    raise NotImplementedError('tools/test.py: main')


if __name__ == '__main__':
    main()


_ = (os, osp, warnings)  # 写完 TODO 后删掉这一行
