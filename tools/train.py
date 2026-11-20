"""tools/train.py —— 训练入口（练习版）

================================================================================
对照原版：mmdetection v3.3.0 的 tools/train.py（约 120 行）。
先自己按 TODO 写完，再打开原版逐行对比，把差异记到 docs/notes/。

用法（写完后应该都能跑）：
    # 单卡
    python tools/train.py configs/xxx.py
    # 指定输出目录 + 混合精度 + 覆盖配置
    python tools/train.py configs/xxx.py --work-dir work_dirs/exp1 --amp \
        --cfg-options train_dataloader.batch_size=4 optim_wrapper.optimizer.lr=0.01
    # 断点续训：自动找 work_dir 下最新的权重 / 指定权重
    python tools/train.py configs/xxx.py --resume
    python tools/train.py configs/xxx.py --resume work_dirs/xxx/epoch_6.pth
    # 多卡：见 tools/dist_train.sh

你会学到
    - 一个训练脚本真正需要做的事有多少：几乎全部工作都交给了 Config + Runner
    - 命令行参数与配置的优先级：命令行 > 配置文件 > 默认值
    - AMP / 自动缩放学习率 / 断点续训 分别是改配置里的哪个字段

整体流程（main 函数）：
    parse_args → Config.fromfile → 用命令行覆盖配置 → 构建 Runner → runner.train()
================================================================================
"""

import argparse
import os
import os.path as osp

# TODO(0) 导入需要的组件（都来自 mmengine，而不是 mmdet）：
#   from mmengine.config import Config, DictAction
#   from mmengine.registry import RUNNERS
#   from mmengine.runner import Runner
#
# 思考：mmdet 自己并没有写 Runner，为什么 mmengine 的 Runner 构建出来的却是
#       mmdet 的模型和数据集？
#   提示：看 mmdet 的 configs/_base_/default_runtime.py 里的 default_scope = 'mmdet'，
#   再看 mmengine/registry/default_scope.py。
#
# 注：原版还会调用 mmdet.utils.setup_cache_size_limit_of_dynamo()（torch.compile 相关），
#     这里省略，好让脚本只依赖 mmengine。


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    # TODO(1) 位置参数 'config'：配置文件路径
    #
    # TODO(2) '--work-dir'：日志和权重的保存目录
    #
    # TODO(3) '--amp'：action='store_true'，开启自动混合精度训练
    #
    # TODO(4) '--auto-scale-lr'：action='store_true'，按实际总 batch size 线性缩放学习率
    #         思考：配置里写的 lr 是按 8 卡 × 2 张图调的，你用 4 卡 × 4 张图要不要改 lr？
    #
    # TODO(5) '--resume'：nargs='?', type=str, const='auto'
    #         这个参数有三种状态，务必弄清楚：
    #           不写 --resume          -> args.resume = None   （不续训）
    #           只写 --resume          -> args.resume = 'auto' （自动找最新权重）
    #           --resume path/to.pth   -> args.resume = 路径    （从指定权重续训）
    #
    # TODO(6) '--cfg-options'：nargs='+', action=DictAction
    #         DictAction 会把 'a.b=1 c=[1,2]' 解析成 {'a.b': 1, 'c': [1, 2]}
    #         —— 这正是你在 mini_mm Config.merge_from_dict 里处理的输入
    #
    # TODO(7) '--launcher'：choices=['none', 'pytorch', 'slurm', 'mpi'], default='none'
    #         单卡用 none；torchrun 启动的多卡用 pytorch
    #
    # TODO(8) '--local_rank' 和 '--local-rank'（两种写法兼容新旧 torch），type=int, default=0
    #         老版本 torch.distributed.launch 会往命令行塞这个参数；
    #         解析完后要执行：
    #             if 'LOCAL_RANK' not in os.environ:
    #                 os.environ['LOCAL_RANK'] = str(args.local_rank)

    raise NotImplementedError('tools/train.py: parse_args')


def main():
    # TODO(9) args = parse_args()

    # TODO(10) 读取配置：cfg = Config.fromfile(args.config)
    #          再写入 cfg.launcher = args.launcher
    #          如果 args.cfg_options 不为 None：cfg.merge_from_dict(args.cfg_options)

    # TODO(11) 决定 work_dir，优先级：命令行 > 配置文件 > 默认
    #          if args.work_dir is not None:          cfg.work_dir = args.work_dir
    #          elif cfg.get('work_dir', None) is None: cfg.work_dir = osp.join(
    #                  './work_dirs', osp.splitext(osp.basename(args.config))[0])

    # TODO(12) 混合精度：if args.amp:
    #          把 cfg.optim_wrapper.type 改成 'AmpOptimWrapper'，并设置 loss_scale='dynamic'
    #          （如果已经是 AmpOptimWrapper，就只打印提示，不重复设置）
    #          思考：为什么 AMP 只需要换 OptimWrapper，而模型代码一行都不用改？
    #               提示：看 mmengine/optim/optimizer/amp_optimizer_wrapper.py 的 optim_context

    # TODO(13) 自动缩放学习率：if args.auto_scale_lr:
    #          要求配置里有 auto_scale_lr=dict(enable=..., base_batch_size=...)，
    #          两个 key 都在就 cfg.auto_scale_lr.enable = True，否则 raise RuntimeError 给出清晰提示

    # TODO(14) 断点续训：
    #          if args.resume == 'auto':
    #              cfg.resume = True;  cfg.load_from = None        # Runner 会自己去 work_dir 找最新的
    #          elif args.resume is not None:
    #              cfg.resume = True;  cfg.load_from = args.resume
    #          思考：load_from 与 resume 的区别是什么？
    #               （只加载模型权重做 fine-tune  vs  连 optimizer / epoch / lr 一起恢复）

    # TODO(15) 构建 Runner：
    #          if 'runner_type' not in cfg:  runner = Runner.from_cfg(cfg)
    #          else:                          runner = RUNNERS.build(cfg)
    #          思考：什么时候需要自定义 Runner？（mmdet 里几乎没有，这本身说明了 Hook 的威力）

    # TODO(16) runner.train()

    raise NotImplementedError('tools/train.py: main')


if __name__ == '__main__':
    main()


_ = (os, osp)  # 写完 TODO 后删掉这一行
