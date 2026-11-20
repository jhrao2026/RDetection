#!/usr/bin/env bash
# tools/dist_test.sh —— 多卡测试启动脚本（练习版）
#
# 对照原版：mmdetection v3.3.0 的 tools/dist_test.sh
#
# 用法（写完后）：
#   bash tools/dist_test.sh configs/xxx.py work_dirs/xxx/epoch_12.pth 8
#   bash tools/dist_test.sh configs/xxx.py ckpt.pth 8 --out results.pkl
#
# 先写完 dist_train.sh，这个就是照着改：
#   - 位置参数变成三个：CONFIG=$1  CHECKPOINT=$2  GPUS=$3
#   - 调用 test.py，把 $CHECKPOINT 也传进去
#   - 透传参数从第 4 个开始：${@:4}
#
# 思考：多卡测试时，每张卡只跑了一部分数据，最终的 mAP 是怎么算出来的？
#   （提示：mmengine/evaluator/metric.py 的 BaseMetric.evaluate 里的 collect_results）

# TODO(1) 读取位置参数和环境变量（同 dist_train.sh）
# TODO(2) PYTHONPATH + torchrun 启动 test.py

echo "TODO: 还没有完成 tools/dist_test.sh 练习" >&2
exit 1
