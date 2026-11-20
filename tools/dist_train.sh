#!/usr/bin/env bash
# tools/dist_train.sh —— 单机 / 多机多卡训练启动脚本（练习版）
#
# 对照原版：mmdetection v3.3.0 的 tools/dist_train.sh（原版用 torch.distributed.launch，
#           它已被弃用，这里改用 torchrun，参数几乎一样）
#
# 用法（写完后）：
#   # 单机 8 卡
#   bash tools/dist_train.sh configs/xxx.py 8
#   # 额外参数原样传给 train.py
#   bash tools/dist_train.sh configs/xxx.py 8 --amp --work-dir work_dirs/exp1
#   # 两台机器各 8 卡（每台机器各执行一次，NODE_RANK 不同）
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash tools/dist_train.sh configs/xxx.py 8
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash tools/dist_train.sh configs/xxx.py 8
#
# 思考：
#   - torchrun 做了什么？它为每张卡起一个进程，并设置 RANK / LOCAL_RANK / WORLD_SIZE /
#     MASTER_ADDR / MASTER_PORT 环境变量。之后谁读这些变量？
#     （提示：train.py 的 --launcher pytorch → mmengine/dist/utils.py 的 init_dist）
#   - 同一台机器上同时跑两个实验，为什么第二个会报 "Address already in use"？怎么解决？

# TODO(1) 读取前两个位置参数：CONFIG=$1 和 GPUS=$2

# TODO(2) 读取环境变量，没设置时使用默认值（bash 语法 ${VAR:-默认值}）：
#         NNODES 默认 1，NODE_RANK 默认 0，PORT 默认 29500，MASTER_ADDR 默认 "127.0.0.1"

# TODO(3) 把项目根目录加入 PYTHONPATH，让 `import rdet` 能找到我们的插件：
#         PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
#         思考：如果已经 pip install -e . 了，这一步还需要吗？

# TODO(4) 启动：
#         torchrun --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR \
#             --nproc_per_node=$GPUS --master_port=$PORT \
#             $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
#         ${@:3} 表示"从第 3 个开始的所有参数"，用来把 --amp 等透传给 train.py
#         提示：PYTHONPATH=... 和 torchrun 写在同一条命令里（用 \ 续行），环境变量才只作用于这条命令

echo "TODO: 还没有完成 tools/dist_train.sh 练习" >&2
exit 1
