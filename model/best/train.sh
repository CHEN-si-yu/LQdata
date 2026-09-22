#!/usr/bin/env bash
# best（实战单元）一键流水线：36 个训练任务（9 组 × 4 折）→ 集成 → 60日门槛 → 出票。
#
# 与 V62/train.sh 同构：memory_guard.py 包住整个进程组，按 cgroup 现读内存上限提前停
# （84 GiB 提前停 / 90 GiB 硬顶 / anon 72 GiB 更严闸门）。训练与分析串行。
#
#   bash train.sh                 # 默认 6 并发
#   JOBS=4 bash train.sh          # 降低并发
#   bash train.sh --skip-analysis # 只训练，不出票
PY="${PYTHON:-/autodl-fs/data/miniconda3/bin/python}"
export MX_THREADS="${MX_THREADS:-3}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$PY" -u memory_guard.py "$PY" -u run.py --jobs "${JOBS:-6}" "$@"
