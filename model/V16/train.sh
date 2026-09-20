#!/usr/bin/env bash
# 单元训练调度 —— 折分批并发 + 断点续跑（与参考工程 train.sh 同构）
#
#   并发折数 = cgroup 内存 GB / 20   ← 参考工程血泪铁律：超限被系统 KILL，无警告
#   单折线程 = 可用核数 × 0.8 / 并发折数（可用 MX_THREADS 覆盖）
#
# 用法：
#   bash train.sh                     # 全部折（续跑：EXIT:0 的折自动跳过）
#   bash train.sh --folds "2 3"       # 只跑指定折
#   bash train.sh --smoke             # 冒烟（小模型，产物在 model_train/smoke/ 等桶内子树）
#   CONC=4 bash train.sh              # 手动指定并发
#   bash train.sh --clean             # 清空产物后全量重跑（★ 保住 logs/ 里的冻结配方）
#
# ★ 下面那句 `cd "$(dirname "$0")"` 是这段脚本能自包含的关键：它先把工作目录切到
#   **单元自己**，于是下面 heredoc 里的 `sys.path.insert(0, ".")` 命中单元自带的 mx/，
#   而不是调用者所在的目录。`ROOT="."` 同理。V12/V13/V14 的 train.sh 已经是这一版，
#   真正坏掉的是它们的 run.py/analysis.py（见那两个文件顶部的说明）。
set -u
cd "$(dirname "$0")"
UNIT="$(basename "$PWD")"
ROOT="."

PY="${PY:-/autodl-fs/data/miniconda3/bin/python}"
if [ ! -x "$PY" ]; then PY="python3"; fi

SMOKE=0
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --folds) FOLDS_LIST="${2:-}"; shift 2 ;;
    --clean)
      # ★ 三个产物桶（2026-09-20 起，契约见 mx/state.py）。
      #   logs/ 用「清空但保住配方」的写法：recipe.yaml / recipe.lock.json 也住在 logs/ 里，
      #   直接 `rm -rf logs` 会把 freeze 的成果一起抹掉（promote 与漂移检测都靠它）。
      rm -rf model_train model_pred
      find logs -mindepth 1 -maxdepth 1 ! -name 'recipe.yaml' ! -name 'recipe.lock.json' \
           -exec rm -rf {} + 2>/dev/null || true
      # 旧名（迁移期兼容：万一老目录还在，一并清掉）
      rm -rf artifacts artifacts_smoke sample_artifacts pred smoke smoke_pred smoke_logs \
             sample_pred sample_logs pics state smoke_state sample_state \
             __pycache__ train_master.log
      shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

export MX_ROOT="$ROOT"
# ★ 用 mapfile **按行**读三行：`read A B C` 会把多出来的字段全塞给 C（踩过：折列表被截成 "1"）
mapfile -t _DEF < <("$PY" - <<'PYEOF'
import os
import sys
sys.path.insert(0, ".")
from mx import unit as U
from mx.config import load
u = U.load_unit(load(), os.path.basename(os.getcwd()), log=lambda *a: None)
conc = U.default_conc(unit=u)
print(" ".join(str(i) for i in U.folds(u)))
print(conc)
print(U.default_threads(conc))
PYEOF
)
FOLDS_LIST="${FOLDS_LIST:-${_DEF[0]:-}}"
CONC="${CONC:-${_DEF[1]:-1}}"
export MX_THREADS="${MX_THREADS:-${_DEF[2]:-8}}"

if [ "$SMOKE" = "1" ]; then LOGDIR=smoke_logs; FLAG="--smoke"; else LOGDIR=logs; FLAG=""; fi
mkdir -p "$LOGDIR"

echo "[train.sh] 单元=$UNIT 折={$FOLDS_LIST} 并发=$CONC 单折线程=$MX_THREADS 冒烟=$SMOKE"
echo "[train.sh] 内存上限 $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo '?')  规则: 并发 = 可用内存×0.8 / 单折预算（按 trainingdata 的 X 体积估）"

run_batch() {
  local items="$1" pids=""
  for x in $items; do
    if grep -q "EXIT:0" "$LOGDIR/fold$x.log" 2>/dev/null; then
      echo "[train.sh] fold$x 已 EXIT:0 —— 跳过（续跑）"
      continue
    fi
    nohup "$PY" -u run.py "$x" $FLAG > "$LOGDIR/fold$x.log" 2>&1 &   # -u: 实时看进度
    pids="$pids $!"
    sleep 3
  done
  [ -n "$pids" ] && wait $pids
  for x in $items; do
    if grep -q "EXIT:0" "$LOGDIR/fold$x.log" 2>/dev/null; then
      echo "[train.sh] fold$x ✔"
    else
      echo "[train.sh] fold$x ✘（看 $LOGDIR/fold$x.log 末尾）"
    fi
  done
}

i=0; B=""
for f in $FOLDS_LIST; do
  B="$B $f"; i=$((i+1))
  if [ $i -ge $CONC ]; then echo "[train.sh] 批次:{ $B }"; run_batch "$B"; B=""; i=0; fi
done
[ -n "${B// /}" ] && { echo "[train.sh] 批次:{ $B }"; run_batch "$B"; }

ok=0
for f in $FOLDS_LIST; do grep -q "EXIT:0" "$LOGDIR/fold$f.log" 2>/dev/null && ok=$((ok+1)); done
echo "[train.sh] TRAIN DONE ($ok/$(echo $FOLDS_LIST | wc -w) 折成功)  下一步： python analysis.py"
