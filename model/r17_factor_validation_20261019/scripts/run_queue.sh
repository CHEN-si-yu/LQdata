#!/bin/bash
set -euo pipefail
ROOT=/autodl-fs/data/model/r17_factor_validation_20261019
PY=/autodl-fs/data/miniconda3/bin/python
export MX_THREADS=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for unit in A_V220 B_V220 A_V230 B_V230 A_V240 B_V240 A_V252 B_V252; do
  if [ "${unit#?}" = "_V252" ] && [ -f "$ROOT/units/$unit/model_info/final_audit.json" ]; then
    echo "[$(date '+%F %T')] $unit already analysed, skip" | tee -a "$ROOT/logs/queue.log"
    continue
  fi
  snap=A_260; [[ "$unit" == B_* ]] && snap=B_310
  echo "[$(date '+%F %T')] START $unit ($snap)" | tee -a "$ROOT/logs/queue.log"
  cd "$ROOT/units/$unit"
  MX_DATA="$ROOT/snapshots/$snap" "$PY" -B run.py --jobs 4 --device cuda >> "$ROOT/logs/${unit}_train.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] TRAIN_DONE $unit rc=$rc" | tee -a "$ROOT/logs/queue.log"
  if [ $rc -ne 0 ]; then echo "FAILED $unit" | tee -a "$ROOT/logs/queue.log"; exit $rc; fi
  MX_DATA="$ROOT/snapshots/$snap" "$PY" -B analysis.py --audit >> "$ROOT/logs/${unit}_analysis.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] ANALYSIS_DONE $unit rc=$rc" | tee -a "$ROOT/logs/queue.log"
  if [ $rc -ne 0 ]; then echo "ANALYSIS FAILED $unit" | tee -a "$ROOT/logs/queue.log"; exit $rc; fi
done
echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$ROOT/logs/queue.log"
