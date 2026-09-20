#!/bin/bash
# 重建后的验证四步（顺序固定，后三步的报告会被下游引用）：
#   ① check      基础体检（universe / 值域 / 复权口径 / 格式统一）   ~3 min
#   ② audit-pit  前视审计（静态扫描 + 抽样截断复算）                        ~20 min
#   ③ eval       有效性（IC / RankIC / 覆盖 / 分层单调性）                  ~5 min
#   ④ dedup      冗余检测（|ρ|≥0.95 簇）                                    ~6 min
#
# 用法（长任务请脱离会话）：
#   setsid nohup bash scripts/post_steps.sh logs/post_steps_$(date +%m%d).log < /dev/null > /dev/null 2>&1 &
cd "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1
PY=/autodl-fs/data/miniconda3/bin/python
LOG="${1:-logs/post_steps_$(date +%m%d_%H%M).log}"

{
  echo "=== post_steps 开始 $(date '+%F %T') ==="
  for cmd in "check" "audit-pit" "eval --years 2026 2026" "dedup"; do
    echo "=== main.py $cmd  $(date '+%F %T') ==="
    $PY main.py $cmd
    rc=$?
    echo "退出码 $rc"
    [ $rc -ne 0 ] && echo "⚠️ $cmd 退出码非 0，后续步骤继续"
  done
  echo "=== 全部完成 $(date '+%F %T') ==="
} >> "$LOG" 2>&1
