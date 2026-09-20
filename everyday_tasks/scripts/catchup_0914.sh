#!/usr/bin/env bash
# 09-14 补跑看门狗 ——「删掉的一天」在厂商 502 时段没能补回，等收盘后再补。
#
# 背景（2026-09-15）：
#   14:2x 做「真删 09-14 → 重新下载」演练时撞上厂商 nginx 的 502 时段
#   （交易时段发作，15~30 分钟一个周期，与我们的限速无关）。
#   那一轮 759 请求 / 545 次重试 / 30.6 分钟，仍有 9 张表没补回
#   （5 张因 502，4 张是 delay=1 表按新规则要等 T 前进）。
#
# 本脚本：按给定时刻表（默认 15:30 / 18:00 / 21:00）逐轮补跑，
#        每轮「跑增量 → 观测 → 验收报告」，**全部补回就提前退出**。
#        所有输出落在 logs/catchup_0914_*.log 与 state/backtest/REPORT_restore_2026-09-14.md。
#
# 启动（脱离会话，CLAUDE.md 的既定做法）：
#   setsid nohup bash scripts/catchup_0914.sh < /dev/null > logs/catchup_0914.out 2>&1 &
# 停止：
#   cat state/catchup_0914.pid | xargs kill      # 只杀这个看门狗，不会误伤别的进程
set -u

PY=/autodl-fs/data/miniconda3/bin/python
ROOT=/root/autodl-fs/everyday_tasks
DATE=2026-09-14
# 时刻表：默认收盘后 3 轮；也可以 `bash catchup_0914.sh 15:40 20:00` 自己给
if [ "$#" -gt 0 ]; then STAMPS=("$@"); else STAMPS=(15:30 18:00 21:00); fi

cd "$ROOT" || exit 1
mkdir -p logs
echo $$ > state/catchup_0914.pid
trap 'rm -f state/catchup_0914.pid' EXIT

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

wait_until() {                      # 等到今天的 HH:MM（已过则立刻返回）
  local target="$1" now wait_s
  now=$(date +%s)
  local t_today
  t_today=$(date -d "today $target" +%s 2>/dev/null) || return 0
  wait_s=$(( t_today - now ))
  if [ "$wait_s" -gt 0 ]; then
    log "等到 $target（还有 $((wait_s / 60)) 分钟）…"
    sleep "$wait_s"
  fi
}

for stamp in "${STAMPS[@]}"; do
  wait_until "$stamp"
  run_log="logs/catchup_0914_$(date +%m%d_%H%M).log"
  log "===== 第 $stamp 轮：跑增量（输出 → $run_log）====="
  "$PY" -u main.py run --no-wait >>"$run_log" 2>&1
  log "增量退出码 $?（日志尾：$(tail -1 "$run_log")）"

  log "观测（main.py monitor）…"
  "$PY" main.py monitor >>"$run_log" 2>&1

  log "验收（scripts/verify_restore.py）…"
  "$PY" scripts/verify_restore.py --date "$DATE" >>"$run_log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    log "✔ 全部补回，看门狗结束。"
    exit 0
  fi
  log "仍有未补回的表（验收退出码 $rc），等下一个时刻再试。"
done

log "本轮时刻表跑完仍有缺口 —— 下次日更（T 前进后）会继续自动补。"
exit 1
