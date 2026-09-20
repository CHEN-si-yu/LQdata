#!/bin/bash
# ============================================================================
#  主下载跑完后，自动补跑两个"被推迟 / 需扩容"的数据集。
#
#  背景（详见 ../QUANT_PLATFORM.md 的 11.2 / 11.3）：
#   1) stock_cyq_chips 被**故意推迟**了 —— 正在跑的主进程加载的是旧版引擎，
#      对单年 1.4 亿行的分区会退化成 O(n²)（实测外推写盘要 120 小时）。
#      现在 state/stock_cyq_chips.json 里是**占位 done 标记，不是真数据**。
#   2) index_weight 的 200 指数扩容要等下一次运行才生效（进程内存里还是旧的 16 个）。
#
#  为什么用脚本而不是等人来敲命令：主进程还要跑好几个小时，
#  而补跑是长任务（cyq_chips 约 13 小时），不能依赖某个交互会话还活着。
#
#  启动方式（幂等，重复启动没有意义，但也不会互相破坏）：
#    cd /root/autodl-fs/datadownload
#    setsid nohup bash scripts/catchup_after_main.sh < /dev/null > /dev/null 2>&1 &
#
#  进度看：logs/catchup.log，以及各数据集自己的 logs/run_full_*.log
# ============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/autodl-fs/data/miniconda3/bin/python
mkdir -p logs
# 自己开日志（而不是靠调用方重定向）：这样文件属主是运行者，
# 不会出现 root 建的文件 claude 用户写不了的情况。
exec >> logs/catchup.log 2>&1

log() { echo "[$(date '+%F %T')] $*"; }
log "=== catchup 启动（PID $$，用户 $(id -un)）==="

# 只在没有别的 catchup 在跑时才继续（防重复启动）
if [ -f state/.catchup.lock ]; then
    other=$(cat state/.catchup.lock 2>/dev/null || echo "")
    if [ -n "$other" ] && kill -0 "$other" 2>/dev/null; then
        echo "[$(date '+%F %T')] 已有 catchup 在跑（PID $other），退出。"
        exit 0
    fi
fi
echo $$ > state/.catchup.lock


# ★ 等主下载结束只能用**心跳**判断，不能用 `ps | grep 'main.py run'`：
#   模块② featureengineering/main.py 也是同名入口，实测会让本脚本永远等下去。
#   state/status.json 被下载进程每 ~15 秒重写一次；超过 5 分钟没更新就算它结束了。
log "等待主下载结束（看 state/status.json 是否还在更新）..."
HB=state/status.json
while true; do
    [ -f "$HB" ] || break
    age=$(( $(date +%s) - $(stat -c %Y "$HB" 2>/dev/null || echo 0) ))
    if [ "$age" -gt 300 ]; then
        log "status.json 已 $age 秒没更新，判定主干已结束"
        break
    fi
    sleep 120
done
log "主进程已结束。当前已落地情况："
$PY main.py status 2>&1 | tail -5

# 跑一个数据集，遇到"已有实例在跑"就等一会儿重试。
# ★ 为什么必须重试：主干如果被人重启（实测 2026-09-13 17:02 发生过一次），
#   心跳会中断 >5 分钟让本脚本误判"主干结束"；此时启动会被单实例保护拒绝（退出码 2），
#   不重试的话脚本会一路跑到底、打印"全部补跑完成"然后退出 —— **补跑其实一次都没跑**。
run_with_retry() {
    ds=$1
    n=0
    while :; do
        $PY main.py run "$ds"
        rc=$?
        if [ "$rc" -ne 2 ]; then
            return "$rc"
        fi
        n=$((n + 1))
        if [ "$n" -ge 72 ]; then            # 最多等 6 小时
            log "✘ $ds 连续被拒 72 次（约 6 小时），放弃。多半是主干一直在跑，下次再补。"
            return 2
        fi
        log "  ↻ $ds 启动被拒（仍有下载实例在跑），5 分钟后重试（第 $n 次）"
        sleep 300
    done
}

# 数据集在 lingqi/spec.py 里是否 enabled。
# ★ 补跑清单**跟随 spec 的开关**，不在这里硬编码 —— 用户想停/想恢复某个数据集时
#   只改 spec.py 一处，看门狗自动跟着变，不会出现"spec 关了但脚本还在下"的错位。
is_enabled() {
    $PY -c "
import sys; sys.path.insert(0, '.')
from lingqi import spec as S
raise SystemExit(0 if S.get('$1').enabled else 1)
" 2>/dev/null
}

# ---------------------------------------------------------------- 1) cyq_chips
if is_enabled stock_cyq_chips; then
    if [ -f state/stock_cyq_chips.json ]; then
        mv state/stock_cyq_chips.json state/stock_cyq_chips.json.deferred.bak
        log "已移走占位状态 → state/stock_cyq_chips.json.deferred.bak（那是假的 done 标记）"
    fi
    log "▶ 开始补跑 stock_cyq_chips（2018 起，约 12.7 亿行 / 预计 13 小时）"
    run_with_retry stock_cyq_chips
    log "✔ stock_cyq_chips 结束（退出码 $?）"
else
    log "⏭ 跳过 stock_cyq_chips（spec.py 里 enabled=False）"
fi

# -------------------------------------------------------------- 2) index_weight
if is_enabled index_weight; then
    log "▶ 开始补跑 index_weight（200 指数扩容；已完成的会自动跳过）"
    run_with_retry index_weight
    log "✔ index_weight 结束（退出码 $?）"
else
    log "⏭ 跳过 index_weight（spec.py 里 enabled=False）"
fi

log "=== 全部补跑完成 ==="
$PY main.py status 2>&1 | tail -8
rm -f state/.catchup.lock
