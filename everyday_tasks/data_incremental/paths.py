"""路径常量 —— 全工程唯一的路径真相源。

★★ 本工程**独立于** `/root/autodl-fs/datadownload` 的爬虫框架：
   **不 import 它的任何模块**，只对齐它的**数据格式**（parquet 分区 + state JSON）。
   接口的 path/参数/主键等配置是"参考"后抄进本工程自己的 `registry.py` 的。

数据落点与旧工程完全一致：
    <SHARED>/data/<数据集>/year=YYYY/data.parquet     年分区
    <SHARED>/data/<数据集>/data.parquet               快照（单文件）
    <SHARED>/state/<数据集>.json                      manifest（coverage/done/...）

本工程私有状态全在 <LOCAL> 下，绝不污染共享 state。
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- 根目录
# ⚠️ 用真实路径 `/autodl-fs/data`，不要用 `/root/autodl-fs` 符号链接 ——
#    网络盘上 resolve 出来的路径更稳定，且日志里打印出来不会有歧义。
SHARED = Path("/autodl-fs/data")
LOCAL = SHARED / "everyday_tasks"

# ---------------------------------------------------------------- 共享（与旧工程同一份）
DATA_ROOT = SHARED / "datadownload" / "data"
STATE_ROOT = SHARED / "datadownload" / "state"
CONF_ROOT = SHARED / "datadownload" / "conf"
API_KEY_FILE = SHARED / "datadownload" / "APIKey.txt"

# 与旧工程共用同一份限速台账 —— 限速是**按 apiKey 全局**的（280/min 红线），
# 两个工程各跑各的会叠加超限。共用这个文件才能保证两边加起来不超过配额。
RATELIMIT_FILE = STATE_ROOT / "ratelimit.json"
# daily_dump 的配额台账（每日期每天限 10 次，超限封禁该日期 3 天）—— 同样是服务端级的，共用
DUMP_QUOTA_FILE = STATE_ROOT / "dump_quota.json"
# 旧工程的单实例 pidfile：新工程启动时要检查它，避免两个工程同时写同一份 parquet
LEGACY_PIDFILE = STATE_ROOT / "run.pid"

# ---------------------------------------------------------------- 本工程私有
CONF_FILE = LOCAL / "conf" / "daily.yaml"
STATE = LOCAL / "state"
LOGS = LOCAL / "logs"

STATUS_FILE = STATE / "status.json"          # 进度可视化（不覆盖旧工程的 status.json！）
LOCK_FILE = STATE / "daily.lock"             # 跨项目单实例锁
DELAY_HISTORY = STATE / "delay_history.json"  # 每日 delay 观测
DELAY_RESOLVED = STATE / "delay_resolved.json"  # 自动标定出的 delay
CALENDAR_EXT = STATE / "calendar_ext.json"   # 从服务端返回值自愈出来的交易日
RUN_LOG = STATE / "run_log.jsonl"            # 每次运行一条记录（趋势图数据源）
LAST_REPORT = STATE / "last_report.json"     # 上次报告（机器可读）
DUMP_CALIB = STATE / "dump_calibration.json"  # daily_dump 的单位标定结果

# ---------------------------------------------------------------- daily_dump 按日缓存
# 用户 2026-09-15 要求：「daily_dump 可以落盘到 /root/autodl-fs/datadownload/data，
# 以单个日期为单位进行保存，以后优先读取本地是否有相关的数据」。
#
# 落点 = `DATA_ROOT/stock_daily_dump/date=YYYY-MM-DD/{data.parquet,meta.json}`
#   —— 目录名与 registry 里那条 `stock_daily_dump`（enabled=False，本身不独立落盘）
#      对齐，所以它一看就知道是「dump 通道的原始数据」，不会被误认成第 40 张表。
#   —— 每个日期一个目录 = 用户说的「以单个日期为单位」：删/看/换某一天互不影响，
#      也不必为了取一天而读整年分区。
#
# ★ 为什么值得单独存一份（而不是只存进 stock_history_5min）：
#   ① 落进 history 之前要过两道加工 —— 48 时刻白名单过滤、vol/amount 单位换算。
#      一旦判据或标定要改（例如将来发现 dump 的 vol 单位变了），有原始数据就
#      **不用重抓**（dump 配额是每日期每天 10 次，重抓一天要按天排队）。
#   ② 「删掉某一天再补回」不必再花请求：本地有就 0 请求补回（用户要的快速重建）。
#   ③ 只有最近 ~90 天能从 dump 拿到；存下来 = 把这 90 天的原始件固化在本地。
DUMP_CACHE_ROOT = DATA_ROOT / "stock_daily_dump"


def dump_cache_dir(day: str) -> Path:
    return DUMP_CACHE_ROOT / f"date={day}"


def dump_cache_path(day: str) -> Path:
    return dump_cache_dir(day) / "data.parquet"


def dump_cache_meta(day: str) -> Path:
    return dump_cache_dir(day) / "meta.json"
DAYHASH_DIR = STATE / "dayhash"              # ★ 单日 MD5 台账（一数据集一文件）
BACKTEST_DIR = STATE / "backtest"            # 回测的基线 / 备份 / 被删行


# 写出的文件一律设成"同组可读写"。`claude` 在 `root` 组里，所以 0664 能让
# root 与 claude 两个用户都读写同一份数据。
SHARED_FILE_MODE = 0o664


def chmod_shared(p) -> None:
    """把刚写出的文件设成同组可读写。

    ★★ 为什么必须这么做（2026-09-15 实测踩到）：
       `tempfile.mkstemp` 建出来的临时文件是 **0600、属主=当前用户**，
       `os.replace` 之后就是 0600 的正式文件。
       本工程**可能被 root 和 claude 两个用户先后执行**（用户要求"每天晚上重启一次
       Agent"，而 Agent 有时以 root 起、有时以 claude 起）。
       于是换一个用户跑的第一轮就出事了：
         · 36 个 manifest 变成 root:600 → claude 读不到；
         · 而 `Manifest.load` 对"读不到"是**静默当空**的 → `partitions={}` `coverage=[]`；
         · 这一轮结束再 `man.save()` → **把真实的 coverage/done/columns 覆盖成空**，
           `_missing_ranges` 随即认为"2010 年至今全是缺口" → 请求风暴 + 状态永久丢失。

       0664 让"换用户"不再有权限面。与之配套的是 `Manifest.load` 现在会**大声报错**
       而不是静默降级（防的是"读不到"被当成"没有"）。
    """
    try:
        import os
        os.chmod(p, SHARED_FILE_MODE)
    except OSError:
        pass


def ensure_dirs() -> None:
    """建齐本工程需要的目录（幂等）。"""
    for p in (STATE, LOGS, DAYHASH_DIR, BACKTEST_DIR):
        p.mkdir(parents=True, exist_ok=True)


def year_partition_path(dataset: str, year: int) -> Path:
    return DATA_ROOT / dataset / f"year={year}" / "data.parquet"


def flat_path(dataset: str) -> Path:
    return DATA_ROOT / dataset / "data.parquet"


def manifest_path(dataset: str) -> Path:
    return STATE_ROOT / f"{dataset}.json"
