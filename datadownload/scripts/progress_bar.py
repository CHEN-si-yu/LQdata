#!/usr/bin/env python
"""进度条渲染 —— 给人在对话框里看的一页纸进度。

零 API 请求、零 parquet 读取：只读 state/*.json 和最新日志的最后一行，
所以随便多频繁地跑都不会影响到正在下载的主进程。

    /autodl-fs/data/miniconda3/bin/python scripts/progress_bar.py
    /autodl-fs/data/miniconda3/bin/python scripts/progress_bar.py --watch 30

判定「已完成」的口径：state/<数据集>.json 里的 last_run.finished_at 非空，
代表这个数据集在本轮里跑完过一次。正在跑的那个 last_run 是空的（跑完才写）。
"""
from __future__ import annotations

import ast
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
LOGS = ROOT / "logs"


def _exec_order() -> list[str]:
    """从 main.py 里读出真实的 DEFAULT_ORDER。

    故意用 ast 解析源码而不是 `import main` —— main 会连带 import engine→pandas，
    启动要一两秒；这个脚本要能随便多勤地跑。也不自己维护一份副本，
    否则 main.py 的顺序一变，进度条就开始说谎。
    """
    try:
        tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "DEFAULT_ORDER" for t in node.targets
            ):
                return list(ast.literal_eval(node.value))
    except (OSError, SyntaxError, ValueError):
        pass
    return []


def _disabled() -> set[str]:
    """`spec.py` 里 enabled=False 的数据集 —— 不参与执行，也不该显示成「待跑」。

    用户会用这个开关砍掉不需要的数据集（见 QUANT_PLATFORM.md §10.9）。
    导入 lingqi.spec 只要 0.07s 且不拉 pandas。
    """
    try:
        sys.path.insert(0, str(ROOT))
        from lingqi import spec as _S
        return {s.name for s in _S.all_specs() if not s.enabled}
    except Exception:
        return set()


ORDER = _exec_order()
DISABLED = _disabled()
ORDER = [n for n in ORDER if n not in DISABLED]
TOTAL = len(ORDER)

# `[index_ths_daily 逐实体]  13.4% 4492/33520 已用 16m22s 259/min ETA 1h53m14s rows=17.96万`
PROG_RE = re.compile(
    r"\[(?P<name>\S+)\s+(?P<mode>[^\]]+)\]\s+"
    r"(?P<pct>[\d.]+)%\s+(?P<cur>\d+)/(?P<tot>\d+)\s+"
    r"已用\s+(?P<elapsed>\S+)\s+(?P<rate>\d+)/min\s+ETA\s+(?P<eta>\S+)"
)


def bar(frac: float, width: int = 28) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def fmt_rows(n: int) -> str:
    if n >= 1e8:
        return f"{n / 1e8:.2f} 亿行"
    if n >= 1e4:
        return f"{n / 1e4:.0f} 万行"
    return f"{n:,} 行"


def load_state(name: str) -> dict:
    p = STATE / f"{name}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def is_done(name: str) -> bool:
    return bool(load_state(name).get("last_run", {}).get("finished_at"))


def rows_of(name: str) -> int:
    parts = load_state(name).get("partitions", {})
    return sum(int(p.get("rows", 0)) for p in parts.values())


def running_pids(pattern: str) -> list[str]:
    """扫 /proc 找进程。不用 pgrep/pkill —— 见 QUANT_PLATFORM.md §13.3。"""
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if pattern in cmd and "progress_bar" not in cmd:
            out.append(p.name)
    return out


def newest_log() -> Path | None:
    """最新的下载日志。

    ⚠️ 模式用 `run_*.log` 而不是 `run_full_*.log` —— 2026-09-14 踩过：
    重启时把日志命名成 `run_cyq_0914_1159.log`（因为那次只跑 cyq_chips），
    而这里只 glob `run_full_*`，于是本函数一直返回**上一个已结束进程的日志**，
    进度条显示的百分比/速率/ETA 全是旧进程的残留值（实测显示 6183/53109，
    而真实进度是 808/26478）。用户看到的是「进度在倒退」的假象。
    `run_*` 能同时覆盖 run_full / run_cyq / run_<数据集> 等各种命名，
    再按 mtime 取最新，天然免疫这个问题。
    """
    cands = sorted(
        list(LOGS.glob("run_*.log")) + list(LOGS.glob("catchup*.log")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def last_progress(log: Path) -> dict | None:
    """从日志尾部往回找最后一条进度行（读尾部 64KB 即可，别读整个大日志）。"""
    try:
        size = log.stat().st_size
        with log.open("rb") as f:
            if size > 65536:
                f.seek(size - 65536)
            tail = f.read().decode(errors="replace")
    except OSError:
        return None
    hits = list(PROG_RE.finditer(tail))
    return hits[-1].groupdict() if hits else None


def render() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []

    log = newest_log()
    prog = last_progress(log) if log else None
    # 日志里那条进度可能来自一个**已被禁用/已跑完**的数据集（日志是历史产物，
    # 进程停了它也不更新）。只认还在 ORDER 里的，否则会拿陈旧数据当"当前"。
    if prog and prog["name"] not in ORDER:
        prog = None
    main_alive = running_pids("main.py run")
    watch_alive = running_pids("catchup_after_main")

    done = [n for n in ORDER if is_done(n)]
    n_done = len(done)
    # 正在跑的既不算「已完成」也不算「待跑」——它单独占第二行
    current = prog["name"] if prog else None
    pending = [n for n in ORDER if not is_done(n) and n != current]

    total_rows = sum(rows_of(n) for n in ORDER)
    # dump 通道按级别落到 stock_dump_*，是 spec 名之外的孤立目录，单独算
    for extra in ("stock_dump_5min", "stock_dump_1min"):
        if (ROOT / "data" / extra).is_dir():
            total_rows += rows_of(extra)

    # ---- 阶段 ----
    if main_alive:
        phase = "① 主干下载中"
    elif prog and prog["name"] == "stock_cyq_chips":
        phase = "② 补跑 cyq_chips（看门狗）"
    elif watch_alive:
        phase = "② 看门狗补跑中"
    elif n_done >= TOTAL:
        phase = "🎉 全部完成"
    else:
        phase = "⚠️ 无进程在跑"

    # ---- 总进度 ----
    if prog:
        # 正在跑的这个算「半个」，进度条才不会卡在整数格上不动
        cur_frac = float(prog["pct"]) / 100.0
        overall = (n_done + cur_frac) / TOTAL
    else:
        cur_frac = 0.0
        overall = n_done / TOTAL

    lines.append(f"**{now}**  ·  {phase}")
    lines.append("")
    lines.append(
        f"总体  `{bar(overall)}`  **{n_done}/{TOTAL}** 数据集已完成  ·  {fmt_rows(total_rows)}"
    )

    # ---- 当前数据集 ----
    if prog:
        frac = float(prog["pct"]) / 100.0
        eta = prog["eta"]
        if eta.endswith("s") and "m" not in eta:
            eta = "即将完成"
        else:
            eta = re.sub(r"(\d+)s$", "", eta)      # 1h49m31s → 1h49m
        lines.append(
            f"当前  `{bar(frac)}`  **{prog['pct']}%**  `{prog['name']}` "
            f"({prog['cur']}/{prog['tot']})  ·  {prog['rate']}/min  ·  ETA {eta}"
        )
    elif main_alive:
        lines.append("当前  `" + bar(0.0) + "`  启动中，等待首条进度…")

    # ---- 剩余清单 ----
    if pending:
        head = " ".join(pending[:6])
        more = f" … 共 {len(pending)} 个" if len(pending) > 6 else ""
        lines.append("")
        lines.append(f"待跑  {head}{more}")

    # ---- 关键提醒 ----
    cyq_pending = (STATE / "stock_cyq_chips.json").exists() and not is_done("stock_cyq_chips")
    notes = []
    if cyq_pending and not watch_alive and not main_alive:
        notes.append("⚠️ **看门狗没在跑，cyq_chips 不会自动补 —— 需要重新拉起**（见文档 §11.2）")
    if not main_alive and not watch_alive and n_done < TOTAL:
        notes.append("⚠️ 两个进程都没了，但数据没跑完 → 检查 `logs/`")
    if notes:
        lines.append("")
        lines.extend(notes)

    return "\n".join(lines)


def main() -> int:
    if "--watch" in sys.argv:
        i = sys.argv.index("--watch")
        interval = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 60
        try:
            while True:
                print("\033[2J\033[H", end="")   # 清屏刷新
                print(render(), flush=True)
                time.sleep(interval)
        except KeyboardInterrupt:
            return 0
    print(render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
