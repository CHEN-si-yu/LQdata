"""进度与可视化原语（纯 stdlib）。

设计取舍：**不依赖 tqdm**，用 ASCII 条自己画 —— 因为：
  ① 环境里没有 matplotlib，图表本来就要手搓；
  ② 本工程要跑在 nohup/日志里，非 TTY 场景才是常态；
  ③ 少一个依赖，跨环境更稳。

输出约定沿用旧工程的惯例（emoji 是状态标记）：
    ✔ 成功   ✘ 失败   ▶ 开始   ⏳ 等待   ⚠️ 警告   🔴 异常   ★ 重点
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

from .. import paths

# ---------------------------------------------------------------- 格式化
def fmt_rows(n: float) -> str:
    n = float(n or 0)
    if n >= 1e8:
        return f"{n / 1e8:.2f}亿"
    if n >= 1e4:
        return f"{n / 1e4:.2f}万"
    return f"{int(n):,}"


def fmt_duration(sec: float) -> str:
    sec = int(sec or 0)
    if sec >= 3600:
        return f"{sec // 3600}h{sec % 3600 // 60}m{sec % 60}s"
    if sec >= 60:
        return f"{sec // 60}m{sec % 60}s"
    return f"{sec}s"


def bar(frac: float, width: int = 24, full: str = "█", empty: str = "░") -> str:
    frac = max(0.0, min(1.0, float(frac)))
    k = int(round(frac * width))
    return full * k + empty * (width - k)


def is_tty() -> bool:
    try:
        return sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


# ---------------------------------------------------------------- 状态文件
class StatusWriter:
    """写 <本工程 state>/status.json（**不覆盖**旧工程的）。

    旧工程的 `progress_bar.py` 读的是 `datadownload/state/status.json`；本工程写自己的，
    两份互不干扰 —— 否则两边会互相覆盖出一堆错乱进度。
    """

    def __init__(self, path: Path | None = None, min_interval: float = 1.0):
        self.path = Path(path) if path else paths.STATUS_FILE
        self.min_interval = min_interval
        self._last = 0.0

    def update(self, **fields: object) -> None:
        now = time.monotonic()
        if now - self._last < self.min_interval:
            return
        self._last = now
        self._write(fields)

    def force(self, **fields: object) -> None:
        self._last = time.monotonic()
        self._write(fields)

    def _write(self, fields: dict) -> None:
        try:
            cur: dict = {}
            if self.path.exists():
                try:
                    v = json.loads(self.path.read_text(encoding="utf-8"))
                    cur = v if isinstance(v, dict) else {}
                except (json.JSONDecodeError, OSError):
                    cur = {}
            cur.update(fields)
            cur["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass          # 共享盘抖动不该炸主流程


# ---------------------------------------------------------------- 进度条
class Bar:
    """一个任务的进度条。TTY 下原地刷新，非 TTY 下按间隔打摘要行。"""

    def __init__(self, total: int, desc: str, status: StatusWriter | None = None,
                 interval: float = 15.0, width: int = 28):
        self.total = max(1, int(total))
        self.desc = desc
        self.n = 0
        self.t0 = time.monotonic()
        self.status = status
        self.interval = interval
        self.width = width
        self._last_print = 0.0
        self._tty = is_tty()
        self.extra = ""

    def tick(self, n: int = 1, **postfix: object) -> None:
        self.n += n
        if postfix:
            self.extra = " ".join(f"{k}={v}" for k, v in postfix.items())
        self._render()

    def _render(self) -> None:
        now = time.monotonic()
        if self._tty:
            el = now - self.t0
            frac = self.n / self.total
            rate = self.n / el * 60 if el > 0 else 0
            eta = (self.total - self.n) / (rate / 60) if rate > 0 else 0
            line = (f"\r  [{self.desc}] {bar(frac, self.width)} {frac * 100:5.1f}% "
                    f"{self.n}/{self.total} 已用 {fmt_duration(el)} "
                    f"{rate:.0f}/min ETA {fmt_duration(eta)} {self.extra}")
            sys.stderr.write(line[:200].ljust(160))
            sys.stderr.flush()
            if self.status:
                self.status.update(current=self.desc, detail=line.strip())
            return
        if now - self._last_print < self.interval and self.n < self.total:
            return
        self._last_print = now
        el = now - self.t0
        frac = self.n / self.total
        rate = self.n / el * 60 if el > 0 else 0
        eta = (self.total - self.n) / (rate / 60) if rate > 0 else 0
        print(f"    [{self.desc}] {frac * 100:5.1f}% {self.n}/{self.total} "
              f"已用 {fmt_duration(el)} {rate:.0f}/min ETA {fmt_duration(eta)} {self.extra}",
              flush=True)
        if self.status:
            self.status.update(current=self.desc, detail=f"{self.desc} {frac * 100:.1f}%")

    def close(self) -> None:
        if self._tty:
            sys.stderr.write("\n")
            sys.stderr.flush()


class Runner:
    """顶层：终端宽度、清屏、状态出口。"""

    def __init__(self, status_path: Path | None = None):
        self.status = StatusWriter(status_path)
        self.width = shutil.get_terminal_size((110, 24)).columns
        self.bar: Bar | None = None

    def note(self, msg: str) -> None:
        """打一行不破坏进度条的日志。"""
        if is_tty() and self.bar is not None:
            sys.stderr.write("\r" + " " * min(160, self.width) + "\r")
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
        self.status.force(detail=msg)

    def task(self, total: int, desc: str, interval: float = 15.0) -> Bar:
        self.bar = Bar(total, desc, status=self.status, interval=interval)
        return self.bar

    def clear_screen(self) -> None:
        if is_tty():
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
