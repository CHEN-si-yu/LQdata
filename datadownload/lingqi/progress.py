"""进度可视化。

两种呈现方式，按运行环境自动切换：
  * 终端（TTY）：tqdm 动态进度条，带 耗时/速率/ETA
  * 重定向（nohup / > log）：不刷屏，改为按时间间隔打印一行摘要

另外持续把状态写进 state/status.json，长跑时随时 `cat` 即可看进度。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm


def fmt_duration(sec: float) -> str:
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def fmt_rows(n: int) -> str:
    n = int(n)
    if n >= 10**8:
        return f"{n / 10**8:.2f}亿"
    if n >= 10**4:
        return f"{n / 10**4:.2f}万"
    return f"{n:,}"


def is_tty() -> bool:
    try:
        return sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


class SilentBar:
    """非 TTY 环境的替身：按时间间隔打印摘要行，不刷屏。"""

    def __init__(self, desc: str, total: int | None, interval: float = 15.0):
        self.desc = desc.strip()
        self.total = total or 0
        self.n = 0
        self.postfix = ""
        self.t0 = time.time()
        self._last_print = 0.0
        self.interval = interval

    def set_postfix(self, **kw) -> None:
        if kw:
            self.postfix = " ".join(f"{k}={v}" for k, v in kw.items())

    def set_description(self, desc: str = "", refresh: bool = True) -> None:
        if desc:
            self.desc = desc.strip()

    def update(self, n: int = 1) -> None:
        self.n += n
        now = time.time()
        if now - self._last_print < self.interval and self.n < self.total:
            return
        self._last_print = now
        el = now - self.t0
        pct = f"{self.n / self.total * 100:5.1f}%" if self.total else "  --- "
        rate = f"{self.n / el * 60:.0f}/min" if el > 1 else "..."
        eta = ""
        if self.total and self.n:
            eta = f" ETA {fmt_duration(el / self.n * (self.total - self.n))}"
        print(f"    [{self.desc}] {pct} {self.n}/{self.total} 已用 {fmt_duration(el)} "
              f"{rate}{eta} {self.postfix}".rstrip(), flush=True)

    def close(self) -> None:
        pass


class StatusWriter:
    """把当前状态写进 state/status.json，供外部随时查看。"""

    def __init__(self, state_dir: Path, enabled: bool = True):
        self.path = Path(state_dir) / "status.json"
        self.enabled = enabled
        self._last = 0.0

    def update(self, force: bool = False, **kw) -> None:
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self._last < 1.0:
            return
        self._last = now
        payload = {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
        try:
            tmp = self.path.with_suffix(".json.tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass


class Board:
    """进度条工厂 + 全局状态维护。"""

    def __init__(self, status: StatusWriter | None = None, tty: bool | None = None):
        self.status = status or StatusWriter(Path("state"))
        self.tty = is_tty() if tty is None else tty
        self.overall_bar = None
        self.t0 = time.time()
        self.current = ""
        self.detail = ""
        self.dataset = ""

    # ---- 进度条 ----
    def overall(self, total: int, desc: str = "总进度"):
        if not self.tty:
            self.overall_bar = SilentBar(desc, total, interval=30.0)
            return self.overall_bar
        self.overall_bar = tqdm(
            total=total, desc=desc, position=0, leave=True,
            bar_format="{desc} |{bar:32}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )
        return self.overall_bar

    def task(self, total: int | None, desc: str):
        self.current = desc
        if not self.tty:
            return SilentBar(desc, total)
        return tqdm(
            total=total, desc=f"   {desc}", position=1, leave=False,
            bar_format="   {desc} |{bar:28}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

    def tick(self, bar, n: int = 1, **postfix) -> None:
        if bar is not None:
            if postfix:
                bar.set_postfix(**postfix)
            bar.update(n)
        if postfix:
            self.detail = " ".join(f"{k}={v}" for k, v in postfix.items())
        self.status.update(current=self.current, dataset=self.dataset,
                           detail=self.detail, elapsed=fmt_duration(time.time() - self.t0))

    def note(self, msg: str) -> None:
        if self.tty and self.overall_bar is not None:
            tqdm.write(msg)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
        self.detail = msg
        self.status.update(force=True, current=self.current, dataset=self.dataset,
                           detail=msg, elapsed=fmt_duration(time.time() - self.t0))
