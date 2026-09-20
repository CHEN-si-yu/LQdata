"""跨项目单实例锁。

为什么需要"跨项目"：本工程与旧工程 `datadownload/main.py run` 写的是**同一份**
`data/*.parquet` 和 `state/*.json`，而共享盘是网络文件系统、**没有文件锁保证**。
两个进程同时读-改-写同一个分区 → 后写的把先写的覆盖掉 → 静默丢数据。

所以本锁做两件事：
    1. 启动时先看旧工程的 pidfile（`datadownload/state/run.pid`）有没有活进程
    2. 再用 `O_CREAT|O_EXCL` 抢自己的锁文件（放在**共享** state 下，两边都能看见）

⚠️ 网络盘上 `O_EXCL` 也不是真锁 —— 极端并发下两个进程仍可能同时创建成功。
   真正的保护是 pidfile 探测 + "每天只跑一次"这个使用前提。
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

from .. import paths

STALE_AFTER_SECONDS = 12 * 3600


class AlreadyRunning(RuntimeError):
    """已有实例在跑（本工程或旧工程）。"""


def _pid_alive(pid: int) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # 存在但没权限发信号 = 活着
    except OSError:
        return False


def _read_json(path: Path) -> dict:
    try:
        v = json.loads(path.read_text(encoding="utf-8"))
        return v if isinstance(v, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def legacy_pids() -> list[int]:
    """旧工程 `datadownload` 的 main.py run 是否在跑。"""
    info = _read_json(paths.LEGACY_PIDFILE)
    pid = int(info.get("pid") or 0)
    if pid and _pid_alive(pid):
        return [pid]
    return []


class LockHandle:
    def __init__(self, path: Path, owned: bool, info: dict | None = None):
        self.path = path
        self.owned = owned
        self.info = info or {}

    def release(self) -> None:
        if not self.owned:
            return
        try:
            cur = _read_json(self.path)
            if int(cur.get("pid") or 0) == os.getpid():
                self.path.unlink(missing_ok=True)
        except OSError:
            pass

    def __enter__(self) -> "LockHandle":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def acquire(force: bool = False) -> LockHandle:
    """抢锁。抢不到抛 AlreadyRunning（除非 force=True 且对方已死）。"""
    # 1) 旧工程在跑 → 直接拒绝（它是全量回填，会长时间占着）
    if not force:
        pids = legacy_pids()
        if pids:
            raise AlreadyRunning(
                f"旧工程 datadownload 的 main.py run 正在运行（PID {pids}）。\n"
                f"   两边写同一份 data/state，共享盘没有文件锁，同时跑会互相覆盖。\n"
                f"   等它跑完，或确认是陈旧 pidfile 后加 --force。")

    # 2) 抢自己的锁
    paths.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "started_ts": time.time(),
        "argv": " ".join(sys.argv[:6]),
    }
    for attempt in (1, 2):
        try:
            fd = os.open(paths.LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            return LockHandle(paths.LOCK_FILE, owned=True, info=payload)
        except FileExistsError:
            info = _read_json(paths.LOCK_FILE)
            pid = int(info.get("pid") or 0)
            age = time.time() - float(info.get("started_ts") or 0)
            alive = _pid_alive(pid)
            stale = (not alive) or age > STALE_AFTER_SECONDS
            if stale or force:
                # 陈旧锁（进程死了 / 超过 12 小时 / 内容坏了）→ 改名归档后重抢一次
                try:
                    os.replace(paths.LOCK_FILE,
                               paths.LOCK_FILE.with_suffix(f".stale.{int(time.time())}"))
                except OSError:
                    pass
                if attempt == 2:
                    break
                continue
            raise AlreadyRunning(
                f"本工程的另一个实例正在运行：PID {pid} @ {info.get('host')} "
                f"（{info.get('started')}）。\n   如确认它是死的，加 --force。")
    # 两次都没抢到
    info = _read_json(paths.LOCK_FILE)
    raise AlreadyRunning(f"抢锁失败，锁文件被占用：{info}")
