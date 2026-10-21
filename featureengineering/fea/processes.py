"""Detect active raw-data writers through writable file descriptors.

No other project's entrypoints, configuration or state are inspected. Only
process metadata and paths beneath the supplied raw upstream directory matter.
"""
from pathlib import Path
import os


def upstream_writer_pids(upstream_root=None, *, proc_root=None):
    if upstream_root is None:
        from .config import load
        upstream_root = load().upstream
    root = Path(upstream_root).resolve()
    proc_root = Path(proc_root or "/proc")
    if not proc_root.exists():
        return []
    busy = []
    for process in proc_root.iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            descriptors = list((process / "fd").iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = descriptor.resolve()
                if not target.is_relative_to(root):
                    continue
                info = (process / "fdinfo" / descriptor.name).read_text()
                flags = next(line.split()[1] for line in info.splitlines()
                             if line.startswith("flags:"))
                # Linux fdinfo flags are octal: O_RDONLY=0, O_WRONLY=1,
                # O_RDWR=2. Directory/read-only factor readers never match.
                if int(flags, 8) & os.O_ACCMODE in (os.O_WRONLY, os.O_RDWR):
                    busy.append(int(process.name))
                    break
            except (OSError, ValueError, StopIteration):
                continue
    return sorted(set(busy))
