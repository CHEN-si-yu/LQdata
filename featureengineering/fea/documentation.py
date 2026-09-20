"""Maintain generated sections of the single project handbook, never replace its prose."""
from __future__ import annotations

import fcntl
import html
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

REPORT_END = "<!-- FEA:REPORTS:END -->"


def update_section(root: Path, key: str, body: str, *, report_title: str | None = None) -> Path:
    if not re.fullmatch(r"[a-z0-9-]+", key):
        raise ValueError("Invalid handbook section key")
    if "<!-- FEA:" in body:
        raise ValueError("Generated content must not contain handbook control markers")
    root = Path(root)
    out = root / "README.md"
    begin = f"<!-- FEA:DOC:{key}:BEGIN -->"
    end = f"<!-- FEA:DOC:{key}:END -->"
    block = begin + "\n" + body.rstrip() + "\n" + end
    lock_dir = root / "state"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / "documentation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = out.read_bytes()  # Missing handbook is an error, not a blank template.
        old = original.decode("utf-8")
        counts = (old.count(begin), old.count(end))
        if counts == (1, 1):
            a, b = old.index(begin), old.index(end)
            if b < a:
                raise ValueError("Reversed handbook section markers")
            new = old[:a] + block + old[b + len(end):]
        elif counts == (0, 0) and report_title is not None:
            report_begin = "<!-- FEA:REPORTS:BEGIN -->"
            if old.count(report_begin) != 1 or old.count(REPORT_END) != 1:
                raise ValueError("Missing or duplicate report area markers")
            if old.index(report_begin) >= old.index(REPORT_END):
                raise ValueError("Reversed report area markers")
            entry = ("<details><summary>" + html.escape(report_title) + "</summary>\n\n"
                     + block + "\n\n</details>\n\n")
            new = old.replace(REPORT_END, entry + REPORT_END)
        else:
            raise ValueError(f"Missing or duplicate handbook markers: {key}")
        if new.encode("utf-8") == original:
            return out
        mode = out.stat().st_mode & 0o777
        temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=root, prefix=".handbook-", suffix=".tmp",
                                             delete=False) as stream:
                temp = Path(stream.name)
                os.fchmod(stream.fileno(), mode)
                stream.write(new.encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            if out.read_bytes() != original:
                raise RuntimeError("README changed during generation; retry after the editor saves")
            os.replace(temp, out)
        finally:
            if temp is not None and temp.exists():
                temp.unlink()
    return out


def append_report(root: Path, kind: str, title: str, body: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return update_section(root, f"{kind}-{stamp}-{uuid4().hex[:8]}", body,
                          report_title=f"{title} · {stamp}")
