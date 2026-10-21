"""Resolve maintenance-script outputs inside the factor project."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_output(value: str | Path) -> Path:
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Output must stay inside the factor project: {path}")
    return path


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: _project_paths.py <project-local output file>")
    try:
        output = project_output(sys.argv[1])
    except ValueError as error:
        raise SystemExit(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    print(output)
