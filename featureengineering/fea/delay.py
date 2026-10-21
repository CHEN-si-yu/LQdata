"""Project-owned availability contracts for after-close factor production.

Only project conf/frequency.yaml and project state/delay_history.json are read.
The positional upstream_root argument remains compatible with older callers;
it is deliberately not used to discover another project's scripts or state.
"""
from __future__ import annotations

import json
from pathlib import Path
import logging

log = logging.getLogger("fea.delay")
_CACHE: dict[tuple[str, str], dict[str, int]] = {}


def _local(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"可得性契约路径必须在因子项目内：{path}")
    return path


def _sources(upstream_root=None, *, project_root=None, contract_path=None,
             history_path=None):
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    freq = _local(root, str(contract_path or "conf/frequency.yaml"))
    hist = _local(root, str(history_path or "state/delay_history.json"))
    return hist, freq


def _from_frequency(path: Path) -> dict[str, int]:
    import yaml
    if not path.exists():
        raise FileNotFoundError(f"缺少项目内可得性契约：{path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw.get("datasets"), dict):
        raise ValueError(f"可得性契约 datasets 必须是映射：{path}")
    out = {}
    for name, rec in raw["datasets"].items():
        if not isinstance(rec, dict):
            raise ValueError(f"可得性契约 {name} 必须是映射")
        value = rec.get("delay_days", 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name}.delay_days 必须是非负交易日整数")
        out[str(name)] = value
    return out


def _recent_by_table(path: Path, n: int = 3) -> dict[str, list[int]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for day in sorted(raw, reverse=True):
        for name, rec in (raw.get(day, {}).get("tables") or {}).items():
            value = rec.get("delay_obs")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                observations = out.setdefault(name, [])
                if len(observations) < n:
                    observations.append(value)
    return out


def load(upstream_root=None, refresh=False, *, project_root=None,
         contract_path=None, history_path=None) -> dict[str, int]:
    hist, freq = _sources(project_root=project_root, contract_path=contract_path,
                          history_path=history_path)
    key = (str(freq), str(hist))
    if key in _CACHE and not refresh:
        return dict(_CACHE[key])
    declared = _from_frequency(freq)
    recent = _recent_by_table(hist)
    out = {}
    for name, value in declared.items():
        observations = recent.get(name, [])
        if len(observations) >= 3 and all(v > value for v in observations[:3]):
            log.warning("%s 最近三次项目内观测 %s 超出声明；请复核 %s",
                        name, observations[:3], freq)
            # Observation is evidence for review, not a semantic rewrite.
            # The audit gate blocks production until the declared contract
            # is explicitly reconciled. Registration/recipes stay immutable.
        if value:
            out[name] = value
    _CACHE[key] = out
    return dict(out)


def audit_contract(upstream_root=None, *, project_root=None,
                   contract_path=None, history_path=None) -> dict:
    hist, freq = _sources(project_root=project_root, contract_path=contract_path,
                          history_path=history_path)
    declared = _from_frequency(freq)
    recent = _recent_by_table(hist)
    rows, unexplained = [], []
    for name in sorted(set(declared) | set(recent)):
        value = declared.get(name)
        observations = recent.get(name, [])[:3]
        row = {"table": name, "declared": value, "observed": observations or None}
        if value is None:
            row["why"] = "项目内观测出现了未声明的数据集"
            unexplained.append(row)
        elif len(observations) >= 3 and all(v > value for v in observations):
            row["why"] = f"最近三次项目内观测 {observations} 超出声明 delay={value}"
            unexplained.append(row)
        rows.append(row)
    return {"declared_n": len(declared), "authoritative_n": len(declared),
            "authoritative_state": "ok", "authoritative_source": str(freq),
            "observed_n": len(recent), "unexplained": unexplained,
            "conservative": [], "acked": [], "ack_file": str(freq),
            "observed": bool(recent), "rows": rows}
