"""Frozen daily cross-sectional task and causal, whole-date split definitions."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import yaml


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Protocol:
    name: str = "fixed_mainboard_daily_v1"
    expected_stocks: int = 2115
    output_start: str = "2018-01-01"
    primary_horizons: tuple = (1, 5)
    auxiliary_horizons: tuple = (3, 10, 20)
    label: str = "open_hfq(t+1+h)/open_hfq(t+1)-1;both_endpoints_traded"
    value_transform: str = "cs_winsor_1_99_zscore_clip10_direction_v1"
    initial_train_end: str = "2021-12-31"
    outer_start: str = "2022-01-01"
    inner_validation_days: int = 126
    min_train_days: int = 252
    block_days: int = 20
    bins: int = 10
    min_pairs: int = 30
    interpretation: str = "retrospective_screening;historical_selection_is_not_untouched_holdout"

    @property
    def horizons(self):
        return tuple(sorted(set(self.primary_horizons + self.auxiliary_horizons)))

    @property
    def purge_days(self):
        # Signal t consumes open at t+1+h. Last training endpoint must precede
        # the next validation signal date. This also excludes boundary equality.
        return max(self.horizons) + 1

    def to_dict(self):
        return asdict(self)

    @property
    def identity(self):
        return digest(self.to_dict())


def load_protocol(cfg, path=None):
    p = path or cfg.root / "conf" / "research" / "protocol.yaml"
    if not p.exists():
        return Protocol()
    payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    unknown = set(payload) - set(Protocol.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown protocol keys: {sorted(unknown)}")
    for key in ("primary_horizons", "auxiliary_horizons"):
        if key in payload:
            payload[key] = tuple(int(h) for h in payload[key])
    result = Protocol(**payload)
    if result.expected_stocks != 2115 or result.output_start != "2018-01-01":
        raise ValueError("Current research contract requires the frozen 2115-stock pool and 2018-01-01 floor")
    if result.primary_horizons != (1, 5) or result.horizons != (1, 3, 5, 10, 20):
        raise ValueError("Current task requires primary horizons 1/5 and auxiliary 3/10/20")
    if result.block_days < result.purge_days - 1 or result.min_pairs < 3:
        raise ValueError("Invalid block_days or min_pairs")
    return result


def _day(v):
    return int(str(v).replace("-", ""))


def walk_forward_splits(dates, protocol=None):
    """Quarterly outer tests, earlier inner validation, expanding earlier train.

    The result is a plan, never evidence that a model has been fitted. All split
    axes are complete trading dates. Purging uses local calendar indices, not
    calendar-day subtraction and never a random stock-day split.
    """
    p = protocol or Protocol()
    dates = np.asarray([_day(d) for d in dates], dtype=np.int32)
    if dates.size and (np.diff(dates) <= 0).any():
        raise ValueError("Trading dates must be strictly increasing and unique")
    dates = dates[dates >= _day(p.output_start)]
    if not dates.size:
        return []
    quarter = [(int(d) // 10000, ((int(d) // 100) % 100 - 1) // 3 + 1) for d in dates]
    folds = []
    for year, q in dict.fromkeys(quarter):
        indices = np.asarray([i for i, z in enumerate(quarter) if z == (year, q)], dtype=np.int64)
        a, b = int(indices[0]), int(indices[-1]) + 1
        if int(dates[a]) < _day(p.outer_start) or int(dates[a]) <= _day(p.initial_train_end):
            continue
        inner_end = a - p.purge_days
        inner_start = inner_end - p.inner_validation_days
        train_end = inner_start - p.purge_days
        if train_end < p.min_train_days or inner_start < 0:
            continue
        train, valid, test = dates[:train_end], dates[inner_start:inner_end], dates[a:b]
        folds.append({
            "fold": f"{year}Q{q}", "train": train.tolist(), "inner_validation": valid.tolist(),
            "outer_test": test.tolist(), "purge_trading_days": p.purge_days,
            "train_label_endpoint_before_validation": True,
            "validation_label_endpoint_before_outer_test": True,
            "interpretation": "causal_split_plan;not_completed_model_validation",
        })
    return folds


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")

