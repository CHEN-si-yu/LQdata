"""配置加载 —— 读 conf/daily.yaml，提供带默认值的取值接口。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import paths

_DEFAULTS: dict[str, Any] = {
    "resources": {"max_address_space_gib": 24, "numeric_threads": 2},
    "api": {
        "base_url": "https://data.diemeng.chat/api",
        "rate_limit_per_min": 260,
        "global_rate_limit": True,
        "concurrency": 3,
        "max_inflight": 2,
        "timeout": 120,
        # ★ 固定重试档位（用户 2026-09-15 指定）：1s → 3s → 5s → 10s → 30s，
        #   共 5 次重试（+首次 = 6 次尝试），五档都失败才记 ⚠️。配 [] 退回指数退避。
        "retry_backoff_seconds": [1, 3, 5, 10, 30],
        "max_retries": 6,
        "backoff_base": 1.6,
        "backoff_max": 30,
        "max_rows_per_query": 100_000,
    },
    # ★★ 2026-09-17 清理：删掉 4 个**从没被任何代码读过**的键 ——
    #   `api.slow_concurrency`（承诺的"低并发通道"不存在，实际走 api.concurrency）、
    #   `api.default_batch`（批量在 registry 的 entity_batch）、
    #   `download.write_buffer_rows`（_Buf 的阈值是代码里的常数）、
    #   `run.{daily_first,regress_check,always_tail_window}`（顺序是硬编码的；
    #   "历史不变回归"从未实现 —— 现在由 `dayhash.sample_old_days` 承担）、
    #   `viz.live_gate`（TTY 下总是重绘）、`dayhash.compare_window`。
    #   留着它们比删掉更危险：改了没效果，会让人以为改了什么。
    "download": {
        "redundancy_days": 5,
        # ★ 与 conf/daily.yaml 对齐（yaml 一直是 0；旧默认 10 只在 yaml 缺失时生效，
        #   容易让人误读成"默认回刷 10 天"）
        "revision_days": 0,
    },
    "gate": {
        "interval_seconds": 300,
        "max_wait_hours": 6.0,          # ★ 与 yaml 对齐（2026-09-15 深夜 4.0 → 6.0）
        "on_timeout": "partial",
        "publish_ratio": 0.7,
        "autobump_after_days": 3,
        # ★★ 新增：delay 自动上调的**上限**（相对声明值最多 +N 天）。
        #   只上不下 + 无上限 = 一旦探针侧出问题（如代表实体取错），
        #   delay 每晚 +1，而取数窗口上界 = T - delay 会随之后退
        #   → 最新的 N 天**永远不会被请求**，且报告仍显示 ✔（静默）。
        "delay_autobump_max": 2,
        "cross_check": True,
    },
    "calendar": {"extend_days": 30},
    "dayhash": {
        "enabled": True,
        # ★ 每轮额外抽查多少个**历史日**（尾窗之外、台账里已有的日期，确定性轮换）。
        #   0 = 不抽查。这是"防止厂商偷偷改历史"的唯一手段（尾窗只有 ~6 天）。
        "sample_old_days": 3,
    },
    # ★ daily_dump 通道
    "dump": {
        # 修复扫描窗口（日历天）：本地行数扫描，零请求；dump 接口只覆盖最近 ~90 天。
        "repair_lookback_days": 45,
    },
    "viz": {"summary": True},
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: Path | None = None) -> dict:
    """加载配置（与默认值深合并）。文件不存在就用全默认值。"""
    p = path or paths.CONF_FILE
    raw: dict = {}
    if p.exists():
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError) as exc:  # noqa: BLE001
            raise SystemExit(f"配置文件解析失败 {p}: {exc}") from exc
    return _deep_merge(_DEFAULTS, raw)


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """cfg 取嵌套值：get(cfg, "gate.interval_seconds")。"""
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
