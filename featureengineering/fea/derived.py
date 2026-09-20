"""派生缓存基类 —— 给「必须先聚合一次、之后被很多因子复用」的重活用的。

两个重活：
  · `stock_history_5min`  6.86 亿行 → 日频宽表（`fea/intraday.py`）
  · `stock_cyq_chips`     6.24 亿行 → 每股票日约 25 个筹码摘要（`fea/chips.py`）

**为什么必须预聚合**：34 个筹码因子若各自扫一遍原始表 = 34 × 6.24 亿行；
日内因子同理。预聚合一次、缓存成小 parquet，之后每个因子只读几十 MB。

## 缓存位置：`data/derived/<name>/year=YYYY/data.parquet`

放在 `data/` 下但**与 `data/factors/` 平级** —— 因子遍历只扫 `data/factors/*`，
不会被误认成因子。**故意不放进 `state/`**：`state/` 会被 `--sandbox` 重定向，
而派生缓存是「上游数据的纯函数」，多个 Agent / 沙箱应当共享同一份，
否则每个沙箱都要重跑几分钟的预聚合。

## 失效判定

manifest 里存**逐年的源指纹** `(行数, 最新时间戳)`。任一分区指纹变了就重建那一年的分区。
这是全量重算的成本下限：只有上游真的变了才重跑。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import store

log = logging.getLogger("fea.derived")


class DerivedCache:
    """派生缓存的公共骨架。子类实现 `build_year()` 与 `source_fingerprint()`。"""

    name: str = "derived"          # 子类覆盖
    version: int = 1               # 聚合逻辑变了就 +1，强制重建

    def __init__(self, up, cfg, codes: np.ndarray):
        self.up = up                      # ★ 自己持有 up：这样 panel() 能自动 ensure()
        self.cfg = cfg
        self.codes = np.asarray(codes)
        self.root = Path(cfg.root) / "data" / "derived" / self.name
        self.man_path = self.root / "_manifest.json"
        self._cache: dict[tuple, np.ndarray] = {}
        self._frame: pd.DataFrame | None = None
        self._frame_window: tuple[int, int] | None = None

    # ---------------------------------------------------------------- manifest
    def _load_man(self) -> dict:
        if self.man_path.exists():
            try:
                return json.loads(self.man_path.read_text())
            except Exception:
                pass
        return {"version": self.version, "years": {}}

    def _save_man(self, man: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        from .manifest import _atomic_json
        _atomic_json(man, self.man_path)

    # ---------------------------------------------------------------- 子类接口
    def source_fingerprint(self, year: int) -> dict:
        """该年源数据的指纹（行数 + 最新时间戳）。变了就重建这一年。

        ⚠️ 子类实现通常要读源 parquet 的一列（`stock_history_5min` 一年 25~63M 行，
           2~4 秒）。**必须走 `_fp()` 包装带记忆化** —— `panel()` 每次都调
           `ready()`/`ensure()`，不记忆化就是每取一个字段就重读一遍全部年份分区，
           实测把日均任务速率从 50 拖到 9。
        """
        raise NotImplementedError

    def _fp(self, year: int) -> dict:
        """带记忆化的 `source_fingerprint`。子类内部一律走这个。"""
        c = self.__dict__.setdefault("_fp_cache", {})
        if year not in c:
            fp = self.source_fingerprint(year)
            # Include the source file identity: same-row-count upstream repairs
            # must invalidate the daily aggregates as well.
            import hashlib
            fp["universe_sha256"] = hashlib.sha256("\n".join(map(str, self.codes)).encode()).hexdigest()
            c[year] = fp
        return c[year]

    def build_year(self, year: int, up) -> pd.DataFrame:
        """重活：把该年的原始数据聚合成 (trade_date, stock_code, ...) 的日频宽表。"""
        raise NotImplementedError

    # ---------------------------------------------------------------- 构建
    def ready(self, y0: int, y1: int) -> bool:
        """[y0, y1] 的缓存是否已经建好（不触发构建，供 panel() 判断）。"""
        man = self._load_man()
        if man.get("version") != self.version:
            return False
        for y in range(y0, y1 + 1):
            fp = self._fp(y)
            if fp.get("exists") and (man["years"].get(str(y)) != fp or not (self.root / f"year={y}" / "data.parquet").exists()):
                return False
        return True

    def ensure(self, y0: int, y1: int) -> None:
        """确保 [y0, y1] 年的缓存存在且是最新的。缺了就**当场构建**。

        ★ 这个函数必须由 `panel()` 自动调用。原来的设计是「调用方记得先 ensure」，
          结果全仓库没有任何地方调用它 —— 缓存缺失时 `panel()` 静默返回空表、
          因子全 NaN 且**不报错**。安全的东西不能靠调用方记得。
        """
        up = self.up
        man = self._load_man()
        if man.get("version") != self.version:
            log.info("%s：聚合版本变更 %s -> %s，全量重建",
                     self.name, man.get("version"), self.version)
            man = {"version": self.version, "years": {}}
        dirty = []
        for y in range(y0, y1 + 1):
            fp = self._fp(y)
            if not fp.get("exists"):
                continue
            if man["years"].get(str(y)) != fp or not (self.root / f"year={y}" / "data.parquet").exists():
                dirty.append(y)
        if not dirty:
            return
        log.info("%s：需要重建 %d 个年份分区 %s", self.name, len(dirty), dirty)
        t0 = time.time()
        for y in dirty:
            t1 = time.time()
            df = self.build_year(y, up)
            if df is None or df.empty:
                man["years"][str(y)] = self._fp(y)
                continue
            store._atomic_write(df, self.root / f"year={y}" / "data.parquet",
                                self.cfg.compression)
            man["years"][str(y)] = self._fp(y)
            self._save_man(man)     # 逐年落盘：中断了不用从头再来
            log.info("  %s year=%d 完成：%d 行 / %.1fs", self.name, y, len(df),
                     time.time() - t1)
        log.info("%s：全部完成 %.1fs", self.name, time.time() - t0)

    # ---------------------------------------------------------------- 读取
    def _read_all(self, y0: int, y1: int) -> pd.DataFrame:
        """读 [y0, y1] 年的缓存。

        ★ 必须**按窗口缓存**。原来写的是 `if self._frame is not None: return`，
          于是第一次读到哪年就永远返回那一份帧 —— 多因子 / 多年份连跑时，
          第二个年份会**静默产出全 NaN**（实测 71.1 万行 → 5.1 万行）。
          与 `PriceLayer._ensure_loaded` 是同一类 bug（见那边的注释）。
        """
        w = self._frame_window
        if self._frame is not None and w is not None and w[0] <= y0 and w[1] >= y1:
            return self._frame
        y0 = min(y0, w[0]) if w else y0
        y1 = max(y1, w[1]) if w else y1
        frames = []
        for y in range(y0, y1 + 1):
            p = self.root / f"year={y}" / "data.parquet"
            if p.exists():
                frames.append(pd.read_parquet(p))
        self._frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self._frame_window = (y0, y1)
        self._cache.clear()               # 窗口变了，面板缓存全部失效
        return self._frame

    def clear_raw(self) -> None:
        """fork 之前调用（COW 保护）。"""
        self._frame = None
        self._frame_window = None
