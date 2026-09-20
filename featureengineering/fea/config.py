"""配置与路径。

所有路径以项目根目录为基准解析；`upstream` 指向模块① 的产出目录。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    raw: dict
    root: Path

    # ---- 路径 ----
    @property
    def upstream(self) -> Path:
        return (self.root / self.raw["paths"]["upstream"]).resolve()

    @property
    def upstream_state(self) -> Path:
        return (self.root / self.raw["paths"]["upstream_state"]).resolve()

    @property
    def factors_dir(self) -> Path:
        return self.root / self.raw["paths"]["factors"]

    @property
    def state_dir(self) -> Path:
        return self.root / self.raw["paths"]["state"]

    @property
    def logs_dir(self) -> Path:
        return self.root / self.raw["paths"]["logs"]

    # ---- 计算参数 ----
    @property
    def revision_days(self) -> int:
        return int(self.raw["compute"]["revision_days"])

    @property
    def default_start(self) -> str:
        return self.raw.get("default_start", "2015-01-01")

    def dep_backfill_days(self, dataset: str) -> int:
        d = self.raw["compute"].get("dep_backfill_days", {})
        return int(d.get(dataset, d.get("default", 400)))

    # ---- universe ----
    @property
    def board_prefixes(self) -> tuple[str, ...]:
        return tuple(self.raw["universe"]["main_board_prefixes"])

    @property
    def exclude_st(self) -> bool:
        return bool(self.raw["universe"].get("exclude_st", False))

    @property
    def min_listed_days(self) -> int:
        return int(self.raw["universe"].get("min_listed_days", 0))

    @property
    def frozen_universe(self) -> Path | None:
        """冻结股票池名单文件；`None` = 不启用（回到"主板全集 + 动态进出"的老口径）。

        ★★ 2026-09-18 用户拍板：把池子固定成 2115 只
          （主板 ∩ 至今存续 ∩ 从未ST ∩ 上市日≤2018-01-01）。
          动机是**历史值稳定**：动态池下一只新股上市就会让全库所有历史截面的
          行数 +1、rank / cs_zscore 平移 → 单日 MD5 台账全红。
          名单由 `scripts/build_frozen_universe.py` 生成，口径与代价写在文件头。

        ⚠️ **配置了却读不到文件 → 直接抛错，绝不静默回退**到动态池：
          静默回退会让"池子变了"这件事只体现在因子值里，没有任何提示。
        """
        p = str(self.raw.get("universe", {}).get("frozen_list") or "").strip()
        if not p:
            return None
        q = Path(p)
        if not q.is_absolute():
            q = self.root / p
        if not q.exists():
            raise FileNotFoundError(
                f"universe.frozen_list 指向的冻结名单不存在：{q}\n"
                f"  要么生成它（python scripts/build_frozen_universe.py），"
                f"要么把 conf/config.yaml 里的 frozen_list 置空以回到动态主板池。")
        return q

    # ---- rank ----
    @property
    def winsor(self) -> tuple[float, float]:
        w = self.raw["rank"]["winsor"]
        return (float(w[0]), float(w[1]))

    @property
    def min_cross_section(self) -> int:
        return int(self.raw["rank"].get("min_cross_section", 100))

    @property
    def compression(self) -> str:
        return self.raw.get("storage", {}).get("compression", "zstd")


def load(path: Path | None = None) -> Config:
    p = path or (ROOT / "conf" / "config.yaml")
    with open(p, encoding="utf-8") as f:
        return Config(raw=yaml.safe_load(f), root=ROOT)


def setup_logging(verbose: bool = False) -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.FileHandler(ROOT / "logs" / "factors.log", encoding="utf-8")
    ]
    if verbose:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
