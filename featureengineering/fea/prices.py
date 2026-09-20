"""日频价格层 —— PIT 安全的价格/成交量面板。

**因子函数不许自己读 `stock_daily`**，一律走 `ctx.px(...)` / `ctx.hfq(...)` / `ctx.ret(...)`。
原因：复权口径、停牌处置、脏行过滤这三件事只要有一处做错，因子就会**静默错**，
而且错得很有说服力（看起来只是"因子有点噪"）。集中在这里一次做对。

## 复权口径（本项目唯一允许的）

    后复权价 hfq(t) = 未复权价(t) × adj_factor(t)

`adj_factor` 是**累计**复权因子（`t` 时刻的值只取决于 `t` 之前发生的分红送转），
所以 `hfq(t)` 一旦算出就**永不改变** —— 这正是 PIT 安全所要的。
前复权是 `raw(t) × f(t)/f(T_latest)`，分母随最新分红变化 → 全部历史重算 → 被明令禁止。

## 三种缺失语义的选择（这是本层最容易做错的地方）

| 字段类 | 语义 | 缺失处置 | 为什么 |
|:--|:--|:--|:--|
| 价格水平（close/open/…）、股本、复权因子 | **状态量** | 前向填充 | 停牌期间"最后一个成交价"确实还在，复权比值才算得对 |
| 成交量 / 成交额 / traded | **流量** | 保持 NaN | 停牌日**不是** 0 成交。补 0 会把停牌算进滚动均值，系统性低估 |

## 实测踩到的坑（都在这里挡掉）

1. **`adj_factor` 有 4 行落在非交易日**（含 `000001.SZ 2023-01-01 = 2.0`，邻值 113.9）。
   不过滤的话，`asof` 会把「≤ d 的最近一条」认成这行 → 造出 −98% 的假收益。
2. **223 个格子 `adj_factor` 相对前值跌 ≥1%**（16 只股，退市整理期 / 上游脏数据）。
   `ret()` 对 `|r| > 0.60` 置 NaN 并记日志。
3. **`stock_daily` 停牌日没有行**（2015 年每天只有 1361~2584 行 / 244 个交易日），
   所以状态量必须前向填充、流量必须留着 NaN。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import mathx as mx
from .dates import series_to_int as day_ints

log = logging.getLogger("fea.prices")

# 逐字段的处置策略
LEVELS = ("open", "high", "low", "close", "pre_close", "change")
FLOWS = ("vol", "amount", "ah_vol", "ah_amount")
RATIOS = ("pct_chg",)

# 收益异常阈值：单日 |收益| 超过它就认定是复权因子脏数据
RET_ABS_MAX = 0.60


class PriceLayer:
    """预建、跨进程 fork 共享、带缓存的日频价格层（仿 `Derivative`）。"""

    def __init__(self, up, cfg, cal, codes: np.ndarray):
        self.up = up
        self.cfg = cfg
        self.cal = cal
        self.codes = np.asarray(codes)
        self._cpos = pd.Series(np.arange(self.codes.size), index=self.codes)
        self._cal_days = np.asarray(cal.days, dtype=np.int32)
        self._daily: pd.DataFrame | None = None          # 原始 stock_daily（fork 前要清）
        self._adj: pd.DataFrame | None = None
        self._fin: pd.DataFrame | None = None
        self._cache: dict[tuple, np.ndarray] = {}
        self._loaded: tuple[int, int] | None = None    # 已加载的年份窗口
        self._panel_window: tuple[int, int] | None = None
        self._n_clipped = 0

    # ---------------------------------------------------------------- 读
    def _ensure_loaded(self, y0: int, y1: int) -> None:
        """确保已加载的年份窗口**覆盖** [y0, y1]。

        ★ 这里踩过一个阻塞级 bug：原来写的是 `if self._daily is not None: return`，
          于是**第一个**碰价格层的面板就永久决定了加载窗口。并行 worker 会跑多个年份，
          窗口外的任务会拿到 100% NaN —— 而且不是报错，是**静默产出空表**，
          再被下游放大成 `zero-size array to reduction` 之类的误导读数。
          （两个独立的因子开发 Agent 都撞上了这个 bug 并各自绕开，值得记一笔。）

        正确做法：记住已加载的窗口，被包含才复用，否则**扩窗重读**并清掉面板缓存
        （缓存键含日期区间，扩窗后旧结果不一定还成立）。
        """
        if self._loaded is not None and self._loaded[0] <= y0 and self._loaded[1] >= y1:
            return
        if self._loaded is not None:
            y0 = min(y0, self._loaded[0])
            y1 = max(y1, self._loaded[1])
            log.info("价格层扩窗 %s -> (%d, %d)，重读并清空面板缓存", self._loaded, y0, y1)
        self._loaded = (y0, y1)
        self._cache.clear()
        cols = ["stock_code", "trade_date"] + list(LEVELS) + list(FLOWS) + list(RATIOS)
        cols = [c for c in cols if c in ("stock_code", "trade_date") or True]
        d = self.up.read("stock_daily", columns=cols, years=(y0, y1), use_cache=False)
        a = self.up.read("stock_adj_factor", columns=["stock_code", "trade_date", "adj_factor"],
                         years=(y0, y1), use_cache=False)
        f = self.up.read("stock_finance",
                         columns=["stock_code", "trade_date", "total_share", "float_share",
                                  "free_share"],
                         years=(y0, y1), use_cache=False)
        # The layer owns its raw window. Keeping every expanded window again in
        # Upstream._cache multiplies memory without helping subsequent access.
        # Out-of-pool rows were discarded by panel placement anyway; discard
        # them before date conversion and retain only the fixed universe.
        d = d[d["stock_code"].isin(self.codes)].copy()
        a = a[a["stock_code"].isin(self.codes)].copy()
        f = f[f["stock_code"].isin(self.codes)].copy()
        for name, df in (("stock_daily", d), ("stock_adj_factor", a), ("stock_finance", f)):
            di = day_ints(df["trade_date"])
            bad = ~np.isin(di, self._cal_days)
            if bad.any():
                # ★ 必须挡掉：非交易日的行会被 asof 当成「<= d 的最近一条」
                log.warning("%s 有 %d 行落在非交易日，已丢弃（实测 adj_factor 的脏行会造出假收益）",
                            name, int(bad.sum()))
                df.drop(index=df.index[bad], inplace=True)
                di = di[~bad]
            # ★★ 日期列**只转一次**，挂成 int32 列给后面所有字段复用（`_place_from`）。
            #   原先每个字段都重转一遍同一列（实测单任务 47 次 × 2.4s = 112s，
            #   占整个任务 84% 的时间）。这一列 4M 行也只有 16 MB。
            df["_day_i"] = di
        self._daily, self._adj, self._fin = d, a, f
        log.info("价格层：stock_daily %d 行 / adj_factor %d 行 / finance %d 行",
                 len(d), len(a), len(f))

    def trim_cache(self, panel) -> None:
        """丢弃属于**其它面板窗口**的面板缓存。

        ★ 与 `Derivative.trim_cache` 同一个理由：`_cache` 按 (日期0, 日期1, C, field)
          缓存，而每个 (因子, 年) 任务的窗口都不同 → 只增不减。
          一条缓存是 (T,C) 的 float64（750×3484×8 ≈ 21 MB），
          一个 worker 跑几百个任务能吃掉好几 GB —— 实测 16 worker 顶到 108 GB 被 OOM。
          同一任务内窗口固定，所以「窗口一变就清」安全且够省。
        """
        if panel.T == 0:
            return
        w = (int(panel.dates[0]), int(panel.dates[-1]))
        if getattr(self, "_panel_window", None) != w:
            self._cache.clear()
            self._panel_window = w

    def clear_raw(self) -> None:
        """fork 之前必须调用：原始 DataFrame 有几百 MB，COW 会被引用计数读操作整页复制。

        ⚠️ 清掉之后 `_loaded` 也要复位，否则下次 `_ensure_loaded` 会以为窗口还在，
           直接返回空的 DataFrame —— 又是「静默全 NaN」。
        """
        self._daily = self._adj = self._fin = None
        self._loaded = None

    # ---------------------------------------------------------------- 落格
    def _place_from(self, df: pd.DataFrame, panel, value_col: str, src_col: str) -> np.ndarray:
        idx = self._cpos.reindex(df[src_col].to_numpy()).to_numpy()
        keep = np.isfinite(idx)
        # ★ 用 `_ensure_loaded` 里预转好的 `_day_i`（缺失时兜底现转一次）
        di = df["_day_i"].to_numpy() if "_day_i" in df.columns else day_ints(df["trade_date"])
        return panel.place(idx[keep].astype(np.int64),
                           di[keep],
                           pd.to_numeric(df[value_col], errors="coerce").to_numpy()[keep])

    def _field_raw(self, panel, field: str) -> np.ndarray:
        """单个原始字段的 (T,C)，未做任何填充。"""
        if field == "adj_factor":
            return self._place_from(self._adj, panel, "adj_factor", "stock_code")
        for src in ("stock_finance", "stock_daily"):
            df = self._fin if src == "stock_finance" else self._daily
            if field in df.columns:
                return self._place_from(df, panel, field, "stock_code")
        known = sorted(set(list(LEVELS) + list(FLOWS)
                           + ["adj_factor", "total_share", "float_share", "free_share"]
                           + ["traded", "ret1", "turnover"]
                           + ["hfq_" + f for f in LEVELS]))
        raise KeyError(f"价格层不认识字段 {field!r}。可用字段：{known}")

    # ---------------------------------------------------------------- 对外
    def panel(self, panel, field: str) -> np.ndarray:
        key = (int(panel.dates[0]), int(panel.dates[-1]), panel.C, field)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        y0 = int(str(panel.dates[0])[:4])
        y1 = int(str(panel.dates[-1])[:4])
        self._ensure_loaded(max(y0 - 1, 2005), y1)

        if field.startswith("hfq_"):
            base = field[4:]
            raw = self._field_raw(panel, base)
            af = self._field_raw(panel, "adj_factor")
            out = mx.nan_fill_ffill(raw) * mx.nan_fill_ffill(af)
            src_level = True
        elif field == "traded":
            out = np.isfinite(self._field_raw(panel, "vol"))
            self._cache[key] = out
            return out
        elif field == "ret1":
            out = mx.safe_div(self._field_raw(panel, "pct_chg"), 100.0)
            self._cache[key] = out
            return out
        elif field == "turnover":
            vol = self._field_raw(panel, "vol")
            fs = mx.nan_fill_ffill(self._field_raw(panel, "float_share"))
            out = mx.safe_div(vol, fs, min_abs_den=1.0) * 100.0
            self._cache[key] = out
            return out
        else:
            out = self._field_raw(panel, field)
            src_level = field in LEVELS or field.endswith("_share")

        if src_level:
            out = mx.nan_fill_ffill(out)      # 状态量：停牌期间沿用最后一个值
        self._cache[key] = out
        return out

    def mask(self, panel, name: str) -> np.ndarray:
        if name != "traded":
            raise KeyError(f"未知掩码 {name!r}（目前只有 'traded'）")
        return self.panel(panel, "traded")

    def daily_ret(self, panel) -> np.ndarray:
        """**单日**后复权收益，带异常清洗。所有多日收益都由它累乘而来。"""
        key = (int(panel.dates[0]), int(panel.dates[-1]), panel.C, "daily_ret")
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        px = self.panel(panel, "hfq_close")
        prev = mx.shift(px, 1)
        r = mx.safe_div(px - prev, np.abs(prev), min_abs_den=1e-12)
        bad = np.isfinite(r) & (np.abs(r) > RET_ABS_MAX)
        n = int(bad.sum())
        if n:
            self._n_clipped += n
            log.warning("单日收益：%d 个格子 |r| > %.0f%%，已置 NaN（复权因子脏数据）",
                        n, RET_ABS_MAX * 100)
            r = np.where(bad, np.nan, r)
        self._cache[key] = r
        return r

    def ret(self, panel, k: int = 1) -> np.ndarray:
        """k 个交易日的后复权收益率。

        ★ 两个曾经的错误（都被因子开发 Agent 实测抓出来，代价很大）：

        1. **把 |r| > 60% 的阈值套在 k 日累计收益上**。250 日涨 109% 是**完全正常**的
           A 股行情，但会被当成复权脏数据整片抹掉 —— 实测 `momentum_250` 因此
           **误杀 317,223 格 = 有效格的 48%**，而且抹掉的正是涨得最多的那批股票
           （对动量因子是最坏方向的截尾，且完全静默）。
           正确做法：**只在单日收益上做异常清洗**，多日收益由单日收益累乘。

        2. **停牌日返回 0.0 而不是 NaN**。`hfq_close` 是前向填充的，
           停牌日 `close/close_prev - 1` 恰好是 0 —— 于是一只**整个窗口都没成交**的
           股票会拿到一个凭空的「0 动量」并混进截面排名。
           正确做法：停牌日贡献 0 收益（价格确实没变，这是对的），
           但**窗口内一个成交日都没有时返回 NaN**。
        """
        r1 = self.daily_ret(panel)
        if k <= 1:
            out = r1
        else:
            # 累乘用 log1p + 滚动求和：O(T·C)，且天然按框架的 NaN 毒化策略处理缺失
            with np.errstate(all="ignore"):
                lg = np.where(np.isfinite(r1), np.log1p(r1), np.nan)
                out = np.expm1(mx.roll_sum(lg, k))
            out[~np.isfinite(out)] = np.nan
        # 窗口内至少要有若干成交日，否则整窗无成交 -> NaN
        traded = self.panel(panel, "traded")
        n_tr = mx.roll_count(traded.astype(np.float64), k)
        out = np.where(n_tr >= 1.0, out, np.nan)
        return out

    def clip_report(self) -> int:
        return self._n_clipped
