"""价格层（自带版）—— 从 `trainingdata/P` 读，**零上游依赖**。

## 与上游 `fea.prices.PriceLayer` 的关系

回测与策略原先直接用模块② 的 `PriceLayer`（经 `fea.engine.Engine.prices_for`），
这带来两条硬依赖：`fea` 包 + 模块① 的 `datadownload/data/`（价格的真身在**模块①**，
不在模块② —— 见 `fea/config.py` 的 `paths.upstream`）。两者都是「外部文件夹」，
正是单元无法独立拷贝的原因之一。

本模块把价格层**只读 `trainingdata/P`** 重实现一遍。派生逻辑**逐条照搬**上游
（`fea/prices.py:168-200`），一处不省：

    hfq_X    = ffill(X) × ffill(adj_factor)
    traded   = isfinite(vol)
    ret1     = safe_div(pct_chg, 100)
    turnover = safe_div(vol, ffill(float_share), min_abs_den=1) × 100
    LEVELS（open/high/low/close）与 *_share：ffill；其余（vol/amount/pct_chg/adj_factor）不填

★ **`ffill` 只在请求窗口内做**，这与上游一致：上游 `_field_raw` 只把 panel 窗口内的
  格子落进数组，`nan_fill_ffill` 于是只能在本窗口内前向填充 —— 窗口首行若无成交即为 NaN。
  本模块刻意**不加跨窗口 lookback**：加了反而会与上游分叉（首行由 NaN 变成"有值"），
  而回测的 `entry_ok` 掩码正是靠这个 NaN 判定"买不进"的。

## 顺带修掉的一个隐患

上游按**列号**落格（`_cpos` → `Panel.place`），两轴不一致时甲乙股票的价会静默互换 ——
2026-09-18 换池时就真的发生过（3484 只 vs 2115 只），所以才有了 `upstream.assert_price_axis`
那道硬闸门。本模块改为按 `(trade_date, stock_code)` **显式对齐**（`Index.get_indexer`），
对不上就是 NaN，不会错位。闸门保留（`_check_axis`），但性质从"防错位"变成"防口径不一致"。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import panel_io as IO
from .config import Cfg

#: 上游 `fea.mathx` 的常量（照搬）
RET_ABS_MAX = 0.60

#: 存进 P 块的 LEVELS 子集。上游 LEVELS 是 6 个（含 pre_close/change），
#: 我们只存回测要用的 4 个；其余可由 close 推出，省 120MB。
_LEVELS = ("open", "high", "low", "close")
#: 直接取原料、**不做 ffill** 的字段（上游 `src_level=False` 的那些）
_RAW = ("vol", "amount", "pct_chg", "adj_factor")


# ================================================================ 数学原语（照搬 fea.mathx）
def _as_f64(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"价格层只接受 (T, C) 二维数组，收到 {a.shape}")
    return a


def ffill(x) -> np.ndarray:
    """前向填充（只在缺失处填上一个有限值），O(T·C) 无 Python 循环。

    ★ 与 `fea.mathx.nan_fill_ffill` 逐字同构 —— 包括"用 `_finite` 而不是 `~isnan`"
      （inf 也算缺失）和"列下标必须另给"（写成 `x[arange(T), idx]` 会张冠李戴）。
    """
    x = _as_f64(x)
    T = x.shape[0]
    if T == 0:
        return x.copy()
    idx = np.where(np.isfinite(x), np.arange(T)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return x[idx, np.arange(x.shape[1])[None, :]]


def safe_div(num, den, min_abs_den: float = 0.0) -> np.ndarray:
    """带分母保护的除法：分母非有限或 |分母| 过小 → NaN（不产生 inf/巨值）。"""
    d = np.asarray(den, dtype=np.float64)
    num = np.asarray(num, dtype=np.float64)
    with np.errstate(all="ignore"):
        out = np.full(np.broadcast_shapes(num.shape, d.shape), np.nan, dtype=np.float64)
        ok = np.isfinite(d) & (np.abs(d) > min_abs_den)
        np.divide(num, d, out=out, where=ok)
    out[~np.isfinite(out)] = np.nan
    return out


# ================================================================ 网格
class Grid:
    """交易日 × 股票的规则网格 —— `fea.panel.Panel` 的**最小同构替身**。

    只保留价格层真正用得到的部分：`dates`（int32 升序）、`codes`、`T`、`C`。
    刻意不实现 `place` / `asof` 等落格方法：本模块按 key 对齐，不需要它们，
    而少一个按列号落格的入口，就少一处静默错位的可能。
    """

    def __init__(self, dates: np.ndarray, codes: np.ndarray):
        self.dates = np.ascontiguousarray(dates, dtype=np.int32)
        self.codes = np.asarray(codes)
        self.T = int(self.dates.size)
        self.C = int(self.codes.size)
        self.shape = (self.T, self.C)
        self.size = self.T * self.C

    def empty(self, fill=np.nan, dtype=np.float32) -> np.ndarray:
        return np.full(self.shape, fill, dtype=dtype)


# ================================================================ 价格层
class PriceLayer:
    """只读 `trainingdata/P` 的价格层。窗口 = 构造时给定的 (dates, codes) 网格。"""

    def __init__(self, root: Path, grid: Grid, *, log=None):
        self.root = Path(root)
        self.grid = grid
        self._raw_cache: dict[str, np.ndarray] = {}
        self._panel_cache: dict[str, np.ndarray] = {}
        self._load(log=log)

    # ---------------------------------------------------------------- 装载
    def _load(self, *, log=None) -> None:
        g = self.grid
        y0 = int(str(g.dates[0])[:4])
        y1 = int(str(g.dates[-1])[:4])
        years = [y for y in IO.years_of(self.root, IO.PK) if y0 <= y <= y1]
        if not years:
            raise SystemExit(
                f"✘ {self.root} 里没有 P（价格块）的年份分区覆盖 {y0}~{y1} —— 回测/策略需要它。\n"
                f"    训练与 IC 层不需要价格块；要用回测请先建：\n"
                f"      cd <模块根> && python preparingdata.py --prices-only\n"
                f"    （★ 它不改 panel_digest，已有结论与配方全部保持有效）")
        df = IO.read_price_frame(self.root, years=years)
        if df.empty:
            raise SystemExit(f"✘ P 块 {years} 年读出来是空的")

        # ★ 显式按 (trade_date, stock_code) 对齐到网格 —— 不按列号落格。
        day_i = df["trade_date"].map(IO.to_int).to_numpy()
        r = pd.Index(g.dates).get_indexer(day_i)
        c = pd.Index(g.codes).get_indexer(df["stock_code"].to_numpy())
        keep = (r >= 0) & (c >= 0)
        if not keep.any():
            raise SystemExit(
                f"✘ P 块与网格**一行都对不上** —— 日期轴或代码轴不一致。\n"
                f"    网格 {g.T} 天 × {g.C} 只（{IO.to_str(g.dates[0])}~{IO.to_str(g.dates[-1])}）\n"
                f"    P 块 {len(df):,} 行（{df['trade_date'].min()}~{df['trade_date'].max()}）")
        n_out = int((~keep).sum())
        if log and n_out:
            log(f"  价格层：{n_out:,}/{len(df):,} 行落在请求窗口之外（已忽略）")

        r, c = r[keep], c[keep]
        for col in IO.PRICE_COLUMNS:
            if col not in df.columns:
                raise SystemExit(f"✘ P 块缺列 {col!r}（现有 {list(df.columns)}）")
            a = np.full(g.shape, np.nan, dtype=np.float64)
            a[r, c] = pd.to_numeric(df[col], errors="coerce").to_numpy()[keep]
            a[~np.isfinite(a)] = np.nan
            self._raw_cache[col] = a
        del df

    def trim_cache(self) -> None:
        """与上游同名方法留个接口。本实现的缓存以窗口为界、不跨窗口增长，故为空操作。"""
        return None

    # ---------------------------------------------------------------- 原料
    def _raw(self, field: str) -> np.ndarray:
        hit = self._raw_cache.get(field)
        if hit is not None:
            return hit
        raise SystemExit(
            f"✘ 价格层不认识字段 {field!r}。\n"
            f"    P 块直接存的：{list(IO.PRICE_COLUMNS)}\n"
            f"    可派生的：hfq_{{open,high,low,close}} · traded · ret1\n"
            f"    ★ `turnover` 不支持：它要 `float_share`，而回测/策略对它零引用，\n"
            f"      故未纳入 P 块。真要用请把它加进 `mx/panel_io.py:PRICE_COLUMNS` 并重建：\n"
            f"        python preparingdata.py --prices-only --years <年份…>")

    # ---------------------------------------------------------------- 对外
    def panel(self, grid: Grid, field: str) -> np.ndarray:
        """(T, C) —— 与上游 `PriceLayer.panel` **同签名同语义**。"""
        if grid is not self.grid:
            if (grid.T, grid.C) != (self.grid.T, self.grid.C):
                raise SystemExit(
                    f"✘ 传入的网格 {(grid.T, grid.C)} 与本价格层的 {self.grid.shape} 不一致 —— "
                    f"价格层按构造时的窗口装载，换窗口要重建")
        hit = self._panel_cache.get(field)
        if hit is not None:
            return hit

        if field.startswith("hfq_"):
            base = field[4:]
            if base not in _LEVELS:
                raise SystemExit(f"✘ 不支持 hfq_{base}（P 块只存了 {list(_LEVELS)} 的行情）")
            out = ffill(self._raw(base)) * ffill(self._raw("adj_factor"))
        elif field == "traded":
            out = np.isfinite(self._raw("vol"))
            self._panel_cache[field] = out
            return out
        elif field == "ret1":
            out = safe_div(self._raw("pct_chg"), 100.0)
            self._panel_cache[field] = out
            return out
        elif field in _LEVELS:
            out = ffill(self._raw(field))
        elif field in _RAW:
            out = self._raw(field)
        else:
            self._raw(field)                      # 交给它抛带说明的错
            raise AssertionError("unreachable")

        self._panel_cache[field] = out
        return out

    def mask(self, grid: Grid, name: str) -> np.ndarray:
        if name != "traded":
            raise KeyError(f"未知掩码 {name!r}（目前只有 'traded'）")
        return self.panel(grid, "traded")

    def daily_ret(self, grid: Grid) -> np.ndarray:
        """**单日**后复权收益，带异常清洗。所有多日收益都由它累乘而来。

        ★ 上游那两条血泪教训在这里同样成立（`fea/prices.py:daily_ret` 的 docstring）：
          ① 清洗阈值只能套在**单日**收益上 —— 套在 k 日累计收益上会误杀正常的大涨股；
          ② 停牌日贡献 0 收益（价格确实没变），但**窗口内一个成交日都没有时要返回 NaN**，
             否则会凭空造出一个"0 动量"混进截面排名。
        """
        hit = self._panel_cache.get("__daily_ret__")
        if hit is not None:
            return hit
        px = self.panel(grid, "hfq_close")
        prev = np.vstack([np.full((1, px.shape[1]), np.nan), px[:-1]])
        r = safe_div(px - prev, np.abs(prev), min_abs_den=1e-12)
        bad = np.isfinite(r) & (np.abs(r) > RET_ABS_MAX)
        if bad.any():
            r = np.where(bad, np.nan, r)
        self._panel_cache["__daily_ret__"] = r
        return r

    def ret(self, grid: Grid, k: int = 1) -> np.ndarray:
        """k 个交易日的后复权收益率（由单日收益累乘，口径见 `daily_ret`）。"""
        r1 = self.daily_ret(grid)
        if k <= 1:
            out = r1
        else:
            with np.errstate(all="ignore"):
                lg = np.where(np.isfinite(r1), np.log1p(r1), np.nan)
                cs = np.cumsum(np.nan_to_num(lg, nan=0.0), axis=0)
                cnt = np.cumsum(np.isfinite(lg), axis=0)
                s = cs - np.vstack([np.zeros((1, lg.shape[1])), cs[:-1]])
                n = cnt - np.vstack([np.zeros((1, lg.shape[1])), cnt[:-1]])
                out = np.expm1(np.where(n >= k, s, np.nan))
            out[~np.isfinite(out)] = np.nan
        # 窗口内至少要有成交日，否则整窗无成交 → NaN
        traded = self.panel(grid, "traded")
        with np.errstate(all="ignore"):
            n_tr = np.cumsum(traded.astype(np.float64), axis=0) - \
                np.vstack([np.zeros((1, traded.shape[1])), np.cumsum(traded.astype(np.float64), axis=0)[:-1]])
        return np.where(n_tr >= 1.0, out, np.nan)


# ================================================================ 工厂
def _check_axis(cfg: Cfg, codes: np.ndarray) -> None:
    """代码轴必须与 `meta.json:axis.codes` 一致 —— 换池/换快照时的硬闸门。

    上游的 `assert_price_axis` 挡的是"按列号落格导致甲乙互换"；本模块按 key 对齐，
    错位已不可能。但"口径不一致"仍要挡：拿新池的网格去读旧池的价格块，
    对不上的格子会全变 NaN —— 不报错，但每个回测数字都悄悄变错。
    """
    want = np.asarray((IO.load_meta(cfg.trainingdata).get("axis") or {}).get("codes") or [],
                      dtype=object)
    got = np.asarray(codes, dtype=object)
    if want.size == 0:
        raise SystemExit(f"✘ {cfg.trainingdata}/meta.json 缺 axis.codes —— 快照不完整")
    if want.size == got.size and np.array_equal(want, got):
        return
    raise SystemExit(
        f"✘ 代码轴与快照不一致：请求 {got.size} 只 vs 快照 {want.size} 只。\n"
        f"    价格块是按快照的轴存的，轴不一致会让**大量格子变成 NaN** —— 不报错，"
        f"但每个回测数字都悄悄变错。\n"
        f"    最常见的原因：拿新池（2115 只）的网格去读旧池（3484 只）的价格块，或反之。")


def load(cfg: Cfg, dates: list[str], codes: np.ndarray, *, log=None) -> tuple[Grid, PriceLayer]:
    """建 (网格, 价格层) —— 回测/策略的统一入口。

    替代原先的三步：
        eng  = upstream._engine(...)         # 要 fea + 模块①
        from fea.panel import Panel          # 要 fea
        dy   = np.asarray([upstream.to_int(d) for d in dates], dtype=np.int32)
        panel = Panel(dy, codes)
        lay  = eng.prices_for(int(dy[0]), int(dy[-1]))
    现在只要 `grid, lay = prices.load(cfg, dates, codes)` —— 零上游依赖。
    """
    di = np.asarray([IO.to_int(d) for d in dates], dtype=np.int32)
    if di.size == 0:
        raise SystemExit("✘ 价格层：请求的日期是空的")
    _check_axis(cfg, codes)
    grid = Grid(di, codes)
    return grid, PriceLayer(cfg.trainingdata, grid, log=log)
