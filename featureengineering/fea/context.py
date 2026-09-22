"""FactorContext —— 因子作者唯一需要打交道的对象。

因子函数的契约是 `(ctx) -> (T, C) float32 数组`，**不接收也不返回 DataFrame**。
这一条约束同时买到三样东西：内存有界、滚动窗口可向量化、增量 warmup 可证明。

常用方法：
    ctx.ttm(field)            累计科目 -> TTM，已按 ann_date 前向填充到日频
    ctx.point(field)          时点科目（资产负债表）-> 日频
    ctx.lag_ttm(field)        TTM 的去年同期值（同比用）
    ctx.asof_daily(...)       任意「逐实体的稀疏披露值」-> 日频
    ctx.event_grid(...)       任意「事件」-> 日频计数
    ctx.roll_sum(mat, n)      滚动求和（cumsum 差分，O(T·C)）
    ctx.universe              (T, C) bool，主板 + 上市窗口
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import mathx as _mx
from .dates import series_to_int
from .deriv import Derivative
from .panel import Panel


class FactorContext:
    def __init__(self, panel: Panel, deriv: Derivative, up, cfg, universe: np.ndarray,
                 prices=None, intraday=None, chips=None, cal=None, factor_io=None,
                 open5=None):
        self.panel = panel
        self.deriv = deriv
        self.up = up
        self.cfg = cfg
        self.universe = universe
        self._pos = {c: i for i, c in enumerate(panel.codes)}
        # 以下都是可选层：不传就是「没有这个能力」，对应家族用不到时零成本
        self.prices = prices          # PriceLayer：日频价格/量/复权
        self.intraday = intraday      # IntradayLayer：5min 派生的日频宽表
        self.chips = chips            # ChipLayer：筹码峰派生的日频摘要
        self.open5 = open5            # Open5Layer：5min 派生的一开盘首段流动性（S-07）
        self.cal = cal                # Calendar：交易日历（日期位移法要用）
        self.factor_io = factor_io    # FactorIO：读已落盘的其它因子（耦合因子用）

    # ---------------------------------------------------------------- 价格
    def _need_prices(self):
        if self.prices is None:
            raise RuntimeError(
                "该因子用到了价格层，但引擎没有预建 PriceLayer。"
                "请确认 engine.prebuild 里已构建 self._prices（或该因子被单独跑）。")
        return self.prices

    def px(self, field: str) -> np.ndarray:
        """未复权价量面板。`field` ∈ open/high/low/close/pre_close/vol/amount/pct_chg。

        ⚠️ 停牌日该股没有行情行 -> 这些格子是 NaN（不是前向填充的旧值）。
           `close` 这类**水平量**会自动前向填充（返回时复权比值才对）；
           `vol`/`amount` 这类**流量**保持 NaN，避免把停牌当成 0 成交。
        """
        return self._need_prices().panel(self.panel, field)

    def hfq(self, field: str = "close") -> np.ndarray:
        """**后复权**价 = 未复权价 × 截至当日的累计复权因子。

        这是本项目唯一允许的价格口径（前复权 qfq 被 `register()` 直接拒绝）。
        `hfq(t)` 只用到 `<= t` 的 `adj_factor`，新分红只影响 `t >= 除权日` 的值，
        所以历史值永远不变 —— 这就是 PIT 安全。
        """
        return self._need_prices().panel(self.panel, f"hfq_{field}")

    def ret(self, k: int = 1) -> np.ndarray:
        """k 个交易日的后复权收益率。

        **停牌日的语义（重要，别靠猜）**：`hfq_close` 是前向填充的，所以停牌日
        `close/close_prev - 1 = 0` —— 停牌日贡献 **0 收益**，不是 NaN。
        这对**动量类**是对的（停牌期间价格确实没变），
        对**波动率类**是错的（把一堆人造的 0 塞进滚动窗口会系统性低估波动，
        实测 2014 年有 9.89% 的有效收益恰好是 0）。

        需要「停牌日 = 缺失」的口径时，用 `ctx.ret_clean(k)`（= `ret` 再按
        `ctx.traded()` 掩码）。**不要**假设 `ret()` 已经挡掉停牌 —— 那是错的。
        """
        return self._need_prices().ret(self.panel, k)

    def ret_clean(self, k: int = 1) -> np.ndarray:
        """k 个交易日收益，**停牌日置 NaN**（波动率 / 风险类因子用这个）。

        k=1 时返回逐日收益并按 `traded` 掩码；k>1 时返回累计收益，
        但窗口内**只要有一天没成交就整窗 NaN**（与 `ret` 的「贡献 0」不同）。
        """
        pl = self._need_prices()
        r = pl.ret(self.panel, k)
        tr = pl.panel(self.panel, "traded")
        if k <= 1:
            return np.where(tr, r, np.nan)
        n_tr = self.roll_count(tr.astype(np.float64), k)
        return np.where(n_tr >= k, r, np.nan)

    def traded(self) -> np.ndarray:
        """(T,C) bool：当日真有成交（停牌 / 未上市 / 已退市为 False）。"""
        return self._need_prices().mask(self.panel, "traded")

    # ---------------------------------------------------------------- 5min 派生
    def intraday_field(self, field: str) -> np.ndarray:
        """读 5min 预聚合出来的日频字段（见 fea/intraday.py 的字段表）。"""
        if self.intraday is None:
            raise RuntimeError("该因子用到了日内层，但引擎没有预建 IntradayLayer。")
        return self.intraday.panel(self.panel, field)

    # ---------------------------------------------------------------- 开盘首段
    def open5_field(self, field: str) -> np.ndarray:
        """读开盘首段（首 5 分钟）预聚合出来的日频字段（见 fea/open5.py 的字段表）。"""
        if self.open5 is None:
            raise RuntimeError("该因子用到了开盘首段层，但引擎没有预建 Open5Layer。")
        return self.open5.panel(self.panel, field)

    # ---------------------------------------------------------------- 筹码
    def chip(self, field: str) -> np.ndarray:
        """读筹码峰预聚合出来的日频摘要字段（见 fea/chips.py 的字段表）。"""
        if self.chips is None:
            raise RuntimeError("该因子用到了筹码层，但引擎没有预建 ChipLayer。")
        return self.chips.panel(self.panel, field)

    # ---------------------------------------------------------------- 耦合
    def load_factor(self, name: str) -> np.ndarray:
        """读另一个已落盘因子的 `value`，切到当前面板的 (T,C)。"""
        if self.factor_io is None:
            raise RuntimeError("该因子用到了因子耦合，但引擎没有预建 FactorIO。")
        return self.factor_io.load(name, self.panel)

    # ---------------------------------------------------------------- 滞后表
    def lag_grid(self, mat: np.ndarray, n: int = 1) -> np.ndarray:
        """把 (T,C) 网格整体下移 n 行 —— 滞后表的标准处置手法。

        面板本身就是交易日历，所以「移一行」=「移一个交易日」，
        语义即 `value(T) := source(T − n 个交易日)`。
        首 n 行变 NaN 无害：它们落在 warmup 区，进不了输出窗口。
        """
        from .mathx import shift
        return shift(mat, int(n))

    def next_trading_day(self, days: np.ndarray) -> np.ndarray:
        """日期位移法的显式版本：把一批交易日映射到**下一个交易日**。"""
        if self.cal is None:
            raise RuntimeError("ctx.next_trading_day 需要交易日历，引擎未注入 cal。")
        pos = np.searchsorted(self.cal.days, np.asarray(days, dtype=np.int32), side="right")
        return self.cal.days[np.clip(pos, 0, self.cal.days.size - 1)]

    def asof_daily_lagged(self, codes, days, values, n: int = 1) -> np.ndarray:
        """`asof_daily` + 滞后位移一步到位（滞后表推荐写法之一）。"""
        return self.lag_grid(self.asof_daily(codes, days, values), n)

    def event_grid_lagged(self, codes, days, weights=None, n: int = 1) -> np.ndarray:
        return self.lag_grid(self.event_grid(codes, days, weights), n)

    # ---------------------------------------------------------------- 财务扩展
    def ind(self, field: str, lag: int = 0) -> np.ndarray:
        """`stock_financial_indicator` 的**时点比率**字段（按 ann_date 做 PIT 前向填充）。

        与 `ttm()/point()` 的区别：这里**不做任何口径变换**，给的就是供应商披露的值。
        ⚠️ 只允许取 `deriv.IND_SAFE` 里的字段（时点比率、同期同比及明确白名单的单季度比率）。
          单季度字段仍有季节性，按报告期 lag=4 比较同期；不能把日频 shift 当财季。
          该表的 `roe / roa / grossprofit_margin / *_turn` 等是**累计 YTD**，
          直接当日频用会得到跨季锯齿（振幅 4 倍）—— 这些必须用 ttm() 从原始三表重算。

        **框架在这里直接拦住违规字段**：因为「用错了口径」不会报错，
        只会让因子的值变成一条锯齿，看起来只是有点噪。
        """
        from .deriv import IND_SAFE
        if field not in IND_SAFE:
            hint = ""
            if field in ("roe", "roa", "roic", "grossprofit_margin",
                         "netprofit_margin", "assets_turn", "inv_turn", "ar_turn"):
                hint = (f"\n  ★ 特别提醒：{field!r} 是**累计 YTD**，不是 TTM。"
                        f"请用 ctx.ttm(...) / ctx.point(...) 从原始三表自己算。")
            raise KeyError(
                f"ctx.ind({field!r}) 不在安全字段表里。\n"
                f"  该表 163 个比率大多数是累计 YTD，当日频用会得到跨季锯齿（振幅 4 倍），"
                f"而且**不报错**。\n"
                f"  允许的字段（{len(IND_SAFE)} 个）：{list(IND_SAFE)}{hint}")
        self._trim()
        return self.deriv.to_panel(self.panel, field, mode="ind", lag=lag)

    # ---------------------------------------------------------------- 数学
    def annual(self, dataset: str, field: str, lag_years: int = 0) -> np.ndarray:
        """仅取当时已公告的12月31日年度报告；年度同比按报告期回看4季。

        供新增字段使用的独立入口，不放宽 ind() 对累计YTD比率的限制。
        允许的源字段显式列在 conf/field_expansion.json；缺年、缺修订保持NaN。
        """
        from .field_expansion import ANNUAL_ALIASES, annual_alias
        alias = annual_alias(dataset, field)
        if alias not in ANNUAL_ALIASES:
            raise KeyError(f"未声明的年度字段：{dataset}.{field}")
        if not isinstance(lag_years, int) or lag_years < 0:
            raise ValueError("年度因子只允许非负整数历史滞后")
        self._trim()
        return self.deriv.to_panel(self.panel, alias, mode="annual", lag=4 * lag_years)

    # ---------------------------------------------------------------- 数学
    # 以下都是 fea/mathx.py 的转发，因子作者不必 import mathx
    def roll_sum(self, mat, n, min_count=None):
        return _mx.roll_sum(mat, n, min_count)

    def shift(self, mat, k):        return _mx.shift(mat, k)
    def diff(self, mat, k=1):       return _mx.diff(mat, k)

    def pct_change(self, mat, k=1, min_abs_den=1e-12):
        """k 期变化率。`min_abs_den` 给分母设地板 —— 变化率类因子的分母经常近零
        （壳公司的账面价值、停牌前的地量成交），不设地板会产出 ±1e3 量级的假值。"""
        return _mx.pct_change(mat, k, min_abs_den)
    def roll_mean(self, mat, n, min_count=None):
        return _mx.roll_mean(mat, n, min_count)
    def roll_std(self, mat, n, min_count=None, ddof=0):
        return _mx.roll_std(mat, n, min_count, ddof)
    def roll_var(self, mat, n, min_count=None, ddof=0):
        return _mx.roll_var(mat, n, min_count, ddof)
    def roll_max(self, mat, n, min_count=None):
        return _mx.roll_max(mat, n, min_count)
    def roll_min(self, mat, n, min_count=None):
        return _mx.roll_min(mat, n, min_count)
    def roll_argmax(self, mat, n):  return _mx.roll_argmax(mat, n)
    def roll_argmin(self, mat, n):  return _mx.roll_argmin(mat, n)
    def roll_rank(self, mat, n, min_count=None):
        return _mx.roll_rank(mat, n, min_count)
    def roll_corr(self, x, y, n, min_count=None):
        return _mx.roll_corr(x, y, n, min_count)
    def roll_cov(self, x, y, n, min_count=None):
        return _mx.roll_cov(x, y, n, min_count)
    def roll_skew(self, mat, n, min_count=None):
        return _mx.roll_skew(mat, n, min_count)
    def roll_kurt(self, mat, n, min_count=None):
        return _mx.roll_kurt(mat, n, min_count)
    def roll_count(self, mat, n, min_count=0):
        return _mx.roll_count(mat, n, min_count)
    def roll_quantile(self, mat, n, q, min_count=None):
        """滚动分位数（历史模拟法的 VaR / CVaR 用）。"""
        return _mx.roll_quantile(mat, n, q, min_count)
    def roll_prod(self, mat, n, min_count=None):
        return _mx.roll_prod(mat, n, min_count)
    def ewm_mean(self, mat, span=None, alpha=None, min_count=1):
        return _mx.ewm_mean(mat, span, alpha, min_count)
    def decay_linear(self, mat, n, min_count=None):
        return _mx.decay_linear(mat, n, min_count)
    def cs_demean(self, mat, mask=None):   return _mx.cs_demean(mat, mask)
    def cs_zscore(self, mat, mask=None):   return _mx.cs_zscore(mat, mask)
    def cs_winsor(self, mat, lo=0.01, hi=0.99, mask=None):
        return _mx.cs_winsor(mat, lo, hi, mask)
    @staticmethod
    def signed_power(mat, p):       return _mx.signed_power(mat, p)
    @staticmethod
    def safe_log(mat):              return _mx.safe_log(mat)
    @staticmethod
    def safe_sqrt(mat):             return _mx.safe_sqrt(mat)

    # ---------------------------------------------------------------- 财务
    # ⚠️ 每次进财务区先 trim 一次面板缓存：不同 (因子, 年) 任务的窗口不同，
    #    不 trim 的话缓存只增不减，一个 worker 能吃掉 6 GB+（实测把内存顶到 OOM）。
    def _trim(self):
        self.deriv.trim_cache(self.panel)
        if self.prices is not None:
            self.prices.trim_cache(self.panel)

    def ttm(self, field: str) -> np.ndarray:
        self._trim()
        return self.deriv.to_panel(self.panel, field, mode="ttm", lag=0)

    def lag_ttm(self, field: str, k: int = 4) -> np.ndarray:
        self._trim()
        return self.deriv.to_panel(self.panel, field, mode="ttm", lag=k)

    def point(self, field: str, lag: int = 0) -> np.ndarray:
        self._trim()
        return self.deriv.to_panel(self.panel, field, mode="point", lag=lag)

    # ---------------------------------------------------------------- 通用
    def code_index(self, codes) -> np.ndarray:
        """股票代码 -> 列下标（不在池子里返回 -1）。

        ★ 传**列下标**（整数）会**静默全 miss** —— 字典的键是代码字符串，
          整数一个都匹配不上，于是所有下游网格变成全 0/全 NaN，而且**不报错**。
          实测有因子开发 Agent 因此让 24 个因子全部产出 0 行、日志却全是 ✓。
          所以这里直接把这种用法拦掉，别让它在数据里留下痕迹。
        """
        arr = np.asarray(codes)
        if arr.dtype.kind in "iu" and arr.size:
            raise TypeError(
                "code_index / event_grid / asof_daily 的第一个参数是**股票代码**"
                "（如 '600519.SH'），不是列下标。传整数会静默全部 miss、"
                "产出全 0 网格而不报错。\n"
                "  如果你手上已经是 `ctx.code_index(...)` 的结果，就直接用，"
                "不要再传回来。")
        return np.array([self._pos.get(c, -1) for c in codes], dtype=np.int64)

    def asof_daily(self, codes, days, values) -> np.ndarray:
        """把 (codes, days) 上的稀疏披露值 as-of 前向填充到日频网格。"""
        idx = self.code_index(codes)
        keep = idx >= 0
        if not keep.any():
            return self.panel.empty()
        c = idx[keep]
        d = np.asarray(days, dtype=np.int32)[keep]
        v = np.asarray(values, dtype=np.float64)[keep]
        order = np.lexsort((d, c))            # 先按 code 再按 day，满足 asof 的前置排序
        return self.panel.asof(c[order], d[order], v[order])

    def event_grid(self, codes, days, weights=None) -> np.ndarray:
        """把稀疏事件铺成 (T, C)，同格多条累加。"""
        idx = self.code_index(codes)
        keep = idx >= 0
        return self.panel.scatter(idx[keep].astype(np.int32),
                                  np.asarray(days, dtype=np.int32)[keep],
                                  None if weights is None else np.asarray(weights, dtype=np.float64)[keep])

    def dataset(self, name: str, columns: list[str] | None = None,
                years: tuple[int, int] | None = None) -> pd.DataFrame:
        return self.up.read(name, columns=columns, years=years)

    def date_col(self, s: pd.Series) -> np.ndarray:
        return series_to_int(s)

    # ---------------------------------------------------------------- 数学
    # ⚠️ 这里**不要**再定义 `roll_sum` —— 上面「数学转发」那一节已经有一个带
    #    `min_count` 的版本。同一个类里定义两次，后一个会静默覆盖前一个，
    #    于是 `ctx.roll_sum(mat, n, min_count=...)` 会抛 TypeError
    #    （实测被因子开发 Agent 撞到，只能绕道 `roll_mean × roll_count`）。

    @staticmethod
    def safe_div(num: np.ndarray, den: np.ndarray, min_abs_den: float = 0.0) -> np.ndarray:
        """带分母保护的除法：分母绝对值过小或为 NaN 时返回 NaN 而不是 inf/巨值。

        财报因子里这个保护是必需的——净利润、经营现金流都可能为负或近零，
        直接相除会产生 ±1e6 量级的假因子值，把截面排名整个带偏。
        """
        d = np.asarray(den, dtype=np.float64)
        ok = np.isfinite(d) & (np.abs(d) > min_abs_den)
        out = np.full(np.broadcast(num, d).shape, np.nan, dtype=np.float64)
        np.divide(num, d, out=out, where=ok)
        return out.astype(np.float32)
