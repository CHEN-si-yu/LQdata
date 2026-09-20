"""资金流「档位动力学」族（12 个）—— 全部日频产出、只主板、跟随 `default_start`。

数据源：`stock_main_fund_flow`（四档买卖 sm/md/lg/elg 的量与额 + 厂商净额，2010-01-04 起）。
本文件是 `fundflow.py` / `fundflow2.py` 的**互补**家族。

═══════════════════════════════════════════════════════════════════════════
一、★★★ 为什么只剩「动力学」这一维 —— 三条恒等式把「档位水平」挖干了
═══════════════════════════════════════════════════════════════════════════

`fundflow.py` 与 `fundflow2.py` 的 docstring 已经实测建档了下面的结构性事实，
本文件的设计**完全建立在它们之上**（不能绕过，绕不过去）：

① **双记口径**：每笔成交同时进买方桶与卖方桶 ⇒ `Σ4 buy ≡ Σ4 sell` ⇒
   四档**净额**之和恒为 0 ⇒ 同日的四档净额只活在 **3 维**（不是 4 维）空间里。
   推论：`big_net − small_net ≡ 2·big_net + md_net` —— **任何窗口平滑都消不掉**
   这个线性恒等关系（`mf_big_small_divergence` 的 note 记的就是这件事）。

② **100 元/手常数**：`netA/netV` 与 `totA/totV` 是同一个换算比 ⇒
   每一个「规模**量**占比」都有与之 rank 相关 0.99999 的「规模**额**占比」版本。
   ⇒ 同一天的量版与额版**只能选一个**（`fundflow2.py` 实测 4 例）。

③ **厂商 `net_mf_amount` 是独立序列**：与 `lg+elg` 净额的 rank 相关只有 0.60~0.67
   ⇒ 它**不是**四档净额的任何线性组合，可以独立使用。

⇒ **水平维已经被挖干**（比较口径的因子、同维重标定的因子都在 2026-09-17 被删了 10 个）。
   本文件只做**动力学与交互**：时序高阶矩、跨档位相关的滚动相关、档位×价格位置、
   相对自身趋势的意外度、以及需要跳出自归一化才能拿到的「单位价格冲击的流量」。

**实测过的删除名单**（`docs/FACTOR_TRIAGE_20260915.md` 的 108 个未注册名 + 34 个已删）：
`mf_small_order_ratio` / `mf_mid_order_ratio` / `mf_elg_order_ratio` /
`mf_order_concentration` / `mf_smart_dumb_divergence` / `mf_flow_streak_5d` /
`mf_flow_acceleration_5d` / `mf_flow_stability_20d` / `mf_cumulative_flow_20d` /
`mf_tier_net_spread_20` —— 本文件**一个都不重建**（写这条时已逐个核过名单）。

═══════════════════════════════════════════════════════════════════════════
二、与 `fundflow.py` 的代码关系
═══════════════════════════════════════════════════════════════════════════

本文件**自带**一份 `_Flow` 接入器，不从 `fundflow.py` import ——
契约「一个 Agent 一个文件」不允许跨家族私有 import（会让两个家族互相耦合，
改一个就动另一个）。接入语义逐条对齐：

  · 先按行算派生量、再**每列一次** `event_grid`（不是四档各铺一遍）；
  · 同格多条用 `safe_div(..., min_abs_den=0.5)` 取均值 ⇒ 天然幂等；
  · **停牌日 → NaN**（`ctx.traded()` 掩码），不是 0；
  · `self.codes` 必须存**原始股票代码**（存列号会让 `event_grid` **静默全 0**）。

═══════════════════════════════════════════════════════════════════════════
三、三处**故意偏离参考库字面公式**（逐条给理由，见各因子 note）
═══════════════════════════════════════════════════════════════════════════

1. `mf_net_vol_surprise_20`：参考库除以 `|MA20(净量)|` —— 股票净流量为负时
   **符号会静默翻转**（流出扩大被读成「意外上升」），本文件改为除以**毛额的 20 日均值**。
2. `mf_net_amount_mom5_to_mv`：参考库原式**未归一化**（单位是万元）⇒
   截面排序实质是**市值排序**；本文件除以自算总市值，变成无尺度的流量动量。
3. `mf_small_order_avg_price_dev`：参考库取比值，本文件取**绝对偏离**
   `|比值 − 1|`（参考库自己的方向标注就是「偏离越小越好」，用绝对值才与之一致）。

"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register

FLOW = "stock_main_fund_flow"
GROUP = "fundflow"
FLOW_START = None

TIERS = ("sm", "md", "lg", "elg")
_BUY_A = tuple(f"buy_{t}_amount" for t in TIERS)
_SELL_A = tuple(f"sell_{t}_amount" for t in TIERS)
_BUY_V = tuple(f"buy_{t}_vol" for t in TIERS)
_SELL_V = tuple(f"sell_{t}_vol" for t in TIERS)
_ALL_COLS = ("stock_code", "trade_date", *_BUY_A, *_SELL_A, *_BUY_V, *_SELL_V,
             "net_mf_amount", "net_mf_vol")

WAN2YUAN = 1e4
W1, W5, W20, W60 = 60, 29, 56, 128

# 双记口径提醒（会拼进受影响的 note）
_DE = ("（双记口径：Σ4buy ≡ Σ4sell ⇒ 四档净额之和恒为 0，见模块 docstring §一.①）")
# 峰度退化守卫的阈值（抄 `ret_kurt_20` 的既有做法）
_KURT_LIM = 1e4


class _Flow:
    """`stock_main_fund_flow` → (T, C)。语义与 `factors/fundflow.py::_Flow` 逐条对齐。"""

    def __init__(self, ctx):
        self.ctx = ctx
        self._rows: dict[str, np.ndarray] = {}
        self._grids: dict[str, np.ndarray] = {}
        self.tr = ctx.traded()
        d0, d1 = int(ctx.panel.dates[0]), int(ctx.panel.dates[-1])
        df = ctx.dataset(FLOW, columns=list(_ALL_COLS), years=(d0 // 10000, d1 // 10000))
        self.df = df
        self.empty = df is None or df.empty
        self.codes = self.days = None
        if not self.empty:
            # ★ 存**原始代码**，不能存 code_index 的结果（否则 event_grid 静默全 0）
            self.codes = df["stock_code"].to_numpy()
            self.days = ctx.date_col(df["trade_date"])
            if int((ctx.code_index(self.codes) >= 0).sum()) == 0:
                raise RuntimeError(
                    f"{FLOW} 的 {len(self.codes)} 行代码没有一个能对上面板，代码格式不匹配")

    def rows(self, kind: str, tier: str | None = None) -> np.ndarray:
        key = kind if tier is None else f"{kind}:{tier}"
        v = self._rows.get(key)
        if v is not None:
            return v
        df = self.df
        if kind == "net_amt":
            v = (df[f"buy_{tier}_amount"] - df[f"sell_{tier}_amount"]).to_numpy(np.float64)
        elif kind == "net_vol":
            v = (df[f"buy_{tier}_vol"] - df[f"sell_{tier}_vol"]).to_numpy(np.float64)
        elif kind == "gross_amt":
            v = (df[f"buy_{tier}_amount"] + df[f"sell_{tier}_amount"]).to_numpy(np.float64)
        elif kind == "gross_vol":
            v = (df[f"buy_{tier}_vol"] + df[f"sell_{tier}_vol"]).to_numpy(np.float64)
        elif kind == "sell_amt":
            v = df[f"sell_{tier}_amount"].to_numpy(np.float64)
        elif kind == "vendor_amt":
            v = df["net_mf_amount"].to_numpy(np.float64)
        elif kind == "vendor_vol":
            v = df["net_mf_vol"].to_numpy(np.float64)
        elif kind == "tot_amt":
            v = df[list(_BUY_A + _SELL_A)].to_numpy(np.float64).sum(axis=1)
        elif kind == "tot_vol":
            v = df[list(_BUY_V + _SELL_V)].to_numpy(np.float64).sum(axis=1)
        else:
            raise KeyError(f"_Flow 不认识的派生量 {kind!r}")
        self._rows[key] = v
        return v

    def g(self, kind: str, tier: str | None = None) -> np.ndarray:
        key = kind if tier is None else f"{kind}:{tier}"
        hit = self._grids.get(key)
        if hit is not None:
            return hit
        ctx = self.ctx
        if self.empty:
            v = ctx.panel.empty()
        else:
            x = self.rows(kind, tier)
            got = ctx.event_grid(self.codes, self.days, np.ones(x.size, dtype=np.float64))
            v = ctx.safe_div(ctx.event_grid(self.codes, self.days, x), got, min_abs_den=0.5)
            v = np.where(self.tr, v, np.nan)
        self._grids[key] = v
        return v

    # ------------------------------------------------------------ 常用组合
    def rate(self, tier: str, base: str = "amt", net: bool = True) -> np.ndarray:
        """单档 (净|毛) / 当日八列总 (额|量)。"""
        k = ("net_" if net else "gross_") + base
        den = "tot_amt" if base == "amt" else "tot_vol"
        return self.ctx.safe_div(self.g(k, tier), self.g(den), min_abs_den=1e-6)

    def big_net(self, base: str = "amt") -> np.ndarray:
        """大单 + 超大单的净额（参考库的「机构资金」口径）。"""
        return self.g("net_" + base, "lg") + self.g("net_" + base, "elg")

    def mktcap(self) -> np.ndarray:
        """自算总市值（元）= 未复权收盘 × 当期已披露股本 —— 与 `quality.py` /
        `mf_net_amount_intensity` 同口径（**不用** `stock_finance.total_mv`，
        那张日频快照表会被厂商事后重算）。"""
        return np.asarray(self.ctx.px("close"), dtype=np.float64) * \
            np.asarray(self.ctx.px("total_share"), dtype=np.float64)


def _ser(x: np.ndarray) -> np.ndarray:
    """把 (T,1) 的市场向量摊成 (T,)，便于 shift。"""
    return np.ascontiguousarray(x[:, 0])


# ══════════════════════════════════════════════════════════════════════
# 1. 档位 × 价格（3 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="large_order_timing_signal",
    group=GROUP,
    deps=(FLOW, "stock_daily", "stock_adj_factor"),
    desc="大单净流入占比 × (1 − 20 日价格位置)：大钱在低位买",
    formula='big_net = (buy_lg+buy_elg-sell_lg-sell_elg)/_total_amount(mf)\n'
            'high_20 = adj.rolling(20,10).max(); low_20 = ...\n'
            'position = (adj-low_20)/(high_20-low_20)\n'
            'signal = big_net*(1-position)',
    start=FLOW_START,
    warmup_days=W20,
    higher_is_better=True,
    note=("逐字抄参考库 fund_flow 的 `large_order_timing_signal`。"
          "★ 参考库用 `_adjusted_close`（**前复权**）算价格位置，本项目**禁止** qfq"
          "（`fea/spec.py` 的红线）⇒ 改用 `ctx.hfq`（截至当日的后复权），"
          "数学上等价于「截至当日的复权序列」，且历史值稳定。"
          "★ 为什么这是一个**交互**而不是两个因子的乘积："
          "「大单净买入」单独看无法区分「低位吸筹」与「高位接盘」，"
          "乘上 `1 − 位置` 之后才表达「在大钱**便宜**的时候买」。"
          "`safe_div` 挡掉 20 日区间为 0（长期一字板）的退化格。"),
))
def large_order_timing_signal(ctx):
    f = _Flow(ctx)
    bn = f.rate("lg", net=True) + f.rate("elg", net=True)
    c = np.asarray(ctx.hfq("close"), dtype=np.float64)
    hi = ctx.roll_max(c, 20, 10)
    lo = ctx.roll_min(c, 20, 10)
    pos = ctx.safe_div(c - lo, hi - lo, 1e-12)
    return bn * (1.0 - pos)


@register(FactorSpec(
    name="mf_small_order_avg_price_dev",
    group=GROUP,
    deps=(FLOW, "stock_daily"),
    desc="小单成交均价相对当日 VWAP 的绝对偏离（散户的成交价劣势）",
    formula='sm_avg_price = safe_divide(sm_total_amt, sm_total_vol)\n'
            'ratio = safe_divide(sm_avg_price, vwap)\n'
            'return cross_sectional_rank(-deviation)   # deviation = (ratio-1).abs()',
    start=FLOW_START,
    warmup_days=W1,
    higher_is_better=False,
    note=("抄参考库 `ext_mf_small_order_average_price_deviation`。"
          "★ 参考库的**方向标注是负**（偏离越小排越前），所以本因子返回"
          "**绝对偏离** `|比值 − 1|`，与参考库排序方向一致；"
          "写成 `比值 − 1` 会让「散户大幅折价买入」和「大幅溢价买入」"
          "分居两端，语义与参考库不符。"
          "★ 为什么**只做小单**版：额版的兄弟 `ext_mf_large_order_avg_price` 已经"
          "以 `mf_large_order_avg_price` 之名注册（超大单成交均价水平），"
          "本因子是散户侧、且是**相对 VWAP 的偏离**（不是绝对价格水平）。"
          "★ 量纲：`sm_amt/sm_vol` 与 `amt/vol` 都是「元/股」，比值自约；"
          "停牌 / 无成交由 `safe_div` 给 NaN。"),
))
def mf_small_order_avg_price_dev(ctx):
    f = _Flow(ctx)
    sm_v = f.g("gross_vol", "sm")
    sm_a = f.g("gross_amt", "sm")
    vwap = ctx.safe_div(f.g("tot_amt"), f.g("tot_vol"), 1e-6)
    sm_px = ctx.safe_div(sm_a, sm_v, 1e-6)
    return np.abs(ctx.safe_div(sm_px, vwap, 1e-8) - 1.0)


@register(FactorSpec(
    name="mf_flow_price_absorption_20",
    group=GROUP,
    deps=(FLOW, "stock_daily", "stock_adj_factor"),
    desc="20 日大单净流入占比累计 / 20 日累计绝对收益（单位价格冲击吸收的流量）",
    formula="flow = roll_sum(big_net_ratio, 20, 10); move = roll_sum(abs(ret), 20, 10); "
            "factor = flow / move",
    start=FLOW_START,
    warmup_days=W20,
    higher_is_better=True,
    note=("**本文件新造**。" + _DE + " 分子用**大单净额占八列毛额之比**"
          "（量纲自约的无尺度流量），分母用 20 日**累计绝对日收益**"
          "（价格走了多远，不看方向）。"
          "比值高 ⇒ 机构净买入很多但价格几乎没动 = **潜伏式建仓**"
          "（吸筹被市场流动性无声吸收）；比值低 ⇒ 流量不大却价格大幅波动 = "
          "**薄盘**或**派发**。"
          "★ 与同日尺度的 `large_order_timing_signal` 互补："
          "那个是「今天在什么位置买」，本因子是「20 天里买进去了多少、价格让了多少」——"
          "一个横截面位置量、一个时序弹性量，窗口与信息都不同。"
          "`roll_sum` 用 (T,C) 原语；`ctx.roll_sum` 支持 `min_count`（2026-09-17 已修"
          "那处重复定义的坑），此处用显式 `min_count=N//2`。"),
))
def mf_flow_price_absorption_20(ctx):
    f = _Flow(ctx)
    flow = f.rate("lg", net=True) + f.rate("elg", net=True)
    move = np.abs(ctx.ret(1))
    s_flow = ctx.roll_sum(flow, 20, 10)
    s_move = ctx.roll_sum(move, 20, 10)
    return ctx.safe_div(s_flow, s_move, 1e-8)


# ══════════════════════════════════════════════════════════════════════
# 2. 相对自身趋势的「意外度」（3 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="mf_net_vol_surprise_20",
    group=GROUP,
    deps=(FLOW,),
    desc="厂商净流入量的 20 日意外度 = (净量 − 20 日均) / 毛量 20 日均",
    formula="ma_20 = net_vol.rolling(20,10).mean()\n"
            "divergence = safe_divide(net_vol, ma_20.abs()+1e-10) - 1.0   # 参考库原式\n"
            "本实现：(net_vol - ma_20) / mean(total_vol, 20)",
    start=FLOW_START,
    warmup_days=W20,
    higher_is_better=True,
    note=("★★ **故意偏离参考库**（`mf_net_vol_ma_divergence`）。参考库原式是"
          "`net_vol / |MA20(net_vol)| − 1`，有两个静默缺陷："
          "① 分母取**绝对值** ⇒ 净流量长期为**负**的股票（持续净流出）"
          "「流出扩大」会被读成「意外**上升**」，符号完全翻转；"
          "② 净流量的 20 日均值可以**跨零**，比值在均值附近会爆成 ±1e3 量级。"
          "本实现改成标准的 z-型意外度 `(本期 − 20 日均) / 毛量 20 日均`："
          "**有符号、有界、无尺度**（分母用总成交量而不是净流量，永远为正）。"
          "★ 为什么用 `vendor_vol`（厂商净量）而不是四档净额之和："
          "模块 docstring §一.③ —— 厂商净额与四档净额是**两个独立序列**"
          "（rank 相关仅 0.60~0.67），四档之和恒为 0，厂商列才是真正可用的「净」量。"
          "★ 量版与额版**只能选一个**（100 元/手常数 ⇒ 两者 rank 相关 0.99999，"
          "见 §一.②）—— 本因子用量版；同族额版已在 `mf_net_amount_intensity` 覆盖。"),
))
def mf_net_vol_surprise_20(ctx):
    f = _Flow(ctx)
    nv = f.g("vendor_vol")
    ma = ctx.roll_mean(nv, 20, 10)
    gross = ctx.roll_mean(f.g("tot_vol"), 20, 10)
    return ctx.safe_div(nv - ma, gross, 1e-6)


@register(FactorSpec(
    name="mf_net_amount_mom5_to_mv",
    group=GROUP,
    deps=(FLOW, "stock_daily"),
    desc="厂商净流入额的 5 日变化 / 总市值（无尺度的流量动量）",
    formula='mom = net_mf_amount.diff(5); return cross_sectional_rank(mom)   # 参考库原式\n'
            '本实现：mom * 1e4 / (close * total_share)',
    start=FLOW_START,
    warmup_days=W5,
    higher_is_better=True,
    note=("源自参考库 `net_mf_amount_momentum_5d`。"
          "★★ **必须归一化**：参考库原式的单位是**万元且未除以任何规模量** ⇒ "
          "它在横截面上的排序**实质上就是市值排序**（大票的万元流量天然大几个量级），"
          "与 `log_mv` rank 相关极高，作为「流量动量」因子完全失效。"
          "本实现除以**自算总市值**（`close × total_share`，未复权价 × 当期已披露股本），"
          "与同家族的 `mf_net_amount_intensity` / `elg_net_60d_to_mv` 同一市值口径"
          "（**不用** `stock_finance.total_mv` —— 日频快照表会被厂商事后重算）。"
          "★ 厂商 `net_mf_amount` 是独立序列（§一.③），本因子不与四档净额重复。"),
))
def mf_net_amount_mom5_to_mv(ctx):
    f = _Flow(ctx)
    n = f.g("vendor_amt") * WAN2YUAN
    mv = f.mktcap()
    return ctx.safe_div(ctx.diff(n, 5), mv, 1e-6)


@register(FactorSpec(
    name="mf_order_size_entropy_chg_5d",
    group=GROUP,
    deps=(FLOW, "stock_daily"),
    desc="四档成交量分布信息熵的 5 日变化（参与结构形状的迁移）",
    formula='sm_share = (buy_sm_vol+sell_sm_vol)/total_vol;  # md/lg/elg 同理\n'
            'entropy = -(Σ p·ln(p+1e-10))/ln(4);  factor = entropy.diff(5)',
    start=FLOW_START,
    warmup_days=W5,
    higher_is_better=True,
    note=("**本文件新造**：在已注册的 `mf_order_size_entropy`（**水平**）上取 5 日差分。"
          "★ 为什么它**不是** `mf_order_size_entropy` 的重复、也不是"
          "`order_size_ratio_change` 的重复：熵是四档占比的**非线性**函数"
          "（`−Σp ln p`），它的变化量不在「某一档占比的变化」所张成的空间里 ——"
          "同样的「大单占比 +2%」，从中档迁移过来与从小单迁移过来，"
          "熵的变化完全不同。"
          "★ 口径与 `mf_order_size_entropy` **逐字一致**（**成交量**口径、"
          "除以 ln4 归一化到 [0,1]、`p+1e-10`），保证差分是同一序列的差分。"
          "★ 分母 = 四档毛量之和（Σp ≡ 1，量纲自约，无需单位换算）。"),
))
def mf_order_size_entropy_chg_5d(ctx):
    f = _Flow(ctx)
    tot = f.g("tot_vol")
    h = np.zeros_like(tot)
    for t in TIERS:
        p = ctx.safe_div(f.g("gross_vol", t), tot, 1e-6)
        h = h - p * np.log(np.maximum(p, 0.0) + 1e-10)
    h = ctx.safe_div(h, np.log(4.0), 1e-12)
    # 四档全缺 -> h 会是 0/ln4 = 0 的假值；用 tot 的有效性把它挡回 NaN
    h = np.where(np.isfinite(tot), h, np.nan)
    return ctx.diff(h, 5)


# ══════════════════════════════════════════════════════════════════════
# 3. 跨档位的**二阶**结构（3 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="mf_big_mid_net_corr_20",
    group=GROUP,
    deps=(FLOW,),
    desc="大单净额 与 中单净额 的 20 日滚动相关（机构与中户是否同向）",
    formula="b = (buy_lg+buy_elg-sell_lg-sell_elg); m = (buy_md-sell_md)\n"
            "factor = roll_corr(b, m, 20, 10)",
    start=FLOW_START,
    warmup_days=W20,
    higher_is_better=True,
    note=("**本文件新造**。" + _DE + " ★ 关键：**滚动相关跳出了恒等式的线性张成** —— "
          "四档净额虽只活在 3 维空间里，但「两个序列的**相关**」是二阶量，"
          "不是任何单一档位净额的线性函数，所以它不是既有因子的重标定。"
          "经济含义：> 0 ⇒ 中户跟着机构同向（**跟风一致**，信号可信度高）；"
          "< 0 ⇒ 机构买、中户卖（**对手盘**，筹码从散户向机构转移）。"
          "与已注册的 `mf_big_small_divergence`（大单净 − 小单净，**方向差**）不同："
          "一个是两序列的**协动性**，一个是两序列的**水平差**，"
          "可以「大单大幅净买、中单也小幅净买」相关为 1 而差值为大。"
          "用**净额占比**（÷八列毛额）而不是净额原值，避免大市值股票主导相关。"),
))
def mf_big_mid_net_corr_20(ctx):
    f = _Flow(ctx)
    b = f.rate("lg", net=True) + f.rate("elg", net=True)
    m = f.rate("md", net=True)
    return ctx.roll_corr(b, m, 20, 10)


@register(FactorSpec(
    name="mf_tier_flow_agreement_20",
    group=GROUP,
    deps=(FLOW,),
    desc="「大单与中单同向」频率 − 「中单与小单同向」频率（20 日）",
    formula="agree(a,b) = sign(a)==sign(b); factor = mean(agree(big,mid),20,10) "
            "- mean(agree(mid,small),20,10)",
    start=FLOW_START,
    warmup_days=W20,
    higher_is_better=True,
    note=("**本文件新造**。" + _DE + " 与 `mf_big_mid_net_corr_20` 的分工："
          "那个用**相关系数**（对幅度敏感，被大流量日主导），"
          "本因子用**符号一致频率**（只看方向、对幅度免疫）——"
          "同样的相关性可以由「每天小幅同向」或「几天大幅同向」产生，"
          "两者对「资金结构是否稳定」的含义完全不同。"
          "★★ 符号判定**必须**先做有限值掩码再比较："
          "`np.sign(NaN)` 是 NaN 而 `NaN != NaN` 为 True，"
          "写成 `sign(a) != sign(b)` 会把缺失日**静默计成「不同向」**"
          "（这类 bug 已在 `mf_flow_stability_20d` 的 note 里记录过一次）。"
          "⚠ **取值卡片化警告**：20 日均值只有 21 个可能取值，"
          "`fea/eval.py` 的 LOWCARD 判据是「每日唯一值中位 ≤ 20」——"
          "沙箱自检阶段必看每日截面唯一值数，不足就地砍。"),
))
def mf_tier_flow_agreement_20(ctx):
    f = _Flow(ctx)
    big = f.rate("lg", net=True) + f.rate("elg", net=True)
    mid = f.rate("md", net=True)
    sml = f.rate("sm", net=True)

    def agree(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return np.where(ok, (np.sign(a) == np.sign(b)).astype(np.float64), np.nan)

    return (ctx.roll_mean(agree(big, mid), 20, 10)
            - ctx.roll_mean(agree(mid, sml), 20, 10))


@register(FactorSpec(
    name="mf_elg_lg_split_20",
    group=GROUP,
    deps=(FLOW,),
    desc="超大单净额占比 − 大单净额占比（20 日均值）：哪一级机构在买",
    formula="e = elg_net/total; l = lg_net/total; factor = mean(e,20,10) - mean(l,20,10)",
    start=FLOW_START,
    warmup_days=W20,
    higher_is_better=True,
    note=("**本文件新造**。" + _DE + " ★ **与「大单」合计口径的区别**："
          "既有的所有因子都用 `big = lg + elg` 把这两档**合并**了 ——"
          "合并会丢掉「**哪一级**在买」这一维。"
          "在双记恒等式下，同日的四档净额只活在 3 维空间里，"
          "但 `elg − lg` 是其中**一个独立方向**（不与 `lg + elg` 共线），"
          "而且本因子取的是**20 日均值之差**（不是同日之差），"
          "在时间维上进一步与同日线性组合分离。"
          "经济含义：超大单 = 公募/保险/量化通道，大单 = 游资/私募 ——"
          "> 0 ⇒ 配置型资金主导，< 0 ⇒ 交易型资金主导。"
          "⚠ 与已注册的 `super_large_order_intensity`（5 日累计超大单）可能有重叠，"
          "`dedup` 阶段定量裁决。"),
))
def mf_elg_lg_split_20(ctx):
    f = _Flow(ctx)
    e = f.rate("elg", net=True)
    l = f.rate("lg", net=True)
    return ctx.roll_mean(e, 20, 10) - ctx.roll_mean(l, 20, 10)


# ══════════════════════════════════════════════════════════════════════
# 4. 时序高阶矩与持续性（3 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="mf_big_order_net_kurt_20",
    group=GROUP,
    deps=(FLOW,),
    desc="大单净额占比的 20 日峰度（脉冲式建仓 vs 匀速滴灌）",
    formula='x = (mf["buy_lg_amount"]+mf["buy_elg_amount"]'
            '-mf["sell_lg_amount"]-mf["sell_elg_amount"])\n'
            'kurt = x.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).kurt())',
    start=FLOW_START,
    warmup_days=W20,
    higher_is_better=True,
    note=("抄参考库 fund_flow 的 `big_order_net_kurt_20`（窗口 20）。"
          + _DE + " 用**净额占比**（÷八列毛额）而不是净额原值："
          "峰度对尺度不变，但原值口径下大市值股票的流量序列会被市值量级主导，"
          "归一化后比较的是**形状**而不是规模。"
          "★★ **必须带退化守卫**（抄 `ret_kurt_20` 的既有做法）："
          "窗口内若二阶矩趋 0（连续多日净额几乎不变），峰度的分母趋 0 会产出"
          "1e8 量级的垃圾值，触发 `main.py check` 的「值域异常」。"
          f"守卫：`|kurt| > {_KURT_LIM:g}` → NaN。"
          "经济含义：峰度高 ⇒ 建仓是**几次大额脉冲**（事件驱动 / 大宗 / 指数调仓），"
          "峰度低 ⇒ **匀速滴灌**（算法拆单的被动配置）。"
          "这两类资金对后续收益的含义完全不同：脉冲式更可能是短期冲击。"),
))
def mf_big_order_net_kurt_20(ctx):
    f = _Flow(ctx)
    x = f.rate("lg", net=True) + f.rate("elg", net=True)
    k = ctx.roll_kurt(x, 20, 10)
    return np.where(np.isfinite(k) & (np.abs(k) <= _KURT_LIM), k, np.nan)


@register(FactorSpec(
    name="mf_big_order_net_ac1_20",
    group=GROUP,
    deps=(FLOW,),
    desc="大单净额占比的一阶自相关（20 日）（资金流入的持续性）",
    formula="x = big_net_ratio; factor = roll_corr(x, shift(x,1), 20, 10)",
    start=FLOW_START,
    warmup_days=W20,
    higher_is_better=True,
    note=("**本文件新造**。" + _DE + " 与「净流入天数 / 连续性」类因子的分工："
          "**已删**的 `mf_flow_streak_5d` / `mf_flow_stability_20d` 用的是"
          "**计数与波动**（几天为正、标准差多大），是**零阶/一阶矩**的时序统计；"
          "本因子是**自相关**（二阶），衡量「今天的流入能不能预测明天的流入」——"
          "一只每天小幅正流入的股票（低波动、高持续性）与一只"
          "「流入流出交替但长期累计为正」的股票（高波动、负持续性）"
          "在计数与波动口径下可能一模一样，本因子能把它们分开。"
          "★ 与已注册的 `mf_flow_factor_momentum_20`（同一序列的 20 日差分）不同："
          "差分是**水平的变化**，自相关是**时序结构**，两者正交性高。"
          "用净额占比（÷八列毛额）归一化，避免大市值主导。"),
))
def mf_big_order_net_ac1_20(ctx):
    f = _Flow(ctx)
    x = f.rate("lg", net=True) + f.rate("elg", net=True)
    return ctx.roll_corr(x, ctx.shift(x, 1), 20, 10)


@register(FactorSpec(
    name="mf_extra_large_sell_pressure",
    group=GROUP,
    deps=(FLOW,),
    desc="超大单**卖出**毛额占八列毛额的比重（顶级资金的派发压力）",
    formula="total = Σ(buy_*_amount) + Σ(sell_*_amount); "
            "elg_sell_ratio = sell_elg_amount / total; return cross_sectional_rank(-elg_sell_ratio)",
    start=FLOW_START,
    warmup_days=W1,
    higher_is_better=False,
    note=("抄参考库 fund_flow 的 `ext_mf_extra_large_sell_pressure`（方向负）。"
          + _DE + " ★ **只有「卖」这一侧是新的**：已注册的 "
          "`order_size_concentration` 用买卖**合计**的四档毛额结构，"
          "`mf_retail_dominance` 是小单（散户）侧，"
          "`mf_big_order_ratio` 是大单的**净**额 —— "
          "「超大单的**毛卖出**占全市场成交的比重」此前**没有**因子覆盖。"
          "它与「超大单净额」（买卖相抵后的差）不是同一个量："
          "一只股票可以超大单净买入为正、但毛卖出占比同时创高"
          "（大资金一边大举卖出、一边更大举买入 = 换手激烈的大资金博弈）。"
          "取值域 [0,1]，分母是当日该股的全部成交额。"),
))
def mf_extra_large_sell_pressure(ctx):
    f = _Flow(ctx)
    return ctx.safe_div(f.g("sell_amt", "elg"), f.g("tot_amt"), 1e-6)
