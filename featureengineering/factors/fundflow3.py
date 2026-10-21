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




# ══════════════════════════════════════════════════════════════════════
# 1. 档位 × 价格（3 个）
# ══════════════════════════════════════════════════════════════════════



@register(FactorSpec(
    name="mf_small_order_avg_price_dev", version=2,
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




# ══════════════════════════════════════════════════════════════════════
# 2. 相对自身趋势的「意外度」（3 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="mf_net_vol_surprise_20", version=2,
    group=GROUP,
    deps=(FLOW, 'stock_daily'),
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






# ══════════════════════════════════════════════════════════════════════
# 3. 跨档位的**二阶**结构（3 个）
# ══════════════════════════════════════════════════════════════════════







# ══════════════════════════════════════════════════════════════════════
# 4. 时序高阶矩与持续性（3 个）
# ══════════════════════════════════════════════════════════════════════



@register(FactorSpec(
    name="mf_big_order_net_ac1_20", version=2,
    group=GROUP,
    deps=(FLOW, 'stock_daily'),
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


