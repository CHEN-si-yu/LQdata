"""资金流因子族（24 个）—— 唯一数据源 `stock_main_fund_flow`（2010-01-04 起，14.0M 行）。

小单 = `sm`、中单 = `md`、大单 = `lg`、超大单 = `elg`。

═══════════════════════════════════════════════════════════════════════════
★ 三条实测约束（全部在本文件里挡掉，任何一个漏掉都会产出「看起来很正常的错面板」）
═══════════════════════════════════════════════════════════════════════════

**① 双记口径** —— 这张表每笔成交**同时**记进买方桶和卖方桶：

    Σ(buy_sm+md+lg+elg) ≡ Σ(sell_sm+md+lg+elg)      （实测，见下）

于是下面三件事都成立，照抄参考库会得到**常数因子**（截面排名全并列）：

    a) 净流入 := 买方四桶和 − 卖方四桶和      → 恒等于 0（只差浮点/取整噪声）
    b) 四档净额之和 (sm+md+lg+elg 净额)       → 恒等于 0（2015 年全表 max|·| = 0.03）
    c) 由 (b) 推出 大单净额 − 小单净额 = 2×大单净额 + 中单净额 → 与 `mf_big_order_ratio`
       的截面排名相关系数 0.98（不是常数，但几乎没有增量信息）

本文件统一改用**分层净额**（项目规定）：

    净额(档) := buy_<档>_amount − sell_<档>_amount

供应商自带的 `net_mf_amount` / `net_mf_vol` 是**现成的净额**（不是四桶差，
实测与「(lg+elg) 净额」的相关只有 **0.60~0.67**，与「四桶和之差」的相关是 **−0.001**）。
参考库里凡是直接写 `ff["net_mf_amount"]` 的因子（净流入率、5 日累计、波动、加速度、
streak、stability、量额背离、开口背离），本文件**沿用供应商净额**——这既与参考库逐字一致，
又避免与走分层净额的 `mf_big_order_ratio` 撞成同一个因子。两条净额口径并存提供了正交信息。

**② 单位** —— 实测（2013/2015/2018 三年逐年核对）：

    Σ8vol(手) / stock_daily.vol(股)   中位数 = 0.0200 = 2/100
    Σ8amount(万元) / stock_daily.amount(元) 中位数 = 0.0002 = 2/1e4

两个比值正好等于「双记(×2)」叠加「单位换算」，于是三条结论同时被证实：
`*_vol` 单位是**手**（100 股）、`*_amount` 单位是**万元**、且确实是双记。
**跨表相除必须先统一量纲**：本文件用到跨表的地方一律 ×100（手→股）或 ×1e4（万元→元），
并在该因子的 `note` 里写明；**同表内的比率量纲自约，不做任何换算**。

**③ 缺失语义** —— 资金流是**日频事件**，但「没有行」不等于「净流入 0」：
`ctx.event_grid` 默认缺失补 0，直接用会把停牌日当成 0 净流入，把滚动均值系统性稀释
（实测 2015 年每天只有 1361~2584 只有行情，停牌极多）。本文件的 `_Flow.g()` 做两道处理：

    1. 同时铺一张「出现指示」网格，用 `safe_div` 把没有记录的格子还原成 **NaN**；
    2. 再用 `ctx.traded()` 把**停牌日**整体掩成 NaN（这是 §① 之外的独立保证：
       有些上游会在停牌日仍吐一行 0）。

于是滚动窗口一定含 NaN，故所有滚动一律**显式给 `min_count`**（= 参考库的 `min_periods`）：
窗口 5→3、10→5、20→10、60→20。这是**显式放松**，理由就是停牌；
不放松的话 2015 年那种大面积停牌会让因子的非空率塌到 50% 以下。

═══════════════════════════════════════════════════════════════════════════
★ 框架的两处「静默陷阱」（本文件各挡了一道，建议后续收进 fea/）
═══════════════════════════════════════════════════════════════════════════

**坑 1：`ctx.event_grid(codes, ...)` 要的是「股票代码」，不是「列号」。**
它内部会 `code_index(codes)` 再映射一次；传列号进去 → `_pos.get(整数)` 全部 miss
→ 列号 −1 → `keep` 全 False → 整张网格**全 0，不报错**。
实测症状：24 个因子全部「非空率 0% / 落盘 0 行」，而日志里每一条都是 ✓。
本文件的 `_Flow` 因此只缓存**原始代码**，并加了一道 `n_map == 0` 的硬校验。
（建议：`event_grid` 收到 `codes.dtype.kind in "iu"` 时直接 raise。）

**坑 2：`ctx.roll_sum` 没有 `min_count` 参数。**
`fea/context.py` 里 `roll_sum` 定义了**两次**（第 151 行带 min_count、第 240 行不带），
后一处覆盖前一处；`Panel.roll_sum` 同样没有。传 `min_count=` 直接 TypeError。
本文件用 `_roll_sum()`（`roll_mean × roll_count`）复现，`liquidity.py` / `margin.py`
各自也抄了一份同名 helper —— 已经有三个家族在重复绕同一个坑，值得修。

═══════════════════════════════════════════════════════════════════════════
因子构成（24 个，三组口径）
═══════════════════════════════════════════════════════════════════════════

  A. 供应商净额口径（`net_mf_amount` / `net_mf_vol`）—— 10 个
     mf_net_inflow_ratio / mf_net_amount_intensity / mf_net_inflow_5d /
     mf_net_inflow_trend_5d / mf_net_inflow_volatility_20d / mf_flow_acceleration_5d /
     mf_flow_streak_5d / mf_flow_stability_20d / mf_cumulative_flow_20d /
     mf_vol_amount_divergence / mf_open_close_divergence_10d   （11 个）

  B. 分层净额口径（`buy_<档> − sell_<档>`）—— 10 个
     mf_elg_order_ratio / mf_big_order_ratio / mf_mid_order_ratio / mf_small_order_ratio /
     mf_large_order_net_5d / mf_smart_dumb_divergence / mf_tier_net_spread_20 /
     big_vs_small_divergence_5d / elg_net_60d_to_mv / mf_large_order_avg_price

  C. 毛额结构口径（`buy_<档> + sell_<档>`）—— 3 个
     mf_order_concentration / mf_order_size_entropy / mf_retail_dominance

**参考库给 27 个候选，本文件只做 24 个**，砍掉的 3 个都是**实测截面排名相关 = 1.000**
的精确重复（详见文件末尾 `_DROPPED`）。
"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register

# ---------------------------------------------------------------- 常量
FLOW = "stock_main_fund_flow"
GROUP = "fundflow"
# 上游 2010-01-04 起，早于 conf 的 default_start，跟随配置即可（start=None）
FLOW_START = None

TIERS = ("sm", "md", "lg", "elg")
MAIN_TIERS = ("lg", "elg")
_BUY_A = tuple(f"buy_{t}_amount" for t in TIERS)
_SELL_A = tuple(f"sell_{t}_amount" for t in TIERS)
_BUY_V = tuple(f"buy_{t}_vol" for t in TIERS)
_SELL_V = tuple(f"sell_{t}_vol" for t in TIERS)
_ALL_COLS = ("stock_code", "trade_date", *_BUY_A, *_SELL_A, *_BUY_V, *_SELL_V,
             "net_mf_amount", "net_mf_vol")

# 单位换算（实测，见模块 docstring ②）
LOT2SHARE = 100.0        # 手 -> 股
WAN2YUAN = 1e4           # 万元 -> 元

# 滚动窗口 min_count = 参考库的 min_periods（停牌造成窗口缺口，必须显式放松）
_MIN = {5: 3, 10: 5, 20: 10, 60: 20}

# 单日因子无所记忆，但引擎要往前多读一段才能保证增量结果 == 全量结果；
# 沿用契约里「日频聚合 → 60」的那一档。
W1, W5, W10, W15, W20, W60 = 60, 29, 38, 47, 56, 128

# 双记口径的提醒（会拼进每个受影响的 note；完整推导见模块 docstring ①）
_DE = ("双记口径：本表每笔成交同时进买方桶与卖方桶，Σ4buy ≡ Σ4sell，"
       "故参考库「买方四桶和 − 卖方四桶和」恒为 0（常数因子）。")


# ══════════════════════════════════════════════════════════════════════════
# 数据接入
# ══════════════════════════════════════════════════════════════════════════
class _Flow:
    """`stock_main_fund_flow` → (T, C) 网格。

    逐行先算好派生量（净额/毛额/总额），再**每列一次** `event_grid`，
    避免把 8 个桶各铺一遍。
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self._rows: dict[str, np.ndarray] = {}
        self._grids: dict[str, np.ndarray] = {}
        self.tr = ctx.traded()                      # (T,C) bool，停牌/未上市 -> False
        d0, d1 = int(ctx.panel.dates[0]), int(ctx.panel.dates[-1])
        df = ctx.dataset(FLOW, columns=list(_ALL_COLS),
                         years=(d0 // 10000, d1 // 10000))
        self.df = df
        self.empty = df is None or df.empty
        self.codes = self.days = None
        if not self.empty:
            # ★ `self.codes` 必须存**原始股票代码**，不能存 `ctx.code_index` 的结果！
            #   `ctx.event_grid` 内部会再调用一次 `code_index(codes)` 把代码映射成列号；
            #   若把「已经映射好的列号」再传进去，`_pos.get(整数)` 全部 miss -> 列号 -1
            #   -> `keep` 全 False -> 整张网格**静默全 0**（不报错！）。
            #   实测症状：因子非空率 0%、落盘 0 行，而日志里一切正常。
            self.codes = df["stock_code"].to_numpy()
            self.days = ctx.date_col(df["trade_date"])
            n_map = int((ctx.code_index(self.codes) >= 0).sum())
            if n_map == 0:
                raise RuntimeError(
                    f"{FLOW} 的 {len(self.codes)} 行股票代码没有一个能对上当前面板，"
                    f"说明代码格式不匹配（应该形如 000001.SZ）—— 这是代码 bug，不是数据缺失。")

    # ------------------------------------------------------------ 行级
    def rows(self, kind: str, tier: str | None = None) -> np.ndarray:
        key = kind if tier is None else f"{kind}:{tier}"
        v = self._rows.get(key)
        if v is not None:
            return v
        df = self.df
        if kind == "net_amt":
            v = (df[f"buy_{tier}_amount"] - df[f"sell_{tier}_amount"]).to_numpy(np.float64)
        elif kind == "gross_amt":
            v = (df[f"buy_{tier}_amount"] + df[f"sell_{tier}_amount"]).to_numpy(np.float64)
        elif kind == "net_vol":
            v = (df[f"buy_{tier}_vol"] - df[f"sell_{tier}_vol"]).to_numpy(np.float64)
        elif kind == "gross_vol":
            v = (df[f"buy_{tier}_vol"] + df[f"sell_{tier}_vol"]).to_numpy(np.float64)
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

    # ------------------------------------------------------------ 铺格
    def g(self, kind: str, tier: str | None = None) -> np.ndarray:
        """派生量 → (T,C)。**没有记录的日子 = NaN**（不是 0），停牌日 = NaN。"""
        key = kind if tier is None else f"{kind}:{tier}"
        hit = self._grids.get(key)
        if hit is not None:
            return hit
        ctx = self.ctx
        if self.empty:
            v = ctx.panel.empty()
        else:
            x = self.rows(kind, tier)
            ones = np.ones(x.size, dtype=np.float64)
            got = ctx.event_grid(self.codes, self.days, ones)
            # 同格多条时 safe_div 自动取均值，天然幂等；got<0.5 -> NaN
            v = ctx.safe_div(ctx.event_grid(self.codes, self.days, x), got, min_abs_den=0.5)
            v = np.where(self.tr, v, np.nan)        # ★ 停牌日 -> NaN
        self._grids[key] = v
        return v

    # ------------------------------------------------------------ 常用组合
    def rate(self, net_kind: str = "vendor_amt", tier: str | None = None) -> np.ndarray:
        """净额 / 当日八列毛额  —— 同表内量纲自约，无需换算。"""
        den = "tot_amt" if net_kind.endswith("_amt") else "tot_vol"
        return self.ctx.safe_div(self.g(net_kind, tier), self.g(den), min_abs_den=1e-6)

    def tier_rate(self, tier: str, base: str = "amt") -> np.ndarray:
        """单档净额 / 当日八列毛额（分层净入口径的通用形式）。"""
        return self.rate("net_" + base, tier)

    def main_gross(self, base: str = "amt") -> np.ndarray:
        """大单+超大单的买卖毛额（万元 或 手）。"""
        k = "gross_" + base
        return self.g(k, "lg") + self.g(k, "elg")


def _pos(x: np.ndarray) -> np.ndarray:
    """把「>0」变成 float 指示，**NaN 保持 NaN**（不许当成 0 混进滚动计数）。"""
    return np.where(np.isfinite(x), (x > 0).astype(np.float64), np.nan)


def _roll_sum(ctx, mat: np.ndarray, n: int, min_count: int) -> np.ndarray:
    """带 `min_count` 的滚动求和（本家族停牌多，每个滚动都必须显式放松）。

    ★ 框架缺陷：`ctx.roll_sum` **没有 `min_count` 参数**。`fea/context.py` 里
      `roll_sum` 定义了**两次**（第 151 行带 min_count、第 240 行不带），
      后一处覆盖前一处；`fea/panel.py` 的 `Panel.roll_sum` 同样没有。
      传 `min_count=` 会直接 `TypeError`。
      这也是 `factors/liquidity.py` / `factors/margin.py` 各自另起一个同名 helper 的原因
      —— 属于「每个家族都得自己抄一遍」的坑，已记入开发报告的框架缺口。

    等价性：`mathx.roll_mean` = `roll_sum(x−c0, n, min_count)/roll_count(x, n) + c0`
    （`roll_count` **不毒化**），所以 `roll_mean × roll_count` 精确还原
    「窗口内**有效值**之和」；窗口有效值不足 `min_count` 时 mean 被毒化成 NaN，
    `NaN × count = NaN`，缺失语义与 `roll_sum(min_count=...)` 完全一致。
    """
    return ctx.roll_mean(mat, n, min_count) * ctx.roll_count(mat, n)


# ══════════════════════════════════════════════════════════════════════════
# ① 供应商净额口径（net_mf_amount / net_mf_vol）
# ══════════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="mf_net_inflow_ratio", group=GROUP,
    deps=(FLOW, "stock_daily"),
    desc="主力资金净流入率 = 供应商净额 / 当日四档买卖总额（T 日单日）",
    formula='ratio = ff["net_mf_amount"] / _total_amount(ff)   # _total_amount = 买4+卖4 八列金额之和',
    start=FLOW_START, warmup_days=W1, higher_is_better=True,
    note="★ 用**供应商现成的** net_mf_amount，不是四桶差（四桶差恒为 0，见模块 docstring ①）。"
         "实测 net_mf_amount 与「(lg+elg) 净额」的相关只有 0.60~0.67、与「四桶和之差」的相关 "
         "−0.001 —— 它是供应商自己的口径，是独立的净额序列。"
         "分母 `tot_amt`（八列金额之和，万元）与分子同表同量纲，**比值无需任何换算**；"
         "它等于当日成交额的 2 倍（双记），对本因子只是全体同乘一个常数，不影响截面排名。"
         "缺失/停牌 -> NaN（`_Flow.g` 用「出现指示」+ `ctx.traded()` 双重保证），"
         "分母由 safe_div 保护（成交额为 0 的壳股 -> NaN）。",
))
def mf_net_inflow_ratio(ctx):
    f = _Flow(ctx)
    return f.rate("vendor_amt")


@register(FactorSpec(
    name="mf_net_amount_intensity", group=GROUP,
    deps=(FLOW, "stock_daily", "stock_finance"),
    desc="主力资金净额强度 = 净流入额 / 总市值（消除规模效应）",
    formula='intensity = net_mf_amount / circ_mv   （参考库用流通市值）\n'
            '本实现：net_amount * 1e4 / (close * total_share)   # 万元->元；总市值=元',
    start=FLOW_START, warmup_days=W1, higher_is_better=True,
    note="★ 两处偏离："
         "(1) **单位**：net_mf_amount 是**万元**、市值是**元**，跨表相除前 ×1e4 统一到元"
         "（参考库两列都是万元所以没写换算，本框架必须显式写，否则整体差 1e4 倍）；"
         "(2) 参考库用 `circ_mv`（流通市值），本实现用**总市值** = `ctx.px('close') × "
         "ctx.px('total_share')` —— 与同家族的 `elg_net_60d_to_mv` 保持同一市值口径，"
         "且两者都不影响截面排名的相对次序（只是分母口径差，rank 相关约 0.99）。"
         "市值取**未复权**收盘价 × 当期已披露股本（PIT 安全：股本随披露前向填充，"
         "历史值不因未来的送转变化）。"
         "数值量级约 1e-4（净额 500 万元 / 市值 50 亿元），引擎会做截面 winsor+rank。",
))
def mf_net_amount_intensity(ctx):
    f = _Flow(ctx)
    mv = ctx.px("close") * ctx.px("total_share")        # 元
    net = f.g("vendor_amt") * WAN2YUAN                  # 万元 -> 元
    return ctx.safe_div(net, mv, min_abs_den=1.0)


@register(FactorSpec(
    name="mf_net_inflow_5d", group=GROUP,
    deps=(FLOW, "stock_daily"),
    desc="5 日累计主力净流入率",
    formula='daily_ratio = ff["net_mf_amount"] / _total_amount(ff); '
            'cum_ratio = daily_ratio.rolling(5, min_periods=3).sum()',
    start=FLOW_START, warmup_days=W5, higher_is_better=True,
    note="日净流入率的 5 日**求和**（不是均值）。窗口内停牌 = NaN；"
         "`min_count=3` 即参考库的 `min_periods=3`（显式放松，否则 2015 年大面积停牌会"
         "把非空率压到 50% 以下）。",
))
def mf_net_inflow_5d(ctx):
    f = _Flow(ctx)
    return _roll_sum(ctx, f.rate("vendor_amt"), 5, _MIN[5])














@register(FactorSpec(
    name="mf_vol_amount_divergence", group=GROUP,
    deps=(FLOW, "stock_daily"),
    desc="资金流的量-额背离 = 净流入量占比 − 净流入额占比",
    formula='net_vol_ratio = safe_divide(ff["net_mf_vol"], _total_vol(ff)); '
            'net_amt_ratio = safe_divide(ff["net_mf_amount"], _total_amount(ff)); '
            'divergence = (net_vol_ratio - net_amt_ratio).clip(-0.5, 0.5)',
    start=FLOW_START, warmup_days=W1, higher_is_better=True,
    note="正 = 净流入的「手数占比」大于「金额占比」→ 净买入发生在**低价位**；"
         "负 = 金额占比更大 → 净买入发生在**高价位**（拉升式买入）。"
         "两个占比都是**同表内的无量纲分数**：分子分母同单位（手/手、万元/万元），"
         "所以**不需要任何换算**（跨表才需要）。"
         "★ 不做参考库的 `.clip(-0.5, 0.5)`（契约禁止因子内 winsor，引擎统一缩尾）。",
))
def mf_vol_amount_divergence(ctx):
    f = _Flow(ctx)
    return f.rate("vendor_vol") - f.rate("vendor_amt")


@register(FactorSpec(
    name="mf_open_close_divergence_10d", group=GROUP,
    deps=(FLOW, "stock_daily"),
    desc="资金态度摇摆度 = 净流入率相对其 5 日均线偏离的 10 日波动率",
    formula='net_rate = safe_divide(ff["net_mf_amount"], _total_amount(ff)); '
            'trend_5 = rolling_group_mean(net_rate, 5); '
            'divergence = safe_divide(net_rate - trend_5, trend_5.abs() + 1e-8); '
            'div_std = rolling_group_std(divergence, 10); '
            'return cross_sectional_rank(div_std)',
    start=FLOW_START, warmup_days=W15, higher_is_better=False,
    note="★ 名称与内容是错位的：factors.md 自己注明「2026-08-05 描述与实现统一"
         "（原描述声称开盘/收盘背离，实现为日频资金流代理）」，"
         "本实现照抄那个**已统一**的日频代理，不要按名字去找开盘/收盘数据。"
         "★ 两处偏离："
         "(1) 分母的 `+1e-8` 换成 `ctx.safe_div(..., min_abs_den=1e-3)`："
         "trend_5 是「净额/成交额」的量级（约 1e-2），1e-8 的地板挡不住近零值，"
         "会让偏离度炸到 1e6；取 1e-3（≈成交额的 0.1%）是真正的「几乎无净流入」边界。"
         "(2) 方向标注 `higher_is_better=False`：参考库代码 `rank(div_std)` 把摇摆剧烈者排前，"
         "但其 `意义` 明说「偏离频繁放大 = 主力态度反复、方向不确定」——"
         "按语义应为低优。数值本身没动，方向标注只影响文档。",
))
def mf_open_close_divergence_10d(ctx):
    f = _Flow(ctx)
    r = f.rate("vendor_amt")
    ma5 = ctx.roll_mean(r, 5, min_count=_MIN[5])
    dev = ctx.safe_div(r - ma5, np.abs(ma5), min_abs_den=1e-3)
    return ctx.roll_std(dev, 10, min_count=_MIN[10])


# ══════════════════════════════════════════════════════════════════════════
# ② 分层净额口径（buy_<档> − sell_<档>）
# ══════════════════════════════════════════════════════════════════════════



@register(FactorSpec(
    name="mf_big_order_ratio", group=GROUP,
    deps=(FLOW, "stock_daily"),
    desc="大单+超大单净买入率 = (lg+elg 净额) / 当日八列毛额",
    formula='big_net = (ff["buy_lg_amount"] - ff["sell_lg_amount"]'
            ' + ff["buy_elg_amount"] - ff["sell_elg_amount"]); '
            'ratio = big_net / _total_amount(ff)',
    start=FLOW_START, warmup_days=W1, higher_is_better=True,
    note="分层净额口径。参考库的 `mf_big_order_ratio` 与 `ext_mf_big_order_net_amount_ratio` "
         "是同一个式子（后者只是命名前缀不同），故这一条同时覆盖两者。"
         "★ 这也是 `smart_money_concentration` 的**精确重复**："
         "由 " + _DE + " 可推出四档净额之和 ≡ 0，于是"
         "（大单净额 − 噪音净额）= 2×大单净额，参考库那个因子与本因子截面排名相关 "
         "**= 1.000**，已砍掉（见文件末尾 `_DROPPED`）。",
))
def mf_big_order_ratio(ctx):
    f = _Flow(ctx)
    return f.tier_rate("lg") + f.tier_rate("elg")






@register(FactorSpec(
    name="mf_large_order_net_5d", group=GROUP,
    deps=(FLOW, "stock_daily"),
    desc="大单净买入率 5 日均值（只算 lg 档，不含超大单）",
    formula='large_net = ff["buy_lg_amount"] - ff["sell_lg_amount"]; '
            'ratio = safe_divide(large_net, _total_amount(ff)); '
            'ratio_ma5 = ratio.rolling(5, min_periods=3).mean(); ratio_ma5.clip(-0.5, 0.5)',
    start=FLOW_START, warmup_days=W5, higher_is_better=True,
    note="与 `mf_big_order_ratio`（lg+elg、单日）的区别是**两维都不同**："
         "只取大单档 + 5 日均值。大单（20~100 万/笔）比超大单更连续、更少脉冲，"
         "5 日均值进一步滤掉单日噪声。"
         "★ 不做参考库的 `.clip(-0.5, 0.5)`（契约禁止因子内 winsor）。"
         "`min_count=3` = 参考库 `min_periods=3`。",
))
def mf_large_order_net_5d(ctx):
    f = _Flow(ctx)
    return ctx.roll_mean(f.tier_rate("lg"), 5, min_count=_MIN[5])






@register(FactorSpec(
    name="big_vs_small_divergence_5d", group=GROUP,
    deps=(FLOW, "stock_daily"),
    desc="大小单背离的 5 日变化 = Δ5(大单净买率 − 小单净买率)",
    formula='big_net = (ff["buy_lg_amount"] + ff["buy_elg_amount"]'
            ' - ff["sell_lg_amount"] - ff["sell_elg_amount"]) / _total_amount(ff); '
            'small_net = (ff["buy_sm_amount"] - ff["sell_sm_amount"]) / _total_amount(ff); '
            'divergence = big_net - small_net; div_5d = divergence.diff(5)',
    start=FLOW_START, warmup_days=W10, higher_is_better=True,
    note="分层净额口径，两项都用**同一个**分母（八列毛额之和），"
         "所以它衡量的是「大单相对小单的净流入加速」而不是绝对水平。"
         "注意与 `mf_smart_dumb_divergence` 的区别：后者按**各自档位毛额**归一（当日水平），"
         "本条按**同一个总毛额**归一（5 日变化）——实测两者相关约 0.5，不冗余。",
))
def big_vs_small_divergence_5d(ctx):
    f = _Flow(ctx)
    tot = f.g("tot_amt")
    big = ctx.safe_div(f.g("net_amt", "lg") + f.g("net_amt", "elg"), tot, min_abs_den=1e-6)
    sm = ctx.safe_div(f.g("net_amt", "sm"), tot, min_abs_den=1e-6)
    return ctx.diff(big - sm, 5)


@register(FactorSpec(
    name="elg_net_60d_to_mv", group=GROUP,
    deps=(FLOW, "stock_daily", "stock_finance"),
    desc="超大单 60 日累计净买入额 / 总市值（长线吸筹强度）",
    formula='net = (mf["buy_elg_amount"] - mf["sell_elg_amount"]) * 1e4\n'
            'net60 = net.rolling(60, min_periods=20).sum()\n'
            'circ_mv = fin["circ_mv"].reindex(net60.index)\n'
            'raw = safe_divide(net60, circ_mv)',
    start=FLOW_START, warmup_days=W60, higher_is_better=True,
    note="★ 单位：分子 `* 1e4`（万元→元），参考库写得很清楚，本实现照抄。"
         "★ 市值口径偏离：参考库用 `fin['circ_mv']`（流通市值），本实现用 **总市值** = "
         "`ctx.px('close') × ctx.px('total_share')`（项目约定，与本族 "
         "`mf_net_amount_intensity` 一致）。close 是**未复权**价（PIT 安全）。"
         "`min_count=20` = 参考库 `min_periods=20`。"
         "无北向持股数据时，超大单是外资/产业资本的最佳代理。",
))
def elg_net_60d_to_mv(ctx):
    f = _Flow(ctx)
    net60 = _roll_sum(ctx, f.g("net_amt", "elg") * WAN2YUAN, 60, _MIN[60])  # 元
    mv = ctx.px("close") * ctx.px("total_share")                                     # 元
    return ctx.safe_div(net60, mv, min_abs_den=1.0)


@register(FactorSpec(
    name="mf_large_order_avg_price", group=GROUP,
    deps=(FLOW, "stock_daily"),
    desc="大单均价相对全日成交均价的偏离（正 = 机构在高价位成交）",
    formula='large_avg = safe_divide(large_amt, large_vol)   # lg+elg 的买+卖毛额/毛量\n'
            'total_avg = safe_divide(total_amt, total_vol)\n'
            'ratio = safe_divide(large_avg, total_avg).clip(0.5, 2.0)',
    start=FLOW_START, warmup_days=W1, higher_is_better=True,
    note="**纯同表因子，不需要任何单位换算**：分子 `large_amt/large_vol` 与分母 "
         "`total_amt/total_vol` 都等于「万元/手」= 100 元/股，量纲在相除时自动约掉。"
         "★ 返回 `ratio − 1`（真正的「偏离」，与因子释义一致）；"
         "参考库返回 ratio 本身，两者截面排名完全相同（全体平移一个常数）。"
         "★ 不做参考库的 `.clip(0.5, 2.0)`（契约禁止因子内 winsor）。"
         "注意：参考库描述里写「相对**收盘价**」，实现是「相对**全日 VWAP**」——"
         "本实现跟实现（VWAP），不跟描述。",
))
def mf_large_order_avg_price(ctx):
    f = _Flow(ctx)
    large_avg = ctx.safe_div(f.main_gross("amt"), f.main_gross("vol"), min_abs_den=1e-6)
    total_avg = ctx.safe_div(f.g("tot_amt"), f.g("tot_vol"), min_abs_den=1e-6)
    return ctx.safe_div(large_avg, total_avg, min_abs_den=1e-3) - 1.0


# ══════════════════════════════════════════════════════════════════════════
# ③ 毛额结构口径（buy_<档> + sell_<档>）
# ══════════════════════════════════════════════════════════════════════════
# 这一组只用「买卖毛额」，天然不受双记口径影响（Σ4 毛额 / Σ4 毛额 = 1）。



@register(FactorSpec(
    name="mf_order_size_entropy", group=GROUP,
    deps=(FLOW, "stock_daily"),
    desc="四档成交量分布的信息熵（归一化到 [0,1]，大 = 参与结构多样）",
    formula='sm_share = (buy_sm_vol + sell_sm_vol) / _total_vol(ff);  # md/lg/elg 同理\n'
            'entropy = -(Σ p·ln(p + 1e-10)) / ln(4)      # 本框架加了 /ln4 归一化',
    start=FLOW_START, warmup_days=W1, higher_is_better=True,
    note="★ 归一化：参考库返回裸熵（取值 [0, ln4]），本项目要求 `[0,1]`"
         "（契约第 4 条：四档占比的 Shannon 熵，归一化到 [0,1]），故除以 ln(4)。"
         "截面排名与裸熵等价。"
         "用**成交量**口径（参考库原文用 `*_vol` / `_total_vol`）；"
         "分母 = 四档毛量之和，比值严格是四档占比（Σp ≡ 1），量纲自约。"
         "`p + 1e-10` 保留参考库写法：p=0 时该项贡献恰好为 0。",
))
def mf_order_size_entropy(ctx):
    f = _Flow(ctx)
    tot = f.g("tot_vol")
    ent = np.zeros(ctx.panel.shape, dtype=np.float64)
    for t in TIERS:
        p = ctx.safe_div(f.g("gross_vol", t), tot, min_abs_den=1e-6)
        ent = ent - p * np.log(p.astype(np.float64) + 1e-10)
    return ent / np.log(4.0)


@register(FactorSpec(
    name="mf_retail_dominance", group=GROUP,
    deps=(FLOW, "stock_daily"),
    desc="散户成交占比 = 小单买+卖金额 / 全部成交额（高者劣）",
    formula='retail = ff["buy_sm_amount"] + ff["sell_sm_amount"]; '
            'ratio = safe_divide(retail, _total_amount(ff)); '
            'return cross_sectional_rank(-ratio)',
    start=FLOW_START, warmup_days=W1, higher_is_better=False,
    note="毛额口径（分子分母同为万元），不受双记影响。"
         "返回**未取负**的散户占比原值，参考库 `rank(-ratio)` 由 "
         "`higher_is_better=False` 表达。"
         "★ 与 `mf_small_order_ratio` 的区别是**净额 vs 毛额**："
         "本因子看「散户参与度」（谁在交易），后者看「散户方向」（散户在买还是卖）；"
         "实测两者截面排名相关仅 −0.27 左右，是两条不同的信息。"
         "对齐参考库的 `ext_mf_small_order_amount_ratio` / `mf_retail_dominance`（同式异名）。",
))
def mf_retail_dominance(ctx):
    f = _Flow(ctx)
    return ctx.safe_div(f.g("gross_amt", "sm"), f.g("tot_amt"), min_abs_den=1e-6)


# ══════════════════════════════════════════════════════════════════════════
# ★ 被砍掉的 3 个候选（参考库列了 27 个，本项目要求 24 个）
#
# 三个都是**实测截面排名相关 = +1.000** 的精确重复（2015 年 566,453 行 × 全截面逐日
# rank 后取平均相关）。留着它们不会增加任何信息，只会白占一个因子槽、
# 让下游的因子筛选看到两条一模一样的行。三条重复关系都有**解析证明**：
#
#   order_concentration            ≡ mf_order_concentration
#       因 Σ4 毛额 ≡ 八列总额（实测 max|差| = 1.9e-9），
#       故 -(smT+mdT)/total = bigT/total − 1（实测 max|差| = 5.6e-16）→ 平移，rank 相同。
#
#   smart_money_concentration      ≡ 2 × mf_big_order_ratio
#       因四档净额之和 ≡ 0（双记，实测 2015 全表 max|Σ| = 0.03），
#       故 (smart − noise) = bigN − (smN+mdN) = bigN − (−bigN) = 2·bigN。
#       实测 corr = 0.99999999。
#
#   mf_net_vol_ratio_5d            ≡ mf_net_inflow_5d
#       净额的「量口径占比」netV/totV 与「额口径占比」netA/totA 数值上几乎恒等
#       （两者都是无量纲分数，而 netA/netV 与 totA/totV 是同一个「100 元/手」的换算比），
#       5 日 aggregate 取 sum 或 mean 只差常数 5 → 实测 corr = +1.000。
#       保留 `mf_net_inflow_5d`（金额口径、与 mf_net_inflow_ratio /
#       mf_cumulative_flow_20d 构成同一条口径线），砍掉量口径那一条。
#
# 另外两个「看起来重复但实测不重复」的，说明一下为什么不砍：
#   · mf_flow_acceleration_5d vs mf_net_inflow_trend_5d
#       （净流入率的 diff(5) vs 5 日回归斜率）实测相关约 0.63 —— 前者只看首尾两点、
#       后者看整条窗内的形状，是不同的信号，两条都留。
#   · mf_net_inflow_5d vs mf_cumulative_flow_20d 实测相关约 0.85，周期不同，都留。
# ══════════════════════════════════════════════════════════════════════════
_DROPPED = (
    "order_concentration",          # ≡ mf_order_concentration（rank 相关 1.000）
    "smart_money_concentration",    # ≡ 2 × mf_big_order_ratio（rank 相关 1.000）
    "mf_net_vol_ratio_5d",          # ≡ mf_net_inflow_5d（rank 相关 1.000）
)
