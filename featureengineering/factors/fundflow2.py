"""资金流「订单分层」×「两融结构」因子族（16 个）—— `fundflow.py` / `margin.py` 的**互补**家族。

数据源与两个已存在的家族完全相同：
    · `stock_main_fund_flow`（订单分层，2010-01-04 起）
    · `stock_margin_detail`（两融，2011-01-04 起，**晚 1 个交易日**）
    · `stock_cyq_chips`（筹码峰，经 `fea/chips.py` 预聚合，**2018-01-02 起、只覆盖主板**）

本文件是**增量**家族：13 个资金流因子（group=`fundflow`）+ 3 个两融因子（group=`margin`）。
`fundflow.py` / `margin.py` 已实现若干（主 Agent 在持续收口精简，数量会变，故不写死），
本族的每一条都在实现前**用上游原始表逐个量过与既有因子的截面 rank 相关**
（见下面的「口径级偏离」表）。

═══════════════════════════════════════════════════════════════════════════
★★ 先说结论：参考库里 4 个因子的**字面公式**与既有因子**重复或近重复**，已改口径
═══════════════════════════════════════════════════════════════════════════
量化方法：把参考公式在上游 `stock_main_fund_flow`（2026 年全量、891,494 行）上逐字复刻，
并与**本项目已落盘的因子**（`data/factors/<name>/year=2026`）逐日做截面 rank 相关。
判定线沿用 `fundflow.py` 的既有标准：**相关 = 1.0000 的精确重复必须砍掉或改口径**。

| 参考因子 | 字面公式 | 实测（逐日截面 rank 相关） | 本文件处置 |
|:--|:--|:--|:--|
| `small_order_crowding` | 小单毛量 / 总毛量 | 与 `mf_retail_dominance` **0.99999** | **改**：相对自身 20 日中枢的抬升幅度 |
| `super_large_order_intensity` | 超大单净额 / 八列毛额 | 与 `mf_elg_order_ratio` **1.00000** | **改**：5 日累计口径（超窗平滑） |
| `mf_net_persistent_5d` | 5 日内 `net_mf_amount>0` 的天数 | 与 `mf_flow_streak_5d` **1.00000** | **改**：改用**分层大单**净额数天数 |
| `margin_buy_momentum_5d` | 5 日均买入额 / 融资余额 | 与 `margin_velocity` **+0.922**（沙箱 2026、单日最高 0.952） | **改**：改成买入速度的 **5 日加速度**（见 note，两轮实测驱动） |

★ **读 note 前请先看这一条**：note 里作为对照对象的同族因子名，取的是**我做判定当时**
`factors/fundflow.py` / `margin.py` 的在册内容。这两个家族随后由主 Agent 按
「精确重复即砍」的原则做了收口精简（`fundflow.py` 里已有多个被裁），
所以 note 里提到的某些名字**现在可能查不到**。
这不影响任何结论：**上面所有相关数字都是从上游原始表（`stock_main_fund_flow`）
复算出来的**，不依赖对照因子是否在册；而且冗余只会因此变少，不会变多。
只有一处是**运行期依赖**，已随精简改过一次：耦合因子
`fundflow_retail_inst_divergence` 的父因子从（已删的）`mf_small_order_ratio`
换成在册的 `mf_retail_dominance`，见其 note。

三条改口径的共同根因，都是 `fundflow.py` docstring ① 记的**双记口径**
（每笔成交同时进买方桶与卖方桶，Σ4 买 ≡ Σ4 卖）：
    · 小单**量**占比与 `mf_retail_dominance` 的**额**占比是同一个数 ——
      因为 `netA/netV` 与 `totA/totV` 是同一个「100 元/手」的换算比（同 docstring ②）；
    · `super_large_order_intensity` 的分母「八列毛额之和」就是 `mf_elg_order_ratio` 的分母；
    · `mf_net_persistent_5d` 的「净流入天数」就是 `mf_flow_streak_5d` 的同一列同一窗。
改口径时**保留了原因子的经济含义**（散户拥挤 / 超大单强度 / 净流入持续性），
只换统计量，并把实测相关压到 0.65 以下（见各因子 note）。

**保留的一处高相关（有意，已量化）**：`mf_big_small_divergence` 的字面公式与
`mf_smart_dumb_divergence`（现已被删）相关 **0.963**、与**仍在**的
`mf_big_order_ratio` **0.938**。
这是**结构性的、无法回避**：「大单净额 − 小单净额」在双记口径下恒等于
`2×大单净额 + 中单净额`（四档净额之和 ≡ 0），任何窗口平滑都消不掉这个线性恒等关系。
它的 note 里写明了这一点与实测数字，**下游若已在用 `mf_big_order_ratio`，本因子可作为冗余项剔除**。

**第二处高相关（保留，已量化 + 已试过解耦、解耦更差）**：
`margin_balance_ma_divergence`（`(rzye − MA20(rzye))/MA20(rzye)`）与
`margin_leverage_trend_10d` **+0.883**（沙箱 2026、单日最高 0.930）。
这也是**结构性**的：参考库那条公式本身就是「余额相对均线的偏离」，
与既有「余额趋势」因子在同一维度上；试过的解耦方案（除以自身 60 日波动做「极端度」归一化）
实测**更差**（把 `margin_flow_asymmetry_10d`/`margin_balance_20d` 的相关从 0.82/0.81 推上去），
故**保留参考库字面式**，并把数字写进 note 供下游去重时取舍。

★ 第四处偏离（`margin_buy_momentum_5d`）是**两轮实测驱动**的：第一轮 0.922（vs `margin_velocity`）
→ 改成自归一化水平量后 0.816（最近邻变成 `vol_ratio_ma5_ma20`）→ 最终改成**加速度**，
max |corr| 降到 **0.509**。全过程数字见该因子 note 与交付报告。

═══════════════════════════════════════════════════════════════════════════
★★ 两融的 PIT 铁律（本文件 3 个 margin 因子全部遵守）
═══════════════════════════════════════════════════════════════════════════
`stock_margin_detail` 的 `trade_date = d` 记录，要到 **d 的下一个交易日**才拿得到
（2026-09-15 实测确认）。处置缺一不可：
    ① `FactorSpec(..., deps=("stock_margin_detail",), lagged_ok=("stock_margin_detail",))`
       —— 不写，`register()` 在注册期直接抛错；
    ② 在**原始网格上把因子算完**，**最后一步** `return ctx.lag_grid(grid, LAGGED_DATASETS.get(MARGIN, 1))`。
不做的话：T 日的因子值今天算对、明天数据到达后**静默变成另一个值**，没有任何报错。
`margin_chip_cost_gap` 同时用 margin 与筹码：**先各自算完、再相减、最后统一下移一格**。

★ 实证（三层，全部真跑）：① 独立 pandas 复现 `value(T) == 上游 raw(T−1)`，逐格比对
2018–2025 共 140 万格，中位相对误差 7.6e-08 / 4.5e-07 / 8.2e-07（三个因子），
而 `raw(T)` 口径的中位误差是 3.0e-01 / 2.9e-01 / 7.8e-02 —— 位移方向与幅度都对；
② `main.py audit-pit`（截断到 T 重算）在 2026-05-18 与 2026-09-14 两个样本日：
三个因子与全量产物**逐格相同，0 个不一致 → 全部通过 ✔**；
③ **增量 = 全量**：`--start 2026-09-01` 的尾部跑与全年跑逐格比对，
19,540 格完全一致（这一条曾不成立 —— 状态量的 asof 语义会让「已退出两融」的股票
在两趟之间从 0 变 NaN，修复见 `_mg_state` 的新鲜度掩码与
`margin_balance_ma_divergence` 的 note）。

★ 给主 Agent 的一条运维提醒：引擎的逻辑指纹 `recipe` = `name@版本|formula|warmup|起点`
（`fea/spec.py::FactorSpec.recipe`）—— **不含函数体与 helper 源码**。所以
① 改了 helper（本文件 2026-09-15 17:2x 的新鲜度掩码就是）**不会**自动失效，
   必须 `--rebuild`；本次已用 `--rebuild` 重算，并把口径写进了相关因子的 `formula`
   字段（指纹变化 → 下次跑自动全量重建）；
② 反之，只改 note/desc 不会触发重建（本次 note 的批量更新就属于这种，产物无需重算）。

═══════════════════════════════════════════════════════════════════════════
★ 数据接入：「出现指示」+ 停牌掩码（照抄 `fundflow.py` 的 `_Flow`，不跨家族 import 私有类）
═══════════════════════════════════════════════════════════════════════════
资金流与两融都是**事件/流量**表：「没有行」≠「值为 0」。本文件的 `_F2.g()` / `_mg_flow()`
一律同时铺一张「出现指示」网格，用 `safe_div` 把覆盖范围外的格子还原成 **NaN**；
再用 `ctx.traded()` 把**停牌日**整体掩成 NaN。滚动一律显式给 `min_count`
（= 参考库的 `min_periods`，窗口 5→3、10→5、20→10），否则 2015 年那种大面积停牌
会把非空率压到 50% 以下。
两融**状态量**（余额）另有一层「新鲜度掩码」：最近 20 个交易日内没有该股的两融记录
（= 已退出两融名单 / 退市）→ NaN，而不是被 asof 无限前向填充成「余额没变化」的 0。
这一层同时保证**增量跑与全量跑逐格一致**（详见 `_mg_state` docstring）。

★ 为什么另起一份接入代码而不 `from .fundflow import _Flow`：跨家族 import 私有类会让
  「改 A 家族顺手弄坏 B 家族」，且 `factors/__init__.py` 的 import 顺序变成隐式依赖
  （契约 §0：一个 Agent 只拥有一个文件）。两份代码的**口径逐条对齐**，差异只在「拿了哪些列」。

═══════════════════════════════════════════════════════════════════════════
★ 两融因子的覆盖率天生约 50%（只有两融标的才有数据），这是**正常**的
═══════════════════════════════════════════════════════════════════════════
非两融标的在上游**没有行**，本文件按「覆盖范围外 = NaN」处理（不是 0），
所以 `margin_*` 三个因子的截面只覆盖两融名单（2026 年主板约 1900 只），
`margin_chip_cost_gap` 还要再与筹码层（只覆盖主板）求交，覆盖率更低 —— 写在各 note 里。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fea.spec import FactorSpec, register, LAGGED_DATASETS

# ---------------------------------------------------------------- 常量
FLOW = "stock_main_fund_flow"
MARGIN = "stock_margin_detail"
CHIPS = "stock_cyq_chips"
DAILY = "stock_daily"
GROUP = "fundflow"

# 上游 2010-01-04 起，早于 conf 的 default_start，跟随配置即可（start=None）
FLOW_START = None
# 筹码层硬起点（stock_cyq_chips 实测 2018-01-02 起，且只覆盖主板）
CHIP_START = "2018-01-02"

TIERS = ("sm", "md", "lg", "elg")
MAIN_TIERS = ("lg", "elg")          # 「主力」= 大单 + 超大单（本项目口径）
_BUY_A = tuple(f"buy_{t}_amount" for t in TIERS)
_SELL_A = tuple(f"sell_{t}_amount" for t in TIERS)
_BUY_V = tuple(f"buy_{t}_vol" for t in TIERS)
_SELL_V = tuple(f"sell_{t}_vol" for t in TIERS)
_ALL_COLS = ("stock_code", "trade_date", *_BUY_A, *_SELL_A, *_BUY_V, *_SELL_V,
             "net_mf_amount")

MARGIN_COLS = ("rzye", "rzmre", "rzche", "rzrqye")

# 滚动窗口 min_count = 参考库的 min_periods（停牌造成窗口缺口，必须显式放松）
_MIN = {5: 3, 10: 5, 20: 10}

# 与 `fundflow.py` 同档的 warmup（N×1.8 + 20）：单日因子沿用契约的「日频聚合 → 60」
W1, W5, W10, W20 = 60, 29, 38, 56
# 两融因子额外 +1（位移损失一天）
M1, M5, M20 = 60, 30, 60

# 分母地板（元）：两融余额/买入额的量纲是元，1 元足以挡掉「余额被清零」的格子
DEN = 1.0

_DE = ("双记口径：本表每笔成交同时进买方桶与卖方桶，Σ4buy ≡ Σ4sell（见 fundflow.py docstring ①）。")


# ══════════════════════════════════════════════════════════════════════════
# 数据接入：stock_main_fund_flow
# ══════════════════════════════════════════════════════════════════════════
class _F2:
    """`stock_main_fund_flow` → (T, C) 网格。逐行先算派生量，再每**列**一次 `event_grid`。

    与 `fundflow.py::_Flow` 同一套口径（出现指示 + 停牌掩码），只提供本族需要的派生量。
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self._rows: dict[str, np.ndarray] = {}
        self._grids: dict[str, np.ndarray] = {}
        self.tr = ctx.traded()
        d0, d1 = int(ctx.panel.dates[0]), int(ctx.panel.dates[-1])
        df = ctx.dataset(FLOW, columns=list(_ALL_COLS),
                         years=(d0 // 10000, d1 // 10000))
        self.df = df
        self.empty = df is None or df.empty
        self.codes = self.days = None
        if not self.empty:
            # ★ 必须存**原始股票代码**：`ctx.event_grid` 内部会再做一次 `code_index`，
            #   传列号进去会全部 miss → 整张网格静默全 0（fundflow.py docstring 坑 1）。
            self.codes = df["stock_code"].to_numpy()
            self.days = ctx.date_col(df["trade_date"])
            n_map = int((ctx.code_index(self.codes) >= 0).sum())
            if n_map == 0:
                raise RuntimeError(
                    f"{FLOW} 的 {len(self.codes)} 行股票代码没有一个能对上当前面板，"
                    f"说明代码格式不匹配（应形如 000001.SZ）—— 这是代码 bug，不是数据缺失。")

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
        elif kind == "tot_amt":
            v = df[list(_BUY_A + _SELL_A)].to_numpy(np.float64).sum(axis=1)
        elif kind == "tot_vol":
            v = df[list(_BUY_V + _SELL_V)].to_numpy(np.float64).sum(axis=1)
        else:
            raise KeyError(f"_F2 不认识的派生量 {kind!r}")
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
            got = ctx.event_grid(self.codes, self.days,
                                 np.ones(x.size, dtype=np.float64))
            # 同格多条时 safe_div 自动取均值，天然幂等；got<0.5 -> NaN
            v = ctx.safe_div(ctx.event_grid(self.codes, self.days, x), got, min_abs_den=0.5)
            v = np.where(self.tr, v, np.nan)          # ★ 停牌日 -> NaN
        self._grids[key] = v
        return v

    def main_net(self, base: str = "amt") -> np.ndarray:
        """大单+超大单的净额（万元 或 手）。"""
        return self.g("net_" + base, "lg") + self.g("net_" + base, "elg")

    def main_gross(self, base: str = "amt") -> np.ndarray:
        return self.g("gross_" + base, "lg") + self.g("gross_" + base, "elg")

    def big_share(self) -> np.ndarray:
        """(大单+超大单) 毛额 / 八列毛额 —— 毛额口径，不受双记影响。"""
        return self.ctx.safe_div(self.main_gross("amt"), self.g("tot_amt"), min_abs_den=1e-6)












# ══════════════════════════════════════════════════════════════════════════
# 数据接入：stock_margin_detail（★ 滞后表，见模块 docstring 的 PIT 铁律）
# ══════════════════════════════════════════════════════════════════════════


def _mg_state(ctx, df, day, col: str, fresh_n: int = 20) -> np.ndarray:
    """**状态量**（余额）→ 日频 as-of 前向填充；**超过 `fresh_n` 个交易日没有新记录 → NaN**。

    ★ 为什么需要「新鲜度」掩码（实测驱动，两条理由都成立）：
      ① **语义**：余额是**状态量**，「没有行」不等于「值为 0」。股票**退出两融名单**后，
         as-of 会把最后一条记录无限前向填充 → 偏离度 `(rzye − MA20)/MA20` 恒等于 **0**
         （余额在窗口内恒定），于是「早已退出两融」被写成「余额没变化」的 0 值，
         在截面上堆在零点附近的并列里，还给 `main.py audit-pit` 制造假告警
         （真值恰为 0，两次复算分别给 0.0 与 ±7e-16，被该工具的 1e-12 分母下限放大成 7e-04）。
      ② **一致性（更要紧）**：旧实现按**面板覆盖的年份**读表，而面板起点随运行方式而变 ——
         全量跑的 warmup 区伸到上一年（读得到上一年的最后一条记录 → 值 0），
         增量 / `--start` 跑的窗口在当年（读不到 → NaN），**同一格在两趟之间从 0 变 NaN**。
         实测：7 只 2025 年 1~4 月退出两融的股票在 2026-09-04~09-14 每天 7~20 格不一致。
         掩码按**面板内交易日距离**判定（不是「读到没读到」），两趟结果一致。
      `fresh_n = 20` = 本族最长窗口（MA20），即「最近 20 个交易日里还挂在该股的两融名单上」。
      只影响**状态量**；`_mg_flow` 走「出现指示」精确落格，本来就只在覆盖日内有值。
    """
    code = df["stock_code"].to_numpy()
    grid = np.asarray(ctx.asof_daily(code, day, df[col].to_numpy(dtype=np.float64)),
                      dtype=np.float64)
    last = np.asarray(ctx.asof_daily(code, day, np.asarray(day, dtype=np.float64)),
                      dtype=np.float64)                       # 该格「最后一条记录」的日期
    dates = np.asarray(ctx.panel.dates, dtype=np.int64)
    row = np.arange(dates.size, dtype=np.int64)[:, None]      # 面板行的位置
    pos = np.searchsorted(dates, last)                        # 最后记录日的位置（NaN → 末尾）
    return np.where((row - pos) <= fresh_n, grid, np.nan)


def _mg_flow(ctx, df, day, col: str) -> np.ndarray:
    """**流量**（当日买入额/偿还额）→ 精确落格，覆盖范围外 **NaN**。

    同时铺一张「出现指示」网格（权重全 1），用 `safe_div` 还原 NaN ——
    与 `margin.py::_flow` 同口径。上游真值里的 0（停牌日 `rzmre=0`）**保持 0**。
    """
    v = df[col].to_numpy(dtype=np.float64)
    got = ctx.event_grid(df["stock_code"].to_numpy(), day,
                         np.ones(v.size, dtype=np.float64))
    raw = ctx.event_grid(df["stock_code"].to_numpy(), day, v)
    return ctx.safe_div(raw, got, min_abs_den=0.5)




# ══════════════════════════════════════════════════════════════════════════
# 一、订单分层族（4 个）
# ══════════════════════════════════════════════════════════════════════════









# ══════════════════════════════════════════════════════════════════════════
# 二、结构 / 持续族（4 个）
# ══════════════════════════════════════════════════════════════════════════









# ══════════════════════════════════════════════════════════════════════════
# 三、量价交叉族（3 个）
# ══════════════════════════════════════════════════════════════════════════





@register(FactorSpec(
    name="mf_amount_weighted_direction", version=2, group=GROUP,
    deps=(FLOW, DAILY),
    desc="金额加权方向 = 四档方向信号 sign(该档净额) 按该档毛额占比加权求和（多空结构综合）",
    formula='sm_dir = np.sign(ff["buy_sm_amount"] - ff["sell_sm_amount"])   # md/lg/elg 同理\n'
            'sm_w = (ff["buy_sm_amount"] + ff["sell_sm_amount"]) / total_amt   # md/lg/elg 同理\n'
            'composite = sm_dir*sm_w + md_dir*md_w + lg_dir*lg_w + elg_dir*elg_w\n'
            'return cross_sectional_rank(composite)',
    start=FLOW_START, warmup_days=W1, higher_is_better=True,
    note="逐字复刻参考库（`fund_flow.py`）。"
         "★ 它**不是**另一个「净流入率」：本因子刻意把每档的**幅度**丢掉、只留**方向**，"
         "再按「这一档在当天的成交里占多大比重」加权 —— 于是「小单巨量卖出、大单微量买入」"
         "与「小单微量卖出、大单巨量买入」得到相反结论，而按净额加总时两者可能都接近 0。"
         "★ 沙箱 2026 实测（vs 全部 259 个既有生产因子的逐日截面 rank 相关）："
         "max|corr| = **−0.441**（`mf_big_order_ratio`），"
         "其次 `big_vs_small_divergence_5d` −0.299、`consecutive_limit_up` −0.121 —— "
         "与全库**最高的相关也才 0.44**，是独立性最好的一批（方向信号与幅度信号不同向，正常）。"
         "取值域：Σw ≡ 1（四档毛额之和 = 八列总额，实测 max|差| < 2e-9），"
         "且四档 net 之和 ≡ 0 使方向不可能全同号，故 composite ∈ [−1, 1]。"
         "实测 2026 年 p1/p50/p99 = −0.50 / +0.06 / +0.47（近似以 0 为心）；"
         "沙箱 |value|max = 0.997。"
         "分子分母同为万元（毛额口径），无量纲。",
))
def mf_amount_weighted_direction(ctx):
    f = _F2(ctx)
    tot = f.g("tot_amt")
    comp = np.zeros(ctx.panel.shape, dtype=np.float64)
    for t in TIERS:
        w = ctx.safe_div(f.g("gross_amt", t), tot, min_abs_den=1e-6)
        comp = comp + np.sign(f.g("net_amt", t)) * w
    return comp


# ══════════════════════════════════════════════════════════════════════════
# 四、两融 × 筹码（1 个）—— ★ 滞后表，lag_grid 是最后一步
# ══════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════
# 五、两融结构（2 个）—— ★ 滞后表，lag_grid 是最后一步
# ══════════════════════════════════════════════════════════════════════════





# ══════════════════════════════════════════════════════════════════════════
# 六、其它（2 个）—— 耦合因子：`deps` 里写的是**别的因子名**
# ══════════════════════════════════════════════════════════════════════════



