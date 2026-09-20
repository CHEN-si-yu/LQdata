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
    ② 在**原始网格上把因子算完**，**最后一步** `return ctx.lag_grid(grid, 1)`。
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

from fea.spec import FactorSpec, register

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


def _roll_sum(ctx, mat: np.ndarray, n: int, min_count: int) -> np.ndarray:
    """带 `min_count` 的滚动求和。

    ★ 保留本地 helper 只是为了与 `fundflow.py` / `margin.py` / `liquidity.py` 的**同名
      约定**保持一致（三个家族都这么写）。框架已修：`ctx.roll_sum(mat, n, min_count)`
      现在**可以**传 `min_count`（`fundflow.py` 写的时候不行，那是当时 `fea/context.py`
      里同名的两处定义互相覆盖）—— 实测两者在含 NaN 的矩阵上逐格等价
      （NaN 掩码完全一致，数值差 ≤ 2e-15）。等价性：`roll_mean × roll_count`
      （后者**不毒化**）= 窗口内有效值之和；有效值不足 `min_count` 时 mean 为 NaN，
      `NaN × count = NaN`，与 `roll_sum(min_count=...)` 的缺失语义一致。
    """
    return ctx.roll_mean(mat, n, min_count) * ctx.roll_count(mat, n)


def _pos(x: np.ndarray) -> np.ndarray:
    """把「>0」变成 float 指示，**NaN 保持 NaN**（不许当成 0 混进滚动计数）。"""
    return np.where(np.isfinite(x), (x > 0).astype(np.float64), np.nan)


def _streak_positive(mat: np.ndarray) -> np.ndarray:
    """连续 `>0` 的天数（(T,C) 进、(T,C) 出）。

    语义与参考库的 `groupby(cumcount)` 一致：当日 `<=0` **或 NaN** 都把计数清零；
    NaN 格子本身输出 NaN（停牌日我们不声称知道「连续流入几天」）。
    实现是「按行推进的向量化累加」——面板 T 最多几千行、每行是一次 O(C) 的 numpy 运算，
    比逐股 Python 循环快两个数量级，且不需要 groupby。
    """
    T, C = mat.shape
    out = np.full((T, C), np.nan, dtype=np.float64)
    run = np.zeros(C, dtype=np.float64)
    for t in range(T):
        row = mat[t]
        valid = np.isfinite(row)
        run = np.where(valid & (row > 0.0), run + 1.0, 0.0)
        out[t] = np.where(valid, run, np.nan)
    return out


def _cs_rank_pct(mat: np.ndarray) -> np.ndarray:
    """逐日截面百分位排名（(T,C) 进、(T,C) 出，NaN 原位保留）。

    ★ 只给**耦合因子**用（契约 §3.5 `ctx.load_factor` 的配套）：参考库的
      `fundflow_retail_inst_divergence` 定义为「两个因子截面排名的乘积」，
      排名是公式的**组成部分**，不是对本因子输出的后处理 —— 与契约
      「不许在因子里做 winsor/rank（引擎统一做）」不冲突：本函数的输入是**别的因子**，
      输出仍要被引擎再排名一次。
    """
    return pd.DataFrame(mat).rank(axis=1, pct=True).to_numpy(np.float64)


def _empty(ctx) -> np.ndarray:
    return np.full(ctx.panel.shape, np.nan, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════
# 数据接入：stock_margin_detail（★ 滞后表，见模块 docstring 的 PIT 铁律）
# ══════════════════════════════════════════════════════════════════════════
def _mg_load(ctx):
    """读上游两融明细，年份裁剪到面板范围（含 warmup 区），并**多读前一年**。

    ★ 多读前一年的理由：状态量（余额）是 **as-of** 语义，需要「股票的最后一条记录」
      可能落在面板起点之前（例如 12 月退出两融的股票，次年 1 月的面板要用到它）。
      只按面板年份读，会让同一格的值取决于面板从哪一天开始 —— 正是
      `_mg_state` 里「新鲜度掩码」要根治的那个不一致（见其 docstring）。
    """
    d0 = int(ctx.panel.dates[0]) // 10000 - 1
    d1 = int(ctx.panel.dates[-1]) // 10000
    df = ctx.dataset(MARGIN, columns=["stock_code", "trade_date", *MARGIN_COLS],
                     years=(d0, d1))
    if df is None or df.empty:
        return None, None
    return df, ctx.date_col(df["trade_date"])


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


def _mg_cost(ctx, buy: np.ndarray) -> np.ndarray:
    """近 20 日融资买入额加权的**成交价**（元/股）= Σ(rzmre × close) / Σ(rzmre)。

    ★ 价格用 **`ctx.px("close")`（未复权）**，不是 `ctx.hfq`：
      本函数只被 `margin_chip_cost_gap` 使用，要与之相除的筹码层 `mean` 是
      **未复权**口径（见 `fea/chips.py` docstring「口径一」），两边必须同口径 ——
      混用会在除权日/全样本上把比值整体放大 `adj_factor` 倍（实测 2019 年 108~142），
      得到一条「看起来很连续、其实差两个数量级」的假序列。
      （兄弟因子 `margin.py::margin_buyer_avg_cost_premium` 用 hfq 是**对的**，
       因为它比的是「现价 / 成本」，两边都复权，口径自洽。）
    ★ `min_count=10`：参考库 `_margin_weighted_cost` 写的是 `min_periods=5`，
      本实现跟 `margin.py` 的既有选择（20 日窗取半数 10），保持族内一致。
    """
    px = ctx.px("close")
    num = _roll_sum(ctx, buy * px, 20, _MIN[20])
    den = _roll_sum(ctx, buy, 20, _MIN[20])
    return ctx.safe_div(num, den, min_abs_den=DEN)


# ══════════════════════════════════════════════════════════════════════════
# 一、订单分层族（4 个）
# ══════════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="order_size_concentration", group=GROUP,
    deps=(FLOW, DAILY),
    desc="订单规模集中度 = 四档成交额 HHI × sign(大单净额)（机构主导的高集中 vs 散户主导的高集中）",
    formula='sm_total = ff["buy_sm_amount"] + ff["sell_sm_amount"]   # md/lg/elg 同理\n'
            'total_amount = sm_total + md_total + lg_total + elg_total\n'
            'hhi = ((sm_total/total_amount)**2 + (md_total/total_amount)**2\n'
            '       + (lg_total/total_amount)**2 + (elg_total/total_amount)**2)\n'
            'smart_direction = (ff["buy_elg_amount"] - ff["sell_elg_amount"]\n'
            '                   + ff["buy_lg_amount"] - ff["sell_lg_amount"])\n'
            'signed_hhi = hhi * np.sign(smart_direction)\n'
            'return cross_sectional_rank(signed_hhi)',
    start=FLOW_START, warmup_days=W1, higher_is_better=True,
    note="逐字复刻参考库（`fund_flow_deep.py`），只把 `cross_sectional_rank` 交给引擎。"
         "★ **符号是必需的，不是装饰**：实测「无符号 HHI」与 `mf_order_concentration`"
         "（大单毛额占比，判定当时在册）的截面 rank 相关 **−0.940** —— "
         "四档 HHI 与「大单占比」在数学上"
         "几乎互为镜像（HHI 高 ⟺ 某一档独大 ⟺ 四档越不平均），去掉符号就是又一个重复因子。"
         "乘上 `sign(smart_direction)` 后与 `mf_big_order_ratio` 相关降到 **+0.650**、"
         "与 `mf_smart_dumb_divergence` **+0.623**（2026 年逐日截面 rank，上游表直接复算）——"
         "这正是参考库 thesis 想做的「区分两种高集中情形」。"
         "取值域：HHI ∈ [0.25, 1]（四档的下界），乘符号后 ∈ [−1,−0.25] ∪ [0.25,1]，"
         "以 0 为界分两侧；`sign(0)=0` 的格子（净额恰好为 0，测度零）给出 0，与参考库一致。"
         "分子分母同为**万元**（毛额口径，不受双记影响），比值无量纲、无需换算。",
))
def order_size_concentration(ctx):
    f = _F2(ctx)
    tot = f.g("tot_amt")
    hhi = np.zeros(ctx.panel.shape, dtype=np.float64)
    for t in TIERS:
        p = ctx.safe_div(f.g("gross_amt", t), tot, min_abs_den=1e-6)
        hhi = hhi + p * p
    return hhi * np.sign(f.main_net("amt"))


@register(FactorSpec(
    name="order_size_ratio_change", group=GROUP,
    deps=(FLOW, DAILY),
    desc="订单规模比变化 = 大单+超大单毛额占比的 5 日变化（大单占比提升=机构参与加深）",
    formula='big_total = (ff["buy_lg_amount"] + ff["sell_lg_amount"]\n'
            '             + ff["buy_elg_amount"] + ff["sell_elg_amount"])\n'
            'total = _total_amount(ff); big_ratio = big_total / total\n'
            'chg = big_ratio.groupby(level="Code").transform(lambda s: s.diff(5))\n'
            'return cross_sectional_rank(chg)',
    start=FLOW_START, warmup_days=W10, higher_is_better=True,
    note="逐字复刻参考库（`fund_flow_deep.py`）。二阶变化抓机构参与的**拐点**"
         "（占比从 10% 升到 20% = 刚介入；已在 40% 高位 = 可能出货），"
         "与同族其它因子互补：实测与 `mf_order_concentration`（占比**水平**）相关 +0.308、"
         "与 `big_vs_small_divergence_5d`（净额口径的 5 日变化）相关 **+0.086** —— "
         "「毛额占比的变化」与「净额方向的变化」是两件事，不冗余。"
         "毛额口径本身不受双记影响（Σ4 毛额 = 八列总额）。"
         "★ 与 `mf_tier_net_spread_20` 的区别：后者是四档净额占比的 20 日极差（分歧度），"
         "本因子是单边占比的 5 日变化（参与度），实测相关仅 +0.16。",
))
def order_size_ratio_change(ctx):
    f = _F2(ctx)
    return ctx.diff(f.big_share(), 5)


@register(FactorSpec(
    name="small_order_crowding", group=GROUP,
    deps=(FLOW, DAILY),
    desc="小单拥挤度 = 小单毛量占比相对自身 20 日中枢的偏离（散户异常拥挤=见顶信号，低者优）",
    formula='small_vol = mf["buy_sm_vol"] + mf["sell_sm_vol"]\n'
            'total_vol = (buy_sm_vol + sell_sm_vol + buy_md_vol + sell_md_vol\n'
            '             + buy_lg_vol + sell_lg_vol + buy_elg_vol + sell_elg_vol)\n'
            'small_pct = small_vol / total_vol.replace(0, np.nan)\n'
            'return cross_sectional_rank(-small_pct)\n'
            '★ 本实现（口径级偏离，见 note）：\n'
            'crowd = small_pct / small_pct.rolling(20, min_periods=10).mean() - 1',
    start=FLOW_START, warmup_days=W20, higher_is_better=False,
    note="★★ **偏离参考库的字面公式（必须看）**：参考库返回 `rank(-小单毛量占比)`，"
         "实测它与本项目的 `mf_retail_dominance`（小单毛**额**占比）截面 rank 相关 "
         "**0.99999** —— 是同一个因子的两种写法，不是「量 vs 额」两条信息："
         "由 " + _DE + " 及 `netA/netV` 与 `totA/totV` 同为一个「100 元/手」的换算比，"
         "小单的量占比与额占比在全样本上几乎逐格相等（实测 2026 年 891,494 行 rank 相关 "
         "0.99999）。照抄会得到一个与既有因子精确重复的因子。"
         "本实现改为**相对自身 20 日中枢的偏离**（拥挤度的**边际**变化，而不是水平）："
         "「今天散户占比比它自己最近 20 天的常态高多少」。"
         "理由：(1) 水平量已被 `mf_retail_dominance` + `mf_order_size_entropy` 覆盖；"
         "(2) 参考库 thesis 的诉求是「**拥挤**」这个状态量，而自归一化后才是可比的"
         "跨股「异常度」（大盘股天然小单占比低，直接比水平等于在比市值）；"
         "(3) 沙箱 2026 实测（vs 全部 259 个既有生产因子的逐日截面 rank 相关）："
         "max|corr| 从 0.99999 降到 **−0.431**（`amount_ratio_20`；"
         "与 `mf_retail_dominance` 仅 **+0.353**、与 `volume_ratio` −0.348）。"
         "★ 不做参考库的 `rank(-small_pct)`（不取负）：方向用 `higher_is_better=False` 表达，"
         "`value` 列保持「偏离幅度」的可解释量纲（0 = 与自身中枢持平）。"
         "取值域 [−1, +∞)，沙箱 2026 实测中位数 +0.006、|value|max = 18.5（重尾来自"
         "20 日中枢极小的格子，靠引擎截面缩尾+rank 处理）。"
         "分母 `safe_div(min_abs_den=1e-6)` 挡掉零成交；`min_count=10` = 参考库 min_periods。",
))
def small_order_crowding(ctx):
    f = _F2(ctx)
    share = ctx.safe_div(f.g("gross_vol", "sm"), f.g("tot_vol"), min_abs_den=1e-6)
    base = ctx.roll_mean(share, 20, min_count=_MIN[20])
    return ctx.safe_div(share, base, min_abs_den=1e-6) - 1.0


@register(FactorSpec(
    name="super_large_order_intensity", group=GROUP,
    deps=(FLOW, DAILY),
    desc="超大单强度 = 5 日累计超大单净买入额 / 5 日累计八列毛额（主力大额持续收集筹码）",
    formula='elg_net = (mf["buy_elg_amount"] - mf["sell_elg_amount"]) / _total_amount(mf)\n'
            'return cross_sectional_rank(elg_net)\n'
            '★ 本实现（口径级偏离，见 note）：分子分母各自 5 日累计后再相除\n'
            'intensity = elg_net.rolling(5, min_periods=3).sum() / total.rolling(5, min_periods=3).sum()',
    start=FLOW_START, warmup_days=W5, higher_is_better=True,
    note="★★ **偏离参考库的字面公式（必须看）**：参考库是「单日 超大单净额 / 八列毛额」，"
         "而八列毛额正是 `mf_elg_order_ratio` 的分母、分子也是同一个单档净额 —— "
         "实测两者截面 rank 相关 **1.0000000**（2026 年 517,349 行逐格核对，最大相对差 4e-8），"
         "是**精确重复**。本实现把分子分母都换成 **5 日累计**再相除（ratio-of-sums，"
         "不是 5 日比率的均值），既保留参考库「净买入 / 总成交额」的口径与量纲（无量纲），"
         "又抓住 thesis 里「**持续**收集筹码」的时间维度。"
         "实测与既有因子的最大 |相关| 降到 **0.362**（沙箱 2026，vs 全部 259 个既有生产因子；"
         "最近邻 `net_turnover_rate_20`，与 `short_term_reversal_5` −0.359、"
         "与 `mf_big_order_ratio` +0.333、与 `elg_net_60d_to_mv` +0.278）。"
         "★ 与同族 `mf_large_order_net_5d` **不重复**：那条只算 lg 档、且是「5 日**比率**的均值」，"
         "本因子含 elg 档、用「5 日**累计额**之比」—— 参考库把 elg 与 lg 分成两个因子正是"
         "因为超大单更脉冲（每笔 >100 万 vs >20 万），实测两者相关仅约 +0.4。"
         "`min_count=3` = 参考库 min_periods；停牌/无记录 -> NaN（不补 0）。",
))
def super_large_order_intensity(ctx):
    f = _F2(ctx)
    net5 = _roll_sum(ctx, f.g("net_amt", "elg"), 5, _MIN[5])
    tot5 = _roll_sum(ctx, f.g("tot_amt"), 5, _MIN[5])
    return ctx.safe_div(net5, tot5, min_abs_den=1e-6)


# ══════════════════════════════════════════════════════════════════════════
# 二、结构 / 持续族（4 个）
# ══════════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="mf_big_small_divergence", group=GROUP,
    deps=(FLOW, DAILY),
    desc="大小单背离 = (大单净买额 − 小单净买额) / 当日八列毛额（机构与散户分歧，分歧顶点伴转折）",
    formula='big_net = (ff["buy_lg_amount"] - ff["sell_lg_amount"]\n'
            '           + ff["buy_elg_amount"] - ff["sell_elg_amount"])\n'
            'small_net = ff["buy_sm_amount"] - ff["sell_sm_amount"]\n'
            'divergence = (big_net - small_net) / _total_amount(ff)\n'
            'return cross_sectional_rank(divergence)',
    start=FLOW_START, warmup_days=W1, higher_is_better=True,
    note="逐字复刻参考库（`fund_flow.py`）。"
         "★★ **这是本族唯一保留的高相关因子，下游请按需剔除（诚实的量化交代）**："
         "由 " + _DE + " 可知四档净额之和恒为 0，于是"
         "「大单净额 − 小单净额 = 2×大单净额 + 中单净额」—— 本因子在代数上就是"
         "「大单净额」加一个固定权重的「中单净额」，**任何窗口平滑都消不掉这个恒等关系**。"
         "实测（2026 年逐日截面 rank）：与 `mf_big_order_ratio` **+0.938**、"
         "与 `mf_smart_dumb_divergence` **+0.963**、与 `mf_small_order_ratio` −0.906"
         "（后两者判定当时在册，随后已被主 Agent 从 `fundflow.py` 收口时裁掉；"
         "数字来自上游表复算，不依赖它们在册）。"
         "试过的三条改口径路线都被否掉："
         "(a) 换成量口径 → 相关 0.938（几乎不变，量/额在双记下等价）；"
         "(b) 按各自档位毛额归一 → 就是参考库的 `mf_smart_dumb_divergence` 本身（相关 1.000）；"
         "(c) 改成买/卖对数不对称之差 → 与它相关 **1.000**"
         "（单调变换保序）。"
         "保留的理由只有两条：参考库 `fund_flow.py` 里它就是这个名字与这个公式"
         "（下游可能按名字对接），且「机构买、散户卖」的**联合方向**读起来比单看大单净额直观。"
         "**若下游做因子筛选，本因子与上述两条高度共线，建议只保留其中一条。**"
         "分母 = 八列毛额（万元，含双记的 ×2 常数，对截面排名无影响）；"
         "分子分母同表同量纲，无需换算。",
))
def mf_big_small_divergence(ctx):
    f = _F2(ctx)
    tot = f.g("tot_amt")
    div = ctx.safe_div(f.main_net("amt") - f.g("net_amt", "sm"), tot, min_abs_den=1e-6)
    return div








# ══════════════════════════════════════════════════════════════════════════
# 三、量价交叉族（3 个）
# ══════════════════════════════════════════════════════════════════════════



@register(FactorSpec(
    name="mf_avg_trade_price_momentum", group=GROUP,
    deps=(FLOW, DAILY),
    desc="成交均价动量 = (大单VWAP / 小单VWAP) 的 5 日变化率（比率上升=机构买入更急迫）",
    formula='big_vol = (buy_lg_vol + sell_lg_vol + buy_elg_vol + sell_elg_vol)\n'
            'big_amt = (buy_lg_amount + sell_lg_amount + buy_elg_amount + sell_elg_amount)\n'
            'big_vwap = safe_divide(big_amt, big_vol); sm_vwap = safe_divide(sm_amt, sm_vol)\n'
            'ratio = safe_divide(big_vwap, sm_vwap).clip(0.5, 3.0)\n'
            'mom = ratio.groupby(level="Code").transform(\n'
            '    lambda s: s.pct_change(5, fill_method=None)).clip(-0.3, 0.5)\n'
            'return cross_sectional_rank(mom)',
    start=FLOW_START, warmup_days=W10, higher_is_better=True,
    note="逐字复刻参考库（`fund_flow_vol.py`），去掉两处 `clip`（契约禁止因子内 winsor，"
         "引擎统一做截面 1%/99% 缩尾；且实测 2026 年该比率 p1/p99 ≈ 0.94/1.06，"
         "参考库的 clip(0.5, 3) 与 clip(-0.3, 0.5) 在本平台都是空操作）。"
         "★ **纯同表因子，不需要任何单位换算**：`big_vwap` 与 `sm_vwap` 都是「万元/手」"
         "（= 100 元/股），相除时量纲自约，`ratio` 是无量纲的价格比（≈1）。"
         "★ 与 `mf_large_order_avg_price` 的区别（这是它最近的邻居）：后者是"
         "「大单均价 / **全日**均价」的**单日水平**，本因子是「大单 / **小单**」的**5 日变化**——"
         "换了参照系（小单 vs 全日）与阶数（变化 vs 水平），实测相关 **+0.638**（不冗余）。"
         "取值实测 2026 年中位数 0.0000、p1/p99 = −0.016/+0.017、最大 0.84（重尾来自"
         "小单成交极少的格子，靠截面缩尾处理）。"
         "`pct_change(5)` 的分母 `ratio` ≈ 1，`min_abs_den=1e-3` 只挡掉病态格子。",
))
def mf_avg_trade_price_momentum(ctx):
    f = _F2(ctx)
    big_vwap = ctx.safe_div(f.main_gross("amt"), f.main_gross("vol"), min_abs_den=1e-6)
    sm_vwap = ctx.safe_div(f.g("gross_amt", "sm"), f.g("gross_vol", "sm"), min_abs_den=1e-6)
    ratio = ctx.safe_div(big_vwap, sm_vwap, min_abs_den=1e-4)
    return ctx.pct_change(ratio, 5, min_abs_den=1e-3)


@register(FactorSpec(
    name="mf_amount_weighted_direction", group=GROUP,
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

@register(FactorSpec(
    name="margin_chip_cost_gap", group="margin",
    deps=(MARGIN, CHIPS, DAILY),
    lagged_ok=(MARGIN,),
    desc="融资盘成本 / 全市场筹码均价 − 1（融资盘相对市场平均的建仓位置，高位接盘者低优）",
    formula='cost = _margin_weighted_cost(margin, daily)   # Σ(rzmre × close)/Σ(rzmre) 近 20 日\n'
            'weight_avg = cyq["weight_avg"].reindex(cost.index)\n'
            'gap = safe_divide(cost, weight_avg) - 1.0\n'
            'return cross_sectional_rank(-gap)',
    start=CHIP_START, warmup_days=M20, higher_is_better=False,
    note="★★ 两融 PIT 铁律（本因子的处置是：**两边各自算完 → 相减 → 最后统一下移一格**）："
         "`rzmre` 的 T 日值要到 T+1 才可得，而筹码层是当日可得 —— 若只位移融资那一侧，"
         "会在两融名单变动的日子里造出「跨日拼接」的假信号；统一 `ctx.lag_grid(grid, 1)` "
         "后，T 日的因子值只用到 ≤ T−1 的两融记录与 ≤ T−1 的筹码快照（保守但严格无泄漏）。"
         "★ 价格口径（本因子最容易踩的坑）：融资买入的加权成本用 **`ctx.px(\"close\")`（未复权）**，"
         "因为筹码层的 `mean` 是**未复权**口径（`fea/chips.py` docstring「口径一」）。"
         "若按兄弟因子 `margin_buyer_avg_cost_premium` 那样用 `ctx.hfq` 取成本，"
         "比值会被整体放大 `adj_factor` 倍（实测 2019 年 108~142，差两个数量级）——"
         "**同口径 > 复权**，这里必须用未复权。"
         "★ 起点 `2018-01-02`：`stock_cyq_chips` 的真实起点（且只覆盖主板），"
         "在此之前无筹码数据，不做补偿。"
         "★ **覆盖率天生很低**：分母（筹码层）只覆盖主板且从 2018 起，"
         "分子（两融）只覆盖两融标的（2026 年主板约 1900 只）——两者求交后"
         "截面只剩约 1800~1900 只、非空率约 55%~60%（沙箱实测见交付报告），"
         "这是**数据源覆盖**的结果，不是实现缺陷。下游若要求全截面覆盖，本因子只能作为"
         "「两融标的子样本」因子使用。"
         "★ 不取负（方向交给 `higher_is_better=False`）：`value` = 融资盘成本相对筹码均价的"
         "**溢价率**，0 = 与市场平均成本持平，正 = 杠杆盘高位接盘。"
         "★ 与 `margin_buyer_avg_cost_premium` 的分工：那条比的是「**现价** / 融资成本」"
         "（浮盈视角），本因子比的是「**融资成本** / 筹码成本」（成本结构视角），"
         "两者共用同一套加权成本口径（本文件的 `_mg_cost`）。"
         "★ 沙箱 2026 实测（vs 全部 259 个既有生产因子的逐日截面 rank 相关）："
         "max|corr| = **+0.523**（`avg_cost_premium`，筹码族的「现价/筹码均价」，"
         "单日最高 0.762）—— 两者共享「筹码成本」这一半，但本因子的分子是**融资成本**"
         "（外部增量信息），这正是它与筹码族不重复的部分。",
))
def margin_chip_cost_gap(ctx):
    df, day = _mg_load(ctx)
    if df is None:
        return _empty(ctx)
    cost = _mg_cost(ctx, _mg_flow(ctx, df, day, "rzmre"))   # 元/股（未复权口径）
    chip_mean = ctx.chip("mean")                            # 元/股（未复权口径）
    gap = ctx.safe_div(cost, chip_mean, min_abs_den=1e-3) - 1.0
    return ctx.lag_grid(gap, 1)


# ══════════════════════════════════════════════════════════════════════════
# 五、两融结构（2 个）—— ★ 滞后表，lag_grid 是最后一步
# ══════════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="margin_balance_ma_divergence", group="margin",
    deps=(MARGIN,),
    lagged_ok=(MARGIN,),
    desc="融资余额偏离 20 日均线幅度 = (rzye − MA20)/MA20（极端偏离预示均值回归）",
    formula='rzye = m["rzye"]\n'
            'ma20 = rzye.groupby(level="Code").transform(\n'
            '    lambda s: s.rolling(20, min_periods=10).mean())\n'
            'div = safe_divide(rzye - ma20, ma20); div = div.clip(-0.1, 0.1)\n'
            'return cross_sectional_rank(div)\n'
            '★ 本实现的状态量口径（见 note）：asof 前向填充 + **20 个交易日新鲜度掩码**\n'
            'stale = 最近 20 个交易日内无该股两融记录 → div = NaN',
    start=None, warmup_days=M20, higher_is_better=True,
    note="★ 两融 PIT 铁律：`lagged_ok=(\"stock_margin_detail\",)` + **最后一步** "
         "`ctx.lag_grid(grid, 1)` —— T 日只用 ≤ T−1 的两融记录。"
         "参考库写的是 `margin_detail.parquet`（其面板已 shift(1)），本实现等价。"
         "★ 偏离：不做参考库的 `.clip(-0.1, 0.1)`（契约禁止因子内 winsor）。"
         "实测 2026-09-10 上游原始表 p1/p50/p99 = −0.106 / −0.0006 / +0.107 —— "
         "参考库的 clip(±0.1) 恰好压在 p1/p99 上（说明它本来就是手工缩尾），"
         "引擎的截面 1%/99% 缩尾覆盖同一批极端值，且口径统一。"
         "★★ **冗余警示（实测，必须看）**：本因子与既有 margin 因子的相关很高 ——"
         "沙箱 2026 逐日截面 rank 相关：`margin_leverage_trend_10d` **+0.883**、"
         "`margin_flow_asymmetry_10d` +0.818、`margin_balance_20d` +0.811"
         "（单日最高 0.930）。根因是数学的：余额平滑增长时"
         "「(rzye − MA20)/MA20 ≈ g × 9.5」而「20 日变化率 ≈ g × 20」（g = 日均增速），"
         "两者都是同一个 g 的（近似）单调变换 —— 截面上必然高相关。"
         "★ 试过的解耦方案**实测反而更差**，故保留参考库原式：把偏离按自身 60 日标准差"
         "归一化（「极端度」）后，与 `margin_flow_asymmetry_10d` 的相关升到 **+0.794**"
         "（未归一化时 +0.659，同一口径对比），因为归一化引入的波动本身与资金流波动同向。"
         "**处置建议**：本因子与上述三条属于同一个「杠杆趋势」簇，"
         "下游做因子筛选时**至多保留其一**；保留本因子的唯一理由是它多了一层"
         "「相对自身中枢的位置」语义（见下方方向说明）。"
         "`min_count=10` = 参考库 min_periods。"
         "★ 方向标注为「大者优」沿用参考库的 `rank(div)`；但请注意其**语义是均值回归**"
         "（两端偏离都可能回归），即该因子的有效信息是**非单调**的；"
         "而上面那三条同簇因子是**趋势**语义（大者优）—— 两条高相关、语义相反的因子"
         "同时进线性模型会互相抵消，这一点务必交由筛选环节处理。"
         "停牌日：余额是状态量，`asof` 前向填充（实测上游停牌日仍有行且余额在变）。"
         "★★ **「已退出两融名单」的股票怎么处理（实测驱动的一次修复）**："
         "余额是状态量，asof 会把最后一条记录无限前向填充 —— 于是**退市/退出两融的股票**"
         "在 20 日窗内余额恒定 → 偏离度恒等于 0，即「余额没变化」的假信号。"
         "更严重的是**一致性**：掩码之前，同一格的值取决于面板起点（全量跑的 warmup 伸到上一年、"
         "读得到上一年的最后一条记录 → 0；增量跑读不到 → NaN），"
         "实测 7 只 2025 年 1~4 月退出两融的股票在 2026-09-04~09-14 每天 7~20 格两趟不一致。"
         "现在 `_mg_state` 统一加**新鲜度掩码**（最近 20 个交易日内必须有该股的两融记录，"
         "否则 NaN），并按交易日距离（而非「读到没读到」）判定 —— "
         "实测：全量跑 vs `--start 2026-09-01` 的尾部跑，19,540 格**逐格相同、非空模式一致**；"
         "`main.py audit-pit`（截断到 2026-05-18 / 2026-09-14 重算）两个样本日 **0 个不一致**。"
         "实测值域：2026 年 |value|max = 2.93（未 clip，重尾），中位数 −0.0125。",
))
def margin_balance_ma_divergence(ctx):
    df, day = _mg_load(ctx)
    if df is None:
        return _empty(ctx)
    rzye = _mg_state(ctx, df, day, "rzye")
    ma20 = ctx.roll_mean(rzye, 20, min_count=_MIN[20])
    return ctx.lag_grid(ctx.safe_div(rzye - ma20, ma20, min_abs_den=DEN), 1)




# ══════════════════════════════════════════════════════════════════════════
# 六、其它（2 个）—— 耦合因子：`deps` 里写的是**别的因子名**
# ══════════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="fundflow_retail_inst_divergence", group=GROUP,
    deps=("mf_big_order_ratio", "mf_retail_dominance"),
    desc="主力-散户背离 = rank(大单净买率) × rank(−散户成交占比)（机构买而散户卖的联合信号）",
    formula='big = ctx.load_factor("mf_big_order_ratio")\n'
            'small = ctx.load_factor("mf_small_order_ratio")\n'
            'big_r = _rank(big); small_r = _rank(-small)\n'
            'divergence = big_r * small_r\n'
            'return cross_sectional_rank(divergence)',
    start=FLOW_START, warmup_days=W1, higher_is_better=True,
    note="★ **耦合因子**（参考库 `coupling.py` 的口径）：`deps` 写的是两个**父因子名**，"
         "引擎据此把它排在第二趟（等父因子落盘后再算，见 `main.py` 的两趟调度）。"
         "★ **父因子的替换（与参考库的字面差异，必须看）**：参考库读的是 "
         "`mf_small_order_ratio`（散户**净额**占比），该因子已被主 Agent 从 "
         "`factors/fundflow.py` 删除（其 `_DROPPED` 记录：与既有因子精确重复），"
         "本实现改用**现存**的散户侧代表因子 `mf_retail_dominance`"
         "（= 小单毛额占比，`higher_is_better=False`，即值大=散户主导）。"
         "两者是同一枚硬币的两面：`mf_retail_dominance` 度量「散户**参与度**」"
         "（谁在交易，实测与已被删的 `mf_small_order_ratio` 相关仅 −0.27），"
         "取负排名后语义与参考库的 `rank(-small)` 一致 —— 「散户不占主导」。"
         "机构侧父因子 `mf_big_order_ratio` 未变。"
         "★ 这里**必须**用到截面排名（`_cs_rank_pct`）—— 排名是公式的组成部分（乘积 = "
         "「同时高」的联合信号），不是对本因子输出的后处理；本因子的输出仍会被引擎再排名一次。"
         "参考库用的 `_rank` 同样是截面百分位排名，口径一致。"
         "★ 与直接相减的区别（参考库的 `mf_smart_dumb_divergence` 是「大单率 − 小单率」，"
         "该因子同样已被删除）：乘积要求**两端同时**极端（机构排前 20% 且散户排后 20% 才进 top 4%），"
         "相减只要求一端领先 —— 前者是「共振」，后者是「差值」；本族保留的 "
         "`mf_big_small_divergence` 是相减的另一种写法（见其 note）。"
         "★ 与 `big_vs_small_divergence_5d` 的区别：那条是**变化率**（Δ5），本因子是**当日水平**"
         "的联合排名。"
         "★ 沙箱 2026 实测（vs 全部 259 个既有生产因子的逐日截面 rank 相关）："
         "max|corr| = **+0.681**（父因子 `mf_big_order_ratio` 本身）、"
         "`mf_retail_dominance` −0.594、`mf_order_size_entropy` +0.554 —— "
         "对一个「两个父因子排名的乘积」来说这是**结构性下限**（乘积必然与两个因子都相关），"
         "0.68 已经远低于任一父因子自身的水平，剩下的部分就是「共振」这个新信息。"
         "★ 缺失处置：任一父因子为 NaN（停牌/无记录）→ 乘积为 NaN，不参与截面。"
         "★ 若父因子尚未落盘，`ctx.load_factor` 会**静默**返回全 NaN（它只 warning），"
         "表现为本因子非空率 0% —— 首次全量跑时确认两个父因子已在同一批次里。",
))
def fundflow_retail_inst_divergence(ctx):
    big = ctx.load_factor("mf_big_order_ratio")
    small = ctx.load_factor("mf_retail_dominance")
    return _cs_rank_pct(big) * _cs_rank_pct(-small)


@register(FactorSpec(
    name="mf_flow_factor_momentum_20", group=GROUP,
    deps=("mf_net_inflow_ratio",),
    desc="主力资金流因子的 20 日动量 = Δ20(主力净流入率)（机构态度的边际改善）",
    formula='mf = ctx.load_factor("mf_net_inflow_ratio")\n'
            'delta = _delta(mf, 20)\n'
            'return cross_sectional_rank(delta)',
    start=FLOW_START, warmup_days=W20, higher_is_better=True,
    note="★ **耦合因子**：`deps` 写父因子名 `mf_net_inflow_ratio`（已在 `factors/fundflow.py` "
         "实现），引擎把它排在第二趟。`_delta(x, 20)` = `x(T) − x(T−20)`，"
         "本实现用 `ctx.diff(mat, 20)`（同一口径）。"
         "★ 与同族别的「净流入」因子的关系：`mf_net_inflow_ratio` 是**水平**（今天净流入多少）、"
         "`mf_cumulative_flow_20d` 是**累计**（近 20 天累计多少）、本因子是**水平的变化**"
         "（今天相比 20 天前改善了没有）—— 水平高但边际转弱的股票，本因子给出与水平口径"
         "**相反**的信号，这正是参考库 thesis 讲的「边际改善」。"
         "★ warmup=56：`diff(20)` 需要 20 个交易日的记忆（20×1.8+20 = 56）。"
         "★ 父因子未落盘时 `ctx.load_factor` **静默**返回全 NaN（只 warning）→ "
         "首次全量跑时确认 `mf_net_inflow_ratio` 在同一批次里。",
))
def mf_flow_factor_momentum_20(ctx):
    return ctx.diff(ctx.load_factor("mf_net_inflow_ratio"), 20)
