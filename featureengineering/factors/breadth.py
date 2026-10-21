"""市场宽度 / 情绪条件因子（12 个）—— 全部日频产出、只主板、跟随 `default_start`。

数据源：`stock_market_distribution_history`（**每分钟**一行，16 个计数：涨/跌/平家数、
涨停/跌停家数、以及 ±3%/5%/7%/10% 的涨跌幅分桶家数），2010-01-04 起。
★★ **该表此前被 225 个因子零引用**，本文件是它的第一次使用。

═══════════════════════════════════════════════════════════════════════════
一、★★★ 本家族最重要的一条设计约束：市场级 ≠ 截面因子
═══════════════════════════════════════════════════════════════════════════

`stock_market_distribution_history` 是**市场级**数据 —— 同一天对所有股票是**同一个值**。
下游任务是 **A 股日横断面回归排序**，一个当天恒定的量在截面上的**离散度恒为 0**，
排序时与截距完全共线，**零信息**。

⇒ 本文件 12 个因子**全部**是「个股对该市场量的**敏感度 / 捕获 / 条件统计量**」：
     β、相关系数、条件均值、同向频率、不对称度。
   **一个「市场宽度水平」因子都没有**，这是刻意的，不是遗漏。
   下一轮的作者请**不要**来「补上」`market_up_share` 这类因子 —— 它做不了横截面排序。

═══════════════════════════════════════════════════════════════════════════
二、实测的管道事实（写下来免得重测）
═══════════════════════════════════════════════════════════════════════════

1. **日期列叫 `trade_time`，是字符串日期时间**（`"2026-09-17 09:30:00"`），
   不是 `trade_date`。`ctx.dataset` + `ctx.date_col` 前必须先 `.str.slice(0, 10)`。
2. **每日 241 行**：09:30–11:30 共 121 分钟 + 13:01–15:00 共 120 分钟；
   11:31–12:59 **整段没有行**（午休）。
3. **`up_count + down_count + flat_count` 在**同一天内**是常数（当天有报价的家数），
   但**跨日会变**（实测 2026 年有 76 个不同取值，范围 5452–5553）—— 上市家数在增长。
   ⇒ **一律用份额（÷总数），绝不用原始计数**（原始计数在 2010→2026 上是非平稳的，
      会把「市场涨家数」变成一个趋势变量）。
4. **涨跌幅分桶精确划分**：`up_over_10 + … + up_0_to_3 ≡ up_count`
   （实测全表为真），`down` 侧同理。
5. ★★ **`limit_up_count` / `limit_down_count` 在 2010–2019 恒等于 0**（实测）
   —— **不是缺测、是整整十年的 `0`**，2020-01-02 起才真正有值
   （与 `stock_limit_list` 的起点一致）。后果：任何直接用这两列做因子的实现
   在 2020 年前会产出一个**恒为 0 的常数因子**，而且 `safe_div` 不会报错、
   `roll_corr` 会因零方差给出全 NaN。
   ⇒ 本文件**一律不用** `limit_up_count` / `limit_down_count`，
     改用**涨跌幅分桶**（见下条），全历史可用。
6. **分桶的可用区间（实测非零率，2010 起）**：
   `up_5_to_7` 99.7~100%、`up_7_to_10` 97~100%、`up_over_10` 92.9~100% —— 全历史可用；
   `down_5_to_7` 78~100% —— 可用；`down_7_to_10` 53~99%、`down_over_10` 12.9~95%
   —— **早年很稀疏，不用**。
   ⇒ 「强势占比」= `(up_5_to_7+up_7_to_10+up_over_10)/总数`，
     「恐慌占比」= `(down_5_to_7+down_7_to_10+down_over_10)/总数`，两者都从 2010 起可用。
7. **`place` 而不是 `asof`**：市场序列某天缺行就应该是 NaN，
   前向填充会造出「0 变化日」，把 β 与相关系数静默拉低
   （与 `factors/volatility.py::_market_ret` 同一条理由）。
   借**第 0 列**落格（`place` 要求 code ∈ [0, C)）。

═══════════════════════════════════════════════════════════════════════════
三、明确不做的（理由）
═══════════════════════════════════════════════════════════════════════════

· **任何市场**水平**因子** —— 见 §一。
· **原始计数**因子 —— 非平稳，见 §二.3。
· **行业相对宽度**（`bw_rel_ind_*`）—— 需要带日期的行业成分股表；
  `tdx_block_stocks` 是**无日期的快照**，
  拿今天的成分股算 2015 年的行业内均值就是前视（参考库自己因此禁用了 6 个 sector 因子）。
  本项目的解法见 `factors/sector.py`（自建动态行业）；宽度 × 行业的交叉属下一轮。
· **真·48 点日内相关**（个股 5min 路径 × 市场宽度路径）—— 日内层不存个股路径
  （见 `factors/intraday2.py` 的 docstring §二）。可得替代：本文件的
  `bw_session_follow_20`（用上/下午两段）与 `bw_intraday_beta_20`。
· ★★ **「自参照滚动分位」做条件化** —— 这是本轮**实测踩到的真实缺陷**，写在这里
  给下一轮：本族首版有两个因子写成「今日宽度量 ≤ 其自身 20 日窗的 5% / 90% 分位」
  再取条件均值，结果**在趋势行情里可以连续几十天一次都不命中**
  ⇒ 整条截面全 NaN ⇒ `main.py check` 报 `✘CONST`（实测 2024 有 103 个全 NaN 日）。
  换成任何自参照分位阈值都会有同样的病：**「今天相对自己过去很弱」在单边行情里
  按定义不成立**。实测：把条件换成**绝对阈值**（`sh_close < 0.5`）后，
  全样本 4059 个交易日里最长连续不命中只有 **8 天** ⇒ 20 日窗恒有命中。
  ⇒ 本族现在**没有任何因子用 `roll_quantile` 做条件**；要用请先验证
     「最长连续不命中 < 窗口长度」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fea.spec import FactorSpec, register

ID_BW_START = None

W_20 = 56     # 20 交易日窗：20×1.8+20
W_60 = 128    # 60 交易日窗：60×1.8+20

PRICE_DEPS = ("stock_daily", "stock_adj_factor")
TBL = "stock_market_distribution_history"
BW_DEPS = (TBL, *PRICE_DEPS)
MKT_INDEX = "000300.SH"

# 每个因子的 note 都要带这句（见模块 docstring §一）
_NOCS = ("★ 本因子是**个股对市场宽度的敏感度**，不是市场水平量 —— "
         "市场级数据在横断面上的离散度恒为 0，排序时与截距共线（见模块 docstring §一）。")


# ══════════════════════════════════════════════════════════════════════
# 市场宽度序列（13 个因子共用）
# ══════════════════════════════════════════════════════════════════════

class _Mkt:
    """把分钟级宽度表聚合成 6 条**与面板日期精确对齐**的 (T,) 市场序列。

    为什么逐因子各建一次也够快：`Upstream.read` 按 (表名, 列, 年份) 缓存，
    同一 worker 内的多次调用只读一次盘；聚合本身是 reduceat，
    单年 4.2 万行 / 双年 8.4 万行，毫秒级。
    """

    def __init__(self, ctx):
        panel = ctx.panel
        y0 = int(str(panel.dates[0])[:4])
        y1 = int(str(panel.dates[-1])[:4])
        self.T = panel.T
        blank = np.full(panel.T, np.nan)
        self.sh_open = self.sh_am = self.sh_close = blank.copy()
        self.bw_vol = self.up_ext = self.dn_ext = blank.copy()
        self.ok = False

        UP5 = ("up_5_to_7", "up_7_to_10", "up_over_10")     # 见 docstring §二.5/§二.6
        DN5 = ("down_5_to_7", "down_7_to_10", "down_over_10")
        df = ctx.dataset(TBL, columns=["trade_time", "up_count", "down_count",
                                       "flat_count", *UP5, *DN5], years=(y0, y1))
        if df is None or df.empty:
            return
        tt = df["trade_time"].astype("string")
        day = ctx.date_col(tt.str.slice(0, 10)).astype(np.int64)
        hm = (tt.str.slice(11, 13).astype("int32") * 100
              + tt.str.slice(14, 16).astype("int32")).to_numpy(dtype=np.int64)
        num = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)
        tot = num("up_count") + num("down_count") + num("flat_count")
        with np.errstate(all="ignore"):
            sh = ctx.safe_div(num("up_count"), tot, 0.0).astype(np.float64)
            up5 = ctx.safe_div(sum(num(c) for c in UP5), tot, 0.0).astype(np.float64)
            dn5 = ctx.safe_div(sum(num(c) for c in DN5), tot, 0.0).astype(np.float64)

        # 组内按 (day, hm) 升序 —— place 要求 (day,code) 唯一，reduceat 要求段连续
        order = np.lexsort((hm, day))
        day, hm = day[order], hm[order]
        sh, up5, dn5 = sh[order], up5[order], dn5[order]
        st = np.flatnonzero(np.concatenate(([True], day[1:] != day[:-1])))
        en = np.concatenate((st[1:], [day.size]))
        days = day[st]

        def place(v):
            return np.ascontiguousarray(
                panel.place(np.zeros(days.size, dtype=np.int64), days, v)[:, 0])

        self.sh_open, self.sh_close = place(sh[st]), place(sh[en - 1])
        # 强势 / 恐慌占比取 15:00 那一刻（收盘定稿）；盘中值对日频因子无意义
        self.up_ext, self.dn_ext = place(up5[en - 1]), place(dn5[en - 1])
        # 当日分钟内 up 份额的波动（市场内部「翻脸」的速度）
        cnt = (en - st).astype(np.float64)
        m1 = np.add.reduceat(sh, st) / cnt
        m2 = np.add.reduceat(sh * sh, st) / cnt
        self.bw_vol = place(np.sqrt(np.maximum(m2 - m1 * m1, 0.0)))
        # 上午收盘（11:30 那一行）的 up 份额：组内按 hm 升序 ⇒ 取每组内
        # 「hm ≤ 1130」的最后一个即可（reduceat 做不到「条件取最后一个」，用下标）
        ai = np.flatnonzero(hm <= 1130)
        ag = np.searchsorted(st, ai, side="right") - 1
        take = np.concatenate((ag[1:] != ag[:-1], [True])) if len(ag) else np.array([], dtype=bool)
        am = np.full(days.size, np.nan)
        am[ag[take]] = sh[ai[take]]
        self.sh_am = place(am)
        self.ok = True


def _m(_M: _Mkt, name: str) -> np.ndarray:
    """取 (T,) 市场序列 -> (T,1)，用于 roll_cov / roll_var / roll_corr 的广播。"""
    return np.asarray(getattr(_M, name), dtype=np.float64)[:, None]




def _ret(ctx) -> np.ndarray:
    return ctx.ret_clean(1)










# ══════════════════════════════════════════════════════════════════════
# 1. 参与度 β + 情绪敏感度（5 个）
# ══════════════════════════════════════════════════════════════════════









@register(FactorSpec(
    name="bw_strength_sensitivity_20", version=2,
    group="breadth",
    deps=BW_DEPS,
    desc="个股收益 与 全市场「涨超 5% 占比」的 20 日相关（对强势情绪的敏感度）",
    formula="up = (up_5_to_7+up_7_to_10+up_over_10)/total; corr(ret, up, 20, 10)",
    start=ID_BW_START,
    warmup_days=W_20,
    higher_is_better=True,
    note=(_NOCS + " **本文件新造**（参考库 `up_down_count_ratio_20` 是最接近的，"
          "但那个是**市场水平量本身**，截面上零离散；本因子把它变成"
          "**个股对该量的相关**）。"
          "与 `bw_beta_60` / `bw_intraday_beta_20` 的回归量**不同**："
          "那两个用**上涨家数份额**（连续、覆盖全市场、代表「扩散度」），"
          "本因子用**涨超 5% 的占比**（稀疏、右偏、代表「赚钱效应/强势股集中度」）——"
          "A 股里「大涨家数」是情绪周期的刻度，与「涨家数多不多」是两个不同的"
          "状态变量（可以普涨但没有强势股，也可以强势股很多但市场收平）。"
          "★ 与已注册的 `limit_up_count_20`（个股**自身**的 20 日涨停次数）"
          "完全不同：那个是自身行为，本因子是对**市场情绪**的反应。"
          "★★ 不用 `limit_up_count`：它在 2010–2019 恒等于 0（模块 docstring §二.5），"
          "`roll_corr` 会因为回归量零方差而在头十年给**全 NaN**。"),
))
def bw_strength_sensitivity_20(ctx):
    _M = _Mkt(ctx)
    if not _M.ok:
        return ctx.panel.empty()
    return ctx.roll_corr(_ret(ctx), _m(_M, "up_ext"), 20, 10)


# ══════════════════════════════════════════════════════════════════════
# 2. 条件收益（3 个）
# ══════════════════════════════════════════════════════════════════════







# ══════════════════════════════════════════════════════════════════════
# 3. 宽度联动与脆弱性（3 个）
# ══════════════════════════════════════════════════════════════════════







