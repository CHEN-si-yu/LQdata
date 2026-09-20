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
  `tdx_block_stocks` / `index_ths_constituent_stocks` 都是**无日期的快照**，
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
        take = np.concatenate((ag[1:] != ag[:-1], [True]))
        am = np.full(days.size, np.nan)
        am[ag[take]] = sh[ai[take]]
        self.sh_am = place(am)
        self.ok = True


def _m(_M: _Mkt, name: str) -> np.ndarray:
    """取 (T,) 市场序列 -> (T,1)，用于 roll_cov / roll_var / roll_corr 的广播。"""
    return np.asarray(getattr(_M, name), dtype=np.float64)[:, None]


def _lag(s: np.ndarray) -> np.ndarray:
    """(T,) 序列下移一行（= 上一交易日）。首行 NaN。"""
    out = np.full_like(s, np.nan)
    out[1:] = s[:-1]
    return out


def _ret(ctx) -> np.ndarray:
    return ctx.ret(1)


def _idx_ret(ctx) -> np.ndarray:
    """(T,1) 沪深300 日收益，精确落格（与 `factors/volatility.py::_market_ret` 同口径）。

    本文件**不**从 `volatility.py` import（一 Agent 一文件的契约不允许跨家族私有 import），
    所以这里重写一份 12 行的版本。
    """
    panel = ctx.panel
    y0 = int(str(panel.dates[0])[:4])
    y1 = int(str(panel.dates[-1])[:4])
    df = ctx.dataset("index_daily", columns=["ts_code", "trade_date", "pct_chg"],
                     years=(y0, y1))
    if df is None or df.empty:
        return np.full((panel.T, 1), np.nan)
    df = df[df["ts_code"] == MKT_INDEX]
    if df.empty:
        return np.full((panel.T, 1), np.nan)
    days = ctx.date_col(df["trade_date"])
    v = pd.to_numeric(df["pct_chg"], errors="coerce").to_numpy(np.float64) / 100.0
    g = panel.place(np.zeros(days.size, dtype=np.int64), days, v)
    return np.ascontiguousarray(g[:, 0])[:, None]


def _beta(ctx, ret, mk, n, mc) -> np.ndarray:
    return ctx.safe_div(ctx.roll_cov(ret, mk, n, mc),
                        ctx.roll_var(mk, n, mc), min_abs_den=1e-12)


def _cond_mean(ctx, r, mask, n: int, min_days: int) -> np.ndarray:
    """滚动窗口内「满足 mask 的那些天」的收益均值。

    ★ 不能用 `roll_mean(where(mask, r, nan), n, mc)`：mask 命中率约 1/20，
    窗口里通常只有 0~2 天命中，任何 mc ≥ 2 都会让整列 NaN。
    改为「命中日的和 ÷ 命中日数」，两者都用 `roll_sum`（输入强制无 NaN，
    所以不会触发窗口毒化）；命中日数不足 `min_days` 时给 NaN。
    """
    m = mask & np.isfinite(r)
    s = ctx.roll_sum(np.where(m, r, 0.0), n)
    c = ctx.roll_sum(m.astype(np.float64), n)
    return np.where(c >= min_days, ctx.safe_div(s, c, 1e-9), np.nan)


def _dsh(ctx, _M: _Mkt) -> np.ndarray:
    """宽度「收益」= 今日 15:00 的上涨份额 − 昨日 15:00 的。"""
    s = _M.sh_close
    return (s - _lag(s))[:, None]


# ══════════════════════════════════════════════════════════════════════
# 1. 参与度 β + 情绪敏感度（5 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="bw_beta_60",
    group="breadth",
    deps=BW_DEPS,
    desc="个股收益对市场宽度变化的 60 日 β（市场参与度暴露）",
    formula="beta = roll_cov(ret, delta_up_share, 60, 30) / roll_var(delta_up_share, 60, 30)",
    start=ID_BW_START,
    warmup_days=W_60,
    higher_is_better=False,
    note=(_NOCS + " 这是全家族的**基准量**：个股对「今天有多少股票在涨」这个变化的敏感度。"
          "与已注册的 `beta_60`（对沪深300）**不是**同一个回归量："
          "沪深300 是被大市值主导的加权指数，宽度变化是**等权**的、"
          "由中小市值主导 —— A 股的小盘股对宽度的 β 显著高于对指数的 β。"
          "⚠ 两者相关性预计较高（0.7~0.9），`dedup` 阶段定量裁决；"
          "若 |ρ| ≥ 0.95 应优先保留 `bw_resid_beta_60`（剔掉指数 β 后的正交部分）。"
          "方向取负（高暴露 = 高系统性风险）。"),
))
def bw_beta_60(ctx):
    _M = _Mkt(ctx)
    if not _M.ok:
        return ctx.panel.empty()
    return _beta(ctx, _ret(ctx), _dsh(ctx, _M), 60, 30)


@register(FactorSpec(
    name="bw_resid_beta_60",
    group="breadth",
    deps=(*BW_DEPS, "index_daily"),
    desc="剔掉沪深300 β 之后的宽度 β（正交的参与度暴露）",
    formula="e = ret - beta_idx*ret_idx; resid_beta = roll_cov(e, d_sh, 60, 30)/roll_var(d_sh, 60, 30)",
    start=ID_BW_START,
    warmup_days=W_60,
    higher_is_better=False,
    note=(_NOCS + " ★ 构造：先对沪深300 回归取残差 `e = r − β_idx·r_idx`（β 同样用 60 日滚动、"
          "只用窗口内数据），再让残差对宽度变化做 β。"
          "**这是宽度暴露里真正与市场 β 正交的那一块** —— 如果 `bw_beta_60` 与 "
          "已注册的 `beta_60` 高度相关，本因子就是那个还剩下信息量的版本。"
          "经济含义：剔掉「随大盘涨跌」之后，还剩多少「随市场**扩散/收敛**」的暴露。"
          "依赖 `index_daily`（沪深300）—— 与 `factors/volatility.py` 同一口径。"),
))
def bw_resid_beta_60(ctx):
    _M = _Mkt(ctx)
    if not _M.ok:
        return ctx.panel.empty()
    r = _ret(ctx)
    idx = _idx_ret(ctx)
    b = _beta(ctx, r, idx, 60, 30)
    e = r - b * idx
    return _beta(ctx, e, _dsh(ctx, _M), 60, 30)


@register(FactorSpec(
    name="bw_beta_change_20",
    group="breadth",
    deps=BW_DEPS,
    desc="宽度 β 的短期变化 = β20 − β60（参与度暴露的抬升）",
    formula="beta20 = rolling_beta(20,10); beta60 = rolling_beta(60,30); change = beta20 - beta60",
    start=ID_BW_START,
    warmup_days=W_60,
    higher_is_better=True,
    note=(_NOCS + " 源自参考库 Class1 `market_beta_change_20`（把回归量从市场收益换成宽度变化）。"
          "★ **方向与 `bw_beta_60` 相反**：水平高是风险（排后），"
          "但**水平在抬升**是时点信号（市场关注度/资金参与度正在向这只股票集中，排前）。"
          "这正是「水平 vs 变化」在 A 股里的经典分野 —— 慢变量的水平几乎没有 alpha"
          "（已删的 `delta_*` 一族就是死在「慢水平的差分也慢」），"
          "但**快窗口相对慢窗口的变化**是对「当下发生了什么事」的直接刻画。"
          "两个窗口的 min_count 分别取 10 / 30（= N//2，见契约）。"),
))
def bw_beta_change_20(ctx):
    _M = _Mkt(ctx)
    if not _M.ok:
        return ctx.panel.empty()
    r, d = _ret(ctx), _dsh(ctx, _M)
    return _beta(ctx, r, d, 20, 10) - _beta(ctx, r, d, 60, 30)


@register(FactorSpec(
    name="bw_beta_asym_20",
    group="breadth",
    deps=BW_DEPS,
    desc="上行宽度日 β − 下行宽度日 β（参与度的不对称）",
    formula="b_up - b_down, 两腿各自用互斥的 NaN 掩码在 20 日窗内回归",
    start=ID_BW_START,
    warmup_days=W_20,
    higher_is_better=True,
    note=(_NOCS + " **本文件新造**（参考库只做了对称的单 β）。"
          "构造：`d_sh > 0` 的交易日只留 `r` 与 `d_sh`，其余置 NaN；"
          "`d_sh < 0` 的日子同理 —— 两腿的掩码**互斥**，两次回归在样本上完全不相交。"
          "每腿在窗口内至少要有 6 个有效日（20 日窗的 30%）——"
          "★ 不按 N//2=10 取：一腿只占窗口中约一半的日子，"
          "`min_count=10` 等于要求「两腿都几乎满格」，会把绝大多数格子打成 NaN；"
          "6 ≈ 该腿期望命中数（10）的 60%，与契约里「N//2」的宽松度对齐。"
          "经济含义：> 0 ⇒ 市场扩散日跟涨、收敛日不跟跌（**顺周期但抗跌**）；"
          "< 0 ⇒ 市场扩散日不涨、收敛日大跌（**脆弱**）。"
          "与已注册的 `downside_vol_ratio_20` / `gain_loss_asymmetry_60` 不同："
          "那几个的不对称是**个股自身**收益分布的不对称，"
          "本因子的不对称是**个股对市场状态的反应**不对称（条件对象完全不同）。"),
))
def bw_beta_asym_20(ctx):
    _M = _Mkt(ctx)
    if not _M.ok:
        return ctx.panel.empty()
    r, d = _ret(ctx), _dsh(ctx, _M)
    up = d > 0
    r_u, d_u = np.where(up, r, np.nan), np.where(up, d, np.nan)
    r_d, d_d = np.where(~up, r, np.nan), np.where(~up, d, np.nan)
    b_u = _beta(ctx, r_u, d_u, 20, 6)
    b_d = _beta(ctx, r_d, d_d, 20, 6)
    return b_u - b_d


@register(FactorSpec(
    name="bw_strength_sensitivity_20",
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

@register(FactorSpec(
    name="bw_capture_asym_20",
    group="breadth",
    deps=BW_DEPS,
    desc="市场收敛日的捕获 / 市场扩散日的捕获（防御性）",
    formula="stock_on_down = mean(ret | d_sh<0, 20d); stock_on_up = mean(ret | d_sh>0, 20d); "
            "sens = |down| / |up|",
    start=ID_BW_START,
    warmup_days=W_20,
    higher_is_better=False,
    note=(_NOCS + " 抄参考库 Class1 `market_regime_sensitivity_60`（窗口 60→20、"
          "市场状态判据由「等权市场收益的符号」换成「宽度变化的符号」）。"
          "参考库原式的方向是负（防御型排前）。"
          "与已注册的 `downside_upside_vol_60` 的区别：那个是**波动**的上下行之比"
          "（离散度），本因子是**条件均值**之比（收益的**方向性**捕获）—— "
          "一只股票可以上下行波动都很小但下行均值显著为负。"
          "分母 `|up|` 趋 0 时由 `safe_div` 给 NaN。"),
))
def bw_capture_asym_20(ctx):
    _M = _Mkt(ctx)
    if not _M.ok:
        return ctx.panel.empty()
    r, d = _ret(ctx), _dsh(ctx, _M)
    dn = _cond_mean(ctx, r, d < 0, 20, 4)
    up = _cond_mean(ctx, r, d > 0, 20, 4)
    return ctx.safe_div(np.abs(dn), np.abs(up), 1e-10)


@register(FactorSpec(
    name="bw_weak_breadth_ret_20",
    group="breadth",
    deps=BW_DEPS,
    desc="市场偏弱日（上涨家数 < 半数）的条件收益 − 自身 20 日均值",
    formula="weak = sh_close < 0.5; cond = mean(ret | weak, 20) - mean(ret, 20)",
    start=ID_BW_START,
    warmup_days=W_20,
    higher_is_better=True,
    note=(_NOCS + " **本文件新造**（与参考库 `tail_corr_60` 是不同构念：那个测**相关性**，"
          "本因子测**条件收益**）。"
          "★ 为什么减去自身均值：不同股票的 β 不同，直接比条件收益会被 β 主导；"
          "减去自身 20 日均值后剩下的是「**相对于自己平时**，在市场偏弱的日子里"
          "表现如何」——这是一个去均值的条件统计量，**不是** β 的重标定。"
          "★★ **条件变量为什么用「上涨家数份额 < 0.5」这个绝对阈值，"
          "而不是滚动分位** —— 这是沙箱 + `main.py check` 抓出来的一个**真实缺陷**："
          "首版写的是「今日宽度变化 ≤ 其自身 20 日窗的 5% 分位」，"
          "看起来是标准的尾部条件化写法，但**在单边上涨行情里"
          "「今天的宽度变化处于自己近 20 日的最差 5%」可以连续几十天一次都不发生**"
          "⇒ 整条截面全 NaN ⇒ `main.py check` 报 **✘CONST**"
          "（实测 2019/2024 各有 69/103 个全 NaN 交易日，2026 也有 29 天）。"
          "这是**构念本身的固有缺陷**，不是实现 bug —— 换成任何自参照的滚动分位阈值"
          "都会有同样的病（趋势市里「今天相对过去很弱」永远不成立）。"
          "改用**绝对水平阈值**后：实测 2012–2026 全样本 `sh_close < 0.5` 占 53.6% 的交易日，"
          "而「连续 ≥0.5」的**最长**区间只有 **8 天** ⇒ 任意 20 日窗内至少 12 天命中，"
          "命中数恒 ≥ 1，**因子在所有交易日都有定义**。"
          "★ 与 `bw_capture_asym_20` 的区别：那个按宽度**变化**的符号分组"
          "（一阶差分），本因子按宽度**水平**是否过半分组 —— "
          "「今天比昨天差」与「今天多数股票在跌」是两个不同的状态。"),
))
def bw_weak_breadth_ret_20(ctx):
    _M = _Mkt(ctx)
    if not _M.ok:
        return ctx.panel.empty()
    r = _ret(ctx)
    weak = _m(_M, "sh_close") < 0.5
    base = ctx.roll_mean(r, 20, 10)
    return _cond_mean(ctx, r, weak, 20, 1) - base


@register(FactorSpec(
    name="bw_overnight_lag_beta_20",
    group="breadth",
    deps=BW_DEPS,
    desc="隔夜跳空 与 昨日宽度变化 的 20 日相关（滞后反应）",
    formula="corr(gap_T, delta_up_share_{T-1}, 20, 10)",
    start=ID_BW_START,
    warmup_days=W_20,
    higher_is_better=True,
    note=(_NOCS + " ★★ **全家族唯一的预测性因子** —— 其余 11 个都是同日（contemporaneous）的。"
          "合法性：`trade_date = T−1` 的宽度在 T−1 的 15:00 就已定稿，"
          "而 T 日的隔夜跳空发生在 T 日 09:25 集合竞价 —— "
          "**用 T−1 收盘后已知的量解释 T 日开盘的跳空，没有用到 T 日的任何信息**。"
          "（与 `fea/delay.py` 的「D 日当晚发布、下游 D+1 交易」同一前提。）"
          "经济含义：> 0 ⇒ 市场昨日的扩散/收敛会在**次日开盘**继续影响这只股票"
          "（滞后反应），< 0 ⇒ 隔夜被过度反应后回吐。"
          "★ 与 `idt_overnight_gap` 的区别：那个是**当日跳空本身**（同期量），"
          "本因子是**跳空与昨日市场状态的关系**（跨期、市场条件）。"
          "隔夜跳空用 `hfq_open(T)/hfq_close(T−1)−1` 并以 `ctx.traded()` 掩码"
          "（停牌日 ffill 会造出假跳空，见 `factors/intraday.py` 的模块 docstring §2）。"),
))
def bw_overnight_lag_beta_20(ctx):
    _M = _Mkt(ctx)
    if not _M.ok:
        return ctx.panel.empty()
    op = np.asarray(ctx.hfq("open"), dtype=np.float64)
    pc = ctx.shift(np.asarray(ctx.hfq("close"), dtype=np.float64), 1)
    gap = ctx.safe_div(op, pc, 1e-8) - 1.0
    gap = np.where(ctx.traded(), gap, np.nan)
    return ctx.roll_corr(gap, _lag(_M.sh_close)[:, None], 20, 10)


# ══════════════════════════════════════════════════════════════════════
# 3. 宽度联动与脆弱性（3 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="bw_breadthvol_response_20",
    group="breadth",
    deps=BW_DEPS,
    desc="|个股收益| 与 当日市场内部翻腾度 的 20 日相关（脆弱性）",
    formula="corr(abs(ret), bw_vol, 20, 10)，bw_vol = 当日分钟内 up 份额的 std",
    start=ID_BW_START,
    warmup_days=W_20,
    higher_is_better=False,
    note=(_NOCS + " **本文件新造**。`bw_vol` 是「当天市场内部来回翻脸的程度」"
          "（分钟内 up 份额的标准差）—— 它与**收益的大小**无关："
          "市场可以收平但全天剧烈震荡（bw_vol 高），也可以单边大涨（bw_vol 低）。"
          "本因子问：这只股票**在市场内部乱的时候会不会跟着乱**。"
          "> 0 ⇒ 高脆弱性（市场一有内部分歧就波动放大）。"
          "与 `vol_of_vol_20`（个股自身波动的波动）的区别："
          "那个是**自身**波动的时间序列变化，本因子是**与市场内部状态的联动**。"
          "★ `bw_vol` 是用分钟内份额算的，不是收益 —— 所以本因子的量纲是"
          "「绝对收益 vs 份额离散度」的相关，同日截面内一致可比。"),
))
def bw_breadthvol_response_20(ctx):
    _M = _Mkt(ctx)
    if not _M.ok:
        return ctx.panel.empty()
    r = _ret(ctx)
    return ctx.roll_corr(np.abs(r), _m(_M, "bw_vol"), 20, 10)


@register(FactorSpec(
    name="bw_tail_comove_60",
    group="breadth",
    deps=BW_DEPS,
    desc="与宽度的尾部共振频率（双方 |z|>1.5 且同向的交易日占比，60 日）",
    formula="z = (x - rolling_mean)/rolling_std; freq(|z_r|>1.5 & |z_m|>1.5 & sign一致) 的 60 日均值",
    start=ID_BW_START,
    warmup_days=W_60,
    higher_is_better=False,
    note=(_NOCS + " 抄参考库 Class1 `tail_corr_60`（把市场收益换成宽度变化）。"
          "★ 用 `roll_mean(布尔, 60, 30)` 而不是自己数个数：框架的 `roll_count`"
          "**不毒化**，所以直接数会给出「0 次」这种看起来有效、其实是「窗口里全是缺失」的值。"
          "★★ 掩码写法：`|z_r|>1.5 & |z_m|>1.5 & sign(z_r)==sign(z_m)` **必须**先要求"
          "两侧 z 都有限，再算真假 —— 否则 `NaN > 1.5` 为 False 会被当成"
          "「不共振」计入分母，把缺失静默摊薄成低频。"
          "判定为 False 时写 0.0（确实不共振），判定为缺失时写 NaN。"
          "方向取负（参考库口径：尾部共振 = 脆弱）。"),
))
def bw_tail_comove_60(ctx):
    _M = _Mkt(ctx)
    if not _M.ok:
        return ctx.panel.empty()
    r, d = _ret(ctx), _dsh(ctx, _M)
    sr = ctx.roll_std(r, 60, 30)
    sm = ctx.roll_std(d, 60, 30)
    zr = ctx.safe_div(r - ctx.roll_mean(r, 60, 30), sr, 1e-12)
    zm = ctx.safe_div(d - ctx.roll_mean(d, 60, 30), sm, 1e-12)
    ok = np.isfinite(zr) & np.isfinite(zm)
    hit = ok & (np.abs(zr) > 1.5) & (np.abs(zm) > 1.5) & (np.sign(zr) == np.sign(zm))
    return ctx.roll_mean(np.where(ok, hit.astype(np.float64), np.nan), 60, 30)


@register(FactorSpec(
    name="bw_session_follow_20",
    group="breadth",
    deps=(*BW_DEPS, "stock_history_5min"),
    desc="上/下午「与市场宽度同向」频率之差（日内跟随的市场一致性）",
    formula="am: sign(am_ret)==sign(d_sh_am); pm: sign(pm_ret)==sign(d_sh_pm); "
            "delta = mean(am,20,10) - mean(pm,20,10)",
    start=ID_BW_START,
    warmup_days=W_20,
    higher_is_better=True,
    note=(_NOCS + " **本文件新造**，它是模块 docstring §三 那条「真·48 点日内相关做不到」的"
          "**可得替代**：日内层不存个股的逐棒路径，个股侧**唯一**能拿到的日内分段"
          "就是上/下午两段（`am_close5/am_open5` 与 `close5/pm_open5`）。"
          "为了让两条腿**同步**，市场侧也切成同样的两段："
          "上午 = `sh_am(11:30) − sh_open(09:30)`，下午 = `sh_close(15:00) − sh_am(11:30)`；"
          "个股侧两个比值是**同日**的（复权因子是常数，无需调整）。"
          "> 0 ⇒ 上午跟得比下午紧（隔夜信息主导的跟随）；"
          "< 0 ⇒ 下午才跟上市场。"
          "★★ 同向判定用 `np.sign(a) == np.sign(b)` 并**先做有限值掩码** —— "
          "`np.sign(NaN)` 是 NaN，`NaN != NaN` 为 True，写成 `!=` 会把缺失日"
          "静默计成「不同向」（这类 bug 已在 `mf_flow_stability_20d` 的 note 里记过一次）。"),
))
def bw_session_follow_20(ctx):
    _M = _Mkt(ctx)
    if not _M.ok:
        return ctx.panel.empty()
    # 个股侧：上/下午两段的同日比值
    f = lambda x: np.asarray(ctx.intraday_field(x), dtype=np.float64)
    am_s = ctx.safe_div(f("am_close5"), f("am_open5"), 1e-8) - 1.0
    pm_s = ctx.safe_div(f("close5"), f("pm_open5"), 1e-8) - 1.0
    # 市场侧：两段的宽度变化
    am_m = (_M.sh_am - _M.sh_open)[:, None]
    pm_m = (_M.sh_close - _M.sh_am)[:, None]

    def agree(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return np.where(ok, (np.sign(a) == np.sign(b)).astype(np.float64), np.nan)

    return (ctx.roll_mean(agree(am_s, am_m), 20, 10)
            - ctx.roll_mean(agree(pm_s, pm_m), 20, 10))


@register(FactorSpec(
    name="bw_intraday_beta_20",
    group="breadth",
    deps=(*BW_DEPS, "stock_history_5min"),
    desc="日内腿的宽度 β = 个股日内收益 对 市场日内宽度变化 的 20 日 β",
    formula="cov(close/open-1, sh_close-sh_open, 20, 10) / var(sh_close-sh_open, 20, 10)",
    start=ID_BW_START,
    warmup_days=W_20,
    higher_is_better=True,
    note=(_NOCS + " 与 `bw_beta_60` 是**同一构念的不同腿与不同期限**："
          "本因子的个股腿是**日内**（`hfq_close/hfq_open − 1`，剔除隔夜跳空），"
          "市场腿也用**日内**（`sh_close − sh_open`，剔除隔夜），"
          "且窗口是 20 日而不是 60 日。"
          "**收盘对收盘**的版本（`bw_beta_60`）含隔夜信息，"
          "而隔夜跳空在 A 股里由完全不同的机制驱动（集合竞价、外盘、公告），"
          "所以剥掉隔夜后的 β 更纯粹地反映「**盘中**跟随市场扩散的能力」。"
          "⚠ 与 `bw_beta_60` 相关性可能较高，`dedup` 阶段定量裁决。"
          "个股腿用 `ctx.hfq`（本项目唯一允许的价格口径），"
          "并以 `ctx.traded()` 掩码挡掉停牌日的 ffill 假收益。"),
))
def bw_intraday_beta_20(ctx):
    _M = _Mkt(ctx)
    if not _M.ok:
        return ctx.panel.empty()
    op = np.asarray(ctx.hfq("open"), dtype=np.float64)
    cl = np.asarray(ctx.hfq("close"), dtype=np.float64)
    ir = ctx.safe_div(cl, op, 1e-8) - 1.0
    ir = np.where(ctx.traded(), ir, np.nan)
    mk = (_M.sh_close - _M.sh_open)[:, None]
    return _beta(ctx, ir, mk, 20, 10)
