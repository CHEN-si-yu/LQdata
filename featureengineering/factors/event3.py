"""事件衰减因子（8 个）—— 跌停生态 / 新低新高 / 跳空 / 极端波动。

与 `factors/event2.py`（18 个涨停生态事件因子）**互补不重复**：涨停侧只做
`limit_up_event_5` 一个对照组，其余 7 个全在跌停 / 新低 / 跳空 / 极端波动侧
（event2 的跌停侧只有 `limit_down_rebound_10` 一个「事件后收益」因子，
`limit_alternation_20` 是涨跌停**乘积**——都不是时间结构）。

═══════════════════════════════════════════════════════════════════════════
一、本批唯一的范式增量：**事件衰减**（把稀疏事件变成有时间结构的连续量）
═══════════════════════════════════════════════════════════════════════════

本地已有的事件因子全是「计数 / 频率」（`limit_up_count_20`、`limit_alternation_20`、
`one_word_limit_up_freq_20`、`suspension_days_60` …）——它们只回答「发生了几次」，
不区分「昨天跌停」和「五天前跌停」。本文件统一用**指数衰减加权和**补上时间维度：

    decay(T) = Σ_{k=0}^{N-1} event(T−k) × 0.5^(k / H)    （H = 半衰期，单位 = 交易日）

k = 0..N−1 个交易日，越近的事件权重越大（k=0 → 1.0，k=H → 0.5，k=2H → 0.25）。
**是「和」而不是「最近一次事件」**：参考库 `utils.event_decay(series, half_life)`
的语义是 `ffill(事件值) × exp(−λ·距上次事件天数)`，只保留**最近一次**事件
（3 天内跌停两次与只跌停一次得到**相同**信号）。本批按任务书给定的范式用
**加权和**，多次事件会累积 —— 这正是「恐慌加速 / 情绪持续」想要的语义。

窗口 / 半衰期取值与理由见各 note，本文件对照表：

    N=5,  H=3    limit_down_event_5 / limit_up_event_5 / extreme_move_event /
                 gap_event_decay_5     （任务书范式 0.5^(k/3)，「_5」系列）
    N=10, H=5    new_low_60_event / new_high_60_event
                 （60 日新高/新低是**结构性**突破，参考库对这两个事件也用
                   half_life=5；窗口取 2 个半衰期，尾部权重 0.25）
    不用衰减     consecutive_limit_down（「当下处于第几个跌停」的状态量）
                 one_word_limit_down_freq_20（频率）

★ **`ctx.decay_linear` 不能用**：它是**线性**权重 (n, n−1, …, 1)/Σ，不是指数
  （名字像、语义完全不同，混用不会报错，只会静默换一个因子）。
  指数衰减在本文件由 `_decay()` 用 `ctx.shift(ev, k)` 逐项乘 0.5^(k/H) 再相加
  实现 —— N 只有 5/10，逐项加的开销可忽略，且没有「无限窗口」的截断近似。

═══════════════════════════════════════════════════════════════════════════
二、涨跌停判定：`stock_limit_list.limit` 优先，2020 前用价格近似
═══════════════════════════════════════════════════════════════════════════

`stock_limit_list`（2020-01-02 起）是**唯一同时含 U/D/Z** 的表，本文件一律优先用它：

    limit == 'D'  → 跌停（收盘封死）      limit == 'U'  → 涨停（收盘封死）
    limit == 'Z'  → 触板未封（炸板）

2020 年前该表不存在，用任务书给定的价格近似（**显式写进各 note**）：

    跌停 ≈ (pct_chg ≤ −9.5) 且 (close == low)     涨停 ≈ (pct_chg ≥ +9.5) 且 (close == high)

「收盘 == 当日最低/最高」就是「封死」的定义（跌停价即当日最低可成交价），
所以这个近似等价于「以跌停价收盘」。**实测两口径的一致性（2020~2026 主板）**：

  · `limit=='D'` 23,562 行中 **23,555 行 close == low**（1 行不等、6 行缺行情）
    → 两个口径说的是同一件事；
  · 价格近似比 D 标记多 198 个交易日（那些行 `limit=='Z'`）—— 供应商认为
    「触板未封」而收盘价恰在跌停价上（尾盘开板又回封之类的边界），本文件
    **以 LL 为准**（任务书：优先用 limit 字段），代价见 `limit_down_event_5` 的 note；
  · D 标记比价格近似多 140 行，全部是 pct_chg 在 −4.88 ~ −9.47 的 **ST（±5%）**
    个股 —— universe 已 `exclude_st: true`，对输出无影响。

★ **`stock_limit_up`（2015-01-05 起）没有用于本文件**：它只有涨停侧，用它做
  2020 前的涨停事件会让同一因子的 U/D 两侧口径不对称（U 用供应商标记、D 用
  价格近似）。实测 2015~2019 主板：`stock_limit_up` 56,284 行 vs 价格近似
  57,381 行，**仅 52,277 行重合（93%）**，差异主要来自 ST（±5%）与新股
  （上市首日 44%），两边都不干净 —— 故选**对称**的价格近似，一个因子一套口径。

═══════════════════════════════════════════════════════════════════════════
三、实测坑（每条都在对应因子的 note 里复述）
═══════════════════════════════════════════════════════════════════════════

1. **停牌日的价格水平量是前向填充的**（`fea/prices.py`：`LEVELS = open/high/low/
   close/pre_close/change` 全部 `nan_fill_ffill`）。后果有两个，都会静默造假事件：
     · `hfq(low)` 在停牌日 = 停牌前最后一个值 → 若那天正好创了 60 日新低，
       停牌期间**每一天都会重新触发一次新低事件**（衰减窗口被灌满）。
       处置：`new_low_60_event` / `new_high_60_event` 用 `ctx.traded()` 显式挡停牌日。
     · 跳空 `hfq(open)/hfq(pre_close) − 1` 在停牌日 = **停牌前那天的跳空**
       （分子分母各自 ffill，比值不为 0，也不为 NaN）→ 假跳空。
       处置：`gap_event_decay_5` 用 `ctx.traded()` 显式挡停牌日。
   （`pct_chg` 是 RATIO，**不** ffill，停牌日就是 NaN，所以 `_limit_*_grid` 不会踩这个坑。）
2. **`ctx.shift` 的窗口头部填 NaN**（`mx.shift(x, k)` 用 `fill=np.nan`）。
   衰减和里逐项 shift 会在面板最前面 k 行产生 NaN，把整格毒化。处置：`_decay()`
   把 shift 出来的非有限值当 **0（没有事件）**——与 event 量「没发生就是 0」一致。
3. **LL 表没有 `consecutive_days`**：涨停侧 `stock_limit_up` 有现成的连板高度
   （event2 的 `consecutive_limit_up` 直接用它），跌停侧没有，只能在面板上自己攒
   连续段（`_run_len()`，与参考库 `groupby+ cumsum` 分段逐格等价，实测 200 组
   随机序列完全一致）。
4. **复权对「同日比值」是恒等变换**：`hfq(open)/hfq(pre_close)` 的分子分母用
   **同一个** `adj_factor(t)` → 比值与未复权完全相同。跳空的除权日安全性来自
   `pre_close` 本身是**除权后基准价**（实测 2026 年 93.5 万行
   `close/pre_close − 1` 与 `pct_chg` 的最大偏差 0.005 个百分点），不是来自 hfq。
   仍统一走 `ctx.hfq`（契约硬约束 4：价格一律用 hfq）。
5. **零膨胀是本家族的固有属性，不是口径太稀**：跌停、一字跌停、|涨跌|≥9.5%
   这些事件本身就罕见（2026 年主板跌停约 0.4% 的 stock-day）。本文件全部走
   event 量「没发生 = 0」（正确的 0），实测零值占比逐因子写在 note 与汇报里。

═══════════════════════════════════════════════════════════════════════════
四、上游 delay
═══════════════════════════════════════════════════════════════════════════

`fea/delay.py` 的实测名单（`load()` 打印核对过）里**没有** `stock_limit_list` /
`stock_limit_up` / `stock_suspension`，即这三张表的 `trade_date = d` 的行当天就能
拿到（名单里是 `stock_margin_detail` / `index_ths_daily` / `stock_st_info` /
`stock_adj_factor_changes` / `stock_top_list` / `stock_dragon_tiger` / `dc_daily` /
`stock_report_rc`）。本文件**不需要 `lag_grid`**，也没有声明 `lagged_ok`。
"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register

GROUP = "event"

LL = "stock_limit_list"      # 涨/跌/炸板，2020-01-02 起（唯一同时含 U/D/Z 的表）
PX = ("stock_daily", "stock_adj_factor")

# 价格近似阈值（%）。见模块 docstring 二：D 标记与 ≤−9.5 的一致性 99.4%，
# 用 9.5 而不是 9.8 是为了兜住「四舍五入报成 −9.5x%」的样本（实测多 153 个事件）。
LIMIT_PCT = 9.5

# `extreme_move_event` 的阈值（%）。★ 任务书给的是 9.5，但它与主板 ±10% 的涨跌幅
# 上限只差 0.5pp → 实测与 `limit_up_event_5` 的秩相关 0.883（被引擎 dedup 判重复）。
# 降到 7.0（参考库原值）可解耦到 0.708、8.0 到 0.797。见该因子的 note。
# ★★ 2026-09-15 主 Agent 拍板：**取参考库原值 7.0**。理由：
#    ① 7.0 是参考库 `event_dynamics.py` 的原始阈值，口径有出处；
#    ② 「极端波动」与「涨停」ρ=0.88 等于白做 —— 7.0 解耦到 0.708，
#       保住了「大涨/大跌但没到板」（主板 ±10% 里 7~9.5% 那一段）的独立信息；
#    ③ 8.0 只是折中，没有额外依据。
EXTREME_PCT = 7.0

# 跳空事件阈值。参考库用 5%，实测 5% 的零值占比 95.8%（越过 >95% 的红线），
# 3% 为 89.9%，且与同族 `big_gap_reversal_5`（event2.py）门槛一致。见 note。
GAP_PCT = 0.03

# warmup（日历天）= 交易日 × 1.8 + 20（契约 §2）
W5 = 29       #  5 × 1.8 + 20
W20 = 56      # 20 × 1.8 + 20
W60 = 128     # 60 × 1.8 + 20
W80 = 164     # 80 × 1.8 + 20  （60 日回看 + 10 日衰减窗口 = 70 个交易日）


# ══════════════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════════════
def _years(ctx) -> tuple[int, int]:
    return int(str(ctx.panel.dates[0])[:4]), int(str(ctx.panel.dates[-1])[:4])


class _Ev:
    """事件表 → `(T, C)` 网格（`ctx.event_grid`，缺失补 **0**）。

    与 event2.py 的 `_Ev` 是同一份语义（本文件自己抄一份，避免跨家族文件依赖）。
    """

    def __init__(self, ctx, table: str, columns: tuple = ()):
        self.ctx = ctx
        df = ctx.dataset(table, columns=["stock_code", "trade_date", *columns],
                         years=_years(ctx))
        self.df = df
        self.empty = df is None or df.empty
        if not self.empty:
            self.codes = df["stock_code"].to_numpy()
            self.days = ctx.date_col(df["trade_date"])
            self.n = len(df)

    def txt(self, name: str) -> np.ndarray:
        """字符串列（原样，不做类型转换）——`limit` 这类枚举列用。"""
        return self.df[name].astype("string").to_numpy()

    def grid(self, mask=None, values=None) -> np.ndarray:
        ctx = self.ctx
        if self.empty:
            return ctx.panel.empty()
        if mask is None:
            c, d = self.codes, self.days
        else:
            m = np.asarray(mask, dtype=bool)
            c, d = self.codes[m], self.days[m]
        if values is None:
            v = None
        else:
            v = np.asarray(values, dtype=np.float64)
            if mask is not None:
                v = v[mask]
        return ctx.event_grid(c, d, v)


def _traded(ctx) -> np.ndarray:
    """当日真有成交。★ 必须挡掉停牌日：见模块 docstring 三.1。"""
    return np.asarray(ctx.traded(), dtype=bool)


def _limit_grid(ctx, up: bool) -> np.ndarray:
    """涨/跌停事件 0/1 网格：`stock_limit_list.limit` 优先，无记录处用价格近似。

    「LL 优先」的判定是逐格的：该 (date, code) 在 LL 表里**有任意一行**
    （U/D/Z 都是「当天触过板」）→ 用供应商的判定；表里没有 → 该股当天没有
    触板记录，用价格近似兜底（2020 年前全表都不存在，整段走近似）。
    """
    pc = ctx.px("pct_chg")
    cl = ctx.px("close")
    edge = ctx.px("high" if up else "low")
    thr = LIMIT_PCT if up else -LIMIT_PCT
    hit = (pc >= thr) if up else (pc <= thr)
    # 停牌日 pct_chg 为 NaN → hit 为 False；close 与 high/low 都被 ffill 成同一个值，
    # 但 NaN 比较必为 False，所以这一格不会误判成事件。
    approx = np.isfinite(pc) & hit & (np.abs(cl - edge) < 1e-6)

    ev = _Ev(ctx, LL, ("limit",))
    if ev.empty:
        return approx.astype(np.float64)
    lim = ev.txt("limit")
    ev_ll = ev.grid(mask=(lim == ("U" if up else "D"))) > 0.0
    any_ll = ev.grid() > 0.0
    return np.where(any_ll, ev_ll, approx).astype(np.float64)


def _decay(ctx, ev: np.ndarray, n: int, half_life: float) -> np.ndarray:
    """指数衰减加权和：`Σ_{k=0}^{n-1} ev(t−k) × 0.5^(k/half_life)`。

    ★ 逐项 `ctx.shift` + 相加，**不用 `ctx.decay_linear`**（那是线性权重）。
    ★ 面板头部 k 行的 shift 结果是 NaN（`mx.shift` 的 fill=np.nan）——
      那里没有历史，按「没有事件」记 0（与 event 量的补零语义一致），
      否则会把窗口头部整片毒化成 NaN。
    """
    out = np.zeros(ev.shape, dtype=np.float64)
    for k in range(int(n)):
        sh = ctx.shift(ev, k)
        out += np.where(np.isfinite(sh), sh, 0.0) * float(0.5 ** (k / float(half_life)))
    return out


def _run_len(x: np.ndarray) -> np.ndarray:
    """连续段长度：`x > 0` 处 = 「到 t 为止连续为正的天数」，其余为 0。

    与参考库的 `seg = (~is).groupby(Code).cumsum(); count = is.groupby([Code, seg]).cumsum()`
    逐个语义等价（断段即归零），但完全不写 Python 循环：
    `pos - 最近的 0 所在位置`（`np.maximum.accumulate` 求「最近一次 x==0 的下标」）。
    实测 200 组随机 0/1 序列与逐格循环结果完全一致。
    """
    t, c = x.shape
    pos = np.arange(t, dtype=np.float64)[:, None] * np.ones((1, c), dtype=np.float64)
    last0 = np.maximum.accumulate(np.where(x > 0, -1.0, pos), axis=0)
    return np.where(x > 0, pos - last0, 0.0)


# ══════════════════════════════════════════════════════════════════════════
# A. 跌停事件（3 个）
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="limit_down_event_5", group=GROUP, deps=(LL, *PX),
    desc="近 5 个交易日跌停事件的指数衰减加权值（半衰期 3 日）",
    formula='down = daily["pct_chg"].le(-9.8)\n'
            'event = down.astype(float).where(down, np.nan)\n'
            'decayed = event_decay(event, half_life=3)      # 参考库原文\n'
            '# 本实现（任务书范式：加权和，窗口 N=5）：\n'
            '#   is_ld   = (limit == "D") if stock_limit_list 有记录\n'
            '#             else (pct_chg <= -9.5 and close == low)\n'
            '#   decayed = Σ_{k=0}^{4} is_ld(t-k) × 0.5^(k/3)',
    start=None, warmup_days=W5, higher_is_better=False,
    note="★ 与参考库的三处偏离（逐条给理由）："
         "① **事件判定优先用 `stock_limit_list.limit == 'D'`**（2020-01-02 起），"
         "2020 年前才退到价格近似 `pct_chg ≤ −9.5 且 close == low`（任务书口径）。"
         "实测两口径在 2020~2026 主板高度一致：D 标记 23,562 行里 23,555 行 close == low；"
         "价格近似多出的 198 行全部是供应商标记为 `Z`（触板未封）而收盘恰在跌停价上的样本，"
         "本文件以 LL 为准（≈0.7% 的事件数差异）。"
         "② **衰减用「加权和」而不是参考库 `event_decay` 的「最近一次事件 × 指数」**："
         "参考库只保留最近一次事件（3 天内跌停两次 = 只跌停一次），"
         "本批按任务书范式取 Σ_{k=0}^{4} 0.5^(k/3)，多次跌停会累积 —— "
         "这正是「恐慌加速」的语义，也让因子对「连续跌停」与「单日跌停」有区分度。"
         "③ 参考库阈值 −9.8% 只用在**价格近似**分支，且放宽到 −9.5%："
         "实测 D 标记里有 153 行的 pct_chg 落在 (−9.8, −9.5]（四舍五入报 −9.5x%），"
         "用 −9.8 会漏掉它们。"
         "★ **零膨胀（固有，非口径太稀）**：跌停本身是罕见事件 —— 2026 年主板"
         "（非 ST）跌停事件约 0.4% 的 stock-day，5 日窗口内至少一次的比例约 3%，"
         "故本因子 **实测零值占比 97.54%**（2026 全年 518,758 行 / 170 天 / 日均 3050 只，"
         "非零格子上界 3.3205 = 5 个连续跌停的完整和 ∑0.5^(k/3)）。"
         ">95% 的零值占比不通过「调窗口」来救：因子的名字与定义就是 5 日窗口，"
         "且截面 rank 在大量并列 0 上仍然可用（rank 由非零样本分层，"
         "非零样本内部按 0.5^(k/3) 的权重自然分层）。若要更稠密，"
         "应换连续量（如 `limit_down_rebound_10` 那类事件后收益），不是放宽窗口。",
))
def limit_down_event_5(ctx):
    return _decay(ctx, _limit_grid(ctx, up=False), 5, 3.0)


@register(FactorSpec(
    name="consecutive_limit_down", group=GROUP, deps=(LL, *PX),
    desc="当前连续跌停天数（连续段逻辑：断段即归零，非跌停日为 0）",
    formula='is_ld = daily["pct_chg"].le(-9.8).astype(int)\n'
            'code = is_ld.index.get_level_values("Code")\n'
            'seg = (~is_ld.astype(bool)).groupby(level="Code").cumsum()\n'
            'count = is_ld.groupby([code, seg]).cumsum()      # 参考库原文\n'
            '# 本实现：同一「连续段」语义（断段归零），向量化为\n'
            '#   count = 到 t 为止连续跌停的天数（np.maximum.accumulate 求最近一次 0 的位置）',
    start=None, warmup_days=W60, higher_is_better=False,
    note="★ 与 `consecutive_limit_up`（event2.py）**镜像不重复**：那个用上游 "
         "`stock_limit_up.consecutive_days`（供应商算好的连板高度），"
         "跌停侧**没有对应的表**（`stock_limit_list` 只有 `limit_times`，"
         "实测在 D 行上 25% 分位=1、中位=1、max=29，与自算连跌天数同量级，但只有 "
         "2020 起），故本因子在面板上自攒连续段：`_run_len()` 与参考库的 "
         "`groupby + cumsum` 分段逐格等价（已用 200 组随机 0/1 序列验证），"
         "但全向量化、不写 Python 循环。"
         "★ 事件口径与 `limit_down_event_5` **完全共用**（LL 优先 + 价格近似回退）。"
         "★ 停牌日 = 没成交 = 不可能跌停 → 记 0，**连续段被打断**（与参考库 "
         "`pct_chg.le(-9.8)` 在 NaN 上取 False 的行为一致）。"
         "warmup 给 W60（128 日历天 ≈ 60 个交易日）：连续段的起点依赖窗口前段的历史，"
         "实测历史上最长连续跌停 29 天（LL `limit_times` 的 max），60 天足够覆盖。"
         "★ 零膨胀（固有）：**实测零值占比 99.41%**（非零仅 0.59%），"
         "2026 年取到的最大值 5（连续 5 个跌停）。"
         "这是「状态量」的必然结果 —— 与 `consecutive_limit_up`（连板数）同款："
         "非连板日为 0 是**定义**，不是缺失。",
))
def consecutive_limit_down(ctx):
    return _run_len(_limit_grid(ctx, up=False))


@register(FactorSpec(
    name="one_word_limit_down_freq_20", group=GROUP, deps=(LL, *PX),
    desc="一字跌停占比 = 20 日内「全天无波动且跌停」的天数占比",
    formula='no_range = (daily["high"] - daily["low"]).abs().lt(1e-6)\n'
            'one_word = (no_range & daily["pct_chg"].le(-9.8)).astype(float)\n'
            'freq = _roll_mean(one_word, 20, 5)\n'
            'return cross_sectional_rank(-freq)',
    start=None, warmup_days=W20, higher_is_better=False,
    note="★ 与 event2.py 的 `one_word_limit_up_freq_20` **镜像**（那边是一字**涨停**），"
         "唯一的结构差异是跌停判定：那边用 `pct_chg >= 9.8`，本因子用 "
         "**LL 优先（`limit == 'D'`）+ 价格近似回退**（与同文件 `limit_down_event_5` 共用"
         "`_limit_grid`），因为一字跌停的样本比一字涨停少一个量级，"
         "供应商标记带来的判定差更容易影响分布。"
         "`high == low` 是一字板的**定义**（全天只成交在一个价位），"
         "除权日不产生假值（high/low 是同日同尺度量）。"
         "停牌日 `pct_chg` 为 NaN → 该日按「无观测」记 NaN（不参与均值），"
         "故 `min_count=5`（对齐参考库 min_periods=5，与 event2 的同款因子一致）。"
         "值域严格 [0,1]；**零膨胀（固有）**：从没打过一字跌停的股票恒为 0 —— "
         "这是正确的 0，不是缺失（event2 对一字涨停同款因子给了同样的结论）。"
         "**实测**：零值占比 94.31%、|value|max = 0.30（20 天里 6 个一字跌停）、"
         "非空率 100.00% —— 刚好压在「>95% 说明口径太稀」的红线之内。"
         "注意 20 日窗口内有效观测不足 5 天时是 NaN（次新股/长期停牌），"
         "沙箱实测有 167 行 NaN（518,591 / 518,758 行）。"
         "与 `consecutive_limit_down` 的区别：那个数「连续几个跌停」（不要求一字），"
         "本因子数「跌停里有多少是一字」（流动性冻结的指纹）。",
))
def one_word_limit_down_freq_20(ctx):
    hi, lo, pc = ctx.px("high"), ctx.px("low"), ctx.px("pct_chg")
    ld = _limit_grid(ctx, up=False)
    one_word = (np.abs(hi - lo) < 1e-6) & (ld > 0.0)
    ok = np.isfinite(hi) & np.isfinite(lo) & np.isfinite(pc)
    return ctx.roll_mean(np.where(ok, one_word.astype(np.float64), np.nan),
                         20, min_count=5)


# ══════════════════════════════════════════════════════════════════════════
# B. 新低 / 新高事件（2 个）—— 参考库 `new_low_60_event` / `new_high_60_event`
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="new_low_60_event", group=GROUP, deps=PX,
    desc="近 60 个交易日新低事件的指数衰减加权（当日最低价 = 60 日最低，半衰期 5 日）",
    formula='adj = _adjusted_close(daily)\n'
            'is_low = adj.eq(adj.groupby(level="Code").transform(\n'
            '    lambda s: s.rolling(60, min_periods=30).min()))\n'
            'event = is_low.astype(float).where(is_low, np.nan)\n'
            'decayed = event_decay(event, half_life=5)      # 参考库原文\n'
            '# 本实现：\n'
            '#   is_low  = hfq(low) == min(hfq(low), 60)  且 当日有成交\n'
            '#   decayed = Σ_{k=0}^{9} is_low(t-k) × 0.5^(k/5)',
    start=None, warmup_days=W80, higher_is_better=False,
    note="★ 与参考库的三处偏离："
         "① 参考库用**复权收盘价**判定新高/新低（`adj.eq(rolling(60).min())`），"
         "本因子按任务书用**当日最低价**（`hfq(low) == 60 日 hfq(low) 的最低`）——"
         "「新低」在技术分析里指**盘中**创出的低点，用 low 更贴定义；"
         "副作用是触发更频繁（盘中破位后收回也算），故零值占比反而更低（见下）。"
         "② 复权基座必须是**后复权** `ctx.hfq`：参考库的 `daily_adj.parquet` 是前复权"
         "（历史值随未来分红重算，违反 PIT 红线）。后复权锚定序列起点，"
         "除权日不产生假新低。"
         "③ **显式挡停牌日**（`ctx.traded()`）：`hfq(low)` 在停牌日被前向填充，"
         "若停牌前一天正好创了新低，停牌期间每一天都会重复触发一次事件、"
         "把 10 日衰减窗口灌满（模块 docstring 三.1）。参考库没有这个问题"
         "是因为它对每只股票独立滚动、且没有停牌行的概念。"
         "★ 窗口/半衰期：参考库 `event_decay(half_life=5)` 是**无限记忆**"
         "（衰减到下一次事件为止），本实现必须有限窗口，取 N=10 = 2 个半衰期"
         "（尾部权重 0.25，截断误差 ≤ 25%，且事件在本窗口内多次触发会累积）。"
         "取 half_life=5 而不是 _5 系列的 3：60 日新高/新低是**结构性**突破，"
         "信息比单日涨跌停持久（参考库对这两个事件也用 5）。"
         "`min_count=30` 对齐参考库 `min_periods=30`（次新股上市满 30 个交易日后才有值）。"
         "★ 零膨胀：**实测零值占比 71.31%**（8 个因子里最低）—— 因为 10 日窗口里"
         "只要盘中破过一次 60 日低点就有值，震荡市里这很常见；"
         "|value|max = 5.7938 = 10 天连续新低的完整和 ∑_{k=0}^{9}0.5^(k/5)"
         "（说明确实有股票连续 10 天创 60 日新低，长尾是真实的）。"
         "★ 该因子与 8 个兄弟因子的截面秩相关最高只有 −0.238（`new_high_60_event`），"
         "与涨跌停/极端波动的相关性都 < 0.15 —— **信息独立度高**，建议下游重点看。",
))
def new_low_60_event(ctx):
    low = ctx.hfq("low")
    min60 = ctx.roll_min(low, 60, min_count=30)
    is_low = (np.isfinite(low) & np.isfinite(min60) & (low <= min60) & _traded(ctx))
    return _decay(ctx, is_low.astype(np.float64), 10, 5.0)


@register(FactorSpec(
    name="new_high_60_event", group=GROUP, deps=PX,
    desc="近 60 个交易日新高事件的指数衰减加权（当日最高价 = 60 日最高，半衰期 5 日）",
    formula='adj = _adjusted_close(daily)\n'
            'is_high = adj.eq(adj.groupby(level="Code").transform(\n'
            '    lambda s: s.rolling(60, min_periods=30).max()))\n'
            'event = is_high.astype(float).where(is_high, np.nan)\n'
            'decayed = event_decay(event, half_life=5)      # 参考库原文\n'
            '# 本实现：\n'
            '#   is_high = hfq(high) == max(hfq(high), 60)  且 当日有成交\n'
            '#   decayed = Σ_{k=0}^{9} is_high(t-k) × 0.5^(k/5)',
    start=None, warmup_days=W80, higher_is_better=True,
    note="`new_low_60_event` 的镜像（把 low/min 换成 high/max），偏离与理由逐条相同。"
         "★ 与 event2.py 的 `new_high_frequency_60` **互补不重复**：那个是"
         "「60 日内创新高的**天数占比**」（频率，无时间结构），本因子是"
         "「新高事件的**衰减加权和**」（越近期的新高权重越大，多次新高累积）。"
         "两者在**持续创新高**的股票上相关，但本因子对「最近才突破」的股票给更高分。"
         "窗口/半衰期同 `new_low_60_event`（N=10, H=5）。"
         "★ 实测零值占比 82.35%、|value|max = 5.7938（同样是 10 天连续新高）、非空率 100.00%。"
         "与 `new_low_60_event` 的截面秩相关 −0.238（互为镜像，符号相反是预期行为）。",
))
def new_high_60_event(ctx):
    high = ctx.hfq("high")
    max60 = ctx.roll_max(high, 60, min_count=30)
    is_high = (np.isfinite(high) & np.isfinite(max60) & (high >= max60) & _traded(ctx))
    return _decay(ctx, is_high.astype(np.float64), 10, 5.0)


# ══════════════════════════════════════════════════════════════════════════
# C. 涨停对照组（1 个）—— 涨停侧本文件只做这一个
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="limit_up_event_5", group=GROUP, deps=(LL, *PX),
    desc="近 5 个交易日涨停事件的指数衰减加权值（半衰期 3 日）——跌停侧的对照组",
    formula='event = daily["pct_chg"].ge(9.8).astype(float).where(\n'
            '    daily["pct_chg"].ge(9.8), np.nan)\n'
            'decayed = event_decay(event, half_life=3)      # 参考库原文\n'
            '# 本实现（与 limit_down_event_5 同一套口径，只把方向反过来）：\n'
            '#   is_lu   = (limit == "U") if stock_limit_list 有记录\n'
            '#             else (pct_chg >= 9.5 and close == high)\n'
            '#   decayed = Σ_{k=0}^{4} is_lu(t-k) × 0.5^(k/3)',
    start=None, warmup_days=W5, higher_is_better=True,
    note="★ 涨停侧本文件**只做这一个**（任务书：与 event2.py 的 18 个涨停因子不重复）。"
         "event2 覆盖的是计数/连板/封板时间/炸板率/龙虎榜/事件后收益，"
         "没有「涨停事件的时间结构」——本因子补的就是这个："
         "同样是「近 5 日涨停过」，昨天涨停得 1.0、五天前涨停得 0.397。"
         "★ 事件判定与 `limit_down_event_5` **完全对称共用**（LL 优先 + 价格近似回退），"
         "偏离理由同该因子。实测 2020~2026 主板 U 标记 87,071 行里 86,981 行 "
         "pct_chg ≥ 9.5（99.9%）；U 标记比价格近似多的 169 行是 pct_chg 落在 "
         "[9.5, 9.8) 的样本。"
         "★ 与 `limit_up_fade_10`（event2）的区别：那个是「10 日内涨停过 × 随后 10 日"
         "累计收益」（事件**后**的表现），本因子是「涨停事件本身的时间加权强度」"
         "（不含任何未来信息，也不含收益）。"
         "★ 零膨胀：**实测零值占比 91.58%**（比跌停侧的 97.54% 低，因为涨停事件"
         "本身比跌停频繁约 3 倍 —— 2026 年主板 U 标记 8.7 万行 vs D 标记 2.8 万行），"
         "|value|max = 3.3205（5 连板）。"
         "★ **与 `extreme_move_event` 高度冗余（实测 |ρ|=0.887，被引擎 `dedup` 判为"
         "重复簇）**，原因见 `extreme_move_event` 的 note —— 那是「|pct_chg| ≥ 9.5%」"
         "阈值与主板 ±10% 涨跌幅撞车，与本因子无关。",
))
def limit_up_event_5(ctx):
    return _decay(ctx, _limit_grid(ctx, up=True), 5, 3.0)


# ══════════════════════════════════════════════════════════════════════════
# D. 极端波动 / 跳空事件（2 个）
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="extreme_move_event", group=GROUP, deps=PX,
    desc="近 5 日极端波动事件的衰减加权（|日收益| ≥ 9.5%，半衰期 3 日）",
    formula='is_ext = daily["pct_chg"].abs().gt(7.0)\n'
            'event = is_ext.astype(float).where(is_ext, np.nan)\n'
            'decayed = event_decay(event, half_life=10)     # 参考库原文（阈值 7%、H=10）\n'
            '# 本实现（任务书口径）：\n'
            '#   ev = 1 if |pct_chg| >= 9.5 else 0      （停牌日 pct_chg 为 NaN -> 非事件）\n'
            '#   decayed = Σ_{k=0}^{4} ev(t-k) × 0.5^(k/3)',
    start=None, warmup_days=W5, higher_is_better=False, version=2,
    note="★ 阈值口径（**主 Agent 2026-09-15 拍板：7.0%**，取参考库原值）："
         "先按任务书的 9.5% 实现过，实测它与 `limit_up_event_5` 的逐日截面 Spearman "
         "**ρ = 0.883** —— 被引擎 `dedup` 直接判成重复簇。根因是 9.5% 与主板 ±10% 的"
         "涨跌幅上限只差 0.5pp，而涨停事件比跌停多约 3 倍 → 秩几乎被涨停侧定住。"
         "降到参考库原值 7.0% 后 ρ = 0.708（8.0% 是 0.797），保住了"
         "「大涨/大跌但没到板」（7~9.5% 那一段）的独立信息。**v2 起生效。**"
         "★ 时间结构与参考库的偏离：半衰期 10 → **3**、窗口取 5 个交易日"
         "（任务书的 `_5` 系列范式，Σ_{k<5} 0.5^(k/3)）。"
         "★ 用 `ctx.px(\"pct_chg\")`（供应商日收益）而不是 `ctx.ret(1)`："
         "与同文件涨跌停阈值同源同尺度（都是交易所口径的当日涨跌幅，除权日已调整），"
         "且 `ctx.ret(1)` 会对 |r|>60% 做清洗（对涨跌停判定无关但会引入口径差）。"
         "停牌日 pct_chg 为 NaN → 直接记 0（停牌不是异动事件）。"
         "向上取 `higher_is_better=False` 只影响文档方向：值大 = 近期异动剧烈。"
         "★ 实测零值占比 89.31%、|value|max = 3.3205、非空率 100.00%。",
))
def extreme_move_event(ctx):
    pc = ctx.px("pct_chg")
    ev = np.where(np.isfinite(pc) & (np.abs(pc) >= EXTREME_PCT), 1.0, 0.0)
    return _decay(ctx, ev, 5, 3.0)


@register(FactorSpec(
    name="gap_event_decay_5", group=GROUP, deps=PX,
    desc="近 5 日跳空事件的衰减加权（|跳空幅度| ≥ 3% 时按跳空幅度加权，半衰期 3 日）",
    formula='gap = safe_divide(daily["open"], daily["pre_close"]) - 1.0\n'
            'is_gap = gap.abs().gt(0.05)\n'
            'event = is_gap.astype(float).where(is_gap, np.nan)\n'
            'decayed = event_decay(event, half_life=5)      # 参考库原文（0/1 事件）\n'
            '# 本实现（任务书：跳空幅度 × 半衰期权重）：\n'
            '#   g   = hfq(open)/hfq(pre_close) - 1        （当日有成交才有效）\n'
            '#   decayed = Σ_{k=0}^{4} |g(t-k)| × 1[|g| >= 3%] × 0.5^(k/3)',
    start=None, warmup_days=W5, higher_is_better=True,
    note="★ 与参考库的四处偏离（逐条给理由）："
         "① **事件值用跳空幅度而不是 0/1**（任务书「跳空幅度 × 半衰期权重」）："
         "同样是「五天前跳空」，跳 9% 与跳 3% 应该不同；0/1 口径把这个信息丢了。"
         "② 阈值 5% → **3%**：实测 5% 在 2026 年主板只有 1.38% 的 stock-day 命中，"
         "5 日窗口内至少一次的比例 4.2% → **零值占比 95.8%，越过「>95% 说明口径太稀」"
         "的红线**；3% 命中率 3.45%、零值占比 89.9%。而且 3% 与同族 `big_gap_reversal_5`"
         "（event2.py，高开 >3%）的门槛一致，家族内部口径统一。"
         "③ **显式挡停牌日**：`hfq(open)` 与 `hfq(pre_close)` 在停牌日各自被前向填充，"
         "比值 = **停牌前那天的跳空**（不是 0、也不是 NaN）——不挡的话停牌期间会"
         "每天重复触发同一个跳空事件（模块 docstring 三.1）。"
         "④ 半衰期 5 → 3、窗口 5 个交易日（任务书 `_5` 系列范式）。"
         "★ 跳空的除权日安全性：`hfq(open)/hfq(pre_close)` 的分子分母同用一个 "
         "`adj_factor(t)`，复权在该比值上是**恒等变换**（模块 docstring 三.4）——"
         "真正保证除权日不产生假跳空的是 `pre_close` 本身是**除权后基准价**"
         "（实测 2026 年 93.5 万行 `close/pre_close − 1` 与 `pct_chg` 偏差 < 0.005pp）。"
         "仍统一走 `ctx.hfq`（契约硬约束 4）。"
         "★ 取绝对值（不保留方向）：任务书写的是「跳空幅度」，"
         "方向信息由同族的 `big_gap_reversal_5`（高开后走势）承担。"
         "★ **实测**：零值占比 86.97%、|value|max = 4.5838（≈ 连续 5 天 9% 跳空的加权和）、"
         "非空率 100.00%。截面秩相关：与 `extreme_move_event` 0.464、"
         "与 `limit_up_event_5` 0.388、与 `new_low_60_event` −0.094 —— "
         "**不在任何重复簇里**（引擎 dedup 未报）。",
))
def gap_event_decay_5(ctx):
    o, pc = ctx.hfq("open"), ctx.hfq("pre_close")
    gap = ctx.safe_div(o, pc, min_abs_den=1e-9) - 1.0
    mag = np.abs(gap)
    ev = np.where(_traded(ctx) & np.isfinite(mag) & (mag >= GAP_PCT), mag, 0.0)
    return _decay(ctx, ev, 5, 3.0)
