"""事件驱动因子（18 个）—— 涨停生态 / 龙虎榜 / 业绩预告 / 质押 / 价格形态事件 / 停牌。

★ 本文件**只新增**因子，不改 `event.py`：`holder_number_chg`（股东户数）与
  `limit_up_count_20`（20 日涨停次数）已经在那边，这里不重复实现。

═══════════════════════════════════════════════════════════════════════════
一、事件量 vs 状态量 —— 本家族唯一会「静默毁掉因子」的选择
═══════════════════════════════════════════════════════════════════════════

`stock_limit_up` / `stock_limit_list` / `stock_top_list` / `stock_dragon_tiger` /
`stock_suspension` 是**事件**：今天涨停/上榜/停牌了没有。用 `ctx.event_grid`
（散点 + **缺失补 0**），稀疏事件才有正确的「没发生 = 0」语义。
`stock_pledge_stat` / `stock_holder_number` 是**状态量**：披露后一直有效，
必须用 `ctx.asof_daily`（前向填充）。用反了的后果：事件当 asof → 退化成
「0 或者一个几个月前的陈旧值」；状态量当事件 → 每天都在跳 0。

`stock_forecast` / `stock_holder_number` / `stock_pledge_stat` 一律按 **ann_date**
（公告日）对齐，不是 end_date（报告期）—— 按 end_date 对齐是前视。

═══════════════════════════════════════════════════════════════════════════
二、实测坑（逐条在对应因子的 note 里复述）
═══════════════════════════════════════════════════════════════════════════

1. `stock_limit_up.first_limit_time` **同列混格式**：`"14:02:36"`（len 8）与
   `"95947"`（= `09:59:47`，丢了冒号的 5 位串）混在一起，另有 14.8% 为 NULL。
   用前必须归一（`_norm_hhmmss`），否则一半样本解析成垃圾。
2. `stock_limit_up.boards` 在 2015–2018 整列为空（2019 才 47%）、`open_count`
   2015 只有 10.9% —— 这两个字段**本文件一个都没用**（原因见「三」）。
3. `stock_limit_up.is_limit_up` 是常量列（唯一值 1），没有任何信息。
4. `stock_dragon_tiger.net_buy_amount` 的**符号不可信**：与 `buy_amount − sell_amount`
   的符号在 32.2% 的行上不一致（实测 178 万行全表）。本文件一律自己算净额。
5. `stock_top_list` 的主键是 `(trade_date, stock_code, reason)`，同一股票同日
   可能有多条上榜原因（实测 17,909 组重复）—— 必须先汇总再算比率，否则分母被切碎。
6. 涨停/跌停阈值用 9.8% 是**安全的**：universe 只含主板（±10% 涨跌幅）且
   `exclude_st: true`（ST 是 ±5%），所以 `|pct_chg| ≥ 9.8` 在面板内就是涨跌停。
   创业板/科创板（±20%）会被 `ctx.code_index` 映射成 −1 直接丢掉。
7. `stock_pledge_stat` **没有 ann_date**（只有 end_date，且 end_date 不规则：
   月末为主，夹杂 05-10 / 09-11 这类日期）—— 见 `pledge_ratio_chg` 的 note。

═══════════════════════════════════════════════════════════════════════════
三、覆盖率为什么天然低（不是 bug，是事件的稀疏性）
═══════════════════════════════════════════════════════════════════════════

涨停、上榜、机构席位、停牌都是**稀疏事件**。本文件有两类因子：

  (a) **频率/计数型**（`one_word_limit_up_freq_20` / `consecutive_limit_up` /
      `limit_alternation_20` / `suspension_days_60` …）：事件网格补 0 + 滚动，
      没发生就是 0。这类因子非空率接近 100%，但**大多数格子是 0**（正确的 0，
      不是缺失）—— 别为了「好看」把 0 抹成 NaN。
  (b) **条件定义型**（`limit_up_open_time_score` 要有涨停才有封板时间、
      `top_list_net_rate_20` 要有上榜才有净买率 …）：事件没发生 → 该量**无定义**，
      用 NaN。这类因子非空率 3%~20% 是正常的，是 §契约「事件类因子覆盖率天然低」
      说的那种。逐个的说明写在各自 note 里。

═══════════════════════════════════════════════════════════════════════════
四、与契约 §7 的「非空率 60%~100%」的关系
═══════════════════════════════════════════════════════════════════════════

契约 §7 那条是针对**稠密**因子（价格/财务）的。本家族中 (b) 类达不到 60%，
这是任务书明确认可的（"limit_up* 类因子覆盖 5%~30% 是正常的"），且每个都在
note 里给了理由；(a) 类全部满足。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fea.spec import FactorSpec, register

GROUP = "event"

# ---------------------------------------------------------------- 上游表
LU = "stock_limit_up"        # 涨停（封板）明细，2015-01-05 起
LL = "stock_limit_list"      # 涨/跌/炸板，2020-01-02 起（唯一同时含 U/D/Z 的表）
TL = "stock_top_list"        # 龙虎榜每日汇总，2010 起
DT = "stock_dragon_tiger"    # 龙虎榜机构/营业部席位明细，2013 起
FC = "stock_forecast"        # 业绩预告，2010 起
PS = "stock_pledge_stat"     # 股权质押，2014-12-31 起
SP = "stock_suspension"      # 停牌事件，2010 起

# 受上游起点限制的因子显式写 start（契约 §1 硬约束 3）
LU_START = "2015-01-05"
LL_START = "2020-01-02"
PS_START = "2014-12-31"

# ---------------------------------------------------------------- warmup（日历天）
# 契约 §2：窗口 N 个交易日 -> N × 1.8 + 20；按 ann_date 对齐的 -> 700
W5 = 29      # 5 × 1.8 + 20
W10 = 38     # 10 × 1.8 + 20
W20 = 56     # 20 × 1.8 + 20
W60 = 128    # 60 × 1.8 + 20
ANN = 700    # 公告类：4 季 + ann_date 最长滞后 15 个月

# 主板涨跌停阈值（%）。见模块 docstring 「二.6」
LIMIT_PCT = 9.8


# ══════════════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════════════
def _years(ctx) -> tuple[int, int]:
    return int(str(ctx.panel.dates[0])[:4]), int(str(ctx.panel.dates[-1])[:4])


class _Ev:
    """事件表 → `(T, C)` 网格（缺失补 0）。

    与「状态量」的分工见模块 docstring §一。`grid()` 不带权重时就是「命中计数」，
    带 `values` 时是「命中日的值之和」；后者配合 `safe_div(..., 命中网格)`
    就能把「没发生」还原成 NaN（fundflow.py 同款手法）。
    """

    def __init__(self, ctx, table: str, columns: tuple = (), day: str = "trade_date"):
        self.ctx = ctx
        df = ctx.dataset(table, columns=["stock_code", day, *columns],
                         years=_years(ctx))
        self.df = df
        self.empty = df is None or df.empty
        if not self.empty:
            self.codes = df["stock_code"].to_numpy()
            self.days = ctx.date_col(df[day])     # ★ 停牌表的日期列叫 suspend_date
            self.n = len(df)

    def col(self, name: str) -> np.ndarray:
        return pd.to_numeric(self.df[name], errors="coerce").to_numpy(np.float64)

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


def _roll_sum(ctx, mat: np.ndarray, n: int, min_count: int) -> np.ndarray:
    """带 `min_count` 的滚动求和 —— 「窗口内有效值之和」。

    ★ 框架缺陷（实测）：`ctx.roll_sum` **没有 `min_count` 参数**。
      `fea/context.py` 里 `roll_sum` 定义了**两次**（第 151 行带 min_count、
      第 240 行不带），后一处覆盖前一处 → 传 `min_count=` 直接 `TypeError`；
      `panel.roll_sum` 走的是另一条实现（不带 min_count，NaN 毒化）。
      `factors/{fundflow,liquidity,margin}.py` 各自抄了同一个 helper —— 同类坑。
      不带 min_count 的场景仍建议用 `ctx.roll_sum(x, n)`（与 `mathx.roll_sum`
      的 NaN 策略一致：窗口内出现过 NaN 就毒化）。

    等价性：`mathx.roll_mean` = `roll_sum(x−c0, n, min_count)/roll_count(x,n) + c0`
    （`roll_count` **不毒化**），故 `roll_mean × roll_count` 精确还原「有效值之和」；
    有效值不足 `min_count` 时 mean 已被毒化成 NaN，`NaN × count = NaN`，语义一致。
    """
    return ctx.roll_mean(mat, n, min_count) * ctx.roll_count(mat, n)


def _ret_k(ctx, k: int) -> np.ndarray:
    """k 个交易日的复利收益 = Π(1 + ret(1)) − 1。

    ★ 不用 `ctx.ret(k)`：它对 **k 日累计**收益套了「单日 |r| > 60%」的清洗阈值，
      k 越大误杀越多（实测 k=20 误杀 15,592 格、k=60 误杀 94,562 格，被误杀的全是
      涨得最多的股票）。这正是「涨停后 10 日」这类因子最不能丢的样本。
      本口径 = 单日异常才毒化窗口，与 prices.py 那条阈值注释的**原意**一致
      （momentum.py 已实测两口径在无异常时逐格相等）。
    停牌日 hfq 水平量被前向填充 → ret(1) = 0（价格层既定语义，不要再用 ffill 去修）；
    但**整窗一次都没成交**的格子必须挡掉，否则会凭空造出一个 0.0 的假收益。
    """
    r1 = ctx.ret(1)
    r = np.expm1(ctx.roll_sum(np.log1p(r1), int(k)))
    zero_trade = ctx.roll_sum(np.asarray(ctx.traded(), dtype=np.float64), int(k)) <= 0.0
    return np.where(zero_trade, np.nan, r)


def _by_pct(ctx, op: str, thr: float = LIMIT_PCT) -> np.ndarray:
    """`pct_chg ≷ thr` 的 0/1 网格；**停牌日（pct_chg 为 NaN）置 NaN**。

    停牌日不是「没涨停」，是「没有观测」—— 置 NaN 让滚动窗口显式放松
    `min_count`（与参考库的 `min_periods` 同义），窗口内有效日不足才缺值。
    """
    pc = ctx.px("pct_chg")
    hit = (pc >= thr) if op == "ge" else (pc <= thr)
    return np.where(np.isfinite(pc), hit.astype(np.float64), np.nan)


def _norm_hhmmss(s: pd.Series) -> np.ndarray:
    """把 `first_limit_time` 归一成「当日分钟数」（float），非法/缺失 → NaN。

    实测该列混了三种形态（同一列！）：
      A. `"14:02:36"` / `"9:30:06"`  —— 带冒号，1~2 位小时（161,919 行，84%）
      B. `"95947"` = `09:59:47`     —— 纯数字 5 位，**丢了前导零**（1,204 行）
      C. `"102121"` = `10:21:21`    —— 纯数字 6 位（1,285 行）
    另有极少量脏值（`final_limit_time` 有 5 行 > 15:01）。统一用
    「09:15 ≤ t ≤ 15:30」做值域校验，越界置 NaN，避免把 15:21 这种噪声算进打分。

    ★ 覆盖率是**分年份**的（实测解析成功率）：2016 年 100%、2020 年 100%，
      但 **2015 年分区整列 89.1% 为 NULL**（该年 32,147 行里只有 3,501 行有时间），
      所以本函数在 2015 年只能给 10.9% 的涨停样本打分 —— 是采集缺失，不是解析错。
      归一化本身在非空样本上 100% 成功。
    """
    t = s.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    t = t.fillna("")
    colon = t.str.contains(":", regex=False)

    # A. 带冒号：按段解析 HH[:MM[:SS]]
    parts = t.str.split(":", expand=True)
    hh = pd.to_numeric(parts[0], errors="coerce")
    mm = (pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1
          else pd.Series(np.nan, index=t.index))
    if parts.shape[1] > 2:
        raw_ss = parts[2]
        # ★ `expand=True` 下「段数不足的行」在尾部列是 None —— 那是「没写秒」= :00，
        #   不是脏值。不补 0 的话 `"9:31"`（只有两段）会被 None 毒化成 NaN（实测踩过）。
        ss = pd.to_numeric(raw_ss, errors="coerce").where(raw_ss.notna(), 0.0)
    else:
        ss = pd.Series(0.0, index=t.index)
    min_a = (hh * 60.0 + mm + ss / 60.0).to_numpy(np.float64)

    # B/C. 纯数字：丢前导零的 HHMMSS，左补零到 6 位
    dig = t.str.replace(r"\D", "", regex=True)
    dig = dig.where(dig.str.len().between(1, 6))
    d6 = dig.str.zfill(6)
    hh2 = pd.to_numeric(d6.str.slice(0, 2), errors="coerce")
    mm2 = pd.to_numeric(d6.str.slice(2, 4), errors="coerce")
    ss2 = pd.to_numeric(d6.str.slice(4, 6), errors="coerce")
    min_b = (hh2 * 60.0 + mm2 + ss2 / 60.0).to_numpy(np.float64)

    mins = np.where(colon.to_numpy(), min_a, min_b)
    ok = np.isfinite(mins) & (mins >= 555.0) & (mins <= 930.0)   # 09:15 ~ 15:30
    return np.where(ok, mins, np.nan)


# ══════════════════════════════════════════════════════════════════════════
# A. 涨停 / 跌停生态（9 个）
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="consecutive_limit_up", group=GROUP, deps=(LU,),
    desc="当前连板数（连续涨停天数），断板即归零；非涨停日为 0",
    formula='is_lu = daily["pct_chg"].ge(9.8).astype(int)\n'
            'code = is_lu.index.get_level_values("Code")\n'
            'seg = (~is_lu.astype(bool)).groupby(level="Code").cumsum()\n'
            'count = is_lu.groupby([code, seg]).cumsum()',
    start=LU_START, warmup_days=W5, higher_is_better=True,
    note="★ 与 `limit_up_count_20`（event.py，20 日累计次数）**互补不重复**："
         "本因子是「当下处于第几板」的状态，断板当天归零。"
         "口径用上游 `stock_limit_up.consecutive_days`（供应商已算好的连板高度，"
         "实测取值 1~30、无 0）而不是自己用 pct_chg 攒连续段——参考库的 cumsum 分段"
         "在面板上要逐列循环，且除权/停牌边界更脆。缺失补 0 = 当天没涨停 = 连板数 0。"
         "分布：约 70% 的涨停样本是首板，长尾到 30 板（大量 0/1 + 长尾）。",
))
def consecutive_limit_up(ctx):
    ev = _Ev(ctx, LU, ("consecutive_days",))
    if ev.empty:
        return ctx.panel.empty()
    return ev.grid(values=ev.col("consecutive_days"))










@register(FactorSpec(
    name="limit_board_streak_mean_60", group=GROUP, deps=(LU,),
    desc="平均连板高度 = 60 日封板天数 / 连板启动次数（历史拉板惯性）",
    formula='sealed = (daily["pct_chg"] >= _LIMIT_UP).astype(float)\n'
            'prev_sealed = sealed.astype(bool).groupby(level="Code").shift(1)\n'
            '    .fillna(False).astype(bool)\n'
            'start = (sealed.astype(bool) & ~prev_sealed).astype(float)\n'
            'days60 = sealed.groupby(level="Code").transform(\n'
            '    lambda s: s.rolling(60, min_periods=1).sum())\n'
            'starts60 = start.astype(float).groupby(level="Code").transform(\n'
            '    lambda s: s.rolling(60, min_periods=1).sum())\n'
            'avg = safe_divide(days60, starts60).fillna(0.0)',
    start=LU_START, warmup_days=W60, higher_is_better=True,
    note="`sealed` 用 `stock_limit_up` 的**行本身**（= 收盘封板），不是 pct_chg≥9.8："
         "上游表只收录封板成功的票，比价格阈值更干净（炸板票不在表里）。"
         "「启动日」在网格上用 `ctx.shift(sealed,1)` 求（面板即交易日历，移一行 = 移一个"
         "交易日），首行 NaN 视为未封板（对齐参考库 fillna(False)）。"
         "★ 偏离参考库一处：`.fillna(0.0)` 保留（60 日内没封过板的股票 avg = 0），"
         "这是参考库明确的选择（「保证全市场覆盖」），0 = 「没有连板基因」，方向上也对。"
         "分布：约 1/3 的股票为 0，非零样本集中在 1.0~3.0（首板/二板），长尾到几十。",
))
def limit_board_streak_mean_60(ctx):
    ev = _Ev(ctx, LU)
    if ev.empty:
        return ctx.panel.empty()
    sealed = ev.grid()
    prev = np.nan_to_num(ctx.shift(sealed, 1), nan=0.0)
    starts = sealed * (1.0 - np.clip(prev, 0.0, 1.0))
    d60 = _roll_sum(ctx, sealed, 60, 1)
    s60 = _roll_sum(ctx, starts, 60, 1)
    return np.nan_to_num(ctx.safe_div(d60, s60, min_abs_den=0.5), nan=0.0)








# ══════════════════════════════════════════════════════════════════════════
# B. 龙虎榜（2 个）
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="top_list_net_rate_20", group=GROUP, deps=(TL,),
    desc="龙虎榜净买率 = 20 日上榜净买额 / 20 日上榜成交额（无上榜则 NaN）",
    formula="NetRate_20 = sum(NetAmount, 20d) / sum(Amount, 20d)\n"
            "# NetAmount = 龙虎榜买入额 - 卖出额；Amount = 龙虎榜成交额\n"
            "# 参考库 net_rate 的日频定义 = NetAmount / Amount * 100，本因子是它的 20 日聚合",
    start=None, warmup_days=W20, higher_is_better=True,
    version=3,
    note="★ **不再做滞后位移**（2026-09-15 晚，v2→v3）：`stock_top_list` 实测**当天可得** ——"
         "日更工程 T=2026-09-15 21:14 的逐日观测 `delay_obs = 0`（服务端当晚就有 T 日上榜记录，"
         "取数窗口已按当天收敛）。用户口径：「服务端当天有数据就必须拿当天」。"
         "所以去掉了 v2 加的 `ctx.lag_grid(grid, 1)`：本因子 T 日直接用 ≤ T 日的上榜记录，"
         "比 v2 多一天信息（整条序列相对 v2 前移一个交易日）。"
         "⚠️ 前提：因子在 D 日收盘后到当晚算，下游按 D+1 及以后交易 —— 这样用 D 当晚发布的数据不构成未来函数。"
         "（更正史：早期曾列为『别误判成滞后』→ 2026-09-15 白天改成滞后表并位移 → 当晚实测证伪、又去掉了位移。）"
         "★ 先汇总再算比率：`stock_top_list` 的键是 `(trade_date, stock_code, reason)`，"
         "同一股票同日可因**多个原因**上榜（实测 17,909 组重复），"
         "直接逐行取 net_rate 会丢掉其余上榜原因的量。用 `event_grid` 散点求和天然完成聚合。"
         "实测 `net_amount / amount × 100` 与表内 `net_rate` 相关 0.9999999，"
         "所以本口径与供应商的「净买率」同源，只是做了 20 日累计（更稳、且能覆盖零星上榜）。"
         "★ 覆盖率天然低且必须说明：没上榜的日子净买率**无定义**（不是 0）→ NaN，"
         "所以非空率 ≈ 「20 日内上过榜的股票占比」（约 10%~20%）。"
         "值域约 [−1, 1]（量纲自约，净额与成交额同为元）。"
         "★★ 精度（实测的框架缺陷，已在本因子内规避）：输入是**元**级大额（1e9~1e12），"
         "而 `ctx.panel.roll_sum` 的 cumsum 在 float32 网格上按 **float32** 累加，"
         "长面板下 cs 到 1e10 量级 → 相邻两期相减的绝对误差可达数百元，"
         "在**分母小**的票上会放大成可见的因子噪声（实测 002061.SZ 2015-12-31："
         "float32 路径 0.80180428、float64 路径 0.80188098、逐行精确复算 0.80188100 ——"
         "同一格子两次运行（面板长度不同）分别落到两个值上，0.38% 的格子有 1e-4 相对偏差）。"
         "本因子在滚动前 `.astype(np.float64)` 消除该误差（误差降到 1e-4 元量级）。"
         "已作为框架缺口报给主 Agent。",
))
def top_list_net_rate_20(ctx):
    ev = _Ev(ctx, TL, ("net_amount", "amount"))
    if ev.empty:
        return ctx.panel.empty()
    # ★ 升 float64 再滚动：见 note「精度」一条（引擎 cumsum 在 float32 网格上按 float32 累加）
    net = ctx.roll_sum(ev.grid(values=ev.col("net_amount")).astype(np.float64), 20)
    amt = ctx.roll_sum(ev.grid(values=ev.col("amount")).astype(np.float64), 20)
    out = ctx.safe_div(net, amt, min_abs_den=1.0)    # 20 日上榜成交额 <= 1 元 -> NaN
    # ★ 2026-09-15 晚：不再 lag_grid(1) —— 上游实测该表当天可得（见 note 的说明）
    return out


@register(FactorSpec(
    name="dragon_tiger_org_net_20", group=GROUP, deps=(DT, "stock_daily"),
    desc="机构席位净买 = 20 日「机构专用」席位净买额 / 20 日成交额（无机构席位则 NaN）",
    formula="OrgNet_20 = sum(OrgNetAmount, 20d) / sum(Amount, 20d)\n"
            "OrgNetAmount = buy_amount - sell_amount, org_name == '机构专用'\n"
            "Amount = 该股当日总成交额（stock_daily.amount，元）",
    # ★ 起点实测 2013-01-04 = 上游 `stock_dragon_tiger` 本身的起点。
    #   跟随 default_start（2012-01-01）会在 2012 整年产出全 NaN 的垃圾分区
    #   （2026-09-17 回填时由空分区监测抓到）。注意本因子**天生稀疏**（只有上龙虎榜的
    #   股票才有值，实测 2013 年非空率 3.5%），那是定义使然，不是缺数据。
    start="2013-01-04", warmup_days=W20, higher_is_better=True,
    version=3,
    note="★ **不再做滞后位移**（2026-09-15 晚，v2→v3）：`stock_dragon_tiger` 实测**当天可得** ——"
         "日更工程 T=2026-09-15 21:14 的逐日观测 `delay_obs = 0`。去掉了 v2 的 `lag_grid(grid, 1)`，"
         "T 日直接用 ≤ T 日的席位明细（整条序列相对 v2 前移一个交易日）。与 `top_list_net_rate_20` "
         "同一处理，前提与更正史见它的 note。"
         "⚠️ 本因子还叠加了一次**输入侧变更**：2026-09-15 晚该表主键由 4 列改 7 列并全量回填"
         "（旧主键把同名机构席位合并，09-14 一天 557→780 行），所以 v3 与 v2 的差异同时来自"
         "「去掉位移」和「机构席位不再被合并」。"
         "★ `stock_dragon_tiger.net_buy_amount` 的**符号不可信**：与 `buy_amount − sell_amount` "
         "★ `stock_dragon_tiger.net_buy_amount` 的**符号不可信**：与 `buy_amount − sell_amount` "
         "在 32.2% 的行上符号不一致（178 万行全表实测，例如某行 buy=778万/sell=0 却给 net=−778万）。"
         "本因子一律自己算净额，不碰该列。"
         "★ 分母用**个股 20 日成交额**（价格层，元）而不是机构席位自身的买卖总额："
         "后者会让「只有一笔小单的机构席位」拿到 ±1 的极端值，丢掉了资金量级信息。"
         "跨表前先确认量纲：机构席位买卖额与 `stock_daily.amount` 同为元"
         "（实测 机构 gross/当日成交额 中位数 4.4%），无需换算。"
         "★ 覆盖率的两个来源：① 20 日内没有任何「机构专用」席位 → NaN（无定义）；"
         "② 停牌造成 20 日成交额窗口不完整 → `min_count=10` 显式放松（对齐契约 §3.4，"
         "理由就是停牌），放松后仍有值的股票占绝大多数。"
         "非空率 ≈ 10%（机构席位本就只出现在少数上榜股票里）。值域约 [−1, 1]。"
         "★ 精度：分子分母都升到 float64 再滚动（元级大额 + 引擎的 float32 cumsum，"
         "见 `top_list_net_rate_20` 的说明）。",
))
def dragon_tiger_org_net_20(ctx):
    ev = _Ev(ctx, DT, ("org_name", "buy_amount", "sell_amount"))
    if ev.empty:
        return ctx.panel.empty()
    is_org = ev.df["org_name"].astype("string").to_numpy() == "机构专用"
    if not is_org.any():
        return ctx.panel.empty()
    net = ev.col("buy_amount") - ev.col("sell_amount")
    # ★ 升 float64 再滚动：元级大额 + `panel.roll_sum` 的 float32 cumsum，见 top_list 的 note
    num = ctx.roll_sum(ev.grid(mask=is_org, values=net).astype(np.float64), 20)
    hit = ctx.roll_sum(ev.grid(mask=is_org), 20)
    amt = _roll_sum(ctx, ctx.px("amount").astype(np.float64), 20, 10)   # 停牌 -> 显式放松
    out = ctx.safe_div(num, amt, min_abs_den=1.0)
    out = np.where(hit > 0.0, out, np.nan)
    # ★ 2026-09-15 晚：不再 lag_grid(1) —— 上游实测该表当天可得（见 note 的说明）
    return out


# ══════════════════════════════════════════════════════════════════════════
# C. 业绩预告（2 个）—— 事件驱动系列之一/之二
# ══════════════════════════════════════════════════════════════════════════
_TYPE_SCORE = {
    # 参考《事件驱动策略之一——把握扭亏、预减公告》：扭亏与预增是最好的两类，
    # 预减/首亏最差；续盈/略增/减亏为弱正，略减/续亏/增亏为负。
    "预增": 2.0, "扭亏": 2.0,
    "略增": 1.0, "续盈": 1.0, "减亏": 1.0,
    "不确定": 0.0, "其他": 0.0,
    "略减": -1.0,
    "预减": -2.0, "首亏": -2.0, "续亏": -2.0, "增亏": -2.0,
}


def _forecast_rows(ctx) -> pd.DataFrame | None:
    """`stock_forecast` → 按 `(stock_code, ann_date)` 聚合后的中位数。

    ★ 对齐口径是 **ann_date**（公告日）不是 end_date（报告期）：按 end_date 对齐
      等于在报告期结束那天就知道预告内容 —— 前视。
    同一 (股票, 公告日) 可能有**多条**记录（不同 end_date / 不同 type，实测 228 组），
    因子名里的「中位数」就在这里落地：对多条取中位数，而不是随便留一条。
    """
    df = ctx.dataset(FC, columns=["stock_code", "ann_date", "type",
                                  "p_change_min", "p_change_max"],
                     years=_years(ctx))
    if df is None or df.empty:
        return None
    lo = pd.to_numeric(df["p_change_min"], errors="coerce")
    hi = pd.to_numeric(df["p_change_max"], errors="coerce")
    # 区间中点；只有一侧披露时取该侧，两侧都缺才 NaN
    mid = pd.concat([lo, hi], axis=1).mean(axis=1, skipna=True)
    out = pd.DataFrame({
        "stock_code": df["stock_code"].to_numpy(),
        "ann_date": df["ann_date"].to_numpy(),
        "mid": mid.to_numpy(np.float64),
        "score": df["type"].map(_TYPE_SCORE).astype(np.float64).to_numpy(),
    })
    g = (out.groupby(["stock_code", "ann_date"], as_index=False)[["mid", "score"]]
            .median())                       # 中位数天然忽略 NaN
    return g




@register(FactorSpec(
    name="forecast_type_score", group=GROUP, deps=(FC,),
    desc="业绩预告类型打分（预增/扭亏 +2 … 预减/首亏 −2），按 ann_date 前向填充",
    formula="Score = map(type): 预增/扭亏=+2, 略增/续盈/减亏=+1, 不确定/其他=0,\n"
            "                  略减=−1, 预减/首亏/续亏/增亏=−2\n"
            "# PIT 对齐: ann_date <= T 的最新一条；同日多条取中位数",
    start=None, warmup_days=ANN, higher_is_better=True,
    note="打分依据《事件驱动策略之一》的结论排序（扭亏最好、预减最差、预增居中偏上；"
         "续盈/略增/略减样本少且特征不明显 → 弱档）。打分是**序数**不是基数，"
         "下游做截面回归时会自动再标准化。未知类型（上游新增枚举）→ NaN，不默认给 0，"
         "避免把「没见过的类型」混进中性档。同 (股票, 公告日) 多条取中位数。"
         "覆盖：与 forecast_p_change_median 同源，但**不依赖 p_change**，"
         "所以少数只有类型没有幅度的预告也能给值。",
))
def forecast_type_score(ctx):
    g = _forecast_rows(ctx)
    if g is None:
        return ctx.panel.empty()
    return ctx.asof_daily(g["stock_code"].to_numpy(),
                          ctx.date_col(g["ann_date"]),
                          g["score"].to_numpy(np.float64))




# ══════════════════════════════════════════════════════════════════════════
# E. 价格形态事件（3 个）
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="new_high_frequency_60", group=GROUP,
    deps=("stock_daily", "stock_adj_factor"),
    desc="60 日新高频率 = 60 日内「收盘价创 60 日新高」的天数占比",
    formula='adj = _adjusted_close(daily)\n'
            'is_high = adj.eq(adj.groupby(level="Code").transform(\n'
            '    lambda s: s.rolling(60, min_periods=30).max()))\n'
            'freq = is_high.groupby(level="Code").transform(\n'
            '    lambda s: s.rolling(60, min_periods=30).mean())',
    start=None, warmup_days=W60, higher_is_better=True,
    note="★ 复权基座必须是**后复权** `ctx.hfq(\"close\")`：参考库用的 daily_adj.parquet 是"
         "前复权（历史值会随未来分红重算，PIT 红线）。后复权锚定序列起点，除权日不产生假新高。"
         "停牌日 hfq_close 被前向填充（= 上一日价），除非整窗横盘否则不会误判为新高。"
         "`min_count=30` 对齐参考库 min_periods=30（窗口内至少半年数据）。"
         "值域 [0,1]；次新股上市满 30 个交易日后才有值 → 上市初期为 NaN（正常）。",
))
def new_high_frequency_60(ctx):
    adj = ctx.hfq("close")
    hi60 = ctx.roll_max(adj, 60, min_count=30)
    is_high = np.where(np.isfinite(adj) & np.isfinite(hi60),
                       (adj >= hi60).astype(np.float64), np.nan)
    return ctx.roll_mean(is_high, 60, min_count=30)




@register(FactorSpec(
    name="high_open_low_close_frac_20", group=GROUP,
    deps=("stock_daily", "stock_adj_factor"),
    desc="高开低走占比 = 20 日内「高开≥2% 且收阴」的天数占比（出货特征）",
    formula='gapup = safe_divide(daily["open"], daily["pre_close"]) - 1.0\n'
            'fade = (gapup >= 0.02) & (daily["close"] < daily["open"])\n'
            'freq = fade.astype(float).groupby(level="Code").transform(\n'
            '    lambda s: s.rolling(20, min_periods=5).mean())',
    start=None, warmup_days=W20, higher_is_better=False,
    note="高开与收阴都是**当日**比较（open 与 pre_close 复权后同尺度、close 与 open 同日），"
         "所以口径上不会踩除权日的坑；仍统一走 `ctx.hfq`（本项目唯一允许的价格口径）。"
         "`min_count=5` 对齐参考库 min_periods=5（停牌日 pct_chg/gap 为 NaN，不参与均值）。"
         "与 `big_gap_reversal_5` 的区别：后者看「高开 3% 之后还涨不涨」（事件+收益），"
         "本因子看「高开 2% 却收阴」的**频率**（当日日内反转的出货指纹）。"
         "值域 [0,1]，绝大多数股票接近 0（高开低走是少数形态）。",
))
def high_open_low_close_frac_20(ctx):
    o, c, pc = ctx.hfq("open"), ctx.hfq("close"), ctx.hfq("pre_close")
    gapup = ctx.safe_div(o, pc, min_abs_den=1e-9) - 1.0
    fade = (gapup >= 0.02) & (c < o)
    ok = np.isfinite(gapup) & np.isfinite(c) & np.isfinite(o)
    return ctx.roll_mean(np.where(ok, fade.astype(np.float64), np.nan), 20, min_count=5)


