"""交易策略引擎 —— **小资金、最多 5 只、每日三动作**（用户 2026-09-17 裁定）。

## 口径（与用户的裁定逐条对齐）

1. **资金量小（个人）** ⇒ 起始资金可配（默认 10 万），**整手 100 股**，
   买不起 1 手的票直接跳过（小资金下这不是细节，是主要约束）；
2. **最多持股 5 只**（`strategy.max_positions`），等权分配；
3. 每天基于**当天打分**决定**第二天早盘**的动作，动作只有三种：
   **持仓不动 `hold` / 清仓 `clear` / 换手 `switch`**；
4. 成交价 = 次日**开盘价（后复权 hfq）**；买不进（次日开盘一字涨停/停牌）与
   卖不掉（次日开盘一字跌停/停牌）都**顺延到下一个能成交的日子**；
5. 费用 = 佣金（万 2.5，最低 5 元）+ 印花税（卖出万 5）+ 过户费（万 0.1）+ 滑点（单边）；
6. 净值 = **现金级真实净值**（`cash + Σ 股数×收盘价`，含未投资现金）。
   ★ 参考工程的实测教训：多仓策略用"逐笔满额复利"会**系统性高估**
     （同一策略 +377% vs 真实 +126%）—— 本引擎只出真实净值，不做逐笔复利。

## 为什么策略要跟模型一起迭代

用户明确："模型的训练和交易的策略应该是同步优化迭代的"。所以策略是**一等公民**：
本引擎提供注册表 + 状态机，单元的 `model.py` 里声明"本版要评估哪些策略"，
`analysis.py` 出**策略 × 头**矩阵（每个策略在每个打分源上单独跑净值），
并且**每个折（种子）各跑一遍** —— 小资金策略的随机干扰很重，只看单一种子的回测
数字是没有意义的（参考工程 V29 的教训）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Cfg

TRADING_DAYS = 244.0


# ================================================================ 价格/可交易性上下文
def price_ctx(cfg: Cfg, dates: list[str], codes: np.ndarray) -> dict:
    """回测要用的价格与可交易性面板（一次算好，多个策略复用）。

    返回：`open`/`close`（hfq，成交与估值用）· `entry_ok`（次日开盘能否买）·
    `exit_ok`（次日开盘能否卖）。
    """
    # ★ 2026-09-19 起价格来自 `trainingdata/P` 快照（`mx/prices.py`），**不再读上游**。
    #   原先要 upstream._engine + fea.panel + assert_price_axis 三件套；
    #   现在与训练一样只需 `trainingdata/` 一个外部文件夹，且轴按 key 对齐（不再有
    #   "按列号落格、甲乙互换"的静默错位隐患）。
    from . import prices as PR
    panel, lay = PR.load(cfg, dates, codes)
    o = lay.panel(panel, "hfq_open")
    c = lay.panel(panel, "hfq_close")
    # 一字板判定用**未复权**价：open==high==low 且当日涨跌幅触及限价
    raw_o = lay.panel(panel, "open")
    raw_h = lay.panel(panel, "high")
    raw_l = lay.panel(panel, "low")
    r1 = lay.panel(panel, "ret1")
    traded = lay.panel(panel, "traded").astype(bool)
    lim = float(cfg.raw.get("strategy", {}).get("limit_pct", 0.098))
    eq = lambda a, b: np.abs(a - b) <= 1e-6 * np.maximum(1.0, np.abs(a))    # noqa: E731
    one_word = eq(raw_o, raw_h) & eq(raw_h, raw_l)
    up_limit = one_word & (r1 >= lim - 1e-3)
    dn_limit = one_word & (r1 <= -(lim - 1e-3))
    # ★ 已实现波动率（`invvol` 权重用）：对数收益的滚动标准差，**第 j 行只用 j 及其之前**。
    #   PIT 安全的做法是"配仓发生在 j 日开盘 ⇒ 只能用 j−1 及之前的数据"，
    #   所以调用方取 `vol[j-1]`（见 `_alloc_weights`）。这里先算好整条序列，避免每轮重算。
    cc = c.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.diff(np.log(np.where(cc > 0, cc, np.nan)), axis=0, prepend=np.nan)
    vol = pd.DataFrame(lr).rolling(20, min_periods=10).std().to_numpy()
    return {"open": o.astype(np.float64), "close": cc,
            "vol20": vol,
            "entry_ok": traded & ~up_limit & np.isfinite(o),
            "exit_ok": traded & ~dn_limit & np.isfinite(o),
            "dates": list(dates), "codes": np.asarray(codes)}


# ================================================================ 状态
@dataclass
class Position:
    code: str
    lots: int
    buy_i: int                 # 买入执行日的日索引
    buy_price: float
    buy_cost: float            # 含费用的总支出
    peak: float = 0.0          # 买入后走过的最高收盘价（供移动止损类策略用）

    @property
    def shares(self) -> int:
        return self.lots * 100


@dataclass
class Decision:
    action: str                       # hold | clear | switch
    target: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class Ctx:
    """策略函数的输入：**只用得到 ≤T 的信息**（打分是当天收盘后算出来的）。"""
    i: int                            # 信号日索引（执行在 i+1）
    scores: np.ndarray                # (C,) 当日打分（NaN = 不可选）
    selectable: np.ndarray            # (C,) 当日可研究/可选（票池 ∩ 有特征 ∩ 有标签口径）
    holdings: list[Position]
    equity: float
    cash: float
    entry_ok: np.ndarray              # (C,) 次日能否买入
    exit_ok: np.ndarray               # (C,) 次日能否卖出
    close: np.ndarray                 # (C,) 当日收盘（hfq）
    params: dict
    cfg: Cfg
    tradable_rank: list[str] = field(default_factory=list)   # 当日"能买"的候选（已按打分降序）
    codes: np.ndarray = None          # (C,) 股票代码轴 —— 用来查"手里这只"的分数/位次
    code_idx: dict = None             # code -> 列下标

    def top(self, n: int) -> list[str]:
        return self.tradable_rank[:n]

    def score_of(self, code: str) -> float:
        """某只股票当日的打分（查不到或无分 ⇒ NaN）。

        ★ 为什么策略需要这个：`tradable_rank` 只给了"能买的候选排序"，
          而"换仓类的规则"（比如"新候选要比手里这只高出多少才值得换"）必须能
          拿到**手里这只**当时的分数。没有它就只能按名次判断，名次在候选集变动时不可比。
        """
        if self.code_idx is None or self.scores is None:
            return float("nan")
        k = self.code_idx.get(code)
        return float("nan") if k is None else float(self.scores[k])

    def rank_of(self, code: str) -> int:
        """这只股票在当日"能买候选"里的位次（0 起；不在候选里 ⇒ 一个很大的数）。"""
        try:
            return self.tradable_rank.index(code)
        except ValueError:
            return 10 ** 9


# ================================================================ 内置策略（注册表）
STRATEGIES: dict[str, callable] = {}


def register(fn):
    STRATEGIES[fn.__name__] = fn
    return fn


@register
def top1_daily(ctx: Ctx) -> Decision:
    """**基线**：持股 top1，次日早盘换成当日 top1（持 1 天）。

    用户 2026-09-17 指定的回测 baseline（"持股 top1 隔日换手"的主口径）。
    """
    t = ctx.top(1)
    if not t:
        return Decision("clear" if ctx.holdings else "hold", note="无可选票")
    return Decision("switch", t)


@register
def top1_alt(ctx: Ctx) -> Decision:
    """持股 top1，**隔日**换手（持 2 天）—— "隔日换手"的另一种读法，做对照。"""
    if ctx.i % 2 == 1 and ctx.holdings:          # 奇数日不动作
        return Decision("hold")
    t = ctx.top(1)
    if not t:
        return Decision("clear" if ctx.holdings else "hold", note="无可选票")
    return Decision("switch", t)


@register
def topn_hold(ctx: Ctx) -> Decision:
    """top-N 等权，到期（h 个交易日）换手。参数：`n`（默认 5，≤ max_positions）、`hold_days`（默认 5）。

    `top5_hold` / `top3_hold` 是它的别名（历史写法，避免"策略名里有数字"导致 n 和名字打架）。
    """
    h = int(ctx.params.get("hold_days", 5))
    if ctx.holdings and all((ctx.i - p.buy_i) < h for p in ctx.holdings):
        return Decision("hold")
    t = ctx.top(int(ctx.params.get("n", 5)))
    if not t:
        return Decision("clear" if ctx.holdings else "hold", note="无可选票")
    return Decision("switch", t)


@register
def topn_ladder(ctx: Ctx) -> Decision:
    """**阶梯换仓**：每天最多只换 `k` 只，换手摊到多天。

    ## 为什么加它（这是冲着本轮量出来的**方差来源**去的）

    种子冗余实测（README §8.1）：同一配方只换随机种子，
    **RankIC 极差只有 0.0037，而 ≤5 只组合的收益极差有 20.4 个百分点**。
    ⇒ 方差**不在模型里，在组合构建里**。

    机制很清楚：`topn_hold{hold_days:h}` 是**整仓同一天换手** ——
    换仓那天的运气决定了后面 h 天的全部持仓。5 只票、一次全换，
    等于每隔 h 天赌一次"这天选出来的 5 只"。
    **阶梯换仓把进出场时间摊开**：每天只动 k 只，持仓的建仓日互相错开，
    单一交易日的运气不再能主导整条净值曲线。

    ★ 实盘上也更顺手：每天最多几笔，不会某天一次性打满（小资金 + 整手约束下尤其明显）。

    ## 规则（三种动作里的 `switch`，但每次只换一部分）

    1. 先把**该走的**挑出来：不在当日 `top-n` 里的，或者**持有已满 `hold_days`** 的；
    2. 从该走的里面，按**持有天数从长到短**最多取 `k` 只卖掉（转得久的先走）；
    3. 用当日 `top-n` 里没持有的票补齐空位。
    ⇒ 「一天最多动 k 只」是把上式里的 ② 截断得到的，① 不直接导致卖出 —— 只有进入 ② 才算。
    """
    n = int(ctx.params.get("n", 5))
    k = max(1, int(ctx.params.get("k", 1)))
    h = int(ctx.params.get("hold_days", 10))
    t = ctx.top(n)
    if not t:
        return Decision("clear" if ctx.holdings else "hold", note="无可选票")
    tset = set(t)
    # ① 该走的：掉出 top-n 的 + 持有满期的
    cand_out = [p for p in ctx.holdings
                if (p.code not in tset) or ((ctx.i - p.buy_i) >= h)]
    # ② 每天最多动 k 只，持有最久的先走
    cand_out = sorted(cand_out, key=lambda p: (ctx.i - p.buy_i), reverse=True)[:k]
    drop = {id(p) for p in cand_out}
    target = [p.code for p in ctx.holdings if id(p) not in drop]
    # ③ 用当日 top-n 补齐空位
    for c in t:
        if len(target) >= n:
            break
        if c not in target:
            target.append(c)
    if not target:
        return Decision("clear" if ctx.holdings else "hold", note="无可选票")
    if not cand_out and target == [p.code for p in ctx.holdings]:
        return Decision("hold", note="无到期、无掉出")
    return Decision("switch", target)


@register
def top5_rank_exit(ctx: Ctx) -> Decision:
    """持有 top-N，**只有跌出前 M 名才卖**（M > N）—— 低换手版本。"""
    n = int(ctx.params.get("n", 5))
    m = int(ctx.params.get("m", 10))
    keep = set(ctx.tradable_rank[:m])
    target = [p.code for p in ctx.holdings if p.code in keep]
    for code in ctx.top(n):
        if len(target) >= n:
            break
        if code not in target:
            target.append(code)
    if not target:
        return Decision("clear" if ctx.holdings else "hold", note="全跌出前 M")
    return Decision("switch", target)


@register
def top5_clear_weak(ctx: Ctx) -> Decision:
    """弱市清仓：当日"能买且打分非空"的股票数 < 阈值（或 top1 打分 < 分位阈值）→ 清仓。

    这是三种动作里 `clear` 的用法示范（参考工程实测"阈值择时"无效，但那是他们的口径，
    我们保留这个策略是为了**在策略矩阵里能被证伪**，不是为了它好用）。
    """
    min_n = int(ctx.params.get("min_selectable", 100))
    if int(np.isfinite(ctx.scores[ctx.selectable]).sum()) < min_n:
        return Decision("clear" if ctx.holdings else "hold", note="可选票过少")
    t = ctx.top(int(ctx.params.get("n", 5)))
    if not t:
        return Decision("clear" if ctx.holdings else "hold", note="无可选票")
    return Decision("switch", t)


@register
def topn_hold_margin(ctx: Ctx) -> Decision:
    """持有 top-N，**只有新候选比手里这只高出 `margin` 分才换** —— 抗抖动的换手版。

    参数：`n`（默认 5）、`margin`（默认 0.05，单位是**当日截面 z 分**）、
    `hold_days`（默认 1 = 每天都允许换，只是加了门槛）、`max_hold`（默认 20，硬上限防赖着不走）。

    为什么要这条：打分本身有噪声，每天都严格按名次换仓会把噪声直接变成手续费。
    门槛的作用是"只在模型**很有把握**的时候才动"。`margin` 的单位取截面 z 分
    （打分在 `analyze` 里已经逐日 z 过），所以它不随打分的量纲漂移。

    ★ 这是"策略与损失耦合"的一个具体体现：如果模型的损失把头部做得很尖锐，
      那 `margin` 的作用就小；如果头部平缓（比如 τ 大的 top 损失），
      换仓门槛就变得关键。两者要一起看，不能单独调。
    """
    n = int(ctx.params.get("n", 5))
    margin = float(ctx.params.get("margin", 0.05))
    max_hold = int(ctx.params.get("max_hold", 20))
    cand = ctx.top(n)

    if not ctx.holdings:
        return Decision("switch", cand) if cand else Decision("hold", note="无可选票")

    # 手里每只：既不在当日候选里、又持有超过 max_hold 天 ⇒ 无条件换掉
    keep: list[str] = []
    dropped: list[str] = []
    for p in ctx.holdings:
        stale = (ctx.i - p.buy_i) >= max_hold
        in_cand = p.code in cand
        # 用"新候选里最好的那只"和"手里这只"比分数：差得多才值得付一次换手成本
        best_new = ctx.score_of(cand[0]) if cand else float("nan")
        mine = ctx.score_of(p.code)
        improve = (best_new - mine) if np.isfinite(best_new) and np.isfinite(mine) else 0.0
        if stale or (not in_cand and improve >= margin):
            dropped.append(p.code)
        else:
            keep.append(p.code)

    if not dropped:
        return Decision("hold", note="都在容忍区内")

    target = list(keep)
    for code in cand:
        if len(target) >= n:
            break
        if code not in target and code not in dropped:
            target.append(code)
    if not target:
        return Decision("clear", note="全体换出且无新候选")
    return Decision("switch", target)


@register
def cash(ctx: Ctx) -> Decision:
    """空仓对照（策略层的 0 基准，净值应恒为 1）。"""
    return Decision("clear" if ctx.holdings else "hold")


# 别名（同一个实现，只是名字里带了 N —— 真参数看 params.n）
STRATEGIES["top5_hold"] = STRATEGIES["topn_hold"]
STRATEGIES["top3_hold"] = STRATEGIES["topn_hold"]


def resolve(name_or_fn):
    if callable(name_or_fn):
        return name_or_fn
    if name_or_fn not in STRATEGIES:
        raise SystemExit(f"未知策略 {name_or_fn}（已注册：{sorted(STRATEGIES)}）")
    return STRATEGIES[name_or_fn]


# ================================================================ 成本
def cost_model(cfg: Cfg) -> dict:
    s = cfg.raw.get("strategy") or {}
    return {"commission_bps": float(s.get("commission_bps", 2.5)),
            "commission_min": float(s.get("commission_min", 5.0)),
            "stamp_bps": float(s.get("stamp_bps", 5.0)),
            "transfer_bps": float(s.get("transfer_bps", 0.1)),
            "slippage_bps": float(s.get("slippage_bps", 3.0))}


def _buy_cost(notional: float, cm: dict) -> float:
    fee = max(notional * cm["commission_bps"] / 1e4, cm["commission_min"]) \
        + notional * cm["transfer_bps"] / 1e4 + notional * cm["slippage_bps"] / 1e4
    return fee


def _sell_cost(notional: float, cm: dict) -> float:
    fee = max(notional * cm["commission_bps"] / 1e4, cm["commission_min"]) \
        + notional * cm["stamp_bps"] / 1e4 + notional * cm["transfer_bps"] / 1e4 \
        + notional * cm["slippage_bps"] / 1e4
    return fee


def _alloc_weights(mode: str, names: list[str], code2idx: dict, ctx_px: dict,
                   j: int, scores: np.ndarray, params: dict) -> dict:
    """给待买名单分配**相对权重**（不必归一；返回 0 表示"这只放弃"）。

    `mode` 由策略的 `params["weight"]` 指定（默认 `equal`）：

    | 口径 | 含义 | 为什么要有它 |
    |:--|:--|:--|
    | `equal` | 全部 1.0 | **默认，与历史口径逐字一致** ⇒ 新旧结果可横比，不许悄悄改掉 |
    | `invvol` | `1/σ`，σ = 近 20 日已实现波动 | 最经典的降方差手段：波动大的少配 |
    | `score` | `softmax(z/τ)`，分数高的多配 | ★ 本轮已**四次**证明"往截面头部押注"是负的（§10）—— 实现它是为了**能被证伪** |

    ★ `invvol` 的 PIT：配仓发生在第 `j` 日**开盘**，所以只能用 **`j−1` 及之前**的收盘价算出的
      波动率（取 `vol[j-1]`）。用 `vol[j]` 就是拿当天的收盘信息去决定当天的开盘下单 —— 前视。
    ★ 波动率缺失（新股 / 长期停牌）时给 0 权重 = 不买这只，**不做均值填补**：
      补一个"平均波动"会把一个未知风险当成已知风险。
    """
    if mode == "equal" or not mode:
        return {c: 1.0 for c in names}
    if mode == "invvol":
        vol = ctx_px.get("vol20")
        if vol is None or j - 1 < 0:
            return {c: 1.0 for c in names}
        row = vol[j - 1]
        out = {}
        for c in names:
            k = code2idx.get(c)
            v = float(row[k]) if (k is not None and np.isfinite(row[k])) else np.nan
            out[c] = (1.0 / v) if (np.isfinite(v) and v > 1e-6) else 0.0
        if sum(out.values()) <= 0:
            return {c: 1.0 for c in names}          # 全拿不到波动率 ⇒ 退回等权，别整仓空着
        cap = float(params.get("w_cap", 0.5))       # 单票上限（占组合的比例）
        if cap > 0 and out:
            tot = sum(out.values())
            out = {c: min(w, cap * tot) for c, w in out.items()}
        return out
    if mode == "score":
        idx = [code2idx.get(c) for c in names]
        vals = np.asarray([scores[k] if k is not None else np.nan for k in idx], dtype=np.float64)
        ok = np.isfinite(vals)
        if not ok.any():
            return {c: 1.0 for c in names}
        tau = float(params.get("w_tau", 1.0))
        e = np.zeros_like(vals)
        e[ok] = np.exp((vals[ok] - vals[ok].max()) / max(1e-6, tau))
        e = e / e.sum()
        return {c: float(w) for c, w in zip(names, e)}
    raise SystemExit(f"✘ 未知的 weight 口径 {mode!r}（可选：equal / invvol / score）")


# ================================================================ 引擎
def simulate(cfg: Cfg, ctx_px: dict, scores_2d: np.ndarray, strategy,
             universe_2d: np.ndarray | None = None, params: dict | None = None,
             capital: float | None = None, lot_size: int = 100, log=None,
             dates_lo: str | None = None, dates_hi: str | None = None) -> dict:
    """按策略跑一遍现金级回测。

    `scores_2d` (T,C)：**第 i 行是第 i 日收盘后算出的打分**（NaN = 不可选）；
    执行发生在第 i+1 日开盘。返回净值/持仓/成交/指标。

    ★ `dates_lo/hi`：只在这个日期区间内**执行**（含两端）。
      做样本外评估时必须给（一般传 test 窗口）——否则会在模型训练过的样本内跑策略，
      得到的净值是自欺（第一版就踩过：全期跑出来 +42%，test 窗口内只有 +11%）。
    """
    s_cfg = cfg.raw.get("strategy") or {}
    n_max = int(s_cfg.get("max_positions", 5))
    capital = float(capital if capital is not None else s_cfg.get("capital", 100_000))
    cm = cost_model(cfg)
    fn = resolve(strategy)
    params = dict(params or {})

    dates = ctx_px["dates"]
    codes = ctx_px["codes"]
    T, C = scores_2d.shape
    assert len(dates) == T and len(codes) == C

    cash = capital
    holdings: list[Position] = []
    equity_hist = np.full(T, np.nan)
    cash_hist = np.full(T, np.nan)
    invested_hist = np.zeros(T)
    pos_count = np.zeros(T, dtype=int)
    trades: list[dict] = []
    n_trades = 0

    # ★ code → 列下标。整条回测里是常量，所以**建一次**：
    #   踩过：它原本是在循环体里、`Ctx` 构造**之后**才建的（那时只有买卖执行用得到它），
    #   给 Ctx 加上 `code_idx` 之后就成了"用后定义"—— 第一轮直接 NameError，
    #   而且就算不报错也会一直用上一轮的值。放到循环外一次解决。
    code2idx = {c: k for k, c in enumerate(codes)}

    for i in range(T - 1):
        j = i + 1                              # 执行日
        if dates_lo and dates[j] < dates_lo:
            continue
        if dates_hi and dates[j] > dates_hi:
            break
        s = scores_2d[i]
        sel = np.isfinite(s)
        if universe_2d is not None:
            sel = sel & universe_2d[i]
        entry_ok, exit_ok, openpx, close_j = ctx_px["entry_ok"][j], ctx_px["exit_ok"][j], \
            ctx_px["open"][j], ctx_px["close"][j]

        # ---- 当日"能买"的候选（按打分降序；买不进的不进候选）
        cand = np.flatnonzero(sel & entry_ok)
        if cand.size:
            order = cand[np.argsort(-s[cand], kind="stable")]
            rank = [codes[k] for k in order]
        else:
            rank = []

        ctx = Ctx(i=i, scores=s, selectable=sel, holdings=holdings,
                  equity=cash + sum(p.shares * ctx_px["close"][j - 1] for p in holdings),
                  cash=cash, entry_ok=entry_ok, exit_ok=exit_ok,
                  close=ctx_px["close"][j - 1] if j >= 1 else np.full(C, np.nan),
                  params=params, cfg=cfg, tradable_rank=rank,
                  codes=codes, code_idx=code2idx)
        dec = fn(ctx)

        # ---- 次日开盘执行：先卖后买
        keep = set(dec.target) if dec.action == "switch" else (
            {p.code for p in holdings} if dec.action == "hold" else set())
        sold_today, bought_today = [], []

        still: list[Position] = []
        for p in holdings:
            if p.code in keep:
                still.append(p)
                continue
            k = code2idx[p.code]
            if not exit_ok[k]:
                still.append(p)                # 卖不掉（停牌/一字跌停）→ 顺延，明天再试
                continue
            px = openpx[k]
            notional = p.shares * px
            fee = _sell_cost(notional, cm)
            cash += notional - fee
            n_trades += 1
            trades.append({"code": p.code, "side": "sell", "i": j, "date": dates[j],
                           "price": float(px), "shares": p.shares, "fee": float(fee),
                           "pnl": float(notional - fee - p.buy_cost),
                           "hold_days": int(j - p.buy_i),
                           "ret": float((notional - fee) / p.buy_cost - 1.0)})
            sold_today.append(p.code)

        # 买入：等权分配（按目标数量），整手取整，买不起 1 手就跳过
        new_codes = [c for c in (dec.target if dec.action == "switch" else []) if c not in
                     {p.code for p in still}]
        slots = max(0, n_max - len(still))
        if new_codes and slots:
            # ★ 预算按**权重口径**切分（删掉权重为 0 的票）。默认 `equal` ⇒ 与历史逐字一致。
            #   买不起 1 手的份额**不重新分配**给其他票（宁可留现金，也不让单票超配 ——
            #   小资金下这是有意的保守选择，`invvol` 下尤其重要：不该因为某只买不起就加码别的）。
            pick = new_codes[:slots]
            wmap = _alloc_weights(str(params.get("weight", "equal")), pick, code2idx,
                                  ctx_px, j, s, params)
            pick = [c for c in pick if wmap.get(c, 0.0) > 0]
            wsum = sum(wmap.get(c, 0.0) for c in pick)
            for c in pick:
                k = code2idx.get(c)
                if k is None or not entry_ok[k]:
                    continue
                px = openpx[k]
                budget_each = cash * (wmap.get(c, 0.0) / wsum) if wsum > 0 else cash / len(pick)
                lots = int(budget_each // (px * lot_size))
                if lots < 1:
                    continue                    # ★ 小资金：买不起 1 手就跳过
                notional = lots * lot_size * px
                fee = _buy_cost(notional, cm)
                if notional + fee > cash:
                    lots -= 1
                    if lots < 1:
                        continue
                    notional = lots * lot_size * px
                    fee = _buy_cost(notional, cm)
                cash -= notional + fee
                still.append(Position(code=c, lots=lots, buy_i=j, buy_price=float(px),
                                      buy_cost=float(notional + fee), peak=float(px)))
                n_trades += 1
                bought_today.append(c)
                trades.append({"code": c, "side": "buy", "i": j, "date": dates[j],
                               "price": float(px), "shares": lots * lot_size,
                               "fee": float(fee), "pnl": 0.0, "hold_days": 0, "ret": 0.0})
        holdings = still

        # ---- 收盘估值
        for p in holdings:
            k = code2idx[p.code]
            if np.isfinite(close_j[k]):
                p.peak = max(p.peak, float(close_j[k]))
        mv = sum(p.shares * (close_j[code2idx[p.code]] if np.isfinite(close_j[code2idx[p.code]])
                             else p.buy_price) for p in holdings)
        equity_hist[j] = cash + mv
        cash_hist[j] = cash
        invested_hist[j] = mv
        pos_count[j] = len(holdings)

    return _metrics(cfg, dates, equity_hist, cash_hist, invested_hist, pos_count,
                    trades, n_trades, capital, fn.__name__, params, cm)


def _metrics(cfg, dates, equity, cash, invested, pos_count, trades, n_trades,
             capital, name, params, cm) -> dict:
    ok = np.isfinite(equity)
    if not ok.any():
        return {"strategy": name, "params": params, "error": "没有净值（数据不足）"}
    eq = equity[ok]
    idx = np.flatnonzero(ok)
    # ★ 前置一格「建仓前本金」：让累计收益从 0 起画（净值口径），返回时配的
    #   `nav_dates` 必须同步前置一格 —— 否则两者差一天（踩过：出图报
    #   `boolean index ... size 234 vs 235`，而且任何"按日对齐净值"的消费方
    #   都会整体错位一天，图上看不出来但数字全串了）。前置的那天用窗口前一个
    #   交易日（那天还没持仓，净值 = 本金，语义正好对上）。
    nav_dates = [dates[max(int(idx[0]) - 1, 0)]] + [dates[k] for k in idx]
    eq = np.concatenate([[capital], eq])
    ret = np.diff(eq) / eq[:-1]
    total = float(eq[-1] / capital - 1.0)
    n = len(ret)
    ann = float((eq[-1] / capital) ** (TRADING_DAYS / max(n, 1)) - 1.0)
    vol = float(np.std(ret, ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = float(np.mean(ret) / (np.std(ret, ddof=0) + 1e-12) * np.sqrt(TRADING_DAYS))
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    # ---- 稳定性层：H1/H2 分半 + bootstrap Sharpe 95%CI + P(Sharpe<0)
    #     小资金 + 短样本 ⇒ 单个数没有意义（参考工程把这条钉在墙上），必须给区间。
    half = max(1, n // 2)
    def _blk(a):
        r = np.diff(a) / a[:-1]
        return float(np.prod(1.0 + r) - 1.0) if len(r) else 0.0
    h1_ret, h2_ret = _blk(eq[:half + 1]), _blk(eq[half:])
    rng = np.random.default_rng(42)
    bs = np.empty(500)
    for b in range(500):
        r = ret[rng.integers(0, n, n)]
        bs[b] = np.mean(r) / (np.std(r, ddof=0) + 1e-12) * np.sqrt(TRADING_DAYS)
    sharpe_lo, sharpe_hi = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    p_neg = float((bs < 0).mean())

    sells = [t for t in trades if t["side"] == "sell"]
    wins = [t for t in sells if t["ret"] > 0]
    traded_notional = sum(t["price"] * t["shares"] for t in trades)
    avg_eq = float(np.mean(eq))
    return {
        "strategy": name, "params": params,
        "days": int(n), "start": dates[idx[0]], "end": dates[idx[-1]],
        "capital": float(capital), "final_equity": float(eq[-1]),
        "total_ret": total, "annual": ann, "vol": vol, "sharpe": sharpe,
        "max_dd": float(dd.min()),
        "h1_ret": h1_ret, "h2_ret": h2_ret,          # 分半（前/后半段累计收益，双正才算稳）
        "sharpe_ci": [sharpe_lo, sharpe_hi], "p_sharpe_neg": p_neg,
        "avg_positions": float(pos_count[ok].mean()),
        "max_positions_held": int(pos_count.max()),
        "exposure": float(np.mean(invested[ok] / np.maximum(equity[ok], 1e-9))),
        "n_trades": int(n_trades), "n_sells": len(sells),
        "win_rate": float(len(wins) / len(sells)) if sells else None,
        "avg_hold_days": float(np.mean([t["hold_days"] for t in sells])) if sells else None,
        "avg_win": float(np.mean([t["ret"] for t in wins])) if wins else None,
        "avg_loss": float(np.mean([t["ret"] for t in sells if t["ret"] <= 0])) if sells else None,
        "daily_turnover": float(traded_notional / max(avg_eq, 1e-9) / max(n, 1)),
        "total_cost": float(sum(t["fee"] for t in trades)),
        "cost_drag": float(sum(t["fee"] for t in trades) / capital),
        "equity": [round(float(x), 6) for x in eq],
        "nav_dates": nav_dates,
        "trades": trades[-40:],           # 只留最近 40 笔（完整成交明细在返回值里被截断）
    }


# ================================================================ 渲染
def render(res: dict) -> str:
    if res.get("error"):
        return f"  ✘ {res['strategy']}: {res['error']}"
    f = lambda v, p=4: "—" if v is None else f"{v:+.{p}f}"      # noqa: E731
    out = ["=" * 92,
           f"  策略 {res['strategy']} {res.get('params') or ''} · "
           f"{res['start']} ~ {res['end']}（{res['days']} 交易日）",
           "-" * 92,
           f"  期末净值 {res['final_equity']:,.0f} / 本金 {res['capital']:,.0f}"
           f" → 累计 {f(res['total_ret']*100, 2)}% · 年化 {f(res['annual']*100, 2)}%",
           f"  波动 {res['vol']*100:.2f}% · 夏普 {res['sharpe']:.2f} · "
           f"最大回撤 {res['max_dd']*100:.2f}%",
           f"  日均持仓 {res['avg_positions']:.2f} 只（上限 {res['max_positions_held']}）· "
           f"仓位暴露 {res['exposure']*100:.1f}%",
           f"  稳定性：H1 {res['h1_ret']*100:+.2f}% / H2 {res['h2_ret']*100:+.2f}% · "
           f"夏普 95%CI [{res['sharpe_ci'][0]:.2f}, {res['sharpe_ci'][1]:.2f}] · "
           f"P(夏普<0) = {res['p_sharpe_neg']*100:.0f}%",
           f"  成交 {res['n_trades']} 笔（卖 {res['n_sells']}）· "
           f"胜率 {('—' if res['win_rate'] is None else f'{res["win_rate"]*100:.1f}%')} · "
           f"平均持有 {('—' if res['avg_hold_days'] is None else f'{res["avg_hold_days"]:.1f}')} 天",
           f"  日均单边换手 {res['daily_turnover']*100:.2f}% · "
           f"总费用 {res['total_cost']:,.0f} 元（占本金 {res['cost_drag']*100:.2f}%）",
           "=" * 92]
    return "\n".join(out)


def nav_curve(res: dict, width: int = 64, height: int = 8) -> str:
    """ASCII 净值曲线（对齐 analyze/report 的朴素风格）。"""
    eq = res.get("equity") or []
    if len(eq) < 3:
        return ""
    e = np.asarray(eq, dtype=np.float64)
    lo, hi = float(e.min()), float(e.max())
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    rows = []
    for r in range(height, -1, -1):
        y = lo + (hi - lo) * r / height
        line = "".join("█" if abs(e[int(k * (len(e) - 1) / (width - 1))] - y) <= (hi - lo) / height / 2
                       else " " for k in range(width))
        rows.append(f"  {y:8.3f} |{line}")
    return "\n".join(rows)
