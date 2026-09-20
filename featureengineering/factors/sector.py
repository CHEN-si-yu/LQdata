"""板块 / 行业相对因子（17 个）—— **自建动态行业**，全部日频产出、只主板。

参考库有 15 个 sector 因子，本项目此前的覆盖是 **0 个**。难点不在公式，在**行业归属从哪来**。

═══════════════════════════════════════════════════════════════════════════
一、★★★ 为什么不用现成的行业成分股表
═══════════════════════════════════════════════════════════════════════════

上游有两张成分股表，**都是无日期的快照**：

  · `tdx_block_stocks`（13 万行，618 个板块 × 成分股）
  · `index_ths_constituent_stocks`（38 万行）

拿**今天的**成分股去算 2015 年的「行业内均值」就是**前视**：今天的行业划分
包含了「哪些公司后来被并入了这个行业」这一未来信息。
参考库自己就是因为这个原因**禁用了 6 个 sector 因子**
（`industry_relative_momentum_20` / `pb_industry_adjusted` / `ps_ttm_sector_neutral` /
`sector_amount_momentum_5d` / `sector_amount_rank` / `sector_mv_rank`，
见 `学习资料/factors.md` §六）。

**本文件的解法：完全不用成分股表，自建「滚动相关性动态行业」。**

═══════════════════════════════════════════════════════════════════════════
二、动态行业的构造（PIT 安全，只用历史数据）
═══════════════════════════════════════════════════════════════════════════

```
1. 行业清单 = tdx_blocks[block_type == 0].block_code（184 个）
   ★ 实测：板块**代码前缀与类型不一一对应**（8805/8806/8807/8808/8809 里
     0/1/2 三种类型混编）⇒ **必须**用 tdx_blocks 当字典筛，不能靠前缀。
   ★ 它只回答「哪些序号是行业指数」，**不**回答「哪只股票属于谁」——
     没有任何成分股信息，所以不引入第一种前视。
     （board 创建得晚 ⇒ 那条指数序列起点就晚，**不会回溯**。）

2. 行业日线收盘 ← `tdx_daily`（2010-01-04 起，618 个板块基本满格）

3. 每年 Y 的归属：取 Y 年首个交易日**之前**的 250 个交易日，
   corr(c, k) = corr(个股 c 的后复权日收益, 行业 k 的日收益)
   assign[Y, c] = 相关性最高的那个 k；峰值相关 < 0.10 或有效日 < 120 → 无归属(NaN)
   ★★ **只用 ≤ Y-01-01 的数据** ⇒ 时点安全、且跨日/跨次运行**可复现**
      （这是「增量 == 全量」的前提：归属是年份的纯函数，不随面板窗口漂移）

4. 逐日 ind_level[T, c] = 行业指数收盘(assign[year(T), c])
   ★ 用**指数点位**而不是累积收益：点位天然连续，不会因为中间某天缺数据
     把 cumprod 整段污染；行业 n 日收益 = level(T)/level(T−n) − 1 一步到位。

5. 行业内均值 / 离散：成员掩码 W (C × 184) 逐年，
   mean_k = (Σ_{c∈k} x_c) / (Σ_{c∈k} 1[x_c 有效]) —— **NaN 感知**，用两个矩阵乘实现
```

实测（2012 年归属，用 2010–2011 数据）：峰值相关 p10/p50/p90 = 0.50/0.64/0.78，
峰值 < 0.10 的股票只占 0.3%，最终用到 142 个不同行业、最大行业占 10.8% —— 分布合理。

═══════════════════════════════════════════════════════════════════════════
三、★ 与参考库 sector 族的**逐条偏离**（都是刻意的）
═══════════════════════════════════════════════════════════════════════════

1. **`ind_ret_ma_*`：参考库把它们当「行业动能」特征，本文件同样，
   但注意它们在截面上是**按行业分组的阶梯函数**（同行业所有股票同值）。
   对横断面排序仍有信息（模型可以学到「哪个行业在动」），但它的**有效自由度
   是 184 而不是 3484** —— 下游若做行业中性化要留意这一点，别误当成个股信号。

2. **`ind_disp_ma_*`：参考库的版本是「行业日收益**横截面** std 的 20 日均值」
   —— 那是一个**市场级常数**（每天全市场一个值），在横断面上的离散度恒为 0，
   **对排序任务零信息**（与 `factors/breadth.py` 模块 docstring §一 同一条理由）。
   ⇒ 本文件改成「**所属行业**成员日收益截面 std 的 20 日均值」：
     每只股票取**自己行业**的分化度，不同行业的股票拿到不同的值，
     截面上重新有区分度，语义也从「市场分化度」变成「我这行的分化度」。

3. **`rel_mom_ind_*`：参考库用 `个股收益 − 行业等权收益`（差）**，本文件沿用。
   ★ 行业收益 = **指数点位收益**（市值加权），不是等权平均 —— 这是与参考库的
     一处实现差异：参考库的 `ind_mean(df,"ret")` 是**等权**行业均值。
     选用指数口径的理由：指数点位是**上游直接给的**，不受本项目股票池
     （只主板、剔 ST、剔次新）影响，跨年可比；等权口径会随股票池规则变化而漂移。

4. **`rel_vol_ind_20d` / `rel_turnover_ind_20d`：参考库用**等权**行业内均值，
   本文件必须同样用等权（指数口径没有「波动」「换手」这两个量）—— 见 §二.5。

═══════════════════════════════════════════════════════════════════════════
四、性能与内存（实测）
═══════════════════════════════════════════════════════════════════════════

归属计算要读 `stock_daily` + `stock_adj_factor` 的**面板起点之前**那一两年，
所以每个 (因子, 年) 任务都会读一次。为避免重复，模块级按面板指纹记忆化
（同一 worker 进程内多个因子/多年任务可复用）。单次归属计算实测 < 2 s。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fea.spec import FactorSpec, register

SEC_START = None
W_3 = 30      # 3 交易日窗（≈5 天，够）
W_5 = 30
W_20 = 56
W_60 = 128
W_250 = 480   # 250 交易日窗：250×1.8+20

IND_DEPS = ("tdx_daily", "tdx_blocks", "stock_daily", "stock_adj_factor")
NOTURN = "stock_finance"

# 归属参数（见模块 docstring §二.3）
_LOOKBACK = 250        # 相关性窗口（交易日）
_MIN_OBS = 120         # 窗口内最少有效交易日
_MIN_PEAK = 0.10       # 峰值相关低于此 → 无归属

# ══════════════════════════════════════════════════════════════════════
# 自建动态行业
# ══════════════════════════════════════════════════════════════════════
_CACHE: dict[tuple, "_Sector"] = {}


class _Sector:
    """一次算清面板覆盖到的所有年份的「行业归属 + 行业指数点位」。

    构造完成后对外只有三样东西：
      · `level`  (T, C) 所属行业的指数点位
      · `ret1`   (T, C) 所属行业的**日**收益（= level/shift(level) − 1）
      · `W`      (C, K) 成员指示矩阵（逐年切换，见 `_member_mean`）
    """

    def __init__(self, ctx):
        self.ok = False
        panel = ctx.panel
        y0 = int(str(panel.dates[0])[:4])
        y1 = int(str(panel.dates[-1])[:4])
        self.T, self.C = panel.T, panel.C
        self.C_ = ctx

        codes = _ind_boards(ctx)
        if codes.size == 0:
            return
        self.kmap = {c: i for i, c in enumerate(codes)}
        K = codes.size

        # ---- 行业指数点位 → 面板网格（先按 (日期, 板块) 摊平，再逐列落格）----
        tx = ctx.dataset("tdx_daily", columns=["board_code", "trade_date", "close"],
                         years=(y0 - 1, y1))
        if tx is None or tx.empty:
            return
        bc = tx["board_code"].astype(str).str.replace(".TDX", "", regex=False)
        sel = bc.isin(self.kmap)
        tx = tx[sel]
        bc = bc[sel]
        day = ctx.date_col(tx["trade_date"])
        lv = pd.to_numeric(tx["close"], errors="coerce").to_numpy(np.float64)
        L = np.full((panel.T, K), np.nan)
        for code, k in self.kmap.items():
            m = (bc.to_numpy() == code)
            if m.any():
                L[:, k] = panel.place(np.zeros(int(m.sum()), dtype=np.int64),
                                      day[m], lv[m])[:, 0]

        # ---- 逐年归属（只用 ≤ 该年 1 月 1 日 的数据）----
        years = sorted({int(str(d)[:4]) for d in panel.dates})
        assigns = {Y: self._assign(ctx, Y, panel) for Y in years}

        # ---- 组装 (T, C)：所属行业的点位与日收益 ----
        lvl = np.full((self.T, self.C), np.nan)
        yr = np.array([int(str(d)[:4]) for d in panel.dates])
        for Y in years:
            a = assigns[Y]
            rm = (yr == Y)
            if not rm.any():
                continue
            col = np.where(a >= 0, a, 0)
            sub = L[np.ix_(np.flatnonzero(rm), col)]
            sub = np.where((a >= 0)[None, :], sub, np.nan)
            lvl[rm] = sub
        self.level = lvl
        prev = np.full_like(lvl, np.nan)
        prev[1:] = lvl[:-1]
        with np.errstate(all="ignore"):
            self.ret1 = np.where(np.isfinite(prev) & (prev > 0), lvl / prev - 1.0, np.nan)

        # ---- 成员掩码（逐年） ----
        self._assigns = assigns
        self._years = years
        self._K = K
        self.ok = True

    # ------------------------------------------------------------------
    def _assign(self, ctx, Y: int, panel) -> np.ndarray:
        """第 Y 年的行业归属 (C,) int64，−1 = 无归属。

        ★ 只读 `[Y−2, Y−1]` 两个年份分区并**再按日期截断**，保证是
          「≤ Y-01-01 的数据」的纯函数 —— 面板窗口怎么变都不影响结果，
          这是增量与全量逐格一致的前提。
        """
        cutoff = f"{Y:04d}-01-01"
        codes = panel.codes
        af = ctx.dataset("stock_adj_factor", years=(Y - 2, Y - 1))
        sd = ctx.dataset("stock_daily", columns=["stock_code", "trade_date", "close"],
                         years=(Y - 2, Y - 1))
        if sd is None or sd.empty or af is None or af.empty:
            return np.full(codes.size, -1, dtype=np.int64)
        sd = sd[sd["trade_date"] < cutoff]
        af = af[af["trade_date"] < cutoff]
        if sd.empty:
            return np.full(codes.size, -1, dtype=np.int64)
        df = sd.merge(af, on=["stock_code", "trade_date"], how="left")
        df["h"] = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64) * \
            pd.to_numeric(df["adj_factor"], errors="coerce").to_numpy(np.float64)
        df = df.sort_values(["stock_code", "trade_date"], kind="stable")
        df["r"] = df.groupby("stock_code", sort=False)["h"].pct_change()
        R = df.pivot_table(index="trade_date", columns="stock_code", values="r")
        # 只用最后 _LOOKBACK 个交易日
        R = R.iloc[-_LOOKBACK:]
        # 行业日收益
        with np.errstate(all="ignore"):
            MR = self._ind_ret(ctx, Y)
        common = R.index.intersection(MR.index)
        if common.size < _MIN_OBS:
            return np.full(codes.size, -1, dtype=np.int64)

        # 只保留面板内的股票（面板外的不需要算）
        pos = pd.Series(np.arange(codes.size), index=codes)
        cols = R.columns
        cidx = pos.reindex(cols).to_numpy()
        keep = np.isfinite(cidx)
        Rv = R.loc[common].to_numpy(np.float64)[:, keep]
        Mv = MR.loc[common].to_numpy(np.float64)
        out = np.full(codes.size, -1, dtype=np.int64)
        if Rv.shape[1] == 0 or Mv.shape[1] == 0:
            return out

        # 相关 = 标准化之后的内积（NaN 按 0 参与、分母用有效计数）
        okR = np.isfinite(Rv)
        okM = np.isfinite(Mv)
        nR = okR.sum(0)
        nM = okM.sum(0)
        Rz = np.where(okR, Rv - np.nansum(Rv, 0) / np.maximum(nR, 1), 0.0)
        Mz = np.where(okM, Mv - np.nansum(Mv, 0) / np.maximum(nM, 1), 0.0)
        Rz = np.where(okR, Rz, 0.0)
        Mz = np.where(okM, Mz, 0.0)
        Rn = np.sqrt((Rz * Rz).sum(0))
        Mn = np.sqrt((Mz * Mz).sum(0))
        C = (Rz / np.maximum(Rn, 1e-12)).T @ (Mz / np.maximum(Mn, 1e-12))
        # 有效共现日数：只有两侧都有值的日子才计数（否则相关被 0 稀释）
        CC = okR.astype(np.float64).T @ okM.astype(np.float64)
        C = np.where(CC >= _MIN_OBS, C, np.nan)
        # ★ 用 −inf 填充后直接 max/argmax，**不用** nanmax/nanargmax：
        #   后者在「整行全 NaN」（窗口里没有任何有效共现日）时会抛 RuntimeWarning，
        #   而这是**正常情形**（新上市 / 长期停牌），一个任务刷几百条警告会淹没日志。
        Cf = np.where(np.isfinite(C), C, -np.inf)
        best = Cf.max(axis=1) if C.shape[1] else np.full(C.shape[0], -np.inf)
        am = Cf.argmax(axis=1)
        ok = np.isfinite(best) & (best >= _MIN_PEAK)
        out[cidx[keep].astype(np.int64)] = np.where(ok, am, -1)
        return out

    def _ind_ret(self, ctx, Y: int) -> pd.DataFrame:
        """[Y−2, Y−1] 的行业日收益（宽表：index=trade_date, columns=board_code）。"""
        key = ("indret", Y)
        hit = _IDXRET.get(key)
        if hit is not None:
            return hit
        tx = ctx.dataset("tdx_daily", columns=["board_code", "trade_date", "close"],
                         years=(Y - 2, Y - 1))
        if tx is None or tx.empty:
            empty = pd.DataFrame()
            _IDXRET[key] = empty
            return empty
        bc = tx["board_code"].astype(str).str.replace(".TDX", "", regex=False)
        tx = tx.assign(board_code=bc)
        tx = tx[tx["board_code"].isin(self.kmap)]
        tx = tx.sort_values(["board_code", "trade_date"], kind="stable")
        tx = tx[tx["trade_date"] < f"{Y:04d}-01-01"]
        tx["r"] = tx.groupby("board_code", sort=False)["close"].pct_change()
        out = tx.pivot_table(index="trade_date", columns="board_code", values="r")
        _IDXRET[key] = out
        return out

    # ------------------------------------------------------------------
    def member_mask(self, Y: int) -> np.ndarray:
        """(C, K) 指示矩阵。无归属的股票整行为 0（不进入任何行业的均值）。"""
        a = self._assigns.get(Y)
        if a is None:
            return np.zeros((self.C, self._K), dtype=np.float64)
        W = np.zeros((self.C, self._K), dtype=np.float64)
        ok = a >= 0
        W[np.flatnonzero(ok), a[ok]] = 1.0
        return W

    def member_stat(self, x: np.ndarray, Y: int, kind: str = "mean") -> np.ndarray:
        """把 (T,C) 的个股量聚合成「所属行业的(有效)均值 / std」，返回 (T,C)。

        ★ NaN 感知：分母是**行业内该日有效值的个数**，不是行业成员数 ——
          否则停牌 / 未上市 / 退市会把行业均值系统性拉低。
        """
        W = self.member_mask(Y)
        okx = np.isfinite(x)
        xz = np.where(okx, x, 0.0)
        cnt = okx.astype(np.float64) @ W                       # (T, K)
        s1 = xz @ W                                            # (T, K)
        cnt = np.maximum(cnt, 1.0)
        m = s1 / cnt
        if kind == "std":
            s2 = (xz * xz) @ W
            v = np.maximum(s2 / cnt - m * m, 0.0)
            m = np.sqrt(v)
        a = self._assigns.get(Y)
        if a is None:
            return np.full_like(x, np.nan)
        col = np.where(a >= 0, a, 0)
        out = m[:, col]
        return np.where((cnt[:, col] > 0) if kind == "mean" else (cnt[:, col] > 1),
                        out, np.nan)

    def stat_by_row(self, x: np.ndarray, kind: str = "mean") -> np.ndarray:
        """逐年版本：面板跨年时逐段调用 `member_stat`。"""
        out = np.full(x.shape, np.nan)
        yr = np.array([int(str(d)[:4]) for d in self.C_.panel.dates])
        for Y in self._years:
            rm = np.flatnonzero(yr == Y)
            if rm.size:
                out[rm] = self.member_stat(x[rm], Y, kind)
        return out


_IDXRET: dict[tuple, pd.DataFrame] = {}


def _ind_boards(ctx) -> np.ndarray:
    """行业板块代码（block_type == 0）。★ 必须查字典，不能靠代码前缀（见 docstring §二.1）。"""
    key = "boards"
    hit = _IDXRET.get(key)
    if hit is not None:
        return hit
    bl = ctx.dataset("tdx_blocks", columns=["block_code", "block_type"])
    out = np.array([], dtype=object)
    if bl is not None and not bl.empty:
        out = np.sort(bl.loc[bl["block_type"] == 0, "block_code"].astype(str).unique())
    _IDXRET[key] = out
    return out


def _sec(ctx) -> _Sector:
    panel = ctx.panel
    key = (int(panel.dates[0]), int(panel.dates[-1]), int(panel.C))
    hit = _CACHE.get(key)
    if hit is None:
        hit = _Sector(ctx)
        _CACHE[key] = hit
        # 缓存只留最近一个面板指纹，避免长跑进程里按年累积（与 fea 的 trim 同理）
        if len(_CACHE) > 2:
            for k in list(_CACHE)[:-1]:
                _CACHE.pop(k, None)
    return hit


def _ind_ret(ctx, k: int) -> np.ndarray:
    """所属行业的 k 个交易日收益（用指数点位算，见 docstring §二.4）。"""
    S = _sec(ctx)
    if not S.ok:
        return ctx.panel.empty()
    lv = S.level
    prev = ctx.shift(lv, k)
    return ctx.safe_div(lv, prev, 1e-8) - 1.0


def _rel(ctx, k: int) -> np.ndarray:
    """行业相对动量 = 个股 k 日收益 − 所属行业 k 日收益。"""
    return ctx.ret(k) - _ind_ret(ctx, k)


# ══════════════════════════════════════════════════════════════════════
# 1. 行业相对动量（7 个）—— 参考库 cat-sector-c1 的核心族
# ══════════════════════════════════════════════════════════════════════

def _mk_rel(k: int, warm: int, note: str):
    name = f"rel_mom_ind_{k}d"

    def fn(ctx):
        return _rel(ctx, k)

    fn.__name__ = name
    register(FactorSpec(
        name=name, group="sector", deps=IND_DEPS,
        desc=f"行业相对动量（个股 {k} 日收益 − 所属行业 {k} 日收益）",
        formula=f"rel_mom_ind = ret_{k}d(stock) - ret_{k}d(industry_index)",
        start=SEC_START, warmup_days=warm, higher_is_better=True, note=note,
    ))(fn)
    return fn


for _k, _w in ((3, W_3), (5, W_5), (10, W_20), (20, W_20), (60, W_60), (250, W_250)):
    _mk_rel(_k, _w,
            "逐字抄参考库 cat-sector-c1 的同名因子定义（`个股 k 日收益 − 行业等权 k 日收益`）。"
            "★★ **行业归属是本项目自建的动态行业**（见模块 docstring §二），"
            "不是参考库那份无日期的快照成分股表 —— 参考库自己就是因为快照非 PIT "
            "才禁用了 6 个 sector 因子。"
            "★ 与参考库的实现差异：行业收益用**指数点位**（市值加权），"
            "参考库用**等权**行业均值；理由见模块 docstring §三.3。"
            "经济含义：剥离板块 β 之后的个股超额 —— 同一行业里跑赢同伴的股票"
            "延续性更强，绝对动量受行业轮动干扰大。")


@register(FactorSpec(
    name="ret_ind_rel_1d",
    group="sector",
    deps=IND_DEPS,
    desc="行业相对 1 日收益 = 个股当日收益 − 所属行业当日收益",
    formula="ind_ret = df.groupby(['Date','industry'])['ret'].transform('mean'); "
            "ret_ind_rel_1d = df['ret'] - ind_ret",
    start=SEC_START,
    warmup_days=W_3,
    higher_is_better=True,
    note=("逐字抄参考库 cat-sector-c1 `ret_ind_rel_1d`（唯一一个 1 日窗口的行业相对因子）。"
          "★★ 行业归属是自建动态行业（模块 docstring §二），非快照成分股。"
          "与 `rel_mom_ind_3d` 的关系：那是 3 日累计，本因子是**当日单日**超额 ——"
          "在 1 日预测期上信息衰减最快，短窗口是必要的（参考库把它单独放在"
          "`fac_cand_daily.py` 而不是常规族里，也是这个理由）。"),
))
def ret_ind_rel_1d(ctx):
    return _rel(ctx, 1)


# ══════════════════════════════════════════════════════════════════════
# 2. 行业动能（5 个）—— 参考库 `ind_ret_ma_*`
# ══════════════════════════════════════════════════════════════════════

def _mk_ind_ma(k: int, warm: int, enum: str):
    name = f"ind_ret_ma_{k}d"

    def fn(ctx):
        S = _sec(ctx)
        if not S.ok:
            return ctx.panel.empty()
        return ctx.roll_mean(S.ret1, k, max(1, k // 2))

    fn.__name__ = name
    register(FactorSpec(
        name=name, group="sector", deps=IND_DEPS,
        desc=f"所属行业的 {k} 日平均日收益（行业动能）",
        formula=f"ind_ret = industry_index_daily_return; factor = ts_mean(ind_ret, {k})",
        start=SEC_START, warmup_days=warm, higher_is_better=True,
        note=("抄参考库 cat-sector-c1 `ind_ret_ma_*`（`行业等权日收益的 k 日均值`）。"
              "★ **本因子的截面自由度是行业数（184）而不是股票数（3484）** ——"
              "同行业的所有股票拿到**同一个值**，是行业层的阶梯函数。"
              "对横断面排序仍有信息（模型能学到「哪个板块在动」），"
              "但下游做行业中性化 / 特征重要性时**必须知道这一点**，"
              "别把它当成个股信号。"
              "★ 行业收益用指数点位口径（模块 docstring §三.3），不是等权均值。"),
    ))(fn)
    return fn


for _k, _w in ((3, W_3), (5, W_5), (20, W_20), (60, W_60)):
    _mk_ind_ma(_k, _w, "")


@register(FactorSpec(
    name="ind_mom_accel",
    group="sector",
    deps=IND_DEPS,
    desc="行业动量加速度 = 行业 5 日均收益 − 行业 20 日均收益",
    formula="accel = ts_mean(ind_ret, 5) - ts_mean(ind_ret, 20)",
    start=SEC_START,
    warmup_days=W_20,
    higher_is_better=True,
    note=("参考库没有这一条，但它是 `ind_ret_ma_5d` 与 `ind_ret_ma_20d` 的自然组合"
          "（同族的 `momentum_accel_60_120` 在个股层已被删为 NOISE，"
          "但那是在**个股**层：个股短长动量差被噪声主导；行业层是 184 个成员的平均，"
          "噪声被压掉一个量级，同样的构念才站得住）。"
          "★ 截面自由度同样是行业数（184），见 `ind_ret_ma_*` 的 note。"
          "> 0 ⇒ 板块动能正在**抬升**（短均线高于长均线）。"),
))
def ind_mom_accel(ctx):
    S = _sec(ctx)
    if not S.ok:
        return ctx.panel.empty()
    return ctx.roll_mean(S.ret1, 5, 2) - ctx.roll_mean(S.ret1, 20, 10)


# ══════════════════════════════════════════════════════════════════════
# 3. 行业内相对波动 / 换手（2 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="rel_vol_ind_20d",
    group="sector",
    deps=IND_DEPS,
    desc="行业内相对波动 = 个股 20 日波动 − 所属行业成员 20 日波动均值",
    formula="vol20 = roll_std(ret, 20, min_periods=5); rel_vol_ind_20d = vol20 - ind_mean(vol20)",
    start=SEC_START,
    warmup_days=W_20,
    higher_is_better=False,
    note=("抄参考库 cat-sector-c1 `rel_vol_ind_20d`（参考库用等权行业内均值，本文件同口径 ——"
          "指数点位没有「波动」这一个量，只能用成员等权，见模块 docstring §三.4）。"
          "★ 成员均值是 **NaN 感知**的（分母 = 该日行业内有效值的个数，不是行业成员数），"
          "否则停牌 / 未上市 / 退市会把行业均值系统性拉低。"
          "与已注册的 `vol_120` / `idio_vol_60` 的区别：那两个是**绝对波动**"
          "（`idio_vol_60` 是相对沪深300 的残差波动），本因子是**行业内的相对**波动 ——"
          "剔除的是板块整体波动水平（某些行业天然波动大）。"),
))
def rel_vol_ind_20d(ctx):
    S = _sec(ctx)
    if not S.ok:
        return ctx.panel.empty()
    v = ctx.roll_std(ctx.ret(1), 20, 10)
    return v - S.stat_by_row(v, "mean")


@register(FactorSpec(
    name="rel_turnover_ind_20d",
    group="sector",
    deps=(*IND_DEPS, NOTURN),
    desc="行业内相对换手 = 个股 20 日平均换手率 − 所属行业成员均值",
    formula="rel_turnover_ind = turnover_rate - ind_mean(turnover_rate)",
    start=SEC_START,
    warmup_days=W_20,
    higher_is_better=False,
    note=("抄参考库 cat-sector-c1 `rel_turnover_ind`（原式是**当日**换手 − 行业均值；"
          "本因子取 20 日均值后再做差 —— 单日换手极噪（涨跌停、大额协议转让），"
          "20 日均值是参考库自己在 `rel_turnover_ind_ma20` 里用的口径）。"
          "★ 数据源 `stock_finance.turnover_rate`（供应商口径），"
          "不是自算 `vol/股本` —— 与已注册的 `turnover_f_20`（自由流通换手）"
          "的股本口径不同，两者**不是**同一个量的重标定。"
          "★ 剔除的是「板块整体交投活跃度」：高换手 = 资金关注但筹码交换剧烈，"
          "低换手 = 惜售/锁筹 —— 只有在**同一行业内部**比较才有意义。"),
))
def rel_turnover_ind_20d(ctx):
    S = _sec(ctx)
    if not S.ok:
        return ctx.panel.empty()
    panel = ctx.panel
    y0 = int(str(panel.dates[0])[:4])
    y1 = int(str(panel.dates[-1])[:4])
    df = ctx.dataset(NOTURN, columns=["stock_code", "trade_date", "turnover_rate"],
                     years=(y0, y1))
    if df is None or df.empty:
        return ctx.panel.empty()
    day = ctx.date_col(df["trade_date"])
    tv = pd.to_numeric(df["turnover_rate"], errors="coerce").to_numpy(np.float64)
    cidx = ctx.code_index(df["stock_code"].to_numpy())
    keep = cidx >= 0
    g = panel.place(cidx[keep].astype(np.int64), day[keep], tv[keep])
    tv20 = ctx.roll_mean(g, 20, 10)
    return tv20 - S.stat_by_row(tv20, "mean")


# ══════════════════════════════════════════════════════════════════════
# 4. 行业内分化度（2 个）—— ★ 口径已改，见 note
# ══════════════════════════════════════════════════════════════════════

def _mk_disp(k: int, warm: int):
    name = f"ind_disp_ma_{k}d"

    def fn(ctx):
        S = _sec(ctx)
        if not S.ok:
            return ctx.panel.empty()
        d = S.stat_by_row(ctx.ret(1), "std")
        return ctx.roll_mean(d, k, max(1, k // 2))

    fn.__name__ = name
    register(FactorSpec(
        name=name, group="sector", deps=IND_DEPS,
        desc=f"所属行业内部的收益分化度（行业成员截面 std 的 {k} 日均值）",
        formula=f"ind_disp = cross_std(ret within my industry); factor = ts_mean(ind_disp, {k})",
        start=SEC_START, warmup_days=warm, higher_is_better=False,
        note=("★★ **口径已改，与参考库不同，这是刻意的**："
              "参考库 cat-sector-c1 的 `ind_disp_ma_*` 是「**行业日收益横截面** std 的均值」"
              "—— 那是一个**市场级常数**（每天全市场一个值），"
              "在横断面上的离散度恒为 0，**对排序任务零信息**（与 "
              "`factors/breadth.py` 模块 docstring §一 同一条理由）。"
              "本文件改成「**所属行业**成员日收益截面 std 的均值」："
              "每只股票取**自己行业**的分化度，不同行业拿不同的值，截面重新有区分度。"
              "语义也随之从「今天市场分化不分化」变成「**我这行**今天分化不分化」。"
              "经济含义：行业内分化高 = 选股空间大但也更难；低 = 板块共振、适合 β 交易。"
              "方向取负（分化 = 不确定性）。"),
    ))(fn)
    return fn


for _k, _w in ((5, W_5), (20, W_20)):
    _mk_disp(_k, _w)


# ══════════════════════════════════════════════════════════════════════
# 5. 个股与所属行业的耦合强度（1 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="ind_beta_60",
    group="sector",
    deps=IND_DEPS,
    desc="个股对**所属行业**的 60 日 β（行业耦合强度）",
    formula="beta = roll_cov(ret, ind_ret, 60, 30) / roll_var(ind_ret, 60, 30)",
    start=SEC_START,
    warmup_days=W_60,
    higher_is_better=False,
    note=("**本文件新造**（参考库没有这一条；它的 `beta_*` 族全部对宽基指数）。"
          "★ 与已注册的 `beta_60`（对沪深300）**不是**同一个回归量："
          "沪深300 是大市值加权，行业指数是行业内市值加权；"
          "一只小盘股对沪深300 的 β 很低，但对自己行业的 β 可以很高。"
          "★ 本因子用**动态行业归属**（逐年变化）：β 的变动里既含"
          "「耦合真的变了」也含「归属换了行业」，**这是有意为之** ——"
          "归属本身是按历史相关性定的，换行业就意味着耦合结构真的重构了。"
          "经济含义：β 高 ⇒ 个股基本由板块驱动（选股空间小、更像 β 交易标的）；"
          "β 低 ⇒ 有独立行情（alpha 标的）。方向取负。"
          "60 日窗、min_count=30。"),
))
def ind_beta_60(ctx):
    S = _sec(ctx)
    if not S.ok:
        return ctx.panel.empty()
    return ctx.safe_div(ctx.roll_cov(ctx.ret(1), S.ret1, 60, 30),
                        ctx.roll_var(S.ret1, 60, 30), min_abs_den=1e-12)
