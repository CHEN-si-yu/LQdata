"""R4-T · V5 — 财报公告前抢跑结构（pre-announcement runup structure）。

三个预登记机制（`artifacts/research/round4_20261004/RESEARCH_BRIEF_B3.md` §V5 事前冻结，
**无变体额度**，执行时不得更改方向或机制集合）：

  1. `pre_ann_runup_20`        窗口内「公告前 20 交易日累计池内超额收益」。方向 false。
  2. `pre_ann_turnover_shock`  窗口内「最近 5 日换手均值 / 前 60 日（第 6–65 日）换手基线均值」。方向 false。
  3. `pre_ann_vol_expansion`   窗口内「最近 5 日收益标准差 / 前 60 日收益标准差基线」。方向 false。

方向先验（B3）：抢跑越强（超额收益越高、换手/波动放大越强）→ 公告落地后收益越低。

──────────────────────────────────────────────────────────────────────────────
窗口口径（执行前固定；探针实测见 delivery/r4t_preann_v1/logs/probe_preann.log）：

  对每只股票、每个季末报告期 p：
    * 法定截止日 dl(p)：Q1 → 04-30、H1 → 08-31、Q3 → 10-31、FY → 次年 04-30
      （沿用 r2_disclosure 规则）；
    * 预期公告日 E(p) = min(first_vis(p − 1 年) + 1 年, dl(p))：
      去年同期报告期的**首个**披露时点 + 整数一年（+10000），超出法定截止日则截到
      截止日；去年同期缺失 → 回退 dl(p)。（整数位移不经过 datetime 解析，无日期歧义。）
    * 窗口 = [E(p) − 20 个交易日, min(公告可见日前最后交易日, dl(p) + 60 日历日)]：
      公告可见 = pit 的**次一交易日**（pit = max(ann_date, f_ann_date, end_date)；
      公告日盘中不可用，保守从次日起算 —— 与 r4a 漂移锚点同一约定）；法定截止 + 60
      日历天为兜底关闭（公告迟迟不披露/数据缺失时窗口不无限延续）。
  「下一个未公告期」口径的并集表达：活跃(T) = 该交易日落在**任一**未公告期窗口内；
  某期公告可见后其窗口自然关闭，因此任何时刻活跃的窗口都只属于尚未公告的报告期。
  （同一股票多个未公告期间窗口在 3 月末–4 月天然重叠，并集即其自然表达，已披露。）

PIT 与 verify 一致性论证（截断 + 两阶段增量必须与全量逐值一致）：
  * E(p) 只用「去年同期首个披露时点」（早于 p 的预期窗口 ≥ 1 年）+ 法定日历，全部为过去信息；
  * 窗口关闭依赖当期公告，但实现按「可见则用其可见日前最后交易日、不可见（被审计截断
    或真缺失）一律退回 grace = dl + 60 日历天」：对任意单元格 T ≤ cutoff 且 vis 被截断
    （vis > cutoff ≥ T），全量口径的关闭日 min(vis 可见前日, grace) 在 T < vis 下等价于
    「以 grace 关闭」，与截断口径（行不可见 → grace）相同；对窗口能延续到 T 的行，其 E
    所依赖的去年同期披露必然 ≤ T − 1 年 < cutoff，两种口径都可见；
  * 时间轴一律走 `ctx.cal.days`（契约表、永不被审计的全历法）+ 面板坐标映射，与
    审计截断无关；因此「全量 / 截断（cutoff）/ 增量（[cutoff, end] 两阶段，末段 cutoff = end）」
    三种环境下活跃布尔与取值完全一致。
  * 数值全部 float64；窗口外为 NaN（主口径，缺「无即将公告」= 结构性缺失，不做零填充——
    项目规范：只有原始表用零表达「无」才可零填充）。「零填充」口径以**离线敏感性分析**
    报告（report §8），不占用候选预算。

──────────────────────────────────────────────────────────────────────────────
数据实测事实（2026-10-04，脚本 delivery/r4t_preann_v1/tools/probe_preann.py，冻结池口径）：
  * `stock_income`：池内 129,447 行，全部季末；同 (code, end) 多版本仅 4 组（max 2）；
    ann/f_ann 无 <1990 哨兵；2018–2025 每年 8460 行 = 4 × 2115（完整）；
    2018-01-01 前 ≥9 个报告期的股票 2094/2115；
  * E 规则（相关期 88,453 个）：来自去年同期 vis 87,407 个、回退法定截止日 1,046 个；
    vis − E 中位数 Q1/H1/Q3/FY = 0/0/0/0 天（share vis ≤ E = 58–64%，E 与实际披露
    基本同步，两侧尾部 FY 达 −26/+89 天）；公告迟到（vis > 法定截止日）5.02%，
    迟到者与 vis 缺失者在 grace 关闭；
  * 有效窗口 117,128 个（closed-by-vis 98,833、closed-by-grace 18,295）：
    窗口长度 p10/25/50/75/90 = 17/20/20/23/29 交易日，mean = 23.1；
  * 并集活跃（冻结池 ∩ listed）：2123 个交易日内 1050 日 ≥100 只活跃（49.5%），
    ≥300 只 879 日（41.4%）、≥500 只 806 日（38.0%）；全期均值 636 只/日、min 0
    （样本：20180115=11、20180410=2114、20230615=57）；月度均值：4 月 ~1900、
    10 月 ~1780、8 月 ~1460、3 月 ~1180、9 月 ~620、7 月 ~440、2 月 ~300，
    5/6 月与 11–12 月 ~0–8 只。**这是一个季节性稀疏候选**：NaN 条件口径下 G2
    覆盖日占比 49.5%（<60% 门槛，如实披露）；零填充口径 100% 覆盖但窗口外
    横截面恒零（G3 张力，见 report §5/§8 双口径，裁决留给 Lead）。
  * 2026-10-04 注：早期探针（probe_preann.py v1）两处「去年同期」合并键 off-by-one
    （E 退化为法定截止日），由 check_windows.py 逐日对照自检捕获并修复；本处数字为
    修复后口径，模块 `_active_windows` 与修复后对照逐值一致（2123 日全等）。
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

from fea import mathx as mx

SIGNALS = (
    "pre_ann_runup_20",
    "pre_ann_turnover_shock",
    "pre_ann_vol_expansion",
)
RUNUP_TD = 20        # 机制 1：公告前 20 个交易日累计超额收益
SHOCK_FAST = 5       # 机制 2/3：5 日快窗
BASE_SLOW = 60       # 机制 2/3：60 日基线（第 6–65 日，与快窗不重叠）
GRACE_DAYS = 60      # 窗口兜底：法定截止日 + 60 日历天
MIN_FAST = SHOCK_FAST // 2    # 房屋规则 min_count = n//2
MIN_SLOW = BASE_SLOW // 2
INCOME_COLS = ["stock_code", "ann_date", "f_ann_date", "end_date"]


def _divide(a, b, floor=1e-12):
    with np.errstate(all="ignore"):
        result = np.asarray(a, dtype=np.float64) / np.asarray(b, dtype=np.float64)
    return np.where(np.isfinite(result) & (np.abs(b) > floor), result, np.nan)


# ══════════════════════════════════════════════════════════════════════
# 日历与报告期工具
# ══════════════════════════════════════════════════════════════════════

def _statutory_deadline(end: np.ndarray) -> np.ndarray:
    """Q1 → 04-30、H1 → 08-31、Q3 → 10-31、FY → 次年 04-30（r2_disclosure 口径）。"""
    y = end // 10000
    m = (end // 100) % 100
    return np.where(m == 3, y * 10000 + 430,
           np.where(m == 6, y * 10000 + 831,
           np.where(m == 9, y * 10000 + 1031,
                           (y + 1) * 10000 + 430))).astype(np.int64)


def _add_calendar_days(dates: np.ndarray, n: int) -> np.ndarray:
    """YYYYMMDD 整数 + n 个日历天。逐唯一值用 datetime.date 计算。

    禁止 str → datetime64[D] 的捷径：numpy ≥ 2 会把 8 位 "YYYYMMDD" 解析成
    多年份（实测 "20230430" → 年份 20230430），产生错误偏移。
    """
    dates = np.asarray(dates, dtype=np.int64)
    uniq, inv = np.unique(dates, return_inverse=True)
    out = np.empty(uniq.size, dtype=np.int64)
    for i, v in enumerate(uniq):
        d0 = _dt.date(int(v) // 10000, (int(v) // 100) % 100, int(v) % 100)
        out[i] = int((d0 + _dt.timedelta(days=int(n))).strftime("%Y%m%d"))
    return out[inv]


def _quarter_ends(y0: int, y1: int) -> np.ndarray:
    out = []
    for y in range(y0, y1 + 1):
        out += [y * 10000 + 331, y * 10000 + 630, y * 10000 + 930, y * 10000 + 1231]
    return np.asarray(out, dtype=np.int64)


def _income_first_versions(ctx):
    """(stock_code, end_i, vis)：每个 (code, end) 的**首个**披露版本。

    vis = max(ann_date, f_ann_date, end_date)（与 fea/upstream 审计口径一致）；
    ann/f_ann 需 ≥1990-01-01（哨兵行视为未披露）；同一 (code, end) 有追溯修正时取
    最早版本（后续修正不重开窗口）。
    """
    frame = ctx.dataset("stock_income", columns=list(INCOME_COLS))
    if frame is None or frame.empty:
        return None
    frame = frame[frame["stock_code"].isin(ctx.panel.codes)]
    if frame.empty:
        return None
    ann = ctx.date_col(frame["ann_date"]).astype(np.int64)
    fann = ctx.date_col(frame["f_ann_date"]).astype(np.int64)
    end = ctx.date_col(frame["end_date"]).astype(np.int64)
    keep = np.isin(end % 10000, (331, 630, 930, 1231)) & ((ann >= 19900101) | (fann >= 19900101))
    if not keep.any():
        return None
    vis = np.maximum(np.maximum(ann, fann), end)[keep]
    rows = pd.DataFrame({"stock_code": frame["stock_code"].to_numpy()[keep],
                         "end_i": end[keep], "vis": vis})
    rows = rows.sort_values(["stock_code", "end_i", "vis"], kind="mergesort")
    return rows.drop_duplicates(["stock_code", "end_i"], keep="first")


# ══════════════════════════════════════════════════════════════════════
# 激活窗口（并集）
# ══════════════════════════════════════════════════════════════════════

def _active_windows(ctx):
    """(T, C) bool：交易日是否落在任一未公告报告期的窗口内；无收入行返回 None。

    端点在全历法 `ctx.cal.days` 上计算（契约表，永不被审计），再映射到面板坐标；
    qtr 网格取面板起止年份 ±1，覆盖所有与面板相交的窗口。
    """
    rows = _income_first_versions(ctx)
    if rows is None:
        return None
    cal = np.asarray(ctx.cal.days, dtype=np.int64)
    panel_days = np.asarray(ctx.panel.dates, dtype=np.int64)
    T = panel_days.size
    visible = rows[["stock_code", "end_i", "vis"]]
    prev = visible.rename(columns={"end_i": "prev_key", "vis": "vis_prev"})

    codes = np.asarray(ctx.panel.codes)
    y0 = int(panel_days[0]) // 10000
    y1 = int(panel_days[-1]) // 10000
    ends = _quarter_ends(y0 - 1, y1 + 1)
    grid = pd.DataFrame({"stock_code": np.repeat(codes, ends.size),
                         "end_i": np.tile(ends, codes.size)})
    grid = grid.merge(visible, on=["stock_code", "end_i"], how="left")
    grid["prev_key"] = grid["end_i"] - 10000
    grid = grid.merge(prev[["stock_code", "prev_key", "vis_prev"]],
                      on=["stock_code", "prev_key"], how="left")

    end_i = grid["end_i"].to_numpy(dtype=np.int64)
    dl = _statutory_deadline(end_i)
    vp = grid["vis_prev"].to_numpy(dtype=np.float64)
    shift = np.where(np.isfinite(vp), vp + 10000.0, np.nan)
    E = np.where(np.isfinite(shift), np.minimum(shift, dl), dl).astype(np.int64)
    vis = grid["vis"].to_numpy(dtype=np.float64)

    # 开盘：E − 20 个交易日（全历法），再映射到面板
    e_pos = np.searchsorted(cal, E, side="left")
    open_pos_cal = np.maximum(e_pos - RUNUP_TD, 0)
    open_pos = np.searchsorted(panel_days, cal[open_pos_cal], side="left")

    # 关闭 1：公告可见（pit 次一交易日）前的最后交易日
    vis_ok = np.isfinite(vis)
    close_vis = np.full(E.size, np.iinfo(np.int64).max, dtype=np.int64)
    if vis_ok.any():
        v = vis[vis_ok].astype(np.int64)
        t0 = np.minimum(np.searchsorted(cal, v, side="right"), cal.size)  # 严格大于 vis 的首日
        close_vis[vis_ok] = cal[np.maximum(t0 - 1, 0)]                    # vis 当日/之前最后交易日
    close_vis_pos = np.full(E.size, np.iinfo(np.int64).max, dtype=np.int64)
    if vis_ok.any():
        close_vis_pos[vis_ok] = np.searchsorted(panel_days, close_vis[vis_ok], side="right") - 1

    # 关闭 2（兜底）：法定截止 + 60 日历天（inclusive）
    grace_pos = np.searchsorted(panel_days, _add_calendar_days(dl, GRACE_DAYS), side="right") - 1
    close_pos = np.minimum(close_vis_pos, grace_pos)

    valid = (close_pos >= open_pos) & (open_pos <= T - 1)
    ci = ctx.code_index(grid["stock_code"].to_numpy(dtype=object))
    sel = valid & (ci >= 0)
    if not sel.any():
        return np.zeros((T, ctx.panel.C), dtype=bool)
    diff = np.zeros((T + 1, ctx.panel.C), dtype=np.int16)
    np.add.at(diff, (np.clip(open_pos[sel], 0, T), ci[sel]), 1)
    np.add.at(diff, (np.clip(close_pos[sel] + 1, 0, T), ci[sel]), -1)
    return np.cumsum(diff, axis=0)[:-1] > 0


# ══════════════════════════════════════════════════════════════════════
# 机制值（窗口内）
# ══════════════════════════════════════════════════════════════════════

def _pool_relative_returns(ctx):
    """日频后复权收益 − 池内已成交股票截面均值（与 r4a 同口径）。

    停牌日按项目口径贡献 0 收益（`ctx.ret` 既定语义），截面均值只用已成交股票；
    未上市/无价格 → NaN。
    """
    returns = np.asarray(ctx.ret(1), dtype=np.float64)
    listed = np.asarray(ctx.universe, dtype=bool)
    traded = np.asarray(ctx.traded(), dtype=bool)
    valid = listed & traded & np.isfinite(returns)
    count = valid.sum(axis=1)
    minimum = max(3, int(ctx.cfg.min_cross_section))
    mean = _divide(np.where(valid, returns, 0.0).sum(axis=1), count, floor=0)
    mean[count < minimum] = np.nan
    return np.where(listed & np.isfinite(returns), returns - mean[:, None], np.nan)


def _runup(ctx):
    """机制 1：窗口内每个交易日 T 的最近 20 个交易日累计池内超额收益。

    sum_{k=0..19} shift(rel_ret, k)；rel_ret 有 NaN（未上市）时累计为 NaN（不编造）。
    """
    relative = _pool_relative_returns(ctx)
    total = np.zeros_like(relative)
    for k in range(RUNUP_TD):
        total += mx.shift(relative, k)
    return total


def _turnover_shock(ctx):
    """机制 2：最近 5 日换手均值 / 前 60 日（第 6–65 日）基线均值。

    换手 = ctx.px("turnover")（vol / 前向填充 float_share × 100，%）；基线与快窗
    不重叠（shift 5 步）；分母 |baseline| ≤ 1e-6 → NaN。
    """
    turnover = np.asarray(ctx.px("turnover"), dtype=np.float64)
    fast = mx.roll_mean(turnover, SHOCK_FAST, min_count=MIN_FAST)
    slow = mx.roll_mean(turnover, BASE_SLOW, min_count=MIN_SLOW)
    baseline = mx.shift(slow, SHOCK_FAST)
    return ctx.safe_div(fast, baseline, min_abs_den=1e-6)


def _vol_expansion(ctx):
    """机制 3：最近 5 日收益标准差 / 前 60 日收益标准差基线。

    收益 = 已成交日 ctx.ret(1)，停牌日 NaN（vol_120 同口径）；std 用总体口径
    （ddof=0，mathx 默认）；分母 ≤ 1e-9 → NaN。
    """
    ret = np.where(np.asarray(ctx.traded(), dtype=bool),
                   np.asarray(ctx.ret(1), dtype=np.float64), np.nan)
    fast = mx.roll_std(ret, SHOCK_FAST, min_count=MIN_FAST)
    slow = mx.roll_std(ret, BASE_SLOW, min_count=MIN_SLOW)
    baseline = mx.shift(slow, SHOCK_FAST)
    return ctx.safe_div(fast, baseline, min_abs_den=1e-9)


# ══════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════

def build(parameters):
    """One of the preregistered V5 signals; directions fixed before any result was seen."""
    params = dict(parameters)
    signal = params.pop("signal", None)
    if signal not in SIGNALS:
        raise ValueError(f"Family permits only preregistered signals: {sorted(SIGNALS)}")
    if params:
        raise ValueError(f"Unregistered parameters: {sorted(params)}")

    def compute(ctx):
        active = _active_windows(ctx)
        if active is None:
            return ctx.panel.empty(dtype=np.float64)
        if signal == "pre_ann_runup_20":
            raw = _runup(ctx)
        elif signal == "pre_ann_turnover_shock":
            raw = _turnover_shock(ctx)
        else:
            raw = _vol_expansion(ctx)
        value = np.where(active, np.asarray(raw, dtype=np.float64), np.nan)
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute
