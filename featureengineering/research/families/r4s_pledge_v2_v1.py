# ══════════════════════════════════════════════════════════════════════
# R4 批 3 · V4 —— 质押二阶：加速度与增解不对称（family r4s_pledge_v2_v1）
#
# 数据：上游只读表 `stock_pledge_stat`（周频快照；字段 stock_code / end_date /
#       pledge_count / unrest_pledge / rest_pledge / total_share / pledge_ratio）
#
# ── 血统与立项依据 ─────────────────────────────────────────────────────
# R2-I1（`research/families/r2_pledge_v2.py`，6 候选）已把「质押比例水平 /
# 变化（13 周、5 周两窗口）/ 高质押×回撤交互 / 限售质押结构」全部证伪：
# declared 侧个别 9/9 同号，但 Lead 的 Fama-MacBeth 增量 t 全不显著或反向
# （pledge_level t*=+1.02/+1.71、structure −1.76/−1.65…，见 doc/历史报告.md
# R2 章与 dsh_scratch/round2/pledge/report.md §5.8/§9）。R2 §9 明示：质押变化的
# **加速度**与**增解不对称**是尚未检验的二阶机制，可作为新一轮事前假设另行
# 预登记——本单元即按该授权立项，**不是**对失败机制换窗口重试。
#
# ── 三个固定机制（Lead 事前登记；方向不因任何 IC 结果改变）────────────
# 记 P(t) = 100 × (unrest_pledge + rest_pledge) / total_share（百分点），
# 经 (股票 × 周) 网格 → as-of 前向填充 → 整体滞后 LAG_DEFAULT=15 个交易日后的
# 可得水平序列（下称"滞后的水平序列"，含本条信息的信号日 = 报告周之后 ≥15 个
# 交易日，绝无未来回填）。
#
# 1) pledge_change_accel（false）：A(t) = P(t) − 2·P(t−S) + P(t−2S)，
#    S = 20 交易日（4 周 ≈ 1 个月）。"本期变化 − 上期变化"，即月度二阶差分。
#    值越高 = 质押加杠杆在加速 = 预期未来收益越低。
# 2) pledge_unpledge_asym（false）：对滞后的水平序列取逐日一阶差分 d，
#    在 W = 65 交易日（13 周 ≈ 1 季）窗口上分解 U = Σ max(d, 0)（增质幅度）、
#    D = Σ max(−d, 0)（解质幅度），A = (U − D) / (U + D) ∈ [−1, +1]；
#    U + D = 0（窗口内无任何变动）记 NaN——"无路径"不是"对称"。
#    值越高 = 增质主导 = 预期未来收益越低。
# 3) pledge_release_streak（true）：截至最近一条报告，**连续**严格递减
#    （L_w < L_{w−1}，在 (股票 × 周) 网格上逐周判定）的周数，cap = 13 周。
#    值越高 = 解押持续 = 风险解除 → 预期未来收益越高。
#
# 机制间关系预判（R3 伪发现模式 4 要求）：三者为同一数据源的不同构念——
# 加速度是水平的二阶导；不对称是有界路径统计（对净变化做了总变差归一，
# 与一阶净变化本身不同）；streak 是离散持续性计数。预期两两 |rho| 中等以下，
# 实测在报告 §6 呈现。
#
# ── PIT 纪律（R2 血泪，本单元硬要求）──────────────────────────────────
# ・`end_date` 只是**截止日**，上游没有发布时刻字段（doc/AGENTS.md F.3）。
# ・R2 实测供应商发布滞后约 13 个交易日（2026-09-30 22:17 落盘里最新
#   end_date = 2026-09-11）；本带在 tools/probe_data.py 中重新实测该水位。
# ・本模块统一使用**保守滞后 LAG_DEFAULT = 15 个交易日**（> 观测到的 13）。
# ・**禁止**复用 R2 被证伪的 lag5 口径：lag5 把发布滞后从 15 缩到 5，
#   增量 t 从 +1.02 跳到 +2.52，而实测供应商滞后约 13 个交易日，属前视变体
#   （doc/历史报告.md R2 §5.5 与 R3 伪发现模式 2）。本模块不含任何 lag<13 的口径。
# ・管线：ctx.asof_daily（只前向填充，绝不用 end_date 之后的未来值回填）
#   → ctx.lag_grid(15) → 信号。两腿共用同一滞后（差值内部不自带额外信息）。
#
# ── 缺失口径（R3 伪发现模式 5：双口径并列报告）────────────────────────
# ・zero 口径（主口径）：(股票 × 周) 稠密网格，缺席记 0。依据是 R2 的字段行为
#   实测（「本周为正质押者下周仍被报告」概率 0.9964；显式 0 行 18,394 vs 821），
#   本带在 tools/probe_data.py 中用当前数据**重新实测**，不引用文档数值。
# ・nan 口径（敏感性）：缺席不记 0——只用真实报告行做 as-of（无新报告即沿用
#   最后一条报告值；从未报告 = NaN）。这是对 R2 §8.3 陷阱的直接检验：朴素
#   asof 会把"还清质押后不再被报告"的股票永远挂着旧的高质押比例。
#   nan 口径下的 streak 只在真实报告周上更新（报告序列上的连续递减），
#   报告缺席不重置计数；as-of 沿用最近一次报告算出的 streak。
#
# ── 框架绕行（doc/AGENTS.md F.1；仅在本模块进程内生效）──────────────────
# fea/upstream.py::Upstream._audit_columns 的日期列白名单只认
# ann_date / f_ann_date / trade_time / trade_date / suspend_date / date；
# stock_pledge_stat 只有 end_date（周频快照截止日）⇒ 任何设置 audit_cutoff 的
# research run 直接抛 ValueError，候选根本无法计算。处置：在**本模块进程内**
# 为该数据集显式声明日期列，仍委托框架原有的 Upstream._audit_frame(mode="dated")
# 执行截断——不关闭审计、不直接读 parquet、不改任何共享源码。
# 截断生效的直接证据（非"前缀==前缀"的平凡比较）：见交付目录
# tools/audit_proof.py —— 调用真实的 _audit_columns / _audit_frame 与
# Upstream.read 完整路径，逐行核对「end_date > cutoff 全丢、<= cutoff 全留」。
#
# ══════════════════════════════════════════════════════════════════════
from __future__ import annotations

import numpy as np
import pandas as pd

from fea import mathx as mx
from fea.dates import series_to_int

# ── 事前冻结的常量（不因任何结果调整）────────────────────────────────
LAG_DEFAULT = 15        # 保守发布滞后（交易日）；R2 实测供应商滞后约 13
ACCEL_STEP = 20         # 机制 1 主口径：4 周 ≈ 1 个月（交易日）
ACCEL_STEP_Q = 65       # 机制 1 敏感性（同代码复算，非候选）：13 周 ≈ 1 季
ASYM_WIN = 65           # 机制 2 主口径：13 周（交易日）
ASYM_WIN_M = 20         # 机制 2 敏感性（同代码复算，非候选）：4 周
STREAK_CAP = 13         # 机制 3：连续解押周数上限（13 周 ≈ 1 季）

# 已登记候选（5 个，family 预算 5）+ 仅用于执行者侧双口径/窗口敏感性复算的
# 3 个同代码口径（不登记为候选，见交付报告 §4/§8 的预登记说明）。
SIGNALS = {
    # id                        kind     missing   参数
    "pledge_change_accel":      ("accel", "zero", ACCEL_STEP),
    "pledge_change_accel_nan":  ("accel", "nan",  ACCEL_STEP),
    "pledge_change_accel_q":    ("accel", "zero", ACCEL_STEP_Q),
    "pledge_unpledge_asym":     ("asym",  "zero", ASYM_WIN),
    "pledge_unpledge_asym_nan": ("asym",  "nan",  ASYM_WIN),
    "pledge_unpledge_asym_m":   ("asym",  "zero", ASYM_WIN_M),
    "pledge_release_streak":    ("streak", "zero", STREAK_CAP),
    "pledge_release_streak_nan": ("streak", "nan", STREAK_CAP),
}

# 本模块只请求必要列；审计截断会自行补上 end_date（见 _audit_frame 调用链）。
_COLUMNS = ("stock_code", "end_date", "unrest_pledge", "rest_pledge", "total_share")


def _divide(a, b, floor=1e-12):
    with np.errstate(all="ignore"):
        out = (np.asarray(a, dtype=np.float64) /
               np.asarray(b, dtype=np.float64))
    return np.where(np.isfinite(out) & (np.abs(b) > floor), out, np.nan)


# ══════════════════════════════════════════════════════════════════════
# 1) 读取 + 清洗：记录级
# ══════════════════════════════════════════════════════════════════════

def _read_frame(ctx):
    days = np.asarray(ctx.panel.dates)
    years = (int(days[0]) // 10000, int(days[-1]) // 10000)
    frame = ctx.dataset("stock_pledge_stat", columns=list(_COLUMNS), years=years)
    if frame is None or len(frame) == 0:
        raise RuntimeError("stock_pledge_stat 在请求年份内没有可见行")
    return frame


def _clean_records(frame, codes):
    """原始行 → (列下标 ci, 周日期 day, 水平 level 百分点)；只保留池内有效行。

    level = 100 × (unrest_pledge + rest_pledge) / total_share（float64 重算，
    不用 pledge_ratio——厂商把它四舍五入到 0.01%，会在低质押区制造假并列）。
    无效判定：非池内 / day < 19000101 / 任一字段非有限 / unrest、rest 为负 /
    total_share ≤ 0。实测这些无效行数会写进交付报告（R2 实测为 0，本带重测）。
    """
    codes_arr = np.asarray(codes).astype(str)
    pos = pd.Series(np.arange(codes_arr.size, dtype=np.int64), index=codes_arr)
    code = frame["stock_code"].astype(str)
    ci = pos.reindex(code.to_numpy()).to_numpy(dtype=np.float64)
    ci = np.where(np.isfinite(ci), ci, -1.0).astype(np.int64)

    day = np.asarray(series_to_int(frame["end_date"]), dtype=np.int64)
    unrest = pd.to_numeric(frame["unrest_pledge"], errors="coerce").to_numpy(np.float64)
    rest = pd.to_numeric(frame["rest_pledge"], errors="coerce").to_numpy(np.float64)
    total = pd.to_numeric(frame["total_share"], errors="coerce").to_numpy(np.float64)

    valid = ((ci >= 0) & (day >= 19000101)
             & np.isfinite(unrest) & np.isfinite(rest) & np.isfinite(total)
             & (unrest >= 0) & (rest >= 0) & (total > 0))
    pledged = np.where(valid, unrest + rest, np.nan)
    level = _divide(100.0 * pledged, np.where(valid, total, np.nan))
    keep = valid & np.isfinite(level)
    return ci[keep], day[keep], level[keep]


def _weekly_matrix(codes, ci, day, level):
    """(W, C) 周水平矩阵：缺席为 NaN；返回 (weeks, mat)。

    要求 (股票, 周) 唯一——上游实测 0 重复；若出现重复直接报错（不静默取一条）。
    """
    weeks = np.unique(day.astype(np.int64))
    key = ci.astype(np.int64) * 100000000 + day.astype(np.int64)
    if np.unique(key).size != key.size:
        raise ValueError("stock_pledge_stat 在 (stock_code, end_date) 上有重复键")
    mat = np.full((weeks.size, codes.size), np.nan, dtype=np.float64)
    mat[np.searchsorted(weeks, day), ci] = level
    return weeks, mat


# ══════════════════════════════════════════════════════════════════════
# 2) 周网格 → 日频（as-of + 保守滞后）。zero / nan 两种缺失口径共用此函数。
# ══════════════════════════════════════════════════════════════════════

def _asof_daily_from_weekly(ctx, codes, weeks, mat, missing):
    """把 (股票 × 周) 网格 as-of 前向填充到 (T, C) 日频面板。

    zero 口径：缺席格先记 0（整网格都是观测）后 as-of。
    nan  口径：只用**有限值**格（真实报告行）做 as-of；缺席 = 不观测，
              日频取"最近一条报告"，从未报告 = NaN。
    """
    W, C = mat.shape
    vals = mat.reshape(-1).astype(np.float64)
    if missing == "zero":
        vals = np.where(np.isfinite(vals), vals, 0.0)
        keep = np.ones(vals.size, dtype=bool)
    elif missing == "nan":
        keep = np.isfinite(vals)
    else:
        raise ValueError(f"未知缺失口径：{missing!r}")
    wgrid = np.repeat(weeks, C)[keep]
    cgrid = np.tile(np.arange(C, dtype=np.int64), W)[keep]
    src_codes = np.asarray(codes).astype(str)[cgrid]
    daily = ctx.asof_daily(src_codes, wgrid, vals[keep])
    return np.asarray(daily, dtype=np.float64)


def _lagged_level(ctx, missing):
    """滞后的水平序列 P（(T,C) float64）：as-of → 整体滞后 LAG_DEFAULT 交易日。"""
    frame = _read_frame(ctx)
    codes = np.asarray(ctx.panel.codes).astype(str)
    ci, day, level = _clean_records(frame, codes)
    weeks, mat = _weekly_matrix(codes, ci, day, level)
    daily = _asof_daily_from_weekly(ctx, codes, weeks, mat, missing)
    return ctx.lag_grid(daily, LAG_DEFAULT)


# ══════════════════════════════════════════════════════════════════════
# 3) 三个机制的信号函数（对滞后水平序列 P 计算；纯矩阵运算，便于同代码复算）
# ══════════════════════════════════════════════════════════════════════

def signal_accel(p, step):
    """P(t) − 2·P(t−step) + P(t−2·step)（百分点²，二阶差分）。"""
    return mx.diff(mx.diff(p, step), step)


def signal_asym(p, win):
    """窗口 win 内增质/解质幅度不对称 (U−D)/(U+D)；U+D=0 → NaN。"""
    d = mx.diff(p, 1)
    up = mx.roll_sum(np.where(np.isfinite(d), np.maximum(d, 0.0), np.nan), win)
    dn = mx.roll_sum(np.where(np.isfinite(d), np.maximum(-d, 0.0), np.nan), win)
    tot = up + dn
    with np.errstate(all="ignore"):
        out = (up - dn) / tot
    # tot 为 0 或 NaN → NaN（"无路径"不是"对称"）
    return np.where(np.isfinite(tot) & (tot > 0.0), out, np.nan)


def _streak_weekly(mat, cap):
    """逐周连续严格递减计数（cap 上限）；返回与 mat 同形的逐周 streak。

    规则（事前冻结）：
    ・zero 口径下 mat 已把缺席填 0——缺席是"没有存续质押"的观测，
      正质押 → 0 的整周记为一次解押。
    ・nan 口径下缺席为 NaN——不观测、不发射（该周输出 NaN，由 as-of 沿用上一条
      报告的值）；真实报告之间跨缺席周继续累积（报告序列意义上的"连续"）。
    ・严格递减（cur < last），不设幅度阈值；相等或上升 → 归零。
    """
    W, C = mat.shape
    out = np.full((W, C), np.nan, dtype=np.float64)
    run = np.zeros(C, dtype=np.float64)
    last = np.full(C, np.nan, dtype=np.float64)
    cap_f = float(cap)
    for i in range(W):
        cur = mat[i]
        finite = np.isfinite(cur)
        dec = finite & np.isfinite(last) & (cur < last)
        run = np.where(dec, np.minimum(run + 1.0, cap_f),
                       np.where(finite, 0.0, run))
        out[i] = np.where(finite, run, np.nan)
        last = np.where(finite, cur, last)
    return out


def _lagged_streak(ctx, missing, cap):
    frame = _read_frame(ctx)
    codes = np.asarray(ctx.panel.codes).astype(str)
    ci, day, level = _clean_records(frame, codes)
    weeks, mat = _weekly_matrix(codes, ci, day, level)
    if missing == "zero":
        mat = np.where(np.isfinite(mat), mat, 0.0)
    wk = _streak_weekly(mat, cap)
    daily = _asof_daily_from_weekly(ctx, codes, weeks, wk, missing)
    return ctx.lag_grid(daily, LAG_DEFAULT)


# ══════════════════════════════════════════════════════════════════════
# 4) 构造器
# ══════════════════════════════════════════════════════════════════════

def build(parameters):
    """八个事前登记 signal 标识之一；不含任何被搜索的窗口或方向。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("r4s_pledge_v2_v1 只允许八个事前登记的 signal 标识")
    kind, missing, param = SIGNALS[parameters["signal"]]

    def compute(ctx):
        if kind == "streak":
            value = _lagged_streak(ctx, missing, param)
        else:
            p = _lagged_level(ctx, missing)
            value = signal_accel(p, param) if kind == "accel" else signal_asym(p, param)
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute


# ══════════════════════════════════════════════════════════════════════
# 5) 读取适配：为 stock_pledge_stat 声明它真实的可得日期列（AGENTS.md F.1）
#
# 只影响本模块进程；其它数据集判定逐字不变（audit_proof.py 逐数据集核对）。
# 委托框架原有的 Upstream._audit_frame(mode="dated") 完成截断：
#     known = end_date 可解析且 > 0；保留 end_date <= audit_cutoff 的行。
# 为什么 end_date 可作为可得日期：本表是按周截止日发布的快照，每一行的
# end_date 是该行数据最早可能被公开知道的日历日；本模块在其之上再整体滞后
# 15 个交易日，比框架自带的 end_date <= cutoff 更严。
# ══════════════════════════════════════════════════════════════════════
from fea import upstream as _fea_upstream  # noqa: E402

_ORIGINAL_AUDIT_COLUMNS = _fea_upstream.Upstream._audit_columns


def _audit_columns_with_pledge_end_date(name, names):
    """只给 stock_pledge_stat 增加 end_date 分支；其余数据集原样转发。"""
    if name == "stock_pledge_stat" and "end_date" in names:
        return "dated", ["end_date"]
    return _ORIGINAL_AUDIT_COLUMNS(name, names)


if getattr(_fea_upstream.Upstream._audit_columns, "__name__", "") != (
        "_audit_columns_with_pledge_end_date"):
    _fea_upstream.Upstream._audit_columns = staticmethod(
        _audit_columns_with_pledge_end_date)
