"""R9-D 股权质押风险：自身历史标准化的异常质押位移（唯一获授权候选）。

数据：上游只读表 `stock_pledge_stat`
      （stock_code, end_date, pledge_count, unrest_pledge, rest_pledge, total_share, pledge_ratio）

════════════════════════════════════════════════════════════════════════
一、立项依据与重蹈审计（写第一行代码之前完成）
════════════════════════════════════════════════════════════════════════
R2-I1（`r2_pledge_v2.py`，6 候选）与 R4 批 3（`r4s_pledge_v2_v1.py`，5 候选）
已经把这张表的 9 个构念全部证伪：
  R2：pledge_level / pledge_change(13周) / pledge_change_fast(5周) /
      pledge_price_stress（高质押 × 回撤交互）/ pledge_structure（rest vs unrest 结构）/
      pledge_level_lag5（被判定为**前视变体**：把发布滞后从 15 缩短到 5，增量 t 从 +1.02
      跳到 +2.52，而实测供应商滞后约 13 个交易日）
  R4s：pledge_change_accel / pledge_unpledge_asym / pledge_release_streak（+ zero/nan 两口径）
因此被指派的 D1（水平 + 价格下跌交互）、D2（变化与加速度）、D3（rest/unrest 构成）
**全部**是既有研究，执行者已如实上报并建议放弃；Lead 复核后**只授权一个新构念**：
**自身历史标准化的异常质押位移**（R2 测过水平与裸一阶差分，R4s 测过二阶与不对称，
**没有任何人测过除以自身历史波动的标准化量**）。

机制陈述：z 是「幅度 ÷ 该股自身波动」的信息比率式对象，与裸一阶差分 `pledge_change`
**不是同一个随机变量**，也不是它的再参数化。方向 higher_is_better=false
（异常加杠杆 = 预期收益越低），事前固定，不因任何 IC/收益结果翻转。

════════════════════════════════════════════════════════════════════════
二、可用性（PIT）—— 本单元最关键的风险，**必须当作未知局限，不是事实**
════════════════════════════════════════════════════════════════════════
`stock_pledge_stat` **只有 `end_date`**（周频快照截止日），上游**没有发布时刻合同**：
`conf/frequency.yaml` 把它标成 `delay_days: 0`，但该值是**没有实测支撑的默认值**，
本项目里本表 `gap_checkable: false`；`doc/AGENTS.md` F.3 亦确认不存在发布时刻字段。
⇒ **本表数据的真实可得时刻是未知的**，任何滞后选择都只能是一个假设。

本候选采用**保守假设**：`LAG_DEFAULT = 15` 个交易日。
依据（可复核的观测，不是文档推断）：R2 于 2026-09-30 22:17 实测上游落盘里最新
`end_date` = **2026-09-11**，即当时最新可见快照已陈旧 **13 个交易日**；
15 > 13，故该假设比直接观测到的最坏陈旧度更保守。
实现：`ctx.asof_daily`（只前向填充，绝不用 end_date 之后的未来值回填）
→ `ctx.lag_grid(..., 15)` → 信号。

**本候选不含任何更短滞后的敏感性变体**：R2 的 `pledge_level_lag5` 已被判定为前视变体，
按 Lead 指令不再重复该错误。

════════════════════════════════════════════════════════════════════════
三、表的真实形状（沿用 R2 的字段行为实测，本带用 tools/audit_proof.py 复测）
════════════════════════════════════════════════════════════════════════
· `end_date` 是**周频**截止日，且**不是每周都给全截面**：整截面日给全池并含大量显式 0 行；
  其余周只报有正质押余额的股票。R2 实测「本周为正质押者下周仍被报告」的概率 0.9964，
  而「两个整截面日均显式为 0」的股票在中间周被报告的概率约 0.001—0.01。
  ⇒ **某只股票在某周缺席 ⇔ 该周没有存续质押**，故本模块建 (股票 × 周) 稠密网格、缺席记 0。
  这是有字段行为证据的零填充，不是「无记录当无事件」的默认推断。
· `pledge_ratio` 是百分数且被厂商四舍五入到 0.01，本模块改用 float64 的
  `(unrest_pledge + rest_pledge) / total_share` 重算，避免量化台阶。

════════════════════════════════════════════════════════════════════════
四、框架缺陷绕行（BRIEF §11.1 / AGENTS.md F.1；**只在本模块进程内生效**）
════════════════════════════════════════════════════════════════════════
`fea/upstream.py::Upstream._audit_columns` 的日期列白名单只认
ann_date / f_ann_date / trade_time / trade_date / suspend_date / date，
而 `stock_pledge_stat` 只有 `end_date` ⇒ 任何设置 `audit_cutoff` 的运行（`research verify`
的截断阶段）会直接抛 ValueError，候选根本无法计算。

处置：在**本模块进程内**为该数据集显式声明日期列，仍委托框架原有的
`Upstream._audit_frame(mode="dated")` 执行截断 —— **不关闭审计、不直接读 parquet、
不修改任何共享源码**。为什么 `end_date` 可作为可得日期：本表是按周截止日发布的快照，
每一行的 `end_date` 是该行数据最早可能被公开知道的日历日；且本模块在其之上又整体滞后
15 个交易日，比框架自带的 `end_date <= cutoff` 更严。
截断生效的直接证据（逐行核对「end_date > cutoff 全丢、<= cutoff 全留」，**不是**
「前缀 == 前缀」的平凡比较）：`artifacts/research/round9_20261005/d_pledge/tools/audit_proof.py`。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── 事前冻结的常量（不因任何结果调整）────────────────────────────────
LAG_DEFAULT = 15      # 保守发布滞后（交易日）；> R2 实测最大陈旧度 13 个交易日
QUARTER_STEP = 65     # 13 周 ≈ 一个季度（交易日）；沿用 R2 冻结值
Z_LOOKBACK = 4        # 用于标准化自身历史波动率的「前序季度变化」个数（互不重叠）
Z_MIN_PRIOR = 3       # 至少要有 3 个有效前序季度变化，否则 z 未定义
STD_FLOOR = 1e-12     # 历史标准差下限；<= 此值视为无波动 -> z 未定义

# 允许的参数组合（单一口径，无敏感性变体）。build() 拒绝任何其它取值。
FROZEN = {("abnormal_pledge_shift", LAG_DEFAULT, QUARTER_STEP, Z_LOOKBACK, Z_MIN_PRIOR): "main"}

_COLUMNS = ("stock_code", "end_date", "unrest_pledge", "rest_pledge", "total_share")


def _divide(a, b, floor=1e-12):
    with np.errstate(all="ignore"):
        out = (np.asarray(a, dtype=np.float64) / np.asarray(b, dtype=np.float64))
    return np.where(np.isfinite(out) & (np.abs(b) > floor), out, np.nan)


def _lagged_level(ctx):
    """(T, C) 的滞后质押水平 P（百分点）：稠密周网格 -> as-of -> 整体滞后 15 交易日。"""
    days = np.asarray(ctx.panel.dates)
    frame = ctx.dataset("stock_pledge_stat", columns=list(_COLUMNS), years=None)
    if frame is None or len(frame) == 0:
        raise RuntimeError("stock_pledge_stat 在请求年份内没有可见行")

    codes = np.asarray(ctx.panel.codes).astype(str)
    code = frame["stock_code"].astype(str).to_numpy()
    end = np.asarray(ctx.date_col(frame["end_date"]), dtype=np.int64)
    unrest = pd.to_numeric(frame["unrest_pledge"], errors="coerce").to_numpy(np.float64)
    rest = pd.to_numeric(frame["rest_pledge"], errors="coerce").to_numpy(np.float64)
    total = pd.to_numeric(frame["total_share"], errors="coerce").to_numpy(np.float64)

    valid = (np.isfinite(unrest) & np.isfinite(rest) & np.isfinite(total)
             & (unrest >= 0) & (rest >= 0) & (total > 0) & (end >= 19000101))
    level = _divide(100.0 * (unrest + rest), np.where(valid, total, np.nan))
    code, end, level = code[valid], end[valid], level[valid]

    obs = pd.DataFrame({"code": code, "day": end, "level": level})
    obs = obs[obs["code"].isin(set(codes.tolist()))]
    if obs.duplicated(["code", "day"]).any():
        raise ValueError("stock_pledge_stat 在 (stock_code, end_date) 上有重复键")

    weeks = np.unique(obs["day"].to_numpy(np.int64))
    grid = pd.MultiIndex.from_product([codes, weeks], names=["code", "day"])
    dense = obs.set_index(["code", "day"]).reindex(grid)
    # 缺席 = 该周没有存续质押 = 0（见模块头 §3 的字段行为证据）
    dense["level"] = dense["level"].fillna(0.0)

    src_codes = dense.index.get_level_values("code").to_numpy().astype(str)
    src_days = dense.index.get_level_values("day").to_numpy().astype(np.int32)
    daily = ctx.asof_daily(src_codes, src_days, dense["level"].to_numpy(np.float64))
    return ctx.lag_grid(np.asarray(daily, dtype=np.float64), LAG_DEFAULT)


def abnormal_shift(p, step=QUARTER_STEP, lookback=Z_LOOKBACK, min_prior=Z_MIN_PRIOR):
    """在滞后水平面板 p 上算自身历史标准化的异常季度位移。

    当前季度变化 d0(t) = p(t) - p(t-step)；
    前序 lookback 个**互不重叠**的季度变化 h_k(t) = p(t-k*step) - p(t-(k+1)*step)；
    z = (d0 - mean(h)) / std(h, ddof=1)，要求有效 h 个数 >= min_prior 且 std > STD_FLOOR。
    """
    import numpy as _np

    def sh(x, n):
        out = _np.full_like(x, _np.nan)
        if n < x.shape[0]:
            out[n:] = x[:-n]
        return out

    d0 = p - sh(p, step)
    hist = []
    for k in range(1, lookback + 1):
        hist.append(sh(p, k * step) - sh(p, (k + 1) * step))
    if not hist:
        return _np.full_like(p, _np.nan)
    H = _np.stack(hist, axis=0)
    finite = _np.isfinite(H)
    cnt = finite.sum(axis=0)
    filled = _np.where(finite, H, 0.0)
    mean = _np.where(cnt > 0, filled.sum(axis=0) / _np.maximum(cnt, 1), _np.nan)
    dev = _np.where(finite, H - mean[None, :, :], 0.0)
    var = _np.where(cnt > 1, (dev ** 2).sum(axis=0) / _np.maximum(cnt - 1, 1), _np.nan)
    sd = _np.sqrt(var)
    ok = (_np.isfinite(d0) & _np.isfinite(mean) & _np.isfinite(sd)
          & (cnt >= min_prior) & (sd > STD_FLOOR))
    with _np.errstate(all="ignore"):
        z = (d0 - mean) / sd
    return _np.where(ok & _np.isfinite(z), z, _np.nan)


def build(parameters):
    """R9-D 唯一获授权口径；不含被搜索的窗口、不含更短滞后的前视变体。"""
    key = (parameters.get("signal"), parameters.get("lag"), parameters.get("step"),
           parameters.get("z_lookback"), parameters.get("z_min_prior"))
    if set(parameters) != {"signal", "lag", "step", "z_lookback", "z_min_prior"} or key not in FROZEN:
        raise ValueError(
            "r9d_abnormal_pledge_shift_v1 只允许唯一事前登记口径 "
            "abnormal_pledge_shift(lag=15, step=65, z_lookback=4, z_min_prior=3)")
    step = int(parameters["step"])
    lookback = int(parameters["z_lookback"])
    min_prior = int(parameters["z_min_prior"])

    def compute(ctx):
        p = _lagged_level(ctx)
        value = abnormal_shift(p, step=step, lookback=lookback, min_prior=min_prior)
        universe = np.asarray(ctx.universe, dtype=bool)
        value = np.where(universe, value, np.nan)
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute


# ══════════════════════════════════════════════════════════════════════
# 读取适配：为 stock_pledge_stat 声明它真实的可得日期列（BRIEF §11.1）
#
# 只影响本模块进程；其它数据集的判定逐字不变（audit_proof.py 逐数据集核对）。
# 委托框架原有的 Upstream._audit_frame(mode="dated") 完成截断：
#     known = end_date 可解析且 > 0；保留 end_date <= audit_cutoff 的行。
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
