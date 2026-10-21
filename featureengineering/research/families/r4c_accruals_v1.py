"""U3 · r4c_accruals_v1 —— 应计项目与盈余质量（Round 4 批 1，机制/方向事前冻结）。

六个预登记信号（4 机制 + 2 变体，方向在任务书中固定，未按任何结果翻转）：
    accrual_ratio_ttm            false  (归母净利TTM − 经营净现金流TTM) / 总资产
    accrual_ratio_q              false  同上表达的单季口径（自算累计差分）
    earnings_conversion_trend    true   OCF/NI 最近4个单季均值 − 前4个单季均值（转化率改善）
    impairment_to_assets         false  |资产减值损失累计|的单季变动 / 总资产
    earnings_conversion_ttm_gap  true   变体：TTM 口径 OCF/NI 同比变化（复杂度对照）
    impairment_to_assets_merged  false  变体：合并信用减值损失口径（范围对照）

单季（单季度）口径的实现说明 —— 改本文件前先读完：
  财务层 `ctx.ttm()/ctx.point()` 只暴露 TTM 与时点值，**没有任何** ctx 入口能给出
  单季水平。本模块按任务书 §2.1「若自算（累计差分）必须说明与供应商单季字段的差异」
  自建单季网格（注意**财年重置**：累计值每年 Q1 归零，Q1 的单季值就是累计本身，
  绝不能拿 Q1 去减上一年年报）：
      S(报告期 q) = cum(q)                       若 q 为 Q1（q % 4 == 0）
                  = cum(q) − cum(q−1)            否则（同一财年内差分）
  其中 q 取值自当日已知的最新报告期 Q 往前 k 期；累计口径与 `fea/deriv.py` 一致：
    · 只保留季末 end_date（331/630/930/1231）；pit = max(ann_date, f_ann_date, end_date)；
    · 公告日不可考（ann 与 f_ann 都 < 1990）的行丢弃；保留历次修订，取 pit ≤ 当日的最新一版；
    · 事件网格 = 该报表每一批 (code, pit) 公告；每个事件上把「当期报告」重新评估
      （与 deriv._revision_events 的语义一致），从而旧期修订也会在它的公告时点影响
      当期单季值 —— 与 ttm() 对修订的响应方式相同；
    · 单季值只在 q 与 q−1 两个累计值都可考时给出，否则 NaN（不做 partial-sum 回退，
      与 deriv._ttm_at 的 fallback 是刻意差异）；
    · 读取走 `ctx.dataset`（Upstream 审计层 + research audit_cutoff 截断），落格走
      `ctx.panel.asof(..., dtype=float64)`（绕开 asof_daily 的 float32 默认，
      财务金额 ~1e11 量级在 float32 下只有 7 位有效数字）。
  与供应商单季字段的差异：白名单里只有 5 个单季比率（q_roe/q_dt_roe/q_npta/
  q_ocf_to_sales/q_sales_yoy），**没有**单季净利/经营现金流水平量，故只能自算差分。

impairment 字段口径（实测，见 delivery/r4c_accruals_v1/report.md）：
  `assets_impair_loss` 的符号约定跨年代/公司不一致（2018 及以前绝大多数为正、2019 格式
  切换后绝大多数为负；少数公司例外），故本模块统一取 **|累计值|** 作为「减值损失规模」，
  单季口径 = |cum_q| − |cum_{q−1}|（负数=单季净转回）。这规避了符号翻转造成的伪跳变，
  代价是有真实转回的小额公司会被计成小额损失（报告披露）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fea.deriv import PERIOD_W, KEY_W, period_of

SIGNALS = (
    "accrual_ratio_ttm", "accrual_ratio_q", "earnings_conversion_trend",
    "impairment_to_assets", "earnings_conversion_ttm_gap", "impairment_to_assets_merged",
)
FIN_TABLES = ("stock_income", "stock_cashflow", "stock_balancesheet")
MIN_ABS_DEN = 1e6      # 分母地板：总资产/净利都必须大于 1e6 元才有意义
CLIP = 10.0            # OCF/NI 单季比率截幅（净利近零时比率会爆炸）


# ---------------------------------------------------------------- 版本表
def _quarter_table(ctx, dataset, field):
    """读一张财报表 -> 干净版本表 DataFrame[code, period, pit, val]。

    清洗规则逐条对齐 fea/deriv._load_one（顺序也一致）：
    季末过滤 -> 公告可考过滤 -> pit=max(ann, f_ann, end) -> (code,period,pit) 去重。
    不修改 Upstream 缓存里的 DataFrame（只读列、只建新数组）。
    """
    frame = ctx.dataset(dataset, columns=["stock_code", "ann_date", "f_ann_date", "end_date", field])
    if frame is None or frame.empty or field not in frame.columns:
        return None
    end_i = ctx.date_col(frame["end_date"])
    ann_i = ctx.date_col(frame["ann_date"])
    fann_i = ctx.date_col(frame["f_ann_date"])
    pit = np.maximum(np.maximum(ann_i, fann_i), end_i)
    keep = np.isin(end_i % 10000, (331, 630, 930, 1231))
    keep &= ((ann_i >= 19900101) | (fann_i >= 19900101)) & (pit > 0)
    if not keep.any():
        return None
    codes = frame["stock_code"].to_numpy(dtype=object)[keep]
    idx = ctx.code_index(codes)
    sel = idx >= 0
    if not sel.any():
        return None
    vals = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=np.float64)[keep]
    out = pd.DataFrame({
        "code": idx[sel].astype(np.int64),
        "period": period_of(end_i[keep][sel]),
        "pit": pit[keep][sel].astype(np.int64),
        "val": vals[sel],
    })
    # 同 (code, period, pit) 多行：与 deriv 一样 keep=last（period 与 end_date 一一对应）
    return out.drop_duplicates(["code", "period", "pit"], keep="last")


def _events(d):
    """事件表：每个 (code, pit) 一行，q = 该时点已知的最新报告期（前缀最大）。"""
    d = d.sort_values(["code", "pit", "period"], kind="stable")
    d = d.assign(q=d.groupby("code", sort=False)["period"].cummax())
    ev = d.drop_duplicates(["code", "pit"], keep="last")
    return ev.sort_values(["code", "pit"], kind="stable").reset_index(drop=True)


def _sorted_vt(d):
    """按 (code, period, pit) 升序的复合键数组，供 asof 查找（键方案与 deriv 相同）。"""
    d2 = d.sort_values(["code", "period", "pit"], kind="stable")
    codes = d2["code"].to_numpy(dtype=np.int64)
    periods = d2["period"].to_numpy(dtype=np.int64)
    pits = d2["pit"].to_numpy(dtype=np.int64)
    keys = (codes * PERIOD_W + periods) * KEY_W + pits
    return {"keys": keys, "code": codes, "period": periods, "val": d2["val"].to_numpy(dtype=np.float64)}


def _table(ctx, cache, dataset, field):
    """惰性构建并缓存某报表字段的 (events, sorted_vt)。空表 -> None。"""
    key = (dataset, field)
    if key not in cache:
        d = _quarter_table(ctx, dataset, field)
        cache[key] = None if d is None or d.empty else {"ev": _events(d), "vt": _sorted_vt(d)}
    return cache[key]


def _lookup(vt, code, period, pit):
    """(code, period, pit) 上取「pit' ≤ pit 的最新一版」的值；同股同期校验（防串行）。"""
    tk = (code * PERIOD_W + period) * KEY_W + pit
    pos = np.searchsorted(vt["keys"], tk, side="right") - 1
    ok = pos >= 0
    p = np.where(ok, pos, 0)
    ok &= (vt["code"][p] == code) & (vt["period"][p] == period)
    return np.where(ok, vt["val"][p], np.nan)


def _cum_grid(ctx, table, k_offset):
    """报告期 (Q−k_offset) 的累计值 -> 日频 (T,C) float64 网格（as-of pit ≤ 当日）。"""
    if table is None:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    ev = table["ev"]
    code = ev["code"].to_numpy(dtype=np.int64)
    q = ev["q"].to_numpy(dtype=np.int64)
    pit = ev["pit"].to_numpy(dtype=np.int64)
    vals = _lookup(table["vt"], code, q - k_offset, pit)
    return ctx.panel.asof(code, pit, vals, dtype=np.float64)


def _q_grid(ctx, table, k):
    """报告期 (Q−k) 本身 -> 日频 float64 网格（用于识别财年重置边界）。"""
    ev = table["ev"]
    q = ev["q"].to_numpy(dtype=np.int64) - k
    return ctx.panel.asof(ev["code"].to_numpy(dtype=np.int64), ev["pit"].to_numpy(dtype=np.int64),
                          q, dtype=np.float64)


def _single_quarter(ctx, table, k):
    """S(报告期 Q−k)。财年重置：q 为 Q1 时单季 = 累计本身；否则同财年内差分。

    q 为 Q1 只需要 cum(q) 可考；其余季度需要 cum(q) 与 cum(q−1) 都可考。
    """
    if table is None:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    qg = _q_grid(ctx, table, k)
    hi = _cum_grid(ctx, table, k)
    lo = _cum_grid(ctx, table, k + 1)
    known = np.isfinite(qg)
    is_q1 = np.rint(np.where(known, qg, 0.0)).astype(np.int64) % 4 == 0
    ok = np.isfinite(hi) & known & (is_q1 | np.isfinite(lo))
    val = np.where(is_q1, hi, hi - lo)
    return np.where(ok, val, np.nan)


def _window_mean(rats):
    """4 个单季比率的均值；要求 4 个全部有效（缺一季则整窗 NaN，不把两季伪装成四季）。"""
    stacked = np.stack(rats)
    ok = np.isfinite(stacked).all(axis=0)
    return np.where(ok, stacked.sum(axis=0) / len(rats), np.nan)


def _ratio(a, b):
    """OCF/NI 单季比率：净利绝对值需 > 1e6，截幅 ±10。"""
    big = np.isfinite(b) & (np.abs(b) > MIN_ABS_DEN)
    r = np.where(big, a / np.where(big, b, np.nan), np.nan)
    return np.clip(r, -CLIP, CLIP)


# ---------------------------------------------------------------- 构建器
def build(parameters):
    """六个固定机制之一；没有可搜索的窗口或方向。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("Batch permits only the six preregistered signal identifiers")
    signal = parameters["signal"]

    def compute(ctx):
        cache = {}
        if signal == "accrual_ratio_ttm":
            ni = np.asarray(ctx.ttm("n_income_attr_p"), dtype=np.float64)
            oc = np.asarray(ctx.ttm("n_cashflow_act"), dtype=np.float64)
            val = ctx.safe_div(ni - oc, ctx.point("total_assets"), min_abs_den=MIN_ABS_DEN)
        elif signal == "accrual_ratio_q":
            ni = _single_quarter(ctx, _table(ctx, cache, "stock_income", "n_income_attr_p"), 0)
            oc = _single_quarter(ctx, _table(ctx, cache, "stock_cashflow", "n_cashflow_act"), 0)
            val = ctx.safe_div(ni - oc, ctx.point("total_assets"), min_abs_den=MIN_ABS_DEN)
        elif signal == "earnings_conversion_trend":
            ni_t = _table(ctx, cache, "stock_income", "n_income_attr_p")
            oc_t = _table(ctx, cache, "stock_cashflow", "n_cashflow_act")
            ratios = [_ratio(_single_quarter(ctx, oc_t, k), _single_quarter(ctx, ni_t, k))
                      for k in range(8)]
            val = _window_mean(ratios[0:4]) - _window_mean(ratios[4:8])
        elif signal == "earnings_conversion_ttm_gap":
            oc = np.asarray(ctx.ttm("n_cashflow_act"), dtype=np.float64)
            ni = np.asarray(ctx.ttm("n_income_attr_p"), dtype=np.float64)
            oc4 = np.asarray(ctx.lag_ttm("n_cashflow_act", 4), dtype=np.float64)
            ni4 = np.asarray(ctx.lag_ttm("n_income_attr_p", 4), dtype=np.float64)
            val = _ratio(oc, ni) - _ratio(oc4, ni4)
        elif signal == "impairment_to_assets":
            imp = _table(ctx, cache, "stock_income", "assets_impair_loss")
            c0 = np.abs(_cum_grid(ctx, imp, 0))
            c1 = np.abs(_cum_grid(ctx, imp, 1))
            val = ctx.safe_div(c0 - c1, ctx.point("total_assets"), min_abs_den=MIN_ABS_DEN)
        else:  # impairment_to_assets_merged
            imp = _table(ctx, cache, "stock_income", "assets_impair_loss")
            cre = _table(ctx, cache, "stock_cashflow", "credit_impa_loss")
            cur = np.abs(_cum_grid(ctx, imp, 0)) + np.abs(_cum_grid(ctx, cre, 0))
            pre = np.abs(_cum_grid(ctx, imp, 1)) + np.abs(_cum_grid(ctx, cre, 1))
            val = ctx.safe_div(cur - pre, ctx.point("total_assets"), min_abs_den=MIN_ABS_DEN)
        return np.where(np.isfinite(val), val, np.nan).astype(np.float64, copy=False)

    return compute
