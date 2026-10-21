"""R12 批 C · C6 现金流八型符号组合（family `r12c_cashflow_sign_v1`；**alpha_v2（B）通道**）。

════════════════════════════════════════════════════════════════════════
一、机制与方向预注册（H2）
════════════════════════════════════════════════════════════════════════
经营/投资/筹资三个现金流净额的**符号组合**刻画公司所处生命周期与财务策略
（Dickinson 2011 的现金流生命周期分类；idea_bank r12u2_028）；连续的金额信息被符号抹平，
同类符号型公司在融资约束与风险特征上系统性不同。

    s1 = sign(n_cashflow_act)        （经营净额）
    s2 = sign(n_cashflow_inv_act)    （投资净额）
    s3 = sign(n_cash_flows_fnc_act)  （筹资净额）
    8 个符号型 → **事前冻结的 5 档“现金流健康序”分值表**（下表；8 型互不混淆地映射到分值）：

      (s1,s2,s3)   型别（Dickinson 类）      分值      先验理由（事前写明，不得事后挑型）
      (+, −, −)    Mature 成熟现金牛          **+2**    造血 + 对外投资 + 不再净融资（还债/分红）
      (+, −, +)    Growth 成长               +1       造血 + 投资扩张，但依赖净融资输血
      (+, +, +)    Shake-out 震荡            0        造血 + 处置/回笼 + 融资（混合信号）
      (+, +, −)    Shake-out 震荡            0        同上
      (−, −, +)    Introduction 导入        −1       烧钱扩张、全靠外部融资
      (−, +, +)    Decline 衰退             −2        主业失血，靠处置资产与融资维持
      (−, +, −)    Decline 衰退             −2        同上次级型
      (−, −, −)    Decline 全面失血          −2        三类全负

    factor(t) = 上述分值（半离散，5 档）

**方向先验（事前冻结，不得反号）**：健康序高（越接近成熟现金牛）→ 未来 h=1/h=5 收益正向
（`higher_is_better=True`，declared = raw）。
**证伪点**：h=1 或 h=5 的 declared RankIC HAC95 下界 ≤ 0、或正向年份 < 6/9（政策 v2 G4）；
更具体地，若 Decline 型（全 −2）显著跑赢 Mature 型（+2），健康序先验被证伪 ⇒ 按 H2 不反号、
不重排型表、直接否决（**禁止事后挑型**）。

**与 idea 原文的关系（README 同步）**：idea r12u2_028 提议两种表达：
(a) 8 个哑变量 —— **不可行**：本框架一个候选 = 一列 (T,C)，8 个哑变量即 8 个候选，
    超出本族预登记预算 2；(b) 连续化符号一致度 —— 原文未给出唯一公式。
本模块登记的实现是 (a) 的**单列化**：把 8 型按**事前冻结的分值表**线性重编码为 5 档健康分。
两个变体只在**符号的测量频率**上区分：
    `cfs_stage_ttm` —— TTM 净额定符号（idea 原文明确推荐：避免单季噪声）
    `cfs_stage_q`   —— **单季**净额定符号（自建单季网格，见下）
第二个变体正是对 idea 原文“用 TTM 避免单季噪声”这一判断的**可证伪对照**。

差异论证（H3/H4）：在库 229 项与 374 已删项中现金流类全部是**科目级量的因子**
（`cfp_ttm` 现金流市值比、`afx_cf_*` 结构占比等），**没有**符号组合/生命周期型构造；
与在库 `quality`/`growth` 族的比值型（roe_ttm 等）不同构。须测量与 `cfp_ttm`、
`accruals_ratio`（在产）及盈利水平类的 pooled |ρ| 并报告（G5）。

**H6/H7（稀疏/半离散，报告侧必做）**：本对象日截面仅 5 个 distinct 值 ⇒ **触发 H7**；
缺失（NaN 条件：池外、任一 sign 不可判、所需报告期缺）与零填充双口径须并报；
报告侧必须做 unique/day 与各型占比统计、以及联合重建/混淆检验（政策 §一 H7、§D.2）。

════════════════════════════════════════════════════════════════════════
二、PIT 与单季网格
════════════════════════════════════════════════════════════════════════
· TTM 变体：`ctx.ttm(field)`（ann_date 锚定版本表；T 日只看 ≤T 的最新已披露版本）。
· 单季变体：ctx 的 `ttm()/point()` **没有单季入口**，故按在产先例
  `research/families/r4c_accruals_v1.py`（= catalog `r4c_accrual_ratio_q_v1`）自建单季网格：
  季末过滤（331/630/930/1231）→ 公告可考过滤（ann/f_ann 均 <1990 丢弃）→
  pit = max(ann_date, f_ann_date, end_date) → (code, period, pit) 去重保留历次修订；
  事件网格 = 每个 (code, pit) 上用「前缀最大报告期」q 取 cum(q) 与 cum(q−1)；
  **财年重置**：q 为 Q1（q % 4 == 0）时单季值 = 累计本身，否则同一财年内差分；
  q 与 q−1 两个累计值都可考才出值，否则 NaN（不做 partial-sum 回退）。
  落格走 `ctx.panel.asof(..., dtype=float64)`（绕开 asof_daily 的 float32 精度）。
· 金额恰为 0.0 的净额 ⇒ 该 sign 不可判 ⇒ 整型 NaN（不判型、不补值）。
· `stock_cashflow` 三字段**不在** POSITIVE_ONLY（允许负值，符号机制的前提）。
· 池外/未上市/ST ⇒ NaN；warmup_days=700（单季网格需要两个相邻报告期 + ann 滞后）。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fea.deriv import KEY_W, PERIOD_W, period_of

log = logging.getLogger("research.families.r12c_cashflow_sign_v1")

OCF = "n_cashflow_act"           # 经营活动现金流净额（累计制）
ICF = "n_cashflow_inv_act"       # 投资活动现金流净额（累计制）
FCF = "n_cash_flows_fnc_act"     # 筹资活动现金流净额（累计制）
CS = "stock_cashflow"
# 事前冻结的 5 档健康序分值表：code = (s1>0)*4 + (s2>0)*2 + (s3>0)，code 0..7
HEALTH_TABLE = np.array([-2.0, -1.0, -2.0, -2.0, 2.0, 1.0, 0.0, 0.0])
# 事前冻结：变体名 -> 符号测量频率
FROZEN = {
    "cfs_stage_ttm": "ttm",
    "cfs_stage_q": "single_quarter",
}


def _frozen_key(parameters):
    if not isinstance(parameters, dict):
        return None
    return parameters.get("variant")


def _score(s1, s2, s3):
    """8 符号型 -> 冻结分值；任一 sign 不可判（0 或 NaN）⇒ NaN。"""
    valid = (np.isfinite(s1) & np.isfinite(s2) & np.isfinite(s3)
             & (s1 != 0.0) & (s2 != 0.0) & (s3 != 0.0))
    code = (np.where(s1 > 0.0, 4, 0) + np.where(s2 > 0.0, 2, 0)
            + np.where(s3 > 0.0, 1, 0)).astype(np.int64)
    return np.where(valid, HEALTH_TABLE[code], np.nan)


# ---------------------------------------------------------------- 单季网格
# 逐条复刻 research/families/r4c_accruals_v1.py（上游 = fea/deriv.py 的版本表口径）。
def _quarter_table(ctx, dataset, field):
    """读一张财报表 -> 干净版本表 DataFrame[code, period, pit, val]。"""
    frame = ctx.dataset(dataset, columns=["stock_code", "ann_date", "f_ann_date",
                                          "end_date", field])
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
    return {"keys": keys, "code": codes, "period": periods,
            "val": d2["val"].to_numpy(dtype=np.float64)}


def _table(ctx, cache, dataset, field):
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
    return ctx.panel.asof(ev["code"].to_numpy(dtype=np.int64),
                          ev["pit"].to_numpy(dtype=np.int64), q, dtype=np.float64)


def _single_quarter(ctx, table, k):
    """S(报告期 Q−k)。财年重置：q 为 Q1 时单季 = 累计本身；否则同财年内差分。"""
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


# ---------------------------------------------------------------- 构建器
def build(parameters):
    """只允许预登记的两个变体；不搜索窗口、不搜索方向、不重排型表。"""
    variant = _frozen_key(parameters)
    if variant not in FROZEN:
        raise ValueError(
            "Only the frozen preregistered variants are permitted; got %r. Allowed: %s"
            % (parameters, sorted(FROZEN)))
    freq = FROZEN[variant]

    def compute(ctx):
        if freq == "ttm":
            s1 = np.sign(np.asarray(ctx.ttm(OCF), dtype=np.float64))
            s2 = np.sign(np.asarray(ctx.ttm(ICF), dtype=np.float64))
            s3 = np.sign(np.asarray(ctx.ttm(FCF), dtype=np.float64))
        else:
            cache = {}
            s1 = np.sign(_single_quarter(ctx, _table(ctx, cache, CS, OCF), 0))
            s2 = np.sign(_single_quarter(ctx, _table(ctx, cache, CS, ICF), 0))
            s3 = np.sign(_single_quarter(ctx, _table(ctx, cache, CS, FCF), 0))
        value = _score(s1, s2, s3)
        grid = np.where(np.asarray(ctx.universe, dtype=bool), value, np.nan)
        log.info("r12c_%s panel_finite=%d", variant, int(np.isfinite(grid).sum()))
        return np.ascontiguousarray(grid, dtype=np.float64)

    return compute
