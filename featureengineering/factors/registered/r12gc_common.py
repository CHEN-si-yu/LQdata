"""R12 绿卡批 A · 共享工具：单季网格 / 年度网格 / 高精度 as-of / 后复权动量 / 自算市值。

本文件是 `r12gc_*` 六个模块的**公共底座**，本身不注册任何因子、不暴露 build。
四个部分：

1. **财务网格（单季 S(q) 与年度 AN(f)）** —— ctx 的 `ttm()/point()` 只暴露 TTM 与时点值，
   **没有任何**入口能给出单季水平或「当时已公告的最新年报」值。这里按在产先例
   `research/families/r4c_accruals_v1.py`（= catalog 里的 `r4c_accrual_ratio_q_v1`）
   逐条复刻其自建网格：季末过滤 → 公告可考过滤 → `pit = max(ann_date, f_ann_date, end_date)`
   → (code, period, pit) 去重 → 事件表（每个 (code, pit) 的 q = 前缀最大报告期）
   → `panel.asof(..., dtype=float64)` 铺到日频网格。

   · 单季（财年重置，改这里之前先读 r4c 文档）：
         S(q) = cum(q)                       若 q 为 Q1（q % 4 == 0）
              = cum(q) − cum(q−1)            否则（同一财年内差分）
     只保留季末 end_date（331/630/930/1231）；q 与 q−1 两个累计值都可考时才出值，
     否则 NaN（不做 partial-sum 回退）。
   · 年度：同一年度网格，`annual_only=True` 只保留 12/31 年报；年报累计值本身即 FY 值
     （与 `fea/deriv.py` 的 `mode="annual"` 注释一致），故 `cum_grid(tbl, 0)` 即
     「当时已公告的最新年报 FY 值」。季度披露绝不能把年报推走（前缀最大只在新年报
     真的公告后才前移）。

2. **`ret_k` / `zero_trade_mask`** —— 逐字复刻 `factors/momentum.py::_ret_k`：
   用 `ret(1)` + `log1p` + `roll_sum(k)` 实现 Π(1+r)−1，避免 `ctx.ret(k)` 把
   「单日 ±60%」的阈值套到 k 日累计上（k=250 时实测误杀 48% 有效格，全是强势股）；
   窗口内一次都没成交（价格全是前向填充的陈旧值）⇒ NaN，而不是凭空造出来的 0。

3. **`mktcap`** —— 与在产 `log_mv` 同一自算市值约定：
   未复权 `close`（状态量，停牌前向填充）× 当期已披露股本（`*_share`，状态量）。

4. **`annual_value`** —— 年度网格取值的具名包装（= `cum_grid(tbl, 0)`）。

★ 哈希口径提醒：`research/catalog.py::implementation` 只对**模块文件本身**取
  sha256 作为 `source_hash`，本共享文件的字节不进入候选定义哈希。因此
  **修改本文件等价于定义变更**（口径变了但候选 id 不会变），必须重新注册并留痕。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fea.deriv import PERIOD_W, KEY_W, period_of


# ---------------------------------------------------------------- 自算市值
def mktcap(ctx, share: str = "total_share") -> np.ndarray:
    """市值 = 未复权 close × 股本（元）。close/*_share 都是状态量（停牌前向填充）。"""
    px = np.asarray(ctx.px("close"), dtype=np.float64)
    sh = np.asarray(ctx.px(share), dtype=np.float64)
    return px * sh


# ---------------------------------------------------------------- 后复权动量
def zero_trade_mask(ctx, n: int) -> np.ndarray:
    """窗口 n 个交易日内一次都没成交 ⇒ True（价格全是前向填充的陈旧值）。"""
    tr = np.asarray(ctx.traded(), dtype=np.float64)
    return ctx.roll_sum(tr, int(n)) <= 0.0


def ret_k(ctx, k: int) -> np.ndarray:
    """k 个交易日后复权累计收益 = Π(1 + ret(1)) − 1（log1p + roll_sum，NaN 毒化）。

    与在产 momentum_* 家族同一实现（factors/momentum.py::_ret_k）；停牌日
    `ret(1)=0`（1 倍，语义正确）；单日 |ret|>60% 由价格层置 NaN 并毒化整窗；
    窗口内零成交 ⇒ NaN。
    """
    r1 = ctx.ret(1)
    r = np.expm1(ctx.roll_sum(np.log1p(r1), int(k)))
    return np.where(zero_trade_mask(ctx, k), np.nan, r)


# ---------------------------------------------------------------- 财务版本表
def quarter_table(ctx, dataset: str, field: str, annual_only: bool = False):
    """读一张财报表 -> 干净版本表 DataFrame[code, period, pit, val]。

    清洗规则逐条对齐 fea/deriv._load_one（顺序也一致）：
    季末过滤（annual_only 时只留 12/31）→ 公告可考过滤 → pit=max(ann, f_ann, end)
    → (code, period, pit) 去重（keep=last）。不修改 Upstream 缓存里的 DataFrame。
    """
    frame = ctx.dataset(dataset, columns=["stock_code", "ann_date", "f_ann_date", "end_date", field])
    if frame is None or frame.empty or field not in frame.columns:
        return None
    end_i = ctx.date_col(frame["end_date"])
    ann_i = ctx.date_col(frame["ann_date"])
    fann_i = ctx.date_col(frame["f_ann_date"])
    pit = np.maximum(np.maximum(ann_i, fann_i), end_i)
    keep = np.isin(end_i % 10000, (331, 630, 930, 1231))
    if annual_only:
        keep &= (end_i % 10000) == 1231
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
    # 同 (code, period, pit) 多行：与 deriv 一样 keep=last
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


def table(ctx, cache: dict, dataset: str, field: str, annual_only: bool = False):
    """惰性构建并缓存某报表字段的 (events, sorted_vt)。空表 -> None。"""
    key = (dataset, field, annual_only)
    if key not in cache:
        d = quarter_table(ctx, dataset, field, annual_only=annual_only)
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


def cum_grid(ctx, tbl, k_offset: int = 0) -> np.ndarray:
    """报告期 (Q − k_offset) 的累计值 -> 日频 (T,C) float64 网格（as-of pit ≤ 当日）。"""
    if tbl is None:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    ev = tbl["ev"]
    code = ev["code"].to_numpy(dtype=np.int64)
    q = ev["q"].to_numpy(dtype=np.int64)
    pit = ev["pit"].to_numpy(dtype=np.int64)
    vals = _lookup(tbl["vt"], code, q - k_offset, pit)
    return ctx.panel.asof(code, pit, vals, dtype=np.float64)


def q_grid(ctx, tbl, k: int = 0) -> np.ndarray:
    """报告期 (Q − k) 本身 -> 日频 float64 网格（用于识别财年重置边界）。"""
    ev = tbl["ev"]
    q = ev["q"].to_numpy(dtype=np.int64) - k
    return ctx.panel.asof(ev["code"].to_numpy(dtype=np.int64), ev["pit"].to_numpy(dtype=np.int64),
                          q, dtype=np.float64)


def single_quarter(ctx, tbl, k: int = 0) -> np.ndarray:
    """S(报告期 Q−k)。财年重置：q 为 Q1 时单季 = 累计本身；否则同财年内差分。

    q 为 Q1 只需要 cum(q) 可考；其余季度需要 cum(q) 与 cum(q−1) 都可考。
    """
    if tbl is None:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    qg = q_grid(ctx, tbl, k)
    hi = cum_grid(ctx, tbl, k)
    lo = cum_grid(ctx, tbl, k + 1)
    known = np.isfinite(qg)
    is_q1 = np.rint(np.where(known, qg, 0.0)).astype(np.int64) % 4 == 0
    ok = np.isfinite(hi) & known & (is_q1 | np.isfinite(lo))
    val = np.where(is_q1, hi, hi - lo)
    return np.where(ok, val, np.nan)


def annual_value(ctx, tbl) -> np.ndarray:
    """当时已公告的最新 12/31 年报的 FY 值（annual_only 网格的 cum(q)）。"""
    return cum_grid(ctx, tbl, 0)
