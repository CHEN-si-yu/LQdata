"""R3-D forecast reason taxonomy (stock_forecast.change_reason text only).

Only ann_date is used as an availability time. end_date may point at a future
report period and is never read as an input. Availability contract matches
fea/upstream.py: a stock_forecast row is visible from ann_date and requires
ann_date >= 1990-01-01.

THE PHRASE LISTS BELOW ARE FROZEN. They were written from the A-share earnings
forecast vocabulary and committed to this module BEFORE any candidate existed and
before any IC was computed. They are not tuned on returns; adding or removing a
phrase requires a new module version and invalidates the previous version's
conclusions. scripts/probe_reason_text.py measures their document frequency only.

Empty change_reason is NOT the same as "matched no category": empty text yields
NaN, non-empty text that matches nothing yields 0.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGNALS = ("main_business", "nonrecurring", "demand_cycle", "quality_composite")

# ---------------------------------------------------------------- frozen phrases
KEYWORDS = {
    "main_business": (
        "主营业务", "主营", "主业", "销量", "销售量", "产销量", "销售规模", "销售收入",
        "营业收入", "产能", "产量", "产能释放", "订单", "市场份额", "业务规模",
        "市场开拓", "业务拓展", "客户需求", "产品结构", "降本增效", "成本控制", "毛利率",
    ),
    "nonrecurring": (
        "非经常性损益", "非经营性", "政府补助", "财政补贴", "补贴", "资产处置", "处置资产",
        "股权转让", "投资收益", "公允价值变动", "债务重组", "税收优惠", "退税", "诉讼",
        "赔偿", "捐赠", "一次性", "非主营", "理财产品", "证券投资",
    ),
    "impairment": (
        "减值", "计提", "商誉", "资产减值", "信用减值", "存货跌价", "坏账", "预计负债",
        "核销", "报废",
    ),
    "demand_cycle": (
        "行业景气", "景气度", "行业需求", "市场需求", "下游需求", "需求旺盛", "需求下降",
        "产品价格", "原材料价格", "价格上涨", "价格下跌", "市场价格", "供需", "周期性",
        "行业竞争", "行业政策", "宏观", "大宗商品", "行情",
    ),
}

# Pre-registered naive baseline reference (same table, existing production factor):
# forecast_type_score. No separate baseline candidate is registered; its published
# values are used directly for the directed comparison.

CATEGORY_OF_SIGNAL = {
    "main_business": "main_business",
    "nonrecurring": "nonrecurring",
    "demand_cycle": "demand_cycle",
}


def _text_rows(ctx):
    frame = ctx.dataset("stock_forecast", columns=["stock_code", "ann_date", "change_reason"])
    if frame is None or frame.empty:
        return None
    frame = frame[frame["stock_code"].isin(ctx.panel.codes)]
    if frame.empty:
        return None
    announced = ctx.date_col(frame["ann_date"]).astype(np.int64)
    keep = announced >= 19900101
    if not keep.any():
        return None
    text = frame["change_reason"].fillna("").astype(str).str.strip()
    return pd.DataFrame({
        "stock_code": frame["stock_code"].to_numpy()[keep],
        "ann_date": announced[keep],
        "text": text.to_numpy()[keep]})


def _hits(text: pd.Series, phrases) -> np.ndarray:
    hit = np.zeros(len(text), dtype=bool)
    for phrase in phrases:
        hit |= text.str.contains(phrase, regex=False).to_numpy()
    return hit


def _aggregate(rows, columns: dict) -> pd.DataFrame:
    """One row per (stock, announcement day); a category is present if any of that
    day's forecast records mentions it."""
    frame = pd.DataFrame({"stock_code": rows["stock_code"].to_numpy(),
                          "ann_date": rows["ann_date"].to_numpy()})
    for name, values in columns.items():
        frame[name] = np.asarray(values)
    grouped = frame.groupby(["stock_code", "ann_date"], as_index=False).max()
    return grouped


def _place(ctx, grouped, values):
    grid = ctx.asof_daily(grouped["stock_code"].to_numpy(),
                          grouped["ann_date"].to_numpy(dtype=np.int64),
                          np.asarray(values, dtype=np.float64))
    grid = np.asarray(grid, dtype=np.float64)
    return np.where(np.isfinite(grid), grid, np.nan)


def reason_taxonomy(parameters):
    """One of the pre-registered R3-D mechanisms; no searched phrases or directions."""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("Family permits only the pre-registered R3-D signal identifiers")
    signal = parameters["signal"]

    def compute(ctx):
        rows = _text_rows(ctx)
        if rows is None:
            return ctx.panel.empty(dtype=np.float64)
        text = rows["text"]
        present = (text != "").to_numpy()
        hits = {name: _hits(text, phrases) for name, phrases in KEYWORDS.items()}
        grouped = _aggregate(rows, {**hits, "present": present})
        if not grouped["present"].to_numpy().any():
            return ctx.panel.empty(dtype=np.float64)
        if signal == "quality_composite":
            value = (grouped["main_business"].to_numpy(dtype=np.float64)
                     + grouped["demand_cycle"].to_numpy(dtype=np.float64)
                     - grouped["nonrecurring"].to_numpy(dtype=np.float64))
        else:
            value = grouped[CATEGORY_OF_SIGNAL[signal]].to_numpy(dtype=np.float64)
        value = np.where(grouped["present"].to_numpy(), value, np.nan)
        keep = np.isfinite(value)
        if not keep.any():
            return ctx.panel.empty(dtype=np.float64)
        return _place(ctx, grouped.loc[keep], value[keep])

    return compute
