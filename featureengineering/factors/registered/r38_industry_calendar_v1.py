"""R38 industry/calendar/announcement factors.

Production-local implementations of four frozen research candidates:
same-month volatility seasonality, industry amount-share shift, quarterly
announcement-delay excess, and the 60-session PEAD reaction state.  The
industry map is the static vendor ``stock_list.industry`` crosswalk used by
the frozen research design; no future constituent data is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAMES = (
    "cal_same_month_vol_seasonality_5y",
    "ind_amount_share_shift_5_60_v1",
    "ann_qtr_delay_excess_v1",
    "ann_car1_pead_60_v1",
)

L1_NAMES = (
    "公用事业", "农林牧渔", "纺织服饰", "煤炭", "食品饮料", "家用电器",
    "石油石化", "有色金属", "钢铁", "传媒", "环保", "美容护理",
    "电力设备", "电子", "房地产", "非银金融", "国防军工", "机械设备",
    "基础化工", "计算机", "建筑材料", "建筑装饰", "交通运输", "汽车",
    "轻工制造", "商贸零售", "社会服务", "通信", "医药生物", "综合", "银行",
)
L1_INDEX = {name: i for i, name in enumerate(L1_NAMES)}
LABEL_TO_L1 = {
    "电气设备": "电力设备", "元器件": "电子", "专用机械": "机械设备",
    "软件服务": "计算机", "汽车配件": "汽车", "化工原料": "基础化工",
    "半导体": "电子", "医疗保健": "医药生物", "化学制药": "医药生物",
    "机械基件": "机械设备", "通信设备": "通信", "建筑工程": "建筑装饰",
    "环境保护": "环保", "电器仪表": "机械设备", "家用电器": "家用电器",
    "食品": "食品饮料", "生物制药": "医药生物", "IT设备": "计算机",
    "互联网": "传媒", "塑料": "基础化工", "家居用品": "轻工制造",
    "中成药": "医药生物", "小金属": "有色金属", "服饰": "纺织服饰",
    "农药化肥": "基础化工", "广告包装": "传媒", "航空": "交通运输",
    "文教休闲": "社会服务", "证券": "非银金融", "运输设备": "机械设备",
    "仓储物流": "交通运输", "供气供热": "公用事业", "纺织": "纺织服饰",
    "区域地产": "房地产", "银行": "银行", "农业综合": "农林牧渔",
    "工程机械": "机械设备", "医药商业": "医药生物", "染料涂料": "基础化工",
    "矿物制品": "建筑材料", "影视音像": "传媒", "百货": "商贸零售",
    "造纸": "轻工制造", "化纤": "基础化工", "新型电力": "公用事业",
    "火力发电": "公用事业", "其他建材": "建筑材料", "铝": "有色金属",
    "出版业": "传媒", "普钢": "钢铁", "全国地产": "房地产",
    "煤炭开采": "煤炭", "装修装饰": "建筑装饰", "机床制造": "机械设备",
    "多元金融": "非银金融", "汽车整车": "汽车", "饲料": "农林牧渔",
    "钢加工": "钢铁", "水泥": "建筑材料", "水力发电": "公用事业",
    "石油开采": "石油石化", "乳制品": "食品饮料", "日用化工": "美容护理",
    "综合类": "综合", "玻璃": "建筑材料", "路桥": "建筑装饰",
    "水运": "交通运输", "白酒": "食品饮料", "种植业": "农林牧渔",
    "旅游景点": "社会服务", "铜": "有色金属", "橡胶": "基础化工",
    "港口": "交通运输", "商贸代理": "商贸零售", "水务": "公用事业",
    "石油加工": "石油石化", "铅锌": "有色金属", "园区开发": "房地产",
    "房产服务": "房地产", "农用机械": "机械设备", "化工机械": "机械设备",
    "摩托车": "汽车", "纺织机械": "机械设备", "其他商业": "商贸零售",
    "船舶": "国防军工", "黄金": "有色金属", "特种钢": "钢铁",
    "汽车服务": "汽车", "酒店餐饮": "社会服务", "软饮料": "食品饮料",
    "红黄酒": "食品饮料", "空运": "交通运输", "旅游服务": "社会服务",
    "啤酒": "食品饮料", "超市连锁": "商贸零售", "铁路": "交通运输",
    "焦炭加工": "煤炭", "渔业": "农林牧渔", "轻工机械": "机械设备",
    "公共交通": "交通运输", "机场": "交通运输", "批发业": "商贸零售",
    "陶瓷": "建筑材料", "电信运营": "通信", "保险": "非银金融",
    "商品城": "商贸零售", "石油贸易": "石油石化", "林业": "农林牧渔",
    "公路": "交通运输", "电器连锁": "商贸零售",
}

_industry_groups_memo: dict[tuple[str, ...], np.ndarray] = {}


def _days(ctx) -> np.ndarray:
    return np.asarray(ctx.panel.dates, dtype=np.int64)


def _final(ctx, value: np.ndarray) -> np.ndarray:
    out = np.where(ctx.universe & ctx.traded(), np.asarray(value, dtype=np.float64), np.nan)
    return np.ascontiguousarray(out, dtype=np.float64)


def _market_ew(r: np.ndarray) -> np.ndarray:
    valid = np.isfinite(r)
    n = valid.sum(axis=1)
    s = np.where(valid, r, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _industry_groups(ctx) -> np.ndarray:
    codes = tuple(np.asarray(ctx.panel.codes).astype(str).tolist())
    hit = _industry_groups_memo.get(codes)
    if hit is not None:
        return hit
    listing = ctx.dataset("stock_list", columns=["stock_code", "industry"])
    stock_to_l1: dict[str, str] = {}
    if listing is not None and not listing.empty:
        for code, ind in zip(listing["stock_code"].astype(str), listing["industry"]):
            label = "" if ind is None else str(ind).strip()
            l1 = LABEL_TO_L1.get(label, "")
            if l1:
                stock_to_l1[code] = l1
    groups = np.array([L1_INDEX.get(stock_to_l1.get(c, ""), -1) for c in codes],
                      dtype=np.int64)
    _industry_groups_memo[codes] = groups
    return groups


def _onehot(g: np.ndarray, n: int = len(L1_NAMES)) -> np.ndarray:
    out = np.zeros((n, g.size), dtype=np.float64)
    ok = g >= 0
    out[g[ok], np.flatnonzero(ok)] = 1.0
    return out


def _month_starts(days: np.ndarray) -> np.ndarray:
    key = days // 100
    return np.flatnonzero(np.r_[True, key[1:] != key[:-1]])


def _month_index(days: np.ndarray) -> np.ndarray:
    y = days // 10000
    m = (days // 100) % 100
    return (y * 12 + m - 1).astype(np.int64)


def _same_month_vol(ctx):
    r = np.asarray(ctx.ret_clean(1), dtype=np.float64)
    days = _days(ctx)
    starts = _month_starts(days)
    ends = np.r_[starts[1:], days.size]
    n_months = starts.size
    T, C = r.shape
    msd = np.full((n_months, C), np.nan, dtype=np.float64)
    for k in range(n_months):
        lo, hi = int(starts[k]), int(ends[k])
        rt = r[lo:hi]
        ok = np.isfinite(rt)
        cnt = ok.sum(axis=0)
        mean = np.where(cnt > 0, np.where(ok, rt, 0.0).sum(axis=0)
                        / np.maximum(cnt, 1), np.nan)
        var = np.where(cnt > 1, np.where(ok, (rt - mean[None, :]) ** 2, 0.0).sum(axis=0)
                       / np.maximum(cnt - 1, 1), np.nan)
        sd = np.sqrt(np.maximum(var, 0.0))
        msd[k] = np.where((cnt >= 10) & np.isfinite(sd) & (sd > 0.0), sd, np.nan)
    month_ord = np.array([int(d // 10000) * 12 + int((d // 100) % 100) - 1
                          for d in days[starts]], dtype=np.int64)
    ord_to_row = {int(o): i for i, o in enumerate(month_ord)}
    mi = _month_index(days)
    out = np.full((T, C), np.nan, dtype=np.float64)
    for t in range(T):
        m = int(mi[t])
        same_rows = [ord_to_row.get(m - 12 * j) for j in range(1, 6)]
        all_rows = [ord_to_row.get(m - j) for j in range(1, 61)]
        same_rows = [x for x in same_rows if x is not None]
        all_rows = [x for x in all_rows if x is not None]
        if len(same_rows) < 3 or len(all_rows) < 24:
            continue
        ss = np.stack([msd[x] for x in same_rows], axis=0)
        aa = np.stack([msd[x] for x in all_rows], axis=0)
        same_cnt = np.isfinite(ss).sum(axis=0)
        all_cnt = np.isfinite(aa).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            same = np.where(same_cnt > 0, np.where(np.isfinite(ss), ss, 0.0).sum(axis=0)
                            / np.maximum(same_cnt, 1), np.nan)
            allm = np.where(all_cnt > 0, np.where(np.isfinite(aa), aa, 0.0).sum(axis=0)
                            / np.maximum(all_cnt, 1), np.nan)
        good = (same_cnt >= 3) & (all_cnt >= 24) & np.isfinite(same) \
            & np.isfinite(allm) & (allm > 1e-8)
        out[t] = np.where(good, np.log(np.where(good, same, 1.0)
                                        / np.where(good, allm, 1.0)), np.nan)
    return _final(ctx, out)


def _amount_share_shift(ctx):
    amt = np.asarray(ctx.px("amount"), dtype=np.float64)
    tr = np.asarray(ctx.traded(), dtype=bool)
    g = _industry_groups(ctx)
    T, C = amt.shape
    valid = np.isfinite(amt) & (amt > 0.0) & tr
    oh = _onehot(g)
    sums = np.where(valid, amt, 0.0) @ oh.T
    total = np.where(valid, amt, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = sums / np.where(total[:, None] > 0, total[:, None], np.nan)
    counts = valid.astype(np.float64) @ oh.T
    shares[counts < 5] = np.nan
    m5 = np.asarray(ctx.roll_mean(shares, 5, min_count=3), dtype=np.float64)
    m60 = np.asarray(ctx.roll_mean(shares, 60, min_count=30), dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = np.log(np.where(m5 > 0, m5, np.nan) / np.where(m60 > 0, m60, np.nan))
    out = np.full((T, C), np.nan, dtype=np.float64)
    ok = g >= 0
    out[:, ok] = vals[:, g[ok]]
    return _final(ctx, out)


def _income_events(ctx):
    days = _days(ctx)
    frame = ctx.dataset("stock_income", columns=[
        "stock_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type"])
    if frame is None or frame.empty:
        return None
    frame = frame.copy()
    frame["stock_code"] = frame["stock_code"].astype(str)
    ann = ctx.date_col(frame["ann_date"]).astype(np.int64)
    fann = ctx.date_col(frame["f_ann_date"]).astype(np.int64)
    end = ctx.date_col(frame["end_date"]).astype(np.int64)
    rt = pd.to_numeric(frame["report_type"], errors="coerce").to_numpy(np.float64)
    ct = pd.to_numeric(frame["comp_type"], errors="coerce").to_numpy(np.float64)
    keep = ((rt == 1.0) & np.isin(ct, (1.0, 2.0, 3.0, 4.0))
            & np.isin(end % 10000, (331, 630, 930, 1231))
            & ((ann >= 19900101) | (fann >= 19900101)))
    if not keep.any():
        return None
    vis = np.maximum(np.maximum(ann, fann), end)[keep]
    code = frame["stock_code"].to_numpy()[keep]
    end = end[keep]
    rows = pd.DataFrame({"stock_code": code, "end": end, "vis": vis})
    rows = rows.sort_values(["stock_code", "end", "vis"], kind="mergesort")
    rows = rows.drop_duplicates(["stock_code", "end"], keep="first").reset_index(drop=True)
    rows["eff"] = np.searchsorted(days, rows["vis"].to_numpy(np.int64), side="right")
    rows["cidx"] = ctx.code_index(rows["stock_code"].to_numpy())
    rows = rows[(rows["cidx"] >= 0) & (rows["vis"] >= int(days[0]))].reset_index(drop=True)
    vdt = pd.to_datetime(rows["vis"].astype(str), format="%Y%m%d", errors="coerce")
    edt = pd.to_datetime(rows["end"].astype(str), format="%Y%m%d", errors="coerce")
    rows["delay"] = (vdt - edt).dt.days.to_numpy(np.float64)
    prev = rows[["stock_code", "end", "delay"]].copy()
    prev["end"] = prev["end"] + 10000
    prev = prev.rename(columns={"delay": "delay_prev"})
    rows = rows.merge(prev, on=["stock_code", "end"], how="left")
    rows["excess"] = rows["delay"] - rows["delay_prev"]
    return rows


def _fill_event_state(rows, start, stop, values, shape):
    T, C = shape
    out = np.full((T, C), np.nan, dtype=np.float64)
    start = np.asarray(start, dtype=np.int64)
    stop = np.asarray(stop, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    cidx = np.asarray(rows["cidx"], dtype=np.int64)
    order = np.lexsort((start, cidx))
    for k in order:
        c = int(cidx[k])
        if c < 0:
            continue
        e = max(int(start[k]), 0)
        s = min(int(stop[k]), T - 1)
        if e >= T or s < e:
            continue
        out[e:s + 1, c] = values[k]
    return out


def _qtr_delay(ctx):
    rows = _income_events(ctx)
    if rows is None or rows.empty:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    days = _days(ctx)
    exp = pd.to_datetime(rows["end"].astype(str), format="%Y%m%d", errors="coerce") \
        + pd.Timedelta(days=550)
    exp_i = exp.dt.strftime("%Y%m%d").astype("Int64").fillna(0).astype(np.int64).to_numpy()
    stop = np.searchsorted(days, exp_i, side="right") - 1
    d = rows["delay"].to_numpy(np.float64)
    dp = rows["delay_prev"].to_numpy(np.float64)
    ok = (np.isfinite(d) & (d >= 0) & (d <= 180)
          & np.isfinite(dp) & (dp >= 0) & (dp <= 180)
          & np.isfinite(rows["excess"].to_numpy(np.float64)))
    vals = np.where(ok, rows["excess"].to_numpy(np.float64), np.nan)
    out = _fill_event_state(rows, rows["eff"].to_numpy(), stop, vals, ctx.panel.shape)
    return _final(ctx, out)


def _car_state(ctx, carry_sessions: int):
    rows = _income_events(ctx)
    if rows is None or rows.empty:
        return None, np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    r = np.asarray(ctx.ret_clean(1), dtype=np.float64)
    mkt = _market_ew(r)
    ab = r - mkt[:, None]
    T, C = r.shape
    eff = rows["eff"].to_numpy(np.int64)
    keep = eff < T
    rows = rows[keep].reset_index(drop=True)
    eff = eff[keep]
    cidx = rows["cidx"].to_numpy(np.int64)
    car = ab[eff, cidx]
    traded = np.asarray(ctx.traded(), dtype=bool)
    car = np.where(traded[eff, cidx] & np.isfinite(car), car, np.nan)
    start = eff + 1
    stop = np.minimum(eff + int(carry_sessions), T - 1)
    out = _fill_event_state(rows, start, stop, car, (T, C))
    return rows, out


def _car_pead(ctx):
    _, out = _car_state(ctx, carry_sessions=60)
    return _final(ctx, out)


def build(parameters):
    variant = (parameters or {}).get("variant")
    if variant not in NAMES:
        raise KeyError(f"unknown R38 industry/calendar variant: {variant!r}")

    def compute(ctx):
        if variant == "cal_same_month_vol_seasonality_5y":
            return _same_month_vol(ctx)
        if variant == "ind_amount_share_shift_5_60_v1":
            return _amount_share_shift(ctx)
        if variant == "ann_qtr_delay_excess_v1":
            return _qtr_delay(ctx)
        return _car_pead(ctx)

    return compute
