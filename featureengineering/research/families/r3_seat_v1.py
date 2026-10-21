"""R3-A 龙虎榜席位结构（dragon-tiger seat structure）；只读上游，不写生产。

数据事实（写第一行公式前实测，见 round3/seat/report.md §2）：
  * stock_dragon_tiger 的键不是 (日期, 股票, 席位)：同一 (日期,股票,席位,方向,原因)
    还会出现「买卖列整块互换」的重复行；32.8% 的行是这种重复/换位副本。
  * net_buy_amount 是权威净额：用 (gross+net)/2 复原的买入额与 stock_top_list.l_buy
    （买榜前5合计）在 2018/2019/2022/2025 全部年份都显著优于直接取 buy_amount；
    且 |net| <= buy+sell 在全部池内行成立。本模块只用 gross = buy+sell（换位不变）
    与 net_buy_amount 复原 buy/sell，不直接采信 buy_amount/sell_amount 的列序。
  * direction：0 = 买入金额前5榜，1 = 卖出金额前5榜。由 buy_ratio/sell_ratio 的方向内
    均值确认（dir0 买入占比 0.86~0.93；99.16% 的股票日 dir0 占比高于 dir1）。
  * 「机构专用」不是单一席位，而是多个匿名机构各自一行；去重必须保留金额不同的多行。

本模块只有一条实现路径：参数 (signal, window) 决定机制与窗口，全部事前固定。
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from fea import mathx as mx

DT = "stock_dragon_tiger"

WINDOWS = {
    "seat_class_divergence": (20, 60),
    "seat_concentration": (60,),
    "seat_persistence": (60, 120),
    "seat_buy_sell_overlap": (60,),
    "seat_novelty": (60,),
}
SIGNALS = tuple(WINDOWS)
HORIZONS = (1, 5)

# 席位分类阈值（事前固定）：非机构、非北向的具名席位，若在信号日之前
# （自 2013 年起、池内口径）已上榜 >= 30 次，归为「游资」（活跃投机席位）。
HOT_MIN_PRIOR = 30

_ANON_EXACT = frozenset({
    "机构专用", "机构", "机构投资者", "机构席位", "机构客户总部",
    "沪股通专用", "沪股通", "深股通专用", "深股通投资者", "深股通",
})
_INST_HINTS = ("机构席位", "机构客户")
_NORTH_HINT = "股通"

COMPANY_ALIASES = (
    ("西藏同信证券", "东方财富证券"),
    ("西藏东方财富证券", "东方财富证券"),
    ("国泰君安证券", "国泰海通证券"),
    ("海通证券", "国泰海通证券"),
    ("安信证券", "国投证券"),
    ("中国中投证券", "中国中金财富证券"),
    ("财富证券", "财信证券"),
    ("国联证券", "国联民生证券"),
    ("申银万国证券", "申万宏源证券"),
    ("宏源证券", "申万宏源证券"),
    ("齐鲁证券", "中泰证券"),
    ("日信证券", "国融证券"),
    ("恒泰证券", "金融街证券"),
    ("联讯证券", "粤开证券"),
)

_WS = re.compile(r"\s+")
_PAREN = re.compile(r"[（(][^)）]*[)）]")
_DIGIT = re.compile(r"\d")
_FIRST_SEC = re.compile(r"证券")


def normalize_seat(name: str) -> str:
    """席位名规范化（规则在看任何 IC 之前冻结）。

    1. NFKC + 去空白
    2. 删除不含数字的括号注释（（对外营业部）/(非营业场所)/（迁址）…）；
       含数字的括号保留（交易单元(353800) 是不同的交易单元，不能合并）
    3. 先保护 分公司/营业部，再删公司后缀（股份有限公司/有限责任公司/有限公司/公司），
       否则 上海分公司 会被 公司 规则啃成 上海分
    4. 统一 证券营业部 -> 营业部
    5. 公司前缀（首个 证券 之前）套用改名别名表
    """
    text = unicodedata.normalize("NFKC", "" if name is None else str(name))
    text = _WS.sub("", text)
    text = _PAREN.sub(lambda m: m.group(0) if _DIGIT.search(m.group(0)) else "", text)
    text = text.replace("分公司", "\u0001").replace("营业部", "\u0002")
    for token in ("股份有限公司", "有限责任公司", "有限公司", "公司"):
        text = text.replace(token, "")
    text = text.replace("\u0001", "分公司").replace("\u0002", "营业部")
    text = re.sub(r"证券营业部$", "营业部", text)
    match = _FIRST_SEC.search(text)
    if match:
        head, tail = text[:match.end()], text[match.end():]
        for old, new in COMPANY_ALIASES:
            if head == old or head.startswith(old):
                head = new + head[len(old):]
                break
        text = head + tail
    return text


def _is_institution(key: str) -> bool:
    if key in _ANON_EXACT:
        return True
    if any(hint in key for hint in _INST_HINTS):
        return True
    return "机构" in key and key.endswith("机构")


def _is_northbound(key: str) -> bool:
    return _NORTH_HINT in key


def _seat_frame(ctx):
    """读上游 -> 复原权威买卖额 -> 去重 -> 规范化 -> PIT 计数 -> 分类。"""
    panel = ctx.panel
    days = panel.dates
    frame = ctx.dataset(
        DT,
        columns=["trade_date", "stock_code", "org_name",
                 "buy_amount", "sell_amount", "net_buy_amount"],
        years=(2013, int(days[-1]) // 10000),
    )
    if frame.empty:
        return None
    frame = frame[frame.stock_code.isin(panel.codes)]
    if frame.empty:
        return None
    buy = pd.to_numeric(frame["buy_amount"], errors="coerce").to_numpy(np.float64)
    sell = pd.to_numeric(frame["sell_amount"], errors="coerce").to_numpy(np.float64)
    net = pd.to_numeric(frame["net_buy_amount"], errors="coerce").to_numpy(np.float64)
    with np.errstate(all="ignore"):
        gross = buy + sell
    ok = (np.isfinite(gross) & np.isfinite(net) & (gross >= 0.)
          & (np.abs(net) <= gross + 1e-6))
    if not ok.any():
        return None
    frame = frame[ok]
    gross, net = gross[ok], net[ok]
    names = frame["org_name"].astype(str)
    lookup = {name: normalize_seat(name) for name in pd.unique(names.to_numpy())}
    records = pd.DataFrame({
        "day": ctx.date_col(frame["trade_date"]),
        "code": frame["stock_code"].to_numpy(),
        "key": names.map(lookup).to_numpy(),
        "gross": np.round(gross, 2),
        "net": np.round(net, 2),
    })
    records = records.drop_duplicates(["day", "code", "key", "gross", "net"])
    records = records.sort_values(["day"], kind="stable")
    seat_day = records.drop_duplicates(["day", "code", "key"])
    seat_day = seat_day.assign(
        prior_key=seat_day.groupby("key", sort=False).cumcount(),
        prior_stock=seat_day.groupby(["code", "key"], sort=False).cumcount(),
    )
    records = records.merge(
        seat_day[["day", "code", "key", "prior_key", "prior_stock"]],
        on=["day", "code", "key"], how="left", validate="many_to_one")
    unique_keys = pd.unique(records["key"].to_numpy())
    inst_map = {k: _is_institution(k) for k in unique_keys}
    north_map = {k: _is_northbound(k) for k in unique_keys}
    records["is_inst"] = records["key"].map(inst_map).to_numpy(dtype=bool)
    records["is_north"] = records["key"].map(north_map).to_numpy(dtype=bool)
    records["trackable"] = ~(records["is_inst"] | records["is_north"])
    records["hot"] = records["trackable"].to_numpy() & (records["prior_key"] >= HOT_MIN_PRIOR)
    records["buy"] = (records["gross"] + records["net"]) / 2.
    records["sell"] = (records["gross"] - records["net"]) / 2.
    return records


def _grid(ctx, records, weight):
    return ctx.event_grid(records["code"].to_numpy(), records["day"].to_numpy(),
                          np.asarray(weight, dtype=np.float64))


def _zero_ratio(numerator, denominator):
    """零填充口径：分母为 0（窗口内无上榜记录）时取 0，而不是 NaN。"""
    safe = np.where(denominator > 0., denominator, 1.)
    with np.errstate(all="ignore"):
        out = np.where(denominator > 0., numerator / safe, 0.)
    return out


def build(parameters):
    """R3-A 唯一的构造器：mechanism x window 都事前登记，不做参数搜索。"""
    if set(parameters) != {"signal", "window"}:
        raise ValueError("R3-A 只接受 {signal, window} 两个参数")
    signal = parameters["signal"]
    window = int(parameters["window"])
    if signal not in SIGNALS or window not in WINDOWS[signal]:
        raise ValueError("未登记的机制/窗口组合")

    def compute(ctx):
        records = _seat_frame(ctx)
        if records is None:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        net = records["net"].to_numpy(np.float64)
        gross = records["gross"].to_numpy(np.float64)
        is_inst = records["is_inst"].to_numpy(bool)
        hot = records["hot"].to_numpy(bool)
        trackable = records["trackable"].to_numpy(bool)
        prior_key = records["prior_key"].to_numpy(np.float64)
        prior_stock = records["prior_stock"].to_numpy(np.float64)
        buy = records["buy"].to_numpy(np.float64)
        sell = records["sell"].to_numpy(np.float64)
        total = lambda grid: mx.roll_sum(grid, window, min_count=1)

        if signal == "seat_class_divergence":
            numerator = _grid(ctx, records, np.where(is_inst, net, 0.))
            counter = _grid(ctx, records, np.where(hot, net, 0.))
            denominator = _grid(ctx, records, gross)
            value = _zero_ratio(total(numerator) - total(counter), total(denominator))
        elif signal == "seat_concentration":
            positive = np.where(net > 0., net, 0.)
            top = _grid(ctx, records, positive)
            squares = _grid(ctx, records, positive * positive)
            value = _zero_ratio(total(squares), total(top) ** 2)
        elif signal == "seat_persistence":
            observations = _grid(ctx, records, np.where(trackable, 1., 0.))
            repeats = _grid(ctx, records, np.where(trackable & (prior_stock >= 1.), 1., 0.))
            value = _zero_ratio(total(repeats), total(observations))
        elif signal == "seat_buy_sell_overlap":
            both = _grid(ctx, records, np.minimum(buy, sell))
            denominator = _grid(ctx, records, gross)
            value = _zero_ratio(total(both), total(denominator))
        else:
            observations = _grid(ctx, records, np.where(trackable, 1., 0.))
            fresh = _grid(ctx, records, np.where(trackable & (prior_key == 0.), 1., 0.))
            value = _zero_ratio(total(fresh), total(observations))

        value = np.asarray(value, dtype=np.float64)
        value = np.where(np.asarray(ctx.universe, dtype=bool), value, np.nan)
        value = np.where(np.isfinite(value), value, np.nan)
        return value.astype(np.float64, copy=False)

    return compute
