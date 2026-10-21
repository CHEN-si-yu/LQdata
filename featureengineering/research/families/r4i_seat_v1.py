"""R4-I 龙虎榜席位收口（seat closeout）：R3-A `class_divergence` 的 NaN 原生重表达
+ 机构净买强度（条件口径）。只读上游，不写生产。

背景与冻结口径（全部在看任何 IC 之前写死；B2 任务书 §4-U9）：
  * R3-A 判定 Revise：`seat_class_divergence` 零填充口径 9/9（FM t=+7.37/+3.16），
    但 NaN 条件口径符号翻转（w60 h1 −0.00194）⇒ 机制对缺失约定不稳健。
    本单元把 NaN 条件口径（未上榜=NaN）做成**原生主口径**重新登记，零填充只作并列对照。
  * 只用 `net_buy_amount`（权威净额）；绝不使用 buy_amount − sell_amount
    （2018 年约 94% 的行买卖列整块互换的污染，R3-A §2.2 实测）。
    gross = buy_amount + sell_amount（换位不变）仅用于归一。
  * 席位分类与规范化**沿用 R3 冻结规则**（逐字复制自 research/families/r3_seat_v2.py，
    不 import 以免引入跨模块运行时耦合）：四分类 机构/北向/游资/普通，
    游资 = 非机构非北向具名席位且 PIT（严格更早记录）池内上榜 >= 30 次；
    14 条公司改名别名表；括号/后缀/交易单元规范化规则不变。
  * PIT：只使用 trade_date <= T 的席位明细；PIT 计数只统计严格早于该记录日的行；
    读表起点 2013（原始表起点）。T 日盘后披露当日可用（与既有正式因子不滞后口径一致，
    见 2026-09-15 的 delay_obs=0 实测记录）。

机制（全部事前登记，不做参数搜索）：
  1) class_divergence_nan_native(window ∈ {20,60})：
     value(T) = [Σ_w 机构净买 − Σ_w 游资净买] / Σ_w gross，W = 尾随窗口（含 T）。
     **NaN 原生**：窗口内该股无任何上榜记录 → NaN（不是 0）。
     方向 true（登记与 R3 相同：机构相对游资净买越强 → 正）。
  2) inst_net_intensity_cond(window=20)：
     上榜日 r = 机构专用席位净买额合计 / 当日总成交额（stock_daily.amount，元）。
     value(T) = decay_linear(r, 20, min_count=1)（WorldQuant DecayLinear：权重 20..1 归一；
     非上榜日不参与、贡献 0；窗口内无任何有效上榜日 → NaN 原生）。
     方向 true。
  3) class_divergence_zerofill_control(window=60)：机制 1 的零填充口径对照
     （未上榜=0，与 R3-A 零填充主口径逐字一致），仅作并列对照，不单独立主张。

窗口内"有记录但 gross 合计为 0"的格子：与 R3 一致取 0.0（不是 NaN）；本模块在
docstring 记录该规则并在报告披露受影响格数（若为 0 则无影响）。

单位：全部为无量纲比率（金额单位为元，分子分母同源相除）。
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from fea import mathx as mx

DT = "stock_dragon_tiger"

WINDOWS = {
    "class_divergence_nan_native": (20, 60),
    "inst_net_intensity_cond": (20,),
    "class_divergence_zerofill_control": (60,),
}
SIGNALS = tuple(WINDOWS)
HORIZONS = (1, 5)

# ---------------------------------------------------------------- 冻结常量（复制自 r3_seat_v2）
# 席位分类阈值（事前固定）：非机构、非北向的具名席位，若在信号日之前
# （自 2013 年起、池内口径）已上榜 >= 30 次，归为「游资」（活跃投机席位）。
HOT_MIN_PRIOR = 30

# 席位四分类（事前固定）：机构 / 北向 / 游资 / 普通营业部。
_INST_EXACT = frozenset({"机构专用", "机构", "机构投资者", "机构席位", "机构客户总部"})
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
    """席位名规范化（规则照 r3_seat_v2 冻结版逐字复制，未做任何改动）。

    1. NFKC + 去空白
    2. 删除不含数字的括号注释；含数字的括号保留（交易单元(353800) 不能合并）
    3. 先保护 分公司/营业部，再删公司后缀
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
    if key in _INST_EXACT:
        return True
    return any(hint in key for hint in _INST_HINTS)


def _is_northbound(key: str) -> bool:
    return _NORTH_HINT in key


# ---------------------------------------------------------------- 记录构建（纯函数，离线分析可复用）
RECORD_COLUMNS = ["trade_date", "stock_code", "org_name",
                  "buy_amount", "sell_amount", "net_buy_amount"]


def seat_records(day, code, org_name, buy_amount, sell_amount, net_buy_amount):
    """上游行 → 去重/规范化/分类后的席位记录 DataFrame；无有效行返回 None。

    与 r3_seat_v2._seat_frame 的几何完全一致（同一过滤、同一去重键、同一 PIT 计数、
    同一分类与买卖额复原），只是把读表部分留给调用方（ctx 或离线脚本）。
    """
    buy = pd.to_numeric(pd.Series(buy_amount), errors="coerce").to_numpy(np.float64)
    sell = pd.to_numeric(pd.Series(sell_amount), errors="coerce").to_numpy(np.float64)
    net = pd.to_numeric(pd.Series(net_buy_amount), errors="coerce").to_numpy(np.float64)
    with np.errstate(all="ignore"):
        gross = buy + sell
    ok = (np.isfinite(gross) & np.isfinite(net) & (gross >= 0.)
          & (np.abs(net) <= gross + 1e-6))
    if not ok.any():
        return None
    day = np.asarray(day, dtype=np.int64)[ok]
    code = np.asarray(code)[ok]
    gross, net = gross[ok], net[ok]
    names = pd.Series(np.asarray(org_name)[ok]).astype(str)
    lookup = {name: normalize_seat(name) for name in pd.unique(names.to_numpy())}
    records = pd.DataFrame({
        "day": day,
        "code": code,
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


def _seat_frame(ctx):
    """ctx 版记录构建：读表 → 池内过滤 → 日期转换 → seat_records()。"""
    panel = ctx.panel
    days = panel.dates
    frame = ctx.dataset(
        DT,
        columns=list(RECORD_COLUMNS),
        years=(2013, int(days[-1]) // 10000),
    )
    if frame.empty:
        return None
    frame = frame[frame.stock_code.isin(panel.codes)]
    if frame.empty:
        return None
    return seat_records(
        ctx.date_col(frame["trade_date"]),
        frame["stock_code"].to_numpy(),
        frame["org_name"].to_numpy(),
        frame["buy_amount"].to_numpy(),
        frame["sell_amount"].to_numpy(),
        frame["net_buy_amount"].to_numpy(),
    )


def _grid(ctx, records, weight):
    return ctx.event_grid(records["code"].to_numpy(), records["day"].to_numpy(),
                          np.asarray(weight, dtype=np.float64))


def build(parameters):
    """R4-I 唯一构造器：mechanism x window 全部事前登记，不做参数搜索。"""
    if set(parameters) != {"signal", "window"}:
        raise ValueError("R4-I 只接受 {signal, window} 两个参数")
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
        total = lambda grid: mx.roll_sum(grid, window, min_count=1)

        if signal in ("class_divergence_nan_native", "class_divergence_zerofill_control"):
            numerator = _grid(ctx, records, np.where(is_inst, net, 0.))
            counter = _grid(ctx, records, np.where(hot, net, 0.))
            denominator = _grid(ctx, records, gross)
            obs = _grid(ctx, records, np.ones(len(records)))
            num = total(numerator) - total(counter)
            den = total(denominator)
            with np.errstate(all="ignore"):
                ratio = num / den
            value = np.where(den > 0., ratio, 0.)
            if signal == "class_divergence_nan_native":
                # NaN 原生主口径：窗口内无任何上榜记录 = NaN（不是 0）。
                value = np.where(total(obs) > 0., value, np.nan)
            # zerofill_control：不设 obs 掩码，未上榜=0（R3-A 零填充口径）。
        elif signal == "inst_net_intensity_cond":
            inst_net = _grid(ctx, records, np.where(is_inst, net, 0.))
            listed = _grid(ctx, records, np.ones(len(records)))
            amount = np.asarray(ctx.px("amount"), dtype=np.float64)
            with np.errstate(all="ignore"):
                ratio = inst_net / amount
            ratio = np.where((listed > 0.) & np.isfinite(amount) & (amount > 0.),
                             ratio, np.nan)
            # DecayLinear(20)：权重 20..1 归一；非上榜日贡献 0；min_count=1
            # ⇒ 窗口内无任何有效上榜日 → NaN 原生。
            value = mx.decay_linear(ratio, window, min_count=1)
        else:  # pragma: no cover - 参数校验已挡住
            raise ValueError(signal)

        value = np.asarray(value, dtype=np.float64)
        value = np.where(np.asarray(ctx.universe, dtype=bool), value, np.nan)
        value = np.where(np.isfinite(value), value, np.nan)
        return value.astype(np.float64, copy=False)

    return compute
