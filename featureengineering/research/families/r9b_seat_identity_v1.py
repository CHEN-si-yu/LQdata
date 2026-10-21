"""R9-B / B-new-1 — 龙虎榜「席位身份宽度」（WHO），不是席位集中度（HOW MUCH）。

一个候选、一条实现路径、一个冻结参数（window=60），不做参数搜索。

────────────────────────────────────────────────────────────────────
数据事实（写第一行公式前实测，本单元自测脚本 + 项目内既有记录）
  * stock_dragon_tiger 全分区 2,590,746 行 / 14 个 year 分区（2013–2026）。
  * 同一 (trade_date, stock_code, org_name) 会出现多行（买卖榜方向、同一席位
    多空两侧、以及 2018 年买卖列整块互换产生的重复行）⇒ 身份计数前必须先按
    (day, code, seat_key) 去重，否则「出现次数」被行数污染。
  * 「机构专用」「沪股通专用」「深股通专用」等匿名名下**不是一个席位**，而是
    多个互不相识的机构/通道各占一行（R3-A 已实测确认）。把它们当成一个席位会
    系统性**低估**身份宽度 ⇒ 本因子把匿名席位（机构类 + 名称含「股通」）
    整体排除在身份计数之外，只统计**可识别**席位。
  * 本因子**不使用任何金额列**：既不读 buy_amount / sell_amount（2018 年约 94%
    行整块互换），也不读 net_buy_amount（已删因子 dragon_tiger_org_net_20 的
    审核记录写明其符号不可信）。构造是纯计数，金额口径的污染与它无关。

────────────────────────────────────────────────────────────────────
去重论证（为什么不是既有因子的重新表达）
  * `dragon_tiger_org_net_20`（已删）= 20 日「机构专用」净买额 / 20 日成交额，
    是金额水平；本因子零金额、且明确把「机构专用」排除。ρ 无法接近 1。
  * `s26_structure_side_hhi_v1`（库内，t=+12.21/+7.29）=
    sum((buy_tier/sum_buy)^2) − sum((sell_tier/sum_sell)^2)，是
    **stock_main_fund_flow 的四档委托规模**（sm/md/lg/elg）在买侧/卖侧的
    集中度之差 —— 度量「多大的单」，输入表完全不同，且它逐日计算、无窗口内
    身份集合的概念。本因子度量「多少人（可识别席位）」，不是「多大的钱」。
  * R3-A `r3seat_*` / R3-A v2：定义在
    `conf/research/round3_20261002/seat/`（14 个候选 + 2 个 family），
    **从未执行**：artifacts/research/runs 的 200 个 run 目录里 0 个 seat、
    ledger 0 条、evaluations 0 个 —— 无 IC、无收益、不构成试验消耗。
    其 seat_concentration 是「净买为正的席位记录之间的金额 HHI」（金额口径），
    seat_persistence 是「该席位是否在本股票上**历史上任何时刻**出现过」
    （全局 cumcount，无窗口），seat_novelty 是「全局首次出现的具名席位占比」。
    本因子是**窗口内**的席位身份集合大小 / 出现次数，与前四者都不同：
    persistence 问「是不是老面孔」（跨全历史），本因子问「窗口里挤不挤」（只看
    窗口内的集合大小）。R4-I `r4i_seat_closeout_v1`（唯一真正跑过并评价过的席位
    单元，4 个候选全部不晋级）测的是净额类分歧与强度，全部是金额机制。
  * 结论：本因子是「席位身份」维度上**从未被评价过**的构造；R3-A 的预登记状态
    按项目自身先例（v1 缺陷、从未跑完 ⇒ 不构成试验消耗）如实披露。

────────────────────────────────────────────────────────────────────
PIT 决策（本单元实测后冻结，见 b_report.md §PIT）
  * `conf/frequency.yaml`（项目权威可得性契约）对 `stock_dragon_tiger` 声明
    `delay_days: 0`；`fea/delay.py::load()` 只把 delay>0 的表放进
    `LAGGED_DATASETS`，所以框架守卫不含该表是**契约的必然结果**，不是漏登记。
  * 项目 `state/delay_history.json` **不存在**（project root 与上游
    datadownload 都找不到），故没有任何 `delay_obs` 观测可引用。
  * 已删因子 `dragon_tiger_org_net_20` 的原定义备注记录了 2026-09-15 晚的实测
    更正：`stock_dragon_tiger` 当天可得（T=2026-09-15 21:14，`delay_obs = 0`），
    并把 v2 的 `lag_grid(grid, 1)` **去掉**；`top_list_net_rate_20` 同一处理。
  * ⇒ 主口径**不做位移**（与两个最新生产对象一致）。BRIEF §2 的「必须 lag」与
    上述三条证据冲突，本单元按 Lead 裁决取「不位移」，并在报告里给出
    位移 1 个交易日的离线敏感性（等价于把输入整体滞后一天：本因子是
    上榜记录的因果滚动函数，输出整体下移一行 ≡ 输入下移一行）。
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

DT = "stock_dragon_tiger"
WINDOW = 60
PARAMETERS = {"signal": "seat_identity_breadth", "window": WINDOW}

# ── R3-A 冻结的席位名规范化规则：逐字沿用，不重新发明 ────────────────────
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


def normalize_seat(name) -> str:
    """席位名规范化（R3-A 冻结规则，逐字沿用）。"""
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


def is_anonymous(key: str) -> bool:
    """匿名席位 = 机构类名下（多个机构共享一个名字）+ 名称含「股通」（北向通道）。

    匿名名不能参与身份计数：把它当一个席位会低估宽度；按行拆开又会把
    「同一天同一匿名名的多行」误当成多个席位。
    """
    if key in _INST_EXACT:
        return True
    if any(hint in key for hint in _INST_HINTS):
        return True
    return _NORTH_HINT in key


def _intervals_to_grid(starts, ends, codes, T, C):
    """把一批 [start, end]（按面板行号）落成 (T, C) 的「覆盖计数」网格。

    用差分数组 + cumsum：每个区间对其覆盖到的每一天贡献 +1。
    """
    diff = np.zeros((T + 1, C), dtype=np.float64)
    np.add.at(diff, (starts, codes), 1.0)
    np.add.at(diff, (np.minimum(ends + 1, T), codes), -1.0)
    return np.cumsum(diff[:T], axis=0)


def build(parameters):
    """R9-B 席位身份宽度：唯一构造器，参数事前冻结。"""
    if dict(parameters) != PARAMETERS:
        raise ValueError("Only the single frozen definition is permitted")
    window = int(PARAMETERS["window"])

    def compute(ctx):
        panel = ctx.panel
        days = panel.dates
        T, C = panel.shape
        out = np.full((T, C), np.nan, dtype=np.float64)

        y0 = max(2010, int(days[0]) // 10000 - 1)
        y1 = int(days[-1]) // 10000
        frame = ctx.dataset(DT, columns=["trade_date", "stock_code", "org_name"],
                            years=(y0, y1))
        if frame is None or len(frame) == 0:
            return out

        day = ctx.date_col(frame["trade_date"])
        code_idx = ctx.code_index(frame["stock_code"].to_numpy())
        keep = code_idx >= 0
        if not keep.any():
            return out
        day = day[keep]
        code_idx = code_idx[keep].astype(np.int64)
        names = frame["org_name"].astype(str).to_numpy()[keep]

        row = np.searchsorted(days, day)
        row_c = np.clip(row, 0, T - 1)
        inrange = (row < T) & (days[row_c] == day)
        if not inrange.any():
            return out
        row = row[inrange].astype(np.int64)
        code_idx = code_idx[inrange]
        names = names[inrange]

        lookup = {n: normalize_seat(n) for n in pd.unique(names)}
        keys = np.array([lookup[n] for n in names], dtype=object)

        rec = pd.DataFrame({"row": row, "code": code_idx, "key": keys})
        rec = rec.drop_duplicates(["row", "code", "key"])          # 一格一席位
        ident = ~rec["key"].map(is_anonymous).to_numpy(dtype=bool)
        rec = rec[ident]
        if len(rec) == 0:
            return out
        rec = rec.sort_values(["code", "key", "row"], kind="stable")
        n = len(rec)
        r = rec["row"].to_numpy(np.int64)
        c = rec["code"].to_numpy(np.int64)
        k = rec["key"].to_numpy()

        # 出现次数网格：第 t 天计数窗口 [t-w+1, t] 内的可识别席位记录数
        obs = _intervals_to_grid(r, np.minimum(r + window - 1, T - 1), c, T, C)

        # 身份集合大小：每个 (code,key) 对覆盖到的天 = 它各次出现区间 [d, d+w-1] 的并集
        newgrp = np.empty(n, dtype=bool)
        newgrp[0] = True
        if n > 1:
            newgrp[1:] = (c[1:] != c[:-1]) | (k[1:] != k[:-1])
        end = np.minimum(r + window - 1, T - 1)
        cummax = pd.Series(end).groupby(np.cumsum(newgrp) - 1).cummax().to_numpy()
        prev = np.empty(n, dtype=np.int64)
        prev[0] = -1
        prev[1:] = cummax[:-1]
        prev[newgrp] = -1
        starts_new = newgrp | (r > prev + 1)
        sidx = np.flatnonzero(starts_new)
        a = r[sidx]
        b = np.empty(sidx.size, dtype=np.int64)
        b[:-1] = cummax[sidx[1:] - 1]
        b[-1] = cummax[-1]
        distinct = _intervals_to_grid(a, b, c[sidx], T, C)

        with np.errstate(all="ignore"):
            breadth = distinct / obs
        breadth = np.where(obs > 0., breadth, np.nan)      # NaN 原生主口径
        breadth = np.where(np.asarray(ctx.universe, dtype=bool), breadth, np.nan)
        breadth = np.where(np.isfinite(breadth), breadth, np.nan)
        return breadth.astype(np.float64, copy=False)

    return compute
