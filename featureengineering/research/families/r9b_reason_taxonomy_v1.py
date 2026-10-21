"""R9-B / B-new-2 — 龙虎榜/上榜原因的**分类学**暴露（reason taxonomy）。

一个候选、一条实现路径、一个冻结参数（window=60），不做参数搜索、不读语义散文。

────────────────────────────────────────────────────────────────────
为什么这是未开采的维度
  * `stock_top_list` / `stock_dragon_tiger` 的 `reason` 列在 177 个正式因子里
    **零引用**；`doc/因子清理记录_20261003.md`（374 个已删个股因子）里也没有
    任何因子对 reason 做过分档。R4-I 之后的席位家族全部是金额/身份机制。
  * 已删的 `efx_dragon_ratio_balance` 用 buy_ratio−sell_ratio（金额占比），
    `efx_top_imbalance` 用 l_buy/l_sell（金额），本因子**不读任何金额列**。

────────────────────────────────────────────────────────────────────
分类学（事前冻结的关键词规则，按顺序首个命中者生效，人工可审计）
  规则作用在 NFKC 规范化 + 去空白后的 reason 字符串上。顺序即优先级：
    1. 含「退市整理」                        → OTHER_DELIST
    2. 含「无价格涨跌幅限制」                → OTHER_NOLIMIT
    3. 含「异常」                            → OTHER_ABNORMAL（含「严重异常」）
    4. 含「融资买入」                        → OTHER_MARGIN
    5. 含「可转债」                          → OTHER_CB
    6. 含「涨跌幅偏离值累计达到+40%」        → OTHER_BIDIR（北交所双向阈值）
    7. 含「换手率」或「振幅」                → CHURN      （流动性/投机型触发）
    8. 含「涨幅」                            → UP_DEV     （上行方向偏离型触发）
    9. 含「跌幅」                            → DOWN_DEV   （下行方向偏离型触发）
   10. 其余                                  → UNMAPPED
  实测覆盖：见 b_report.md §reason 分类学（distinct 值计数 + 行覆盖率 + 未命中清单）。

────────────────────────────────────────────────────────────────────
构造
  churn = 窗口内 CHURN 类上榜记录数
  dev   = 窗口内 UP_DEV + DOWN_DEV 类上榜记录数
  value = (churn − dev) / (churn + dev)，窗口内 churn+dev = 0 → NaN（原生口径）
  取值域 [−1, 1]：+1 = 只被流动性型原因带上榜，−1 = 只被方向偏离型原因带上榜。

────────────────────────────────────────────────────────────────────
去重论证
  * 与 `dragon_tiger_org_net_20`（已删）无关：那是机构席位净买额/成交额，金额。
  * 与 `s26_structure_side_hhi_v1`（库内）无关：那是 stock_main_fund_flow 四档
    委托规模的买/卖侧 HHI 之差，输入表、对象（单子大小）都不同。
  * 与 `limit_broken_ratio_20` / `limit_days_since_up_20`（库内，stock_limit_list）
    不同表、不同事件；与已删 `efx_limit_*` 系列（涨停封板微结构）也不同表。
  * R3-A `r3seat_*`（从未执行）与 R4-I `r4i_seat_closeout_v1`（4 个候选全部
    不晋级）全部是**席位/金额**机制，没有一个碰 reason 列。
  * 与 C1 `r9b_seat_identity_breadth_w60_v1` 的表、键、机制都不同
    （C1=stock_dragon_tiger 席位身份集合；本因子=stock_top_list 上榜原因档位）。

────────────────────────────────────────────────────────────────────
PIT：同 C1 —— `conf/frequency.yaml` 对 `stock_top_list` 声明 `delay_days: 0`；
项目与上游都没有 `state/delay_history.json`；已删 `top_list_net_rate_20` 的
原定义备注记录 2026-09-15 晚实测当天可得并去掉位移。主口径不位移，报告给出
lag=1 的离线敏感性。
"""
from __future__ import annotations

import unicodedata

import numpy as np

from fea import mathx as mx

TL = "stock_top_list"
WINDOW = 60
PARAMETERS = {"signal": "reason_trigger_churn", "window": WINDOW}

CATEGORIES = ("OTHER_DELIST", "OTHER_NOLIMIT", "OTHER_ABNORMAL", "OTHER_MARGIN",
              "OTHER_CB", "OTHER_BIDIR", "CHURN", "UP_DEV", "DOWN_DEV", "UNMAPPED")

# 冻结的关键词规则：按顺序首个命中者生效。
_RULES = (
    ("OTHER_DELIST", ("退市整理",)),
    ("OTHER_NOLIMIT", ("无价格涨跌幅限制",)),
    ("OTHER_ABNORMAL", ("异常",)),
    ("OTHER_MARGIN", ("融资买入",)),
    ("OTHER_CB", ("可转债",)),
    ("OTHER_BIDIR", ("涨跌幅偏离值累计达到+40%",)),
    ("CHURN", ("换手率", "振幅")),
    ("UP_DEV", ("涨幅",)),
    ("DOWN_DEV", ("跌幅",)),
)


def classify_reason(text) -> str:
    """冻结分类器：输入原始 reason，输出 CATEGORIES 中的一类。"""
    s = unicodedata.normalize("NFKC", "" if text is None else str(text))
    s = "".join(s.split())
    for name, needles in _RULES:
        for needle in needles:
            if needle in s:
                return name
    return "UNMAPPED"


def build(parameters):
    """R9-B reason 分类学：唯一构造器，参数事前冻结。"""
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
        frame = ctx.dataset(TL, columns=["trade_date", "stock_code", "reason"],
                            years=(y0, y1))
        if frame is None or len(frame) == 0:
            return out

        code_idx = ctx.code_index(frame["stock_code"].to_numpy())
        keep = code_idx >= 0
        if not keep.any():
            return out
        day = ctx.date_col(frame["trade_date"])[keep]
        row = np.searchsorted(days, day)
        row_c = np.clip(row, 0, T - 1)
        inrange = (row < T) & (days[row_c] == day)
        if not inrange.any():
            return out
        codes = frame["stock_code"].to_numpy()[keep][inrange]
        rows = row[inrange]
        reason = frame["reason"].astype(str).to_numpy()[keep][inrange]

        cat = np.array([classify_reason(x) for x in reason], dtype=object)
        is_churn = (cat == "CHURN").astype(np.float64)
        is_dev = np.isin(cat, ("UP_DEV", "DOWN_DEV")).astype(np.float64)

        evday = days[rows]
        churn = ctx.event_grid(codes, evday, is_churn)
        dev = ctx.event_grid(codes, evday, is_dev)

        w_churn = mx.roll_sum(churn, window, min_count=1)
        w_dev = mx.roll_sum(dev, window, min_count=1)
        den = w_churn + w_dev
        with np.errstate(all="ignore"):
            value = (w_churn - w_dev) / den
        value = np.where(den > 0., value, np.nan)          # NaN 原生主口径
        value = np.where(np.asarray(ctx.universe, dtype=bool), value, np.nan)
        value = np.where(np.isfinite(value), value, np.nan)
        return value.astype(np.float64, copy=False)

    return compute
