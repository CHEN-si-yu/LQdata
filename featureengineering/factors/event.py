"""事件 / 筹码类因子（2 个）—— 数据源与基本面无关，提供正交的信息。

这类因子用的是「事件」语义而不是「状态」语义：
涨停是当天发生的事（要散点 + 补零 + 滚动计数），
股东户数是一个会保持到下次披露的状态量（要 as-of 前向填充）。
两者在 `FactorContext` 里是**不同的方法**（`event_grid` vs `asof_daily`），
用一个 as-of 糊弄两者会让涨停次数退化成「0 或永远的陈旧值」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fea.spec import FactorSpec, register


@register(FactorSpec(
    name="holder_number_chg", group="sentiment",
    deps=("stock_holder_number",),
    desc="股东户数变化率 = 本期 / 上期 − 1（户数减少 = 筹码集中，低者优）",
    formula="Chg = HolderNum_t / HolderNum_{t-1} - 1（两条腿都取「该公告日为止最新一版」）",
    start="2016-01-04",          # 上游 stock_holder_number 实测最早 2016-01-04
    version=2,                   # ★ 2026-09-17 修 PIT 违规，强制重建（见 note）
    warmup_days=700, higher_is_better=False,
    note="上游披露期不规律（1~12 月都有），所以对「与上一期间的间隔」做保护，"
         "间隔超过 6 个月的样本置 NaN，避免拿两个不同年份的观察值相比。"
         "★ 2026-09-17 修 PIT 违规：旧实现对**全表** `drop_duplicates(keep='last')`，"
         "而同一个 `end_date` 会在一周内被不同公司陆续披露 —— 新公告一到就挤掉旧公告"
         "并把自己的 `ann_date` 盖上去，于是**历史生效日被改写**（实测一次重建改写 9 天）。"
         "现在改成 **as-of 状态机**：逐条公告推进「最新两期」的读数，每条公告只看得到"
         "它自己那天（含）之前的信息 → 历史值不再随新公告变化。",
))
def holder_number_chg(ctx):
    hn = ctx.dataset(
        "stock_holder_number",
        columns=["stock_code", "ann_date", "end_date", "holder_num"],
    )
    if hn.empty:
        return ctx.panel.empty()

    hn = hn[hn["holder_num"].notna() & (hn["holder_num"] > 0)].copy()
    hn["ann"] = ctx.date_col(hn["ann_date"])
    hn["end"] = ctx.date_col(hn["end_date"])
    # ★ 保留**全部公告**（不做任何全表去重），按 (股票, 公告日, 报告期) 排序后逐条推进。
    #   同一天同一期的多条公告在这里天然去重（后一条覆盖前一条）。
    hn = hn.sort_values(["stock_code", "ann", "end"], kind="stable")

    codes: list[str] = []
    days: list[int] = []
    b_ends: list[int] = []      # 最新一期的报告期
    s_ends: list[int] = []      # 上一期的报告期
    ratio: list[float] = []     # 本期 / 上期
    for code, sub in hn.groupby("stock_code", sort=False):
        b_end = b_num = s_end = s_num = None
        last: tuple | None = None
        for ann, end, num in zip(sub["ann"].to_numpy(), sub["end"].to_numpy(),
                                 sub["holder_num"].to_numpy()):
            # ---- 维护「最新两期」：任何一条公告只影响它自己那一期 ----
            if b_end is None or end > b_end:
                s_end, s_num = b_end, b_num
                b_end, b_num = end, num
            elif end == b_end:
                b_num = num                      # 同一期的新版本（重述）
            elif s_end is None or end > s_end:
                s_end, s_num = end, num
            elif end == s_end:
                s_num = num
            else:
                continue                         # 更早的期：不影响最新两期
            if s_end is None:
                continue
            state = (b_end, b_num, s_end, s_num)
            if state == last:                    # 状态没变 → 不必发新行（as-of 会沿用上一行）
                continue
            last = state
            codes.append(code)
            days.append(int(ann))
            b_ends.append(int(b_end))
            s_ends.append(int(s_end))
            ratio.append(float(b_num) / float(s_num))
    if not codes:
        return ctx.panel.empty()

    # 间隔保护：两个报告期之间不应超过 ~6 个月（用真实天数，不是 YYYYMMDD 差值）
    e1 = pd.to_datetime(pd.Series(b_ends).astype(str), format="%Y%m%d")
    e0 = pd.to_datetime(pd.Series(s_ends).astype(str), format="%Y%m%d")
    gap = (e1 - e0).dt.days.to_numpy()
    r = np.asarray(ratio)
    val = np.where((gap > 0) & (gap <= 200), r - 1.0, np.nan)
    return ctx.asof_daily(np.asarray(codes), np.asarray(days, dtype=np.int32), val)


@register(FactorSpec(
    name="limit_up_count_20", group="event",
    deps=("stock_limit_up",),
    desc="过去 20 个交易日的涨停次数",
    formula="Count = sum(IsLimitUp, window=20 trading days)",
    start="2015-01-05",          # 上游 stock_limit_up 实测最早 2015-01-05
    warmup_days=60, higher_is_better=True,
    note="纯交易日滚动计数，与财务口径无关，是这批因子里唯一的行为面因子",
))
def limit_up_count_20(ctx):
    lu = ctx.dataset("stock_limit_up", columns=["stock_code", "trade_date"])
    if lu is None or lu.empty:
        return ctx.panel.empty()
    grid = ctx.event_grid(lu["stock_code"].to_numpy(), ctx.date_col(lu["trade_date"]))
    return ctx.roll_sum(grid, 20)
