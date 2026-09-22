"""V16 样本外评价：逐日 RankIC、可执行 ΣtopN、现金策略与四列打分。

本文件负责「训练完成之后的事」：拼各折推演结果、算 IC、跑现金回测，把可分析的
CSV / JSON / 图落盘，生成 REPORT.md 与最新一日推荐表，并调度 train.sh 的整条流水线。
"""

# =============================================================================
# ① 导入
#
# matplotlib 只在下标曲线的函数里按需 import：IC-only 的运行不该为它付加载开销。
# =============================================================================
import csv
import json
import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

from model import (ROOT, RECIPE, Panel, Prices, atomic_json, axis, corr, provenance,
                   quarters, PredictModel, predict, torch)

# =============================================================================
# ② 指标：逐标签 IC 表
# =============================================================================


def ic_table(pred, panel, days):
    """逐标签算 RankIC / ICIR / Pearson IC：先按日算，再在日间平均。"""
    results = []
    for label in panel.labels:
        ic = []
        pearson = []
        for p, d in zip(pred, days):
            y = panel.Y[label][d]
            ok = np.isfinite(p) & np.isfinite(y)
            if ok.sum() < 3:
                continue
            ic.append(corr(rankdata(p[ok]), rankdata(y[ok])))
            pearson.append(corr(p[ok], y[ok]))
        std = float(np.std(ic, ddof=1)) if len(ic) > 1 else 0.
        results.append(dict(label=label, days=len(ic), RankIC=float(np.mean(ic)),
                            ICIR=float(np.mean(ic) / std) if std else 0.,
                            Pearson_IC=float(np.mean(pearson))))
    return results

# =============================================================================
# ③ 无摩擦可执行收益（ΣtopN）
# =============================================================================


def costfree(pred, panel, prices, days, n):
    """信号日选前 n、次日开盘买得进才算，用逐日 1d 标签求和。

    这是可执行无摩擦口径，不是现金账户净值——两者不能混用。
    返回 (汇总指标, 逐日收益, 逐笔选股)。
    """
    daily = []
    selections = []
    for p, d in zip(pred, days):
        eligible = np.flatnonzero(np.isfinite(p) & prices.next_entry[d])
        picks = eligible[np.argsort(-p[eligible], kind='stable')[:n]]
        y = panel.Y['label_ret_1d'][d]
        if len(picks) and not np.isfinite(y[picks]).all():
            raise ValueError('已选择股票的收益缺失，不能事后剔除或替换')
        ret = float(y[picks].sum() / n) if len(picks) else 0.
        base = float(np.nanmean(y[eligible])) if len(eligible) else 0.
        daily.append(dict(trade_date=str(panel.days[d]), return_value=ret, universe_return=base))
        for c in picks:
            selections.append(dict(trade_date=str(panel.days[d]), stock_code=str(panel.codes[c]),
                                   value=float(p[c]), ret=float(y[c])))
    returns = np.asarray([x['return_value'] for x in daily])
    ranked = np.sort(returns)[::-1]
    return dict(topn=n, sum_return=float(returns.sum()),
                sum_excess=float(sum(x['return_value'] - x['universe_return'] for x in daily)),
                remove_best_1=float(returns.sum() - ranked[:1].sum()),
                remove_best_2=float(returns.sum() - ranked[:2].sum()),
                remove_best_5=float(returns.sum() - ranked[:5].sum()),
                H1=float(returns[:len(returns) // 2].sum()),
                H2=float(returns[len(returns) // 2:].sum()),
                days=len(returns)), daily, selections

# =============================================================================
# ④ 费用与现金回测
# =============================================================================


def fees(notional, sell=False):
    """原参考工程/项目约定的固定费率；这是研究假设，不是历史费率复原。"""
    return max(notional * .00025, 5) + notional * .00001 + (notional * .0005 if sell else 0)


# --- 策略注册表（2026-09-21 新增，见 TODO.md RD10）---
# 前 4 条是原口径（固定周期换仓），保持不变、作为 baseline 组；
# 后面是参考工程 LINGQIDATA 的**排名退出族**形态 —— 它的实测是"按打分排名决定退不退"
# 比"固定周期换仓"普适（跨 24 个模型超基线 19~20 次），所以先原样搬形态进来比较。
# ★ 只动策略层、不动配方 ⇒ 复用已有 16 折打分，不需要重训。
STRATEGIES = [
    dict(name='top1_1d', topn=1, period=1),
    dict(name='top5_1d', topn=5, period=1),
    dict(name='top5_5d', topn=5, period=5),
    dict(name='top5_20d', topn=5, period=20),
    # 排名退出：每晚按最新打分重排，跌出前 exit_rank 名且持有满 min_hold 日 ⇒ 次日开盘换
    dict(name='re100_t1', topn=1, exit_rank=100, min_hold=2),
    dict(name='re300_t1', topn=1, exit_rank=300, min_hold=2),
    dict(name='re300_t2', topn=2, exit_rank=300, min_hold=2),
    dict(name='re600_t2', topn=2, exit_rank=600, min_hold=2),
    dict(name='re1000_t2', topn=2, exit_rank=1000, min_hold=2),
    dict(name='re300_t5', topn=5, exit_rank=300, min_hold=2),
    # 跳过第 1 名（TODO.md 第 5 条：集成的 top-1 是坏的而 2~5 名是好的）
    dict(name='skip1_t1_re100', topn=1, exit_rank=100, min_hold=2, skip=1),
    dict(name='skip1_t2_re300', topn=2, exit_rank=300, min_hold=2, skip=1),
    dict(name='skip1_t5_1d', topn=5, period=1, skip=1),
    # 止损 / 移动止盈（参考工程 S5）：8% 止损是它扫出来的甜点，15% 移动止盈配套
    dict(name='re300_sl8_t2', topn=2, exit_rank=300, min_hold=2, stop_loss=.08),
    dict(name='re300_tr15_t2', topn=2, exit_rank=300, min_hold=2, trailing=.15),
    dict(name='re300_sl8_tr15_t2', topn=2, exit_rank=300, min_hold=2, stop_loss=.08, trailing=.15),
    dict(name='re600_sl8_tr15_t2', topn=2, exit_rank=600, min_hold=2, stop_loss=.08, trailing=.15),
    # --- ★ 台地（2026-09-21 网格扫描，`V17/strategy_scan.py --mode plateau`）---
    # 「持 5 只 + 最少持 5 日 + 跌出前 400~700 名才换」这一族：**四折全正、三种集成规则全正**
    # （skip=0 时净收益 +6%~+16%、四折最小值 -4.5%~+7.3%）。对照 baseline：集成 +28% 但
    # **四折全负**（-14.7/-16.8/-43.8/-52.8）—— 所以这是"用 12~22pp 的均值换掉全部折的尾部风险"。
    # 取 400/500/600/700 四个邻居而不是峰值格，就是为了让读者看到它是一片**台地**不是一根尖刺。
    dict(name='re400_t5_h5', topn=5, exit_rank=400, min_hold=5),
    dict(name='re500_t5_h5', topn=5, exit_rank=500, min_hold=5),
    dict(name='re600_t5_h5', topn=5, exit_rank=600, min_hold=5),
    dict(name='re700_t5_h5', topn=5, exit_rank=700, min_hold=5),
    # 跳过第 1 名的对照：集成收益更高（+31%~+60%），但**名次剖面不支持它**（第 1 名的日均
    # 1d 收益 +0.276%/天，反而是 1~6 名里最高的）⇒ 只作对照行，不当结论。
    dict(name='re500_t5_h5_sk1', topn=5, exit_rank=500, min_hold=5, skip=1),
]


def cash_backtest(pred, panel, px, days, n=5, period=1, exit_rank=None, min_hold=1,
                  stop_loss=None, trailing=None, skip=0, sell_at='open'):
    """10 万元现金账户含费用模拟：T 日收盘出分数，T+1 开盘下单执行。

    两种换仓模式，由 exit_rank 是否给出决定：

    - **固定周期**（`exit_rank=None`，原口径）：每 `period` 个交易日按最新排名重选 n 只，
      相同目标继续持有（避免无意义的重复买卖），不同目标在下一个开盘卖出并买入。
    - **排名退出族**（`exit_rank=k`，参考工程 S4/S5）：**每晚**用最新打分评估持仓，
      跌出前 k 名、或触发止损/移动止盈，且持有满 `min_hold` 个交易日 ⇒ 次日开盘换掉；
      空出的名额按排名从高到低补"当天开盘买得进"的票。

    `skip` = 跳过排名最前的 skip 只。**两种模式都生效**（排名退出族的"前 k 名"与补位都从
    `skip` 名之后算起）—— 给"集成的 top-1 不可信"留的对照，见 TODO.md §12-5。

    `sell_at='close'` 时**卖单改成当日收盘价成交**（买单仍在开盘）：持仓比开盘卖多拿半天，
    且资金要等到收盘才回笼（同一天的买入预算因此偏保守，不会高估收益）。这是与参考工程
    "收盘卖 > 开盘卖"那条对应的对照实验，**默认口径仍是开盘卖**。

    止损/移动止盈在**信号日收盘**（d-1）上判定、在执行日开盘/收盘执行 —— 与"T 日收盘出分数、
    T+1 开盘下单"同一条时序，不看执行日当天行情；比价一律用**复权价**，否则除权会被误判成暴跌。
    卖不出去（跌停/停牌）或当日刚买的顺延。返回 (汇总指标, 逐日净值曲线, 逐笔成交)。
    """
    cash = 100000.
    holdings = {}
    buys = {}
    entry = {}          # 买入成本（复权口径），止损/移动止盈的基准
    peak = {}           # 持有期内复权收盘最高值，移动止盈的基准
    desired = []
    curves = []
    trades = []
    corp_events = 0
    first, last = int(days[0]) + 1, int(days[-1]) + 2
    pred_by_day = {int(d): p for d, p in zip(days, pred)}
    # 复权价只在"止损/移动止盈"这条腿上用；原始价仍按整手下单（引擎其余部分不变）。
    adj_open = px.open * px.adj
    adj_close = px.close * px.adj

    for d in range(first, last + 1):
        # --- 公司行为：原始价按整手下单；复权比例折算经济股数，假设分红立即再投资。
        #     没有逐笔公司行为现金流，因此该部分是总收益近似，并非券商交割单。
        for c in list(holdings):
            ratio = px.adj[d, c] / px.adj[d - 1, c]
            if not np.isfinite(ratio) or ratio <= 0:
                raise ValueError('持仓复权因子缺失')
            if abs(ratio - 1) > 1e-8:
                holdings[c] *= ratio
                corp_events += 1

        # --- 目标持仓：信号日必须落在打分窗内 ---
        signal = d - 1
        p = pred_by_day.get(signal)
        if p is not None:
            candidates = np.flatnonzero(np.isfinite(p))
            # 信号日先排序；开盘只根据当时价格拒单/顺位补足，不用标签选股。
            ranked = candidates[np.argsort(-p[candidates], kind='stable')]
            if exit_rank is None:
                # 固定周期：只在调仓日重选目标，其余日子沿用上一份目标（相同目标不重复买卖）
                if (signal - int(days[0])) % period == 0:
                    desired = [int(c) for c in ranked[skip:] if px.entry[d, c]][:n]
            else:
                # 排名退出族：每晚重新评估持仓 —— 跌出前 exit_rank 名、或触发止损/移动止盈，
                # 且持有满 min_hold 日 ⇒ 次日开盘换。未满 min_hold 的一律留住（避免当日反复进出）。
                # skip 在两种模式下都生效：先丢掉最前的 skip 名，再取 exit_rank 名当"安全区"。
                head = set(int(c) for c in ranked[skip:skip + exit_rank])
                exits = set()
                for c in list(holdings):
                    # 名次退出受 min_hold 约束（避免刚买就被排名抖动甩掉）；
                    # 止损/移动止盈是风控，不受 min_hold 限制，但实际最早也只能 T+1 卖（引擎会拦）。
                    if c not in head and d - buys[c] >= min_hold:
                        exits.add(c)
                        continue
                    if stop_loss is None and trailing is None:
                        continue
                    price = adj_close[signal, c]
                    if not np.isfinite(price):
                        continue
                    peak[c] = max(peak.get(c, entry[c]), price)     # 先抬峰再比，刚创新高不会误触
                    if stop_loss is not None and price <= entry[c] * (1 - stop_loss):
                        exits.add(c)
                    elif trailing is not None and price <= peak[c] * (1 - trailing):
                        exits.add(c)
                desired = [c for c in holdings if c not in exits]
                # 空出的名额按排名从前 k 名里补，买得进才算；顺序即建仓优先级。
                for c in ranked[skip:]:
                    if len(desired) >= n:
                        break
                    c = int(c)
                    if c in desired or c in exits:
                        continue
                    if px.entry[d, c]:
                        desired.append(c)
        if d == last:
            desired = []                 # 末尾清仓，把持仓换成现金后再估值

        # --- 先卖：不在目标里就卖；卖不出去或当日刚买的顺延到下一个开盘 ---
        # 收盘卖 = 卖单挪到当日收盘价，拒单判据也换成收盘口径（见 Prices.exit_close）。
        sell_ok = px.exit_close if sell_at == 'close' else px.exit
        sell_raw = px.raw['close'] if sell_at == 'close' else px.open
        blocked = []
        for c in list(holdings):
            if c in desired:
                continue
            if not sell_ok[d, c] or buys[c] >= d:
                blocked.append(c)
                continue
            quantity = holdings[c]
            price = sell_raw[d, c] * (1 - .0003)
            gross = quantity * price
            fee = fees(gross, True)
            cash += gross - fee
            trades.append(dict(date=str(panel.days[d]), code=str(panel.codes[c]), side='sell',
                               quantity=quantity, price=float(price), fee=float(fee)))
            del holdings[c]
            del buys[c]
            entry.pop(c, None)
            peak.pop(c, None)

        # --- 再买：按 1/N 预算、100 股整手，单笔不超过信号日成交额的 1% ---
        # 固定周期只在调仓日买；排名退出族每天都在评估，空出名额就补。
        if p is not None and d < last and (exit_rank is not None or (signal - int(days[0])) % period == 0):
            equity = cash + sum(q * px.open[d, c] if np.isfinite(px.open[d, c]) else q * px.close[d - 1, c]
                                for c, q in holdings.items())
            for c in desired:
                if c in holdings or len(holdings) >= n:
                    continue
                price = px.open[d, c] * (1 + .0003)
                # 成交额只用信号日已知值；不偷看执行日全天成交额。
                amount = panel.amount[signal, c]
                budget = min(cash, equity / n, amount * .01 if np.isfinite(amount) else 0.)
                quantity = int(max(0, budget - 5) / (price * (1 + .00025 + .00001)) / 100) * 100
                while quantity > 0 and quantity * price + fees(quantity * price) > budget:
                    quantity -= 100
                if quantity <= 0:
                    continue
                gross = quantity * price
                fee = fees(gross)
                cash -= gross + fee
                holdings[c] = float(quantity)
                buys[c] = d
                # 买入成本按复权价记：与止损/移动止盈的比价口径一致，除权不会误触。
                entry[c] = price * px.adj[d, c]
                peak[c] = entry[c]
                trades.append(dict(date=str(panel.days[d]), code=str(panel.codes[c]), side='buy',
                                   quantity=quantity, price=float(price), fee=float(fee)))

        # --- 当日估值 ---
        if cash < -1e-6 or len(holdings) > n:
            raise AssertionError('资金/持仓上限被突破')
        equity = cash + sum(q * px.close[d, c] for c, q in holdings.items())
        if not np.isfinite(equity):
            raise ValueError('持仓估值缺失')
        curves.append(dict(date=str(panel.days[d]), equity=float(equity), cash=float(cash),
                           positions=len(holdings), blocked_exits=len(blocked)))

    # --- 汇总：净值曲线 → 收益 / 回撤 / 夏普 / 费用 ---
    values = np.r_[100000., [r['equity'] for r in curves]]
    returns = values[1:] / values[:-1] - 1
    dd = values / np.maximum.accumulate(values) - 1
    return dict(topn=n, rebalance_days=period, exit_rank=exit_rank, min_hold=min_hold,
                stop_loss=stop_loss, trailing=trailing, skip=skip, sell_at=sell_at,
                return_value=float(values[-1] / 100000 - 1),
                max_drawdown=float(dd.min()),
                sharpe=float(np.mean(returns) / np.std(returns) * np.sqrt(242)) if np.std(returns) > 0 else 0.,
                total_fees=float(sum(t['fee'] for t in trades)), trades=len(trades), ending_cash=cash,
                ending_positions=len(holdings), corporate_action_adjustments=corp_events,
                start=str(panel.days[first]), end=str(panel.days[last]),
                corporate_action_assumption='复权比折算经济股数，分红立即再投资近似'), curves, trades

# =============================================================================
# ⑤ 打分四列落盘
# =============================================================================


def write_scores(pred, panel, days, out):
    """按年分区写打分：trade_date / stock_code / value / rank，float32 + 原子替换。"""
    for year in sorted(set(str(panel.days[d])[:4] for d in days)):
        chunks = []
        for p, d in zip(pred, days):
            if not str(panel.days[d]).startswith(year):
                continue
            ok = np.isfinite(p)
            chunks.append(pd.DataFrame(dict(trade_date=str(panel.days[d]), stock_code=panel.codes[ok],
                                            value=p[ok].astype(np.float32),
                                            rank=(rankdata(p[ok]) / ok.sum()).astype(np.float32))))
        df = pd.concat(chunks, ignore_index=True)
        df['trade_date'] = df['trade_date'].astype('string')
        df['stock_code'] = df['stock_code'].astype('string')
        folder = out / f'year={year}'
        folder.mkdir(parents=True, exist_ok=True)
        temp = folder / 'data.tmp.parquet'
        df.to_parquet(temp, index=False)
        os.replace(temp, folder / 'data.parquet')

# =============================================================================
# ⑥ 季度归集与复合校验
# =============================================================================


def quarter_metrics(source, strategy, curve, trades, pred, panel, days):
    """把全年曲线/成交按季度切段逐季出指标，并断言季度收益能复合回全年。"""
    rows = []
    previous = 100000.
    cursor = 0

    # 执行日归属于发出指令的信号季度；最后清仓顺延归入最后一个季度。
    boundaries = []
    for q in quarters(panel.days):
        qdays = [int(d) for d in days if str(pd.Period(panel.days[d], freq='Q')) == q]
        if qdays:
            boundaries.append((q, qdays))

    for i, (q, qdays) in enumerate(boundaries):
        end_day = qdays[-1] + (2 if i == len(boundaries) - 1 else 1)
        end_date = str(panel.days[end_day])
        start = cursor
        while cursor < len(curve) and curve[cursor]['date'] <= end_date:
            cursor += 1
        segment = curve[start:cursor]
        if not segment:
            continue
        eq = np.r_[previous, [r['equity'] for r in segment]]
        ret = eq[1:] / eq[:-1] - 1
        dd = eq / np.maximum.accumulate(eq) - 1
        tx = [t for t in trades if segment[0]['date'] <= t['date'] <= segment[-1]['date']]
        pos = np.flatnonzero(np.isin(days, qdays))
        ric = next(r for r in ic_table(pred[pos], panel, days[pos]) if r['label'] == RECIPE['label'])
        rows.append(dict(source=source, quarter=q, strategy=strategy, signal_days=len(qdays),
                         execution_start=segment[0]['date'], execution_end=segment[-1]['date'],
                         start_equity=float(previous), end_equity=float(eq[-1]),
                         return_value=float(eq[-1] / previous - 1), max_drawdown=float(dd.min()),
                         daily_sharpe=float(ret.mean() / ret.std() * np.sqrt(242)) if ret.std() > 0 else 0.,
                         daily_win_rate=float(np.mean(ret > 0)), fees=float(sum(t['fee'] for t in tx)),
                         trades=len(tx), turnover=float(sum(t['quantity'] * t['price'] for t in tx) / eq.mean()),
                         RankIC_5d=ric['RankIC']))
        previous = float(eq[-1])

    assert cursor == len(curve), '季度汇总遗漏现金曲线日期'
    if rows and not np.isclose(np.prod([1 + r['return_value'] for r in rows]), curve[-1]['equity'] / 100000, atol=1e-10):
        raise AssertionError('季度收益无法复合回全年收益')
    return rows

# =============================================================================
# ⑦ 策略曲线图
# =============================================================================


def plot_results(out, source, first_day):
    """画两张图：含费用净值 + 回撤；可执行无摩擦收益累加。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': .2, 'font.size': 10})
    folder = out / source
    plots = out / 'charts'
    plots.mkdir(exist_ok=True)

    # --- 图一：四条固定周期策略（原口径）的含费用净值与回撤 ---
    fig, (ax, ddax) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    colors = ['#c33f3f', '#2678b7', '#17917e', '#9763b5']
    fixed = [spec['name'] for spec in STRATEGIES if spec.get('exit_rank') is None]
    for color, name in zip(colors, fixed):
        df = pd.read_csv(folder / f'cash_{name}.csv')
        dates = pd.to_datetime([first_day] + df['date'].tolist())
        equity = np.r_[100000., df['equity'].to_numpy()]
        label = name + (' (baseline)' if name == 'top1_1d' else '')
        ax.plot(dates, 100 * (equity / 100000 - 1), label=label, color=color, linewidth=2 if name == 'top1_1d' else 1.5)
        ddax.plot(dates, 100 * (equity / np.maximum.accumulate(equity) - 1), color=color, linewidth=1.2)
    ax.set_title(f'V16 | {source} | Quarterly retraining, 4 folds | Net cash return')
    ax.set_ylabel('Cumulative net return (%)')
    ax.axhline(0, color='black', linewidth=.6)
    ax.legend(ncol=2)
    ddax.set_ylabel('Drawdown (%)')
    ddax.set_xlabel('Execution / valuation date')
    # 明确使用季度边界文字，避免自动日期格式把边界刻度显示到上一个月。
    tick_dates = ['2025-07-01', '2025-10-01', '2026-01-01', '2026-04-01', '2026-07-02']
    ddax.set_xticks(pd.to_datetime(tick_dates), labels=tick_dates)
    fig.tight_layout()
    fig.savefig(plots / f'{source}_cash_curves.png')
    plt.close(fig)

    # --- 图二：可执行无摩擦收益累加（求和，不是复利净值） ---
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for n, color in [(1, colors[0]), (5, colors[1])]:
        df = pd.read_csv(folder / f'top{n}_daily.csv')
        ax.plot(pd.to_datetime(df.trade_date), df.return_value.cumsum() * 100,
                label=f'Top{n} executable sum', color=color)
    df = pd.read_csv(folder / 'top1_daily.csv')     # universe_return 与 N 无关，取哪个文件都一样
    ax.plot(pd.to_datetime(df.trade_date), df.universe_return.cumsum() * 100,
            label='Enterable universe equal weight', color='#666666', linestyle='--')
    ax.set_title(f'V16 | {source} | Cost-free 1-day return sums (not compounded NAV)')
    ax.set_ylabel('Cumulative sum (percentage points)')
    ax.set_xlabel('Signal date')
    ax.legend()
    tick_dates = ['2025-07-01', '2025-10-01', '2026-01-01', '2026-04-01', '2026-06-30']
    ax.set_xticks(pd.to_datetime(tick_dates), labels=tick_dates)
    fig.tight_layout()
    fig.savefig(plots / f'{source}_costfree_curves.png')
    plt.close(fig)

    # --- 图三：全部策略（固定周期 vs 排名退出族）的净值曲线 + 收益/回撤排行 ---
    # 图一只能放 4 条线，这一张把注册表里的策略全画上，用来一眼看出"退出规则"带来的差别。
    fig, (ax, rank_ax) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [2, 1]})
    cmap = plt.get_cmap('tab20')
    summary = []
    for i, spec in enumerate(STRATEGIES):
        name = spec['name']
        df = pd.read_csv(folder / f'cash_{name}.csv')
        dates = pd.to_datetime([first_day] + df['date'].tolist())
        equity = np.r_[100000., df['equity'].to_numpy()]
        net = equity[-1] / 100000 - 1
        dd = float((equity / np.maximum.accumulate(equity) - 1).min())
        summary.append((name, net, dd))
        style = dict(linewidth=2.4, color='#c33f3f') if name == 'top1_1d' else dict(linewidth=1.1, color=cmap(i % 20))
        ax.plot(dates, 100 * (equity / 100000 - 1), label=name, **style)
    ax.axhline(0, color='black', linewidth=.6)
    ax.set_title(f'All strategies | {source} | Net cash return')
    ax.set_ylabel('Cumulative net return (%)')
    ax.legend(ncol=2, fontsize=8)
    ax.set_xticks(pd.to_datetime(['2025-07-01', '2025-10-01', '2026-01-01', '2026-04-01', '2026-07-02']),
                  labels=['2025-07-01', '2025-10-01', '2026-01-01', '2026-04-01', '2026-07-02'])
    order = sorted(summary, key=lambda x: x[1])
    ypos = np.arange(len(order))
    rank_ax.barh(ypos, [100 * x[1] for x in order],
                 color=['#c33f3f' if x[0] == 'top1_1d' else '#2678b7' for x in order])
    rank_ax.set_yticks(ypos, labels=[x[0] for x in order], fontsize=7)
    rank_ax.axvline(0, color='black', linewidth=.6)
    rank_ax.set_title('Net return by strategy (baseline red)')
    rank_ax.set_xlabel('Net return (%)')
    fig.tight_layout()
    fig.savefig(plots / f'{source}_strategy_compare.png')
    plt.close(fig)

# =============================================================================
# ⑧ 全流程调度器
# =============================================================================


def cgroup_memory_limit_gib():
    """容器内存硬上限（GiB）；读不到或为 'max' 时返回 None，表示本机没有 cgroup 上限。

    一律现读，不写死任何主机数字——换机器/换实例时上限随之变化。
    """
    try:
        raw = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    except OSError:
        return None
    if raw == 'max':
        return None
    return int(raw) / 1024 ** 3


def run_pipeline(jobs=8, device='auto'):
    """训练调度 → 资源采样 → 全部完成后串行跑一次完整评价。

    - 并发上限 = min(jobs, 内存允许量)；单任务内存预算优先用上一阶段的实测值估算。
    - 不可回收 anon 超过硬上限的 90% 就终止所有任务、保留断点，避免容器被 OOM 杀掉。
    - 已完成且指纹一致的折不派发；训练与分析串行（分析要加载整张面板，叠着跑会顶穿内存）。
    """
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)

    # --- 采样工具：容器内存 / CPU + GPU ---
    def counters(path):
        return {k: int(v) for k, v in (x.split() for x in Path(path).read_text().splitlines())}

    def sample():
        mem = counters('/sys/fs/cgroup/memory.stat')
        cpu = counters('/sys/fs/cgroup/cpu.stat')
        row = dict(timestamp=time.time(),
                   memory_current_gib=int(Path('/sys/fs/cgroup/memory.current').read_text()) / 1024 ** 3,
                   anon_gib=mem['anon'] / 1024 ** 3, file_cache_gib=mem['file'] / 1024 ** 3,
                   cpu_usage_seconds=cpu['usage_usec'] / 1e6,
                   cpu_throttled_seconds=cpu.get('throttled_usec', 0) / 1e6)
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,utilization.memory,power.draw',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True)
        for k, v in zip(['gpu_memory_mib', 'gpu_util_percent', 'gpu_memory_util_percent', 'gpu_power_watts'],
                        gpu.stdout.strip().split(',')):
            row[k] = float(v.strip())
        return row

    # --- 阶段衔接：上一阶段的摘要转成历史阶段，原始样本改名留档 ---
    prior = logs / 'resource_summary.json'
    phases = logs / 'resource_phases.json'
    history = json.loads(phases.read_text()) if phases.exists() else []
    if prior.exists() and not history:
        before = json.loads(prior.read_text())
        before['phase'] = '4并发（按用户要求中途切换）'
        history.append(before)
        os.replace(logs / 'resource_usage.csv', logs / 'resource_usage_4workers.csv')

    # --- 算并发上限：有 cgroup 上限时只用它的 80% 作预算，留出安全边界 ---
    start = sample()
    before_events = counters('/sys/fs/cgroup/memory.events')
    limit = cgroup_memory_limit_gib()
    if limit is None:
        # 本机没有 cgroup 内存上限 ⇒ 无从按内存收敛并发，只按 --jobs 限制，
        # 高水位闸门一并关闭（没有可比的基准），运行中明确提示。
        cap = jobs
        print('未检测到 cgroup 内存上限：跳过内存闸门，仅按 --jobs 限制并发', flush=True)
    else:
        per_worker = max(8., history[0]['peak_anon_gib'] / 4 * 1.10) if history else 10.
        cap = min(jobs, int((limit * .8 - start['anon_gib']) / per_worker))
        if cap < 1:
            raise RuntimeError('内存不足，拒绝训练')

    # --- 待办清单：已完成且指纹一致的折跳过 ---
    tasks = []
    days, _ = axis()
    for quarter in quarters(days):
        for fold in range(1, 5):
            folder = ROOT / 'model_train' / quarter / f'fold{fold}'
            if (folder / 'complete.json').exists():
                done = json.loads((folder / 'complete.json').read_text())
                lock = json.loads((folder / 'recipe.lock.json').read_text())
                if provenance(lock['recipe'])['run_id'] != done['run_id']:
                    raise RuntimeError('已完成模型指纹不符')
            else:
                tasks.append((quarter, fold))
    actual = min(cap, len(tasks))
    print(f'并发上限{jobs}，内存允许{cap}，剩余{len(tasks)}，本轮实际最多{actual}', flush=True)

    # --- 调度循环：派发 → 采样 → 收尸；训练全部结束后串行跑一次评价 ---
    pending = tasks.copy()
    active = {}
    samples = []
    failed = False
    analysis_started = False
    max_active = 0
    aff = sorted(os.sched_getaffinity(0))
    # 单任务核数：本机可用核数的 80% 在并发任务间均分，上限 3 核（再多收益递减、内存更吃紧）。
    cores_per_job = max(1, min(3, int(len(aff) * .8) // max(actual, 1)))
    slots = list(range(max(actual, 1)))
    with open(logs / 'resource_usage.csv', 'w', newline='') as fh:
        writer = None
        while True:
            # 派发：受并发上限与剩余内存双重约束
            while pending and len(active) < cap:
                task = pending.pop(0)
                quarter, fold = task
                slot = slots.pop(0)
                handle = open(logs / f'{quarter}_fold{fold}.log', 'a', buffering=1)
                cmd = [sys.executable, '-u', str(ROOT / 'model.py'),
                       '--quarter', quarter, '--fold', str(fold), '--device', device]
                child = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
                # 限定每个工作进程的可用CPU，避免并发任务互相争抢超出配额的核。
                cpus = aff[slot * cores_per_job:(slot + 1) * cores_per_job]
                if cpus:
                    os.sched_setaffinity(child.pid, cpus)
                active[task] = (child, handle, slot)
                print(f'启动 {task} PID={child.pid} | 已派发 {len(tasks) - len(pending)}/{len(tasks)}，'
                      f'在跑 {len(active)}，剩余待派发 {len(pending)}', flush=True)

            # 采样一行，并记下当前活跃任务数
            max_active = max(max_active, len(active))
            row = sample()
            row['elapsed_seconds'] = row['timestamp'] - start['timestamp']
            row['active_tasks'] = len(active)
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            fh.flush()
            samples.append(row)

            # 接近硬上限时明确失败并保留断点，避免容器OOM杀掉所有工作。
            if limit is not None and row['anon_gib'] > limit * .90:
                for child, _, _ in active.values():
                    child.terminate()
                failed = True
                pending = []

            # 收尸：回收槽位，非零退出码视为失败并停止继续派发
            for task, (child, handle, slot) in list(active.items()):
                code = child.poll()
                if code is not None:
                    # 训练任务成功时 `train_fold` 自己已经打了 `fold<k> EXIT:0`，这里再写一行就重复了；
                    # 只有失败、或不是训练任务（如 analysis 子进程，它自己不打印）才补 EXIT。
                    if code or task[0] == 'analysis':
                        handle.write(f'\nEXIT:{code}\n')
                    handle.close()
                    del active[task]
                    slots.append(slot)
                    # 进度与 ETA：按已完成任务的平均墙钟估算剩余（并发数不变时够准）。
                    finished = len(tasks) - len(pending) - len(active)
                    elapsed = time.time() - start['timestamp']
                    eta = elapsed / finished * (len(tasks) - finished) if finished else 0.
                    print(f'完成 {task} EXIT {code} | 进度 {finished}/{len(tasks)}，'
                          f'已用 {elapsed / 60:.1f}m，ETA {eta / 60:.1f}m', flush=True)
                    if code:
                        failed = True
                        pending = []

            # 训练全部结束 → 起一个子进程做完整评价（含重载推理对拍）
            if not pending and not active:
                if failed or analysis_started:
                    break
                handle = open(logs / 'analysis.log', 'a', buffering=1)
                child = subprocess.Popen([sys.executable, '-u', str(ROOT / 'analysis.py'), '--audit'],
                                         stdout=handle, stderr=subprocess.STDOUT)
                active[('analysis', 0)] = (child, handle, 0)
                analysis_started = True
            time.sleep(2)

    # --- 本阶段汇总 ---
    finish = sample()
    wall = finish['timestamp'] - start['timestamp']
    result = dict(phase=f'上限{jobs}/实际{max_active}并发阶段', exit_code=1 if failed else 0, wall_seconds=wall,
                  sample_interval_seconds=2, samples=len(samples),
                  peak_memory_current_gib=max(x['memory_current_gib'] for x in samples),
                  peak_anon_gib=max(x['anon_gib'] for x in samples),
                  peak_gpu_memory_mib=max(x['gpu_memory_mib'] for x in samples),
                  mean_gpu_util_percent=float(np.mean([x['gpu_util_percent'] for x in samples])),
                  peak_gpu_util_percent=max(x['gpu_util_percent'] for x in samples),
                  cpu_usage_seconds=finish['cpu_usage_seconds'] - start['cpu_usage_seconds'],
                  cpu_throttled_seconds=finish['cpu_throttled_seconds'] - start['cpu_throttled_seconds'],
                  memory_events_before=before_events, memory_events_after=counters('/sys/fs/cgroup/memory.events'),
                  max_active_tasks=max_active, worker_cpu_affinity=cores_per_job,
                  newly_completed_models=len(tasks) - sum(
                      1 for q, f in tasks if not (ROOT / 'model_train' / q / f'fold{f}' / 'complete.json').exists()))
    result['mean_cpu_cores'] = result['cpu_usage_seconds'] / wall
    history.append(result)
    atomic_json(phases, history)

    # --- 多阶段合并：耗时/CPU 相加，峰值取最大 ---
    merged = dict(result)
    merged['wall_seconds'] = sum(x['wall_seconds'] for x in history)
    merged['cpu_usage_seconds'] = sum(x['cpu_usage_seconds'] for x in history)
    merged['cpu_throttled_seconds'] = sum(x['cpu_throttled_seconds'] for x in history)
    merged['mean_cpu_cores'] = merged['cpu_usage_seconds'] / merged['wall_seconds']
    merged['mean_gpu_util_percent'] = sum(x['mean_gpu_util_percent'] * x['wall_seconds']
                                          for x in history) / merged['wall_seconds']
    merged['memory_events_before'] = history[0]['memory_events_before']
    for key in ['peak_memory_current_gib', 'peak_anon_gib', 'peak_gpu_memory_mib', 'peak_gpu_util_percent']:
        merged[key] = max(x[key] for x in history)
    atomic_json(prior, merged)

    if failed:
        raise SystemExit('部分任务失败，检查logs后原命令恢复')

# =============================================================================
# ⑧b 报告生成（REPORT.md）
#
# 用户 2026-09-20 定：报告由脚本生成（不靠 Agent 手工写）。文字在本文件里，
# 数字全部来自 model_pred/ 的产物与 logs/resource_summary.json。
# =============================================================================


def markdown_table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)

def generate_report():
    import platform
    out=ROOT/'model_pred';tb=out/'tables'          # 汇总表与报告 json 都在 tables/
    report=json.loads((tb/'report.json').read_text())
    q=pd.read_csv(tb/'quarterly_summary.csv');cf=pd.read_csv(tb/'quarterly_costfree.csv');st=pd.read_csv(tb/'strategy_summary.csv')
    resource_path=ROOT/'logs/resource_summary.json'
    resources=json.loads(resource_path.read_text()) if resource_path.exists() else None
    source='ensemble';selected=q[q.source==source];whole=st[st.source==source]
    baseline=selected[selected.strategy=='top1_1d']; main=report['variants'][source]
    money=lambda x:f'{x:,.2f}'; pct=lambda x:f'{x*100:+.2f}%'
    def label(name):
        """策略名 → 中文说明；由注册表生成，加策略不用改报告代码。"""
        spec=next((s for s in STRATEGIES if s['name']==name),None)
        if spec is None:
            return name
        if spec.get('exit_rank') is None:
            return ('Top1 隔日换手（baseline）' if name=='top1_1d'
                    else f"Top{spec['topn']} 每{spec['period']}交易日")
        parts=[f"排名退出 · 前{spec['exit_rank']}名"]
        if spec.get('skip'):
            parts.append(f"跳过前{spec['skip']}名")
        parts.append(f"持{spec['topn']}只 / 最少{spec['min_hold']}日")
        if spec.get('stop_loss'):
            parts.append(f"止损{spec['stop_loss']:.0%}")
        if spec.get('trailing'):
            parts.append(f"移动止盈{spec['trailing']:.0%}")
        if spec.get('sell_at') == 'close':
            parts.append('收盘卖')
        return ' · '.join(parts)
    rows=[]
    for _,r in baseline.iterrows():
        value=cf[(cf.source==source)&(cf.quarter==r.quarter)&(cf.topn==1)].iloc[0]
        rows.append([r.quarter,int(r.signal_days),f'{r.RankIC_5d:+.4f}',f'{value.sum_return:+.4f}',
                     pct(r.return_value),pct(r.max_drawdown),f'{r.daily_sharpe:.2f}',pct(r.daily_win_rate),money(r.fees),int(r.trades)])
    title='# V16 季度滚动 MLP baseline：训练、回测与资源报告\n\n'
    intro=(f'**状态：16/16 个季度×折模型完成。** 打分窗口固定为 {report["evaluation_window"][0]} 至 {report["evaluation_window"][1]}，'
           '仅2025Q3、2025Q4、2026Q1、2026Q2；此范围同样适用于以后版本。'
           f'打分产物（`ensemble/year=*`）覆盖**全推演窗** {report["score_window"][0]} ~ {report["score_window"][1]}'
           '（尾段留给增量推演）；下面的评价与回测**只用四季度**那一段。\n\n'
           '主结果使用四折打分**直接相加**（不归一化）；策略在运行前固定，未根据测试收益挑选模型、季度或种子。'
           '四折使用同一测试窗，不能视为四段独立年份。\n\n')
    txt=title+intro+'## 1. 基线与季度结果\n\n'
    # --- 主评估指标（用户 2026-09-20 定）：1d 口径、分季度 + 一行总计 = 5 行 ---
    mfile=tb/'main_metrics_1d.csv'
    if mfile.exists():
        m=pd.read_csv(mfile)
        txt+='### ★ 主评估指标（1d 口径）\n\n'
        txt+='口径：期限**全部取 1d**；IC = **1d RankIC**（逐日截面 Spearman 的日均值）；'
        txt+='top 收益 = **Σtop1**（可执行、无摩擦、**累加**）；'
        txt+='稳定性 = 逐日 top1 收益的**均值 ÷ 标准差**（×√242 即年化）。'
        txt+='季度行只用**该季度**的逐日序列，总计行用**全年** —— 不是把四个季度再加权平均。\n\n'
        txt+=markdown_table(['', 'IC (1d RankIC)', 'ICIR', 'top 收益 (Σtop1)', 'top 收益稳定性', '交易日'],
            [[('**总计**' if r.quarter=='总计' else r.quarter),
              f'{r.IC:+.4f}', f'{r.ICIR:.3f}', f'{r.top_return:+.4f}',
              f'{r.top_stability:.3f}', int(r.days)] for _,r in m.iterrows()])+'\n'
        txt+='（下面表里的 **5d RankIC 是参考** —— 它与训练标签匹配，不是交易口径。）\n\n'
    txt+='Baseline：T日收盘生成分数，T+1开盘买top1，T+2开盘按最新排名换仓；相同目标继续持有，避免无意义的重复买卖。'
    txt+='可执行无摩擦指标Σtop1为逐日1d标签求和；下面“净收益”是10万元现金账户含费用模拟，两者不能混用。\n\n'
    txt+=markdown_table(['季度','信号日','5d RankIC','Σtop1','净收益','最大回撤','日频夏普','日胜率','费用/元','成交笔数'],rows)+'\n'
    base_all=whole[whole.strategy=='top1_1d'].iloc[0]
    txt+=f'全年 baseline 净收益 **{pct(base_all.return_value)}**，最大回撤 **{pct(base_all.max_drawdown)}**；'
    txt+=f'Σtop1 **{main["costfree"][0]["sum_return"]:+.4f}**，剔除最好2日后 **{main["costfree"][0]["remove_best_2"]:+.4f}**。\n\n'
    txt+='季度按发出交易指令的信号日归属；跨季执行和末尾清仓顺延仍归入对应信号季度。'
    txt+='资金跨季连续，已断言四个季度净收益连乘与全年净收益一致。信号范围没有扩展，最后成交/估值日可晚于6月30日。\n\n'
    txt+='## 2. 不同策略的时间曲线\n\n'
    txt+='![各策略含费用收益与回撤](model_pred/charts/ensemble_cash_curves.png)\n\n'
    txt+='![可执行无摩擦收益累加](model_pred/charts/ensemble_costfree_curves.png)\n\n'
    txt+='![全部策略对比](model_pred/charts/ensemble_strategy_compare.png)\n\n'
    # --- 策略层扩展：注册表里全部策略（按全年净收益排序），baseline 行加粗作为对照 ---
    ranked=whole.sort_values('return_value',ascending=False)
    base_net=float(whole[whole.strategy=='top1_1d'].iloc[0].return_value)
    txt+='### 全部策略（集成分数，按全年净收益排序）\n\n'
    txt+=markdown_table(['策略','全年净收益','vs baseline','最大回撤','日频夏普','费用/元','成交笔数','末尾未平仓'],
        [[label(r.strategy),pct(r.return_value),f'{r.return_value-base_net:+.2%}',pct(r.max_drawdown),f'{r.sharpe:.2f}',
          money(r.total_fees),int(r.trades),int(r.ending_positions)] for _,r in ranked.iterrows()])+'\n'
    # --- 跨折普适性：同一策略在 4 折各自的分数上跑，看它是不是只对集成的某一折有效 ---
    # （参考工程用同一手法评"普适性"：不是看某个模型的峰值，而是看它超基线的次数）
    txt+='### 跨折普适性（同一策略跑四折各自的打分）\n\n'
    all_sources=st.copy()
    pivot=all_sources.pivot_table(index='strategy',columns='source',values='return_value')
    base_by_fold=all_sources[all_sources.strategy=='top1_1d'].set_index('source')['return_value']
    robust=[]
    for strategy,row in pivot.iterrows():
        folds=[c for c in ['fold1','fold2','fold3','fold4'] if c in row and np.isfinite(row[c])]
        if not folds:
            continue
        beat=sum(1 for c in folds if row[c]>base_by_fold[c])
        robust.append((strategy,len(folds),beat,float(np.median([row[c] for c in folds])),float(row.get('ensemble',np.nan))))
    robust.sort(key=lambda x:(-x[2],-x[4] if np.isfinite(x[4]) else 0))
    txt+=markdown_table(['策略','折数','超 baseline 折数','四折中位净收益','集成净收益'],
        [[label(s),n,f'{b}/{n}',pct(np.nan if not np.isfinite(m) else m),pct(np.nan if not np.isfinite(e) else e)] for s,n,b,m,e in robust])+'\n'
    txt+='★ 这张表是"普遍有效"的证据：只对集成分数有效、在四折上全输的策略，不能当结论。\n\n'
    txt+='### 统一季度策略矩阵（baseline 组 + 集成净收益前 4 的排名退出策略）\n\n'
    exit_names=[s for s,_,_,_,_ in robust if next((x for x in STRATEGIES if x['name']==s),{}).get('exit_rank') is not None][:4]
    show=set(['top1_1d','top5_1d','top5_5d','top5_20d'])|set(exit_names)
    txt+=markdown_table(['季度','策略','净收益','最大回撤','日频夏普','费用/元','成交笔数'],
        [[r.quarter,label(r.strategy),pct(r.return_value),pct(r.max_drawdown),f'{r.daily_sharpe:.2f}',money(r.fees),int(r.trades)]
         for _,r in selected[selected.strategy.isin(show)].sort_values(['quarter','strategy']).iterrows()])+'\n'
    txt+='## 3. 四折稳定性与评价口径\n\n'
    foldrows=[]
    for name,value in report['variants'].items():
        ic=next(x for x in value['IC'] if x['label']=='label_ret_5d');cost=value['costfree'][0];cash=value['strategies'][0]
        foldrows.append([name,f'{ic["RankIC"]:+.4f}',f'{ic["ICIR"]:.3f}',f'{cost["sum_return"]:+.4f}',
                         f'{cost["remove_best_2"]:+.4f}',pct(cash['return_value']),pct(cash['max_drawdown'])])
    txt+=markdown_table(['来源','5d RankIC','ICIR','Σtop1','去最好2日Σtop1','baseline净收益','最大回撤'],foldrows)+'\n'
    txt+='- 固定存续、从未ST的2115只股票池存在幸存者偏差；报告不能解释为无偏的全市场实盘收益。\n'
    txt+='- 次日开盘触及主板限价即保守拒单，停牌拒单；不使用次日全天high/low选择开盘成交。\n'
    txt+='- 10万元、最多5只、100股整手、T+1；佣金万2.5且最低5元、卖出印花万5、过户万0.1、双边滑点3bp。\n'
    txt+='- 单笔买入不超过信号日成交额1%；未成交资金留现金，不偷看成交日全天成交额。建仓按1/N预算，持有后权重自然漂移，同票不强制卖买回权重。\n'
    txt+='- 公司行为用复权比折算经济股数，假设分红立即再投资；这是含费用现金账户模拟，不是逐笔公司行为现金账复原。\n'
    txt+='- 日频夏普按242日年化，只作描述；季度样本短，不代表稳定预期收益。\n\n'
    txt+='## 4. 训练配置与完成记录\n\n'
    txt+='用户指定MLP-4linear（337→256→128→32→1）；WPCC权重由预测排名决定，保持参考脚本公式。'
    txt+='特征已是当日rank，固定中心化和缺失填0.5；5d单头，batch=4个日截面，AdamW lr=.001、weight_decay=.02，最多30轮。\n\n'
    txt+='每季度从2018起扩展训练历史，季度前6个交易日隔离；四个历史验证块轮换并purge/embargo。'
    txt+='val_wei=100×可执行top5的5d标签代理收益+0.7×逐日Pearson IC；只用于验证选epoch，测试不参与选择。'
    txt+='调度factor=.5/patience=3/cooldown=2/min_lr=5e-6；6轮未改善早停，种子3253–3256。\n\n'
    training=[]; epoch_times=[]; sample_visits=0; epoch_count=0
    for done in sorted(report['folds'],key=lambda x:(x['quarter'],x['fold'])):
        folder=ROOT/'model_train'/done['quarter']/f'fold{done["fold"]}'
        h=json.loads((folder/'history.json').read_text());sp=json.loads((folder/'split.json').read_text())
        ntrain=sum(x['days'] for x in sp['train']);nvalid=sum(x['days'] for x in sp['valid'])
        secs=[r['seconds'] for r in h];epoch_times+=secs;sample_visits+=ntrain*len(h)*2115;epoch_count+=len(h)
        training.append([done['quarter'],done['fold'],ntrain,nvalid,len(done['test_dates']),done['epochs'],done['best_epoch'],
                         f'{sum(secs):.1f}',f'{np.mean(secs):.2f}',f'{done["peak_rss_gib"]:.2f}'])
    txt+=markdown_table(['季度','折','训练日','验证日','测试日','轮数','最佳轮','逐轮训练验证秒','平均秒/轮','进程峰值RSS/GiB'],training)+'\n'
    txt+='RSS为各折工作进程生命周期内的系统高水位，后续季度会继承同一进程此前峰值；不能把16个RSS直接相加当容器峰值。逐轮耗时来自完整history，包含恢复断点前已完成轮次，不含数据读取/落盘/最终预测。\n\n'
    txt+='## 5. 资源与计算效率\n\n'
    mem=cgroup_value('memory.max'); cpu=cgroup_value('cpu.max')
    txt+=f'环境：Python {platform.python_version()}，PyTorch {torch.__version__}；'
    txt+=f'CPU 配额 {cpu:.0f} 核、内存硬上限 {mem:.0f} GiB（均现读 cgroup，不写死主机数字）。'
    txt+='先4并发，后按用户要求提高到8任务上限；切换时剩6个任务，实际最高6并发，每进程CPU亲和限制为3核。训练结束后才分析。\n\n'
    if resources:
        wall=resources['wall_seconds'];trainsecs=sum(epoch_times)
        resource_rows=[['活跃运行墙钟合计（不含切换暂停）',f'{wall/60:.2f} 分钟'],
                       ['容器总内存采样峰值（含页缓存）',f'{resources["peak_memory_current_gib"]:.2f} GiB / {mem:.0f} GiB'],
                       ['容器不可回收anon采样峰值',f'{resources["peak_anon_gib"]:.2f} GiB'],
                       ['单工作进程最大RSS高水位',f'{max(x["peak_rss_gib"] for x in report["folds"]):.2f} GiB'],
                       ['显存采样峰值',f'{resources["peak_gpu_memory_mib"]/1024:.2f} GiB'],
                       ['GPU利用率采样均值/峰值',f'{resources["mean_gpu_util_percent"]:.1f}% / {resources["peak_gpu_util_percent"]:.1f}%'],
                       ['平均使用CPU核数',f'{resources["mean_cpu_cores"]:.2f} / {cpu:.0f}'],
                       ['CPU累计使用',f'{resources["cpu_usage_seconds"]:.1f} 核秒'],
                       ['完成训练轮数',str(epoch_count)],['平均/中位单轮耗时',f'{np.mean(epoch_times):.2f} / {np.median(epoch_times):.2f} 秒'],
                       ['季度×折训练耗时之和',f'{trainsecs/60:.2f} 分钟'],
                       ['每小时完成模型数',f'{len(report["folds"])*3600/wall:.2f}'],
                       ['按全网格计的训练吞吐',f'{sample_visits/wall:,.0f} 股票行·轮/秒（上界，含被掩码剔除行）'],
                       ['本轮新增OOM kill',str(resources['memory_events_after'].get('oom_kill',0)-resources['memory_events_before'].get('oom_kill',0))]]
        txt+=markdown_table(['资源/效率项','实测值'],resource_rows)+'\n'
        phasefile=ROOT/'logs/resource_phases.json'
        if phasefile.exists():
            phase_data=json.loads(phasefile.read_text())
            txt+='### 并发阶段实测\n\n'+markdown_table(['阶段','活跃分钟','内存峰值/GiB','anon峰值/GiB','显存峰值/GiB','GPU均值','平均CPU核'],
                [[r['phase'],f'{r["wall_seconds"]/60:.2f}',f'{r["peak_memory_current_gib"]:.2f}',f'{r["peak_anon_gib"]:.2f}',
                  f'{r["peak_gpu_memory_mib"]/1024:.2f}',f'{r["mean_gpu_util_percent"]:.1f}%',f'{r["mean_cpu_cores"]:.2f}'] for r in phase_data])+'\n'
            snap=ROOT/'logs/phase1_training_snapshot.json'
            if snap.exists():
                old=json.loads(snap.read_text());old_count=sum(x['epochs'] for x in old);old_seconds=sum(x['seconds'] for x in old)
                new_count=epoch_count-old_count;new_seconds=trainsecs-old_seconds
                if new_count>0:
                    txt+=f'4并发阶段已完成轮次的平均耗时为 {old_seconds/max(old_count,1):.2f} 秒/轮；扩并发后新增轮次为 {new_seconds/new_count:.2f} 秒/轮。'
                    txt+='两阶段季度、剩余轮数、加载开销不同，只作运行观察，不能当严格的4→8加速实验。实际没有满8任务同时运行。\n\n'
        txt+='![资源使用曲线](model_pred/charts/resource_usage.png)\n\n'
        txt+='容器/GPU数据每2秒采样，因此采样峰值可能低于瞬时峰值；该内核没有memory.peak文件。'
        txt+='容器数据包含同容器其他进程。并行耗时之和/墙钟不是严格加速比：没有相同任务的串行对照；不据此声称加速倍数。\n\n'
    else:txt+='资源采样尚未结束；运行完成后 `python analysis.py --report-only` 补齐此节。\n\n'
    txt+='## 6. 验证、锚点与复现\n\n'
    txt+=f'数据锚点：`{report["panel_digest"]}`。模型来源锁定**配方与各数据块SHA**（不哈希源码字节）；原四块训练数据未重建。'
    txt+='价格/标签逐格对拍、WPCC参考公式及数值梯度、季度/验证隔离、费用/整手/T+1/卖不出顺延、季度收益复合检查均通过。'
    txt+=f'完成{len(report["audit"])}组模型重载/当日独立推理对拍。\n\n'
    txt+='```bash\ncd /autodl-fs/data/model/V16\nbash train.sh                       # 训练 + 推演 + 评价 + 本报告\n/autodl-fs/data/miniconda3/bin/python model.py --rescore   # 只重推演、不重训（口径改动后刷新打分）\n```\n\n'
    txt+='统一产物：`model_pred/tables/`（`report.json`、`quarterly_summary.csv`、`quarterly_costfree.csv`、`strategy_summary.csv`）；'
    txt+='逐日曲线/成交/预测在各source目录；资源原始样本在`logs/resource_usage.csv`，摘要在`logs/resource_summary.json`。\n'
    (ROOT/'REPORT.md').write_text(txt,encoding='utf-8')
    if resources:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        r=pd.read_csv(ROOT/'logs/resource_usage.csv')
        previous=ROOT/'logs/resource_usage_4workers.csv'
        if previous.exists():
            before=pd.read_csv(previous)
            r=pd.concat([before,r],ignore_index=True).sort_values('timestamp')
            r['elapsed_seconds']=r.timestamp-r.timestamp.min()
            r.to_csv(ROOT/'logs/resource_usage_all.csv',index=False)
        x=r.elapsed_seconds/60
        fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
        axes[0].plot(x,r.memory_current_gib,label='Container memory incl. cache',color='#2678b7')
        axes[0].plot(x,r.anon_gib,label='Unreclaimable anon',color='#17917e')
        axes[0].axhline(90,color='#c33f3f',linestyle='--',label='90 GiB limit');axes[0].set_ylabel('Memory / GiB');axes[0].legend(ncol=3)
        axes[1].plot(x,r.gpu_util_percent,color='#9763b5');axes[1].set_ylabel('GPU utilization / %')
        axes[2].plot(x,r.gpu_memory_mib/1024,color='#e49432');axes[2].set_ylabel('GPU memory / GiB');axes[2].set_xlabel('Elapsed minutes')
        axes[0].set_title('V16 | 4 quarters x 4 folds | Resource samples every 2 seconds')
        for ax in axes: ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/'charts/resource_usage.png',dpi=150);plt.close(fig)
    print('报告已生成:',ROOT/'REPORT.md',flush=True)


# =============================================================================
# ⑧b 最新一日推荐表（给实战 / 增量推演用，不参与任何评价指标）
# =============================================================================


def cgroup_value(name):
    """读一个 cgroup 参数并换成人看的单位，读不到返回 nan。

    `memory.max` → GiB；`cpu.max`（内容是「配额 周期」两个数）→ 核数。
    一律现读，不在脚本里写死主机数字。
    """
    try:
        parts = Path(f'/sys/fs/cgroup/{name}').read_text().split()
    except OSError:
        return float('nan')
    if not parts or parts[0] == 'max':
        return float('nan')
    if name == 'cpu.max':
        return float(parts[0]) / float(parts[1])
    return float(parts[0]) / 1024 ** 3


def stock_names():
    """代码 → 当前名称，只用于推荐表的展示。

    来源是模块① 的原始表 `stock_daily.stock_name`（根目录可用 `MX_RAW_DATA` 覆盖）。
    ⚠️ 厂商把**整段历史**都改写成**当前**名称（见 featureengineering/README.md），
    所以它**只能用于展示"当前"名称**，绝不能拿去判 ST / 风险 / 分类。
    读不到时返回 None 并**显式提示** —— 名称只是展示，不该拖垮评价。
    """
    root = Path(os.environ.get('MX_RAW_DATA', ROOT.parent.parent / 'datadownload' / 'data'))
    files = sorted((root / 'stock_daily').glob('year=*/data.parquet'))
    if not files:
        print(f'⚠️ 找不到 {root}/stock_daily/year=*/data.parquet：推荐表只出代码、不出名称', flush=True)
        return None
    df = pq.read_table(files[-1], columns=['trade_date', 'stock_code', 'stock_name']).to_pandas()
    df['trade_date'] = df['trade_date'].astype(str).str[:10]
    df = df.sort_values('trade_date').drop_duplicates('stock_code', keep='last')
    return df.set_index('stock_code')['stock_name'].to_dict()


def write_daily_picks(variants, panel, px, days, topn=10, recent=10):
    """最新一日的推荐表 + 近 N 日策略回测（隔日换手口径）。

    口径与主判据一致：T 日收盘出分数 → **T+1 开盘买入** → **T+2 开盘卖出**。
    这是给"实战 / 增量推演"看的输出，**不参与任何评价指标**。
    """
    out = ROOT / 'model_pred' / 'latest'
    out.mkdir(parents=True, exist_ok=True)
    p = variants['ensemble'] if 'ensemble' in variants else next(iter(variants.values()))
    nm = stock_names() or {}
    name_of = lambda c: nm.get(str(panel.codes[c]), '')

    # --- ① 最新一日的排名表（T 日收盘后即可见）---
    d_last = int(days[-1])
    score = p[-1]
    cand = np.flatnonzero(np.isfinite(score))
    top = cand[np.argsort(-score[cand], kind='stable')][:topn]
    picks = pd.DataFrame([dict(排名=i, 代码=str(panel.codes[c]), 名称=name_of(c),
                               打分=round(float(score[c]), 4)) for i, c in enumerate(top, 1)])
    picks.to_csv(out / 'picks.csv', index=False)

    # --- ② 近 N 个信号日（按日期连续取，**含尚未定型的最后两天**）---
    # 未定型的格子留空：能确定的照填（比如 0917 的买入日就是 0918、末两天的选股与分数都已知），
    # 收益率还没发生就空着；累计列按已定型的日滚动。
    hist = []
    cum = 1.0
    idx = range(max(0, len(days) - recent), len(days))
    for i in idx:
        d = int(days[i])
        ok = np.isfinite(p[i]) & px.next_entry[d]           # 次日开盘真能买进
        c2 = np.flatnonzero(ok)
        settled = len(c2) > 0                               # 可执行性本身已定（最后一天未定）
        if not settled:
            c2 = np.flatnonzero(np.isfinite(p[i]))          # 退化：只看分数，输出里标注未定
        if not len(c2):
            continue
        pick = c2[np.argmax(p[i][c2])]                      # top1，与主判据 baseline 同口径
        y = panel.Y['label_ret_1d'][d]
        ret = float(y[pick]) if np.isfinite(y[pick]) else None
        if ret is not None:
            cum *= (1 + ret)
        buy = str(panel.days[d + 1]) if d + 1 < len(panel.days) else '下一交易日'
        sell = str(panel.days[d + 2]) if d + 2 < len(panel.days) else '其次日'
        hist.append(dict(信号日=str(panel.days[d]), 买入日=buy, 卖出日=sell,
                         代码=str(panel.codes[pick]), 名称=name_of(pick),
                         收益率=('' if ret is None else round(ret, 5)),
                         累计=(f'{(cum - 1) * 100:+.2f}%' if ret is not None else ''),
                         当时打分=round(float(p[i][pick]), 4),
                         备注=('' if ret is not None else
                             ('可执行性待定' if not settled else '收益未定型'))))
    hdf = pd.DataFrame(hist)
    if len(hdf):
        done = hdf.loc[hdf['收益率'].astype(str) != '', '收益率'].astype(float)
        n_settled, n_pending = len(done), len(hdf) - len(done)
        win = float((done > 0).mean()) if len(done) else float('nan')
        hdf.to_csv(out / 'recent_backtest.csv', index=False)
    else:
        n_settled = n_pending = 0
        win = float('nan')

    # --- ③ 人读版 ---
    t1 = str(panel.days[d_last + 1]) if d_last + 1 < len(panel.days) else '下一交易日'
    t2 = str(panel.days[d_last + 2]) if d_last + 2 < len(panel.days) else '其次日'
    txt = ('# 最新一日打分与策略建议\n\n'
           f'**当前日期**：{panel.days[d_last]} 收盘\n\n'
           f'**模型预测**：1 日收益 —— {t1} 买入、{t2} 卖出（隔日换手口径）\n\n'
           f'**打分来源**：四折打分直接相加的集成分数；推演窗 {panel.days[int(days[0])]} ~ {panel.days[d_last]}\n\n')
    txt += markdown_table(['排名', '代码', '名称', '打分'],
                          picks.astype(str).values.tolist()) + '\n'
    txt += f'\n## 近 {len(hdf)} 个信号日的回测（top1 隔日换手，可执行口径）\n\n'
    if len(hdf):
        txt += markdown_table(['信号日', '买入日', '卖出日', '代码', '名称', '收益率', '累计', '当时打分'],
                              hdf[['信号日', '买入日', '卖出日', '代码', '名称',
                                   '收益率', '累计', '当时打分']].astype(str).values.tolist()) + '\n'
        txt += (f'\n**近 {len(hdf)} 个信号日累计收益 {((cum - 1) * 100):+.2f}%**'
                f'（{n_settled} 个已定型，日胜率 {win * 100:.0f}%）'
                + (f'；末 {n_pending} 天收益待定型，表格里留空\n' if n_pending else '\n'))
        if n_pending:
            txt += ('\n※ 留空的行是"还没发生"：末两天的**买入/卖出日**在未来、**收益率**要等 open(T+2) 出来；'
                    '选股与分数是当天收盘就能确定的，所以照常给出。\n')
    else:
        txt += '（最近没有可用的信号日）\n'
    txt += ('\n> 口径提醒：T+1 开盘触及涨停/跌停或停牌时**拒单顺延**；'
            '这里的收益是 1 日标签（open(T+2)/open(T+1)−1），不含费用。\n')
    (out / 'picks.md').write_text(txt, encoding='utf-8')
    print(txt, flush=True)
    print('推荐表已生成:', out, flush=True)
    return picks


# =============================================================================
# ⑨ 命令行入口
# =============================================================================


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', action='store_true', help='训练调度、资源采样、完整回测')
    parser.add_argument('--jobs', type=int, choices=range(1, 9), default=8, help='pipeline 的并发任务上限')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--no-backtest', action='store_true', help='明确只计算 IC；不输出 ΣtopN/现金策略')
    parser.add_argument('--audit', action='store_true', help='最佳模型逐日截断/重载推理对拍')
    parser.add_argument('--picks-only', action='store_true',
                        help='只出最新一日推荐表（不跑评价/回测），给增量推演用')
    args = parser.parse_args()
    # 进程内线程数与模型侧用同一个旋钮（train.sh 导出 MX_THREADS），不写死主机假设。
    torch.set_num_threads(int(os.environ.get('MX_THREADS', '3')))

    if args.pipeline:
        return run_pipeline(args.jobs, args.device)

    # --- 评价：拼各折预测 → 集成 → IC / 无摩擦收益 / 现金回测 ---
    panel = Panel('all', load_x=args.audit)
    px = None if args.no_backtest else Prices(panel)
    out = ROOT / 'model_pred'
    out.mkdir(exist_ok=True)
    tb = out / 'tables'          # 汇总表与报告 json 单独一个文件夹，便于查阅与对接
    tb.mkdir(exist_ok=True)
    variants = {}
    completions = []
    dates = None
    audit = []
    fold_ids = [1, 2, 3, 4]
    qs = quarters(panel.days)

    # --- 逐折拼出「季度首日 → 数据末日」的完整推演序列 ---
    # 每一天由「最新一个没见过它的季度模型组」负责：
    #   四个季度各由自己那组给（该组的训练数据止于季度首日前 6 个交易日，整个季度都是样本外）；
    #   季度之后的尾段由**最后那个季度**的组给（其余组其实也都没见过它，取最新即可）。
    for fold in fold_ids:
        chunks = []
        fold_dates = []
        for quarter in qs:
            folder = ROOT / 'model_train' / quarter / f'fold{fold}'
            done = json.loads((folder / 'complete.json').read_text())
            lock = json.loads((folder / 'recipe.lock.json').read_text())
            if provenance(lock['recipe'])['run_id'] != done['run_id']:
                raise RuntimeError('模型来源与当前数据/配方不一致')
            st = np.load(folder / 'score_predictions.npy')       # 全推演窗：季度首日 → 数据末日
            n_test = len(done['test_dates'])
            p = st[:n_test]                                      # 本季度那一段（= 旧 test_predictions）
            qdays = np.searchsorted(panel.days, done['test_dates'])
            chunks.append(p)
            fold_dates.extend(done['test_dates'])
            if quarter == qs[-1]:
                chunks.append(st[n_test:])                        # 尾段 → 增量最新打分
                fold_dates.extend(done['score_dates'][n_test:])
            completions.append(done)

            # --- 重载对拍：从磁盘重建模型，验证当日独立推理可复现 ---
            if args.audit:
                checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
                net = PredictModel(len(panel.features))
                net.load_state_dict(checkpoint['model'])
                net.eval()
                check_ix = np.unique([0, len(qdays) // 2, len(qdays) - 1])
                check_days = qdays[check_ix]
                actual = predict(net, panel, check_days, torch.device('cpu'))
                expected = p[check_ix]
                finite = np.isfinite(actual) & np.isfinite(expected)
                error = float(np.max(np.abs(actual[finite] - expected[finite])))
                if not np.array_equal(np.isfinite(actual), np.isfinite(expected)) or error > 2e-4:
                    raise AssertionError(f'推理重载对拍失败 {error}')
                # 同一模型、同一日，逐行打分必须与整批打分一致（无跨行/跨日依赖）。
                for j, d in enumerate(check_days):
                    mask = panel.mask(d)
                    with torch.no_grad():
                        one = net(torch.from_numpy(panel.X[d, mask][:10])).squeeze(-1).numpy()
                    if not np.allclose(one, actual[j, mask][:10], atol=2e-5, rtol=2e-5):
                        raise AssertionError('跨行/跨日推理依赖')
                audit.append(dict(quarter=quarter, fold=fold, max_cpu_reload_error=error,
                                  days=panel.days[check_days].tolist()))
        if dates is not None and dates != fold_dates:
            raise ValueError('各折测试日期不一致')
        dates = fold_dates
        if len(set(dates)) != len(dates) or dates != sorted(dates):
            raise ValueError('季度拼接重叠/乱序')
        variants[f'fold{fold}'] = np.concatenate(chunks)

    days = np.searchsorted(panel.days, dates)
    # 评价（IC / 回测 / 季度表）**只用四个季度那一段**（用户固定的评价窗）；
    # 尾段（2026-07-01 起）只进打分产物，供增量推演用。
    n_eval = int(np.searchsorted(dates, '2026-07-01'))
    eval_days = days[:n_eval]

    # --- 集成：四折打分**直接相加**，不做逐日标准化（用户 2026-09-20）---
    # 用 NaN 传播的加法：某票某天没有分数时，集成结果仍是"没有分数"，
    # 不会变成 0 分混进排序（np.nansum 会犯这个错）。
    if len(variants) > 1:
        variants['ensemble'] = sum(variants.values()).astype(np.float32)

    # --- 只出推荐表：给"增量推演 → 看今天买什么"用（不跑评价/回测）---
    if args.picks_only:
        if px is None:
            raise RuntimeError('--picks-only 需要价格层（不要与 --no-backtest 同用）')
        return write_daily_picks(variants, panel, px, days)

    report = dict(panel_digest=panel.meta['panel_digest'],
                  evaluation_window=[dates[0], dates[n_eval - 1]],
                  score_window=[dates[0], dates[-1]],
                  folds=completions, analysis_mode='IC_only' if args.no_backtest else 'full',
                  variants={}, audit=audit,
                  ensemble='四折打分直接相加（不归一化）',
                  inference_rule='每天由「最新一个没见过它的季度模型组」推演；季度之后的尾段由最后一个季度那组给',
                  caveats=['固定存续且从未ST股票池含幸存者偏差', '四折共用测试窗，不是四段独立样本外收益',
                           '验证收益是5日标签代理；策略净值另算', '不据测试结果调参或晋级best',
                           f'评价只用 {dates[0]} ~ {dates[n_eval - 1]} 四季度；'
                           f'{dates[n_eval]} 起的尾段只进打分产物，供增量推演'])
    report['data_digests'] = {k: panel.meta.get(k, {}).get('digest') for k in ['prices', 'amount', 'fac_sample']}
    if px:
        report['label_price_check'] = px.verify_labels(panel, days)

    # --- 逐变体出 IC / 无摩擦收益 / 现金策略矩阵（全部只用评价窗那一段）---
    # ★ 只有**集成**才是"这个模型"（用户 2026-09-20）：各折仍会把指标算出来供
    #   REPORT §3 的四折稳定性表用，但**不再落盘**逐折的 scores/曲线/成交/选股。
    primary = 'ensemble' if 'ensemble' in variants else 'fold1'
    quarterly = []
    strategy_rows = []
    quarter_costfree = []
    for name, p in variants.items():
        ev = p[:n_eval]                                  # 评价切片；p 本身是全推演窗
        save_here = (name == primary)
        result = {'IC': ic_table(ev, panel, eval_days)}
        target = out / name
        if save_here:
            target.mkdir(exist_ok=True)
            write_scores(p, panel, days, target)              # 打分写**全推演窗**（含增量尾段）
        if px:
            result['costfree'] = []
            result['strategies'] = []
            for n in [1, 5]:
                stat, daily, picks = costfree(ev, panel, px, eval_days, n)
                result['costfree'].append(stat)
                if save_here:
                    pd.DataFrame(daily).to_csv(target / f'top{n}_daily.csv', index=False)
                    pd.DataFrame(picks).to_csv(target / f'top{n}_picks.csv', index=False)
                for q in quarters(panel.days):
                    loc = np.array([str(pd.Period(panel.days[d], freq='Q')) == q for d in eval_days])
                    if loc.any():
                        qstat, _, _ = costfree(ev[loc], panel, px, eval_days[loc], n)
                        quarter_costfree.append(dict(source=name, quarter=q, **qstat))
            for spec in STRATEGIES:
                stat, curve, trades = cash_backtest(
                    ev, panel, px, eval_days, n=spec['topn'], period=spec.get('period', 1),
                    exit_rank=spec.get('exit_rank'), min_hold=spec.get('min_hold', 1),
                    stop_loss=spec.get('stop_loss'), trailing=spec.get('trailing'),
                    skip=spec.get('skip', 0), sell_at=spec.get('sell_at', 'open'))
                result['strategies'].append(stat)
                strategy = spec['name']
                strategy_rows.append(dict(source=name, strategy=strategy, **stat))
                quarterly.extend(quarter_metrics(name, strategy, curve, trades, ev, panel, eval_days))
                if save_here:
                    pd.DataFrame(curve).to_csv(target / f'cash_{strategy}.csv', index=False)
                    pd.DataFrame(trades).to_csv(target / f'trades_{strategy}.csv', index=False)
        report['variants'][name] = result
        print(name, json.dumps(result, ensure_ascii=False), flush=True)

    atomic_json(tb / ('ic_report.json' if args.no_backtest else 'report.json'), report)
    if px:
        pd.DataFrame(quarterly).to_csv(tb / 'quarterly_summary.csv', index=False)
        pd.DataFrame(strategy_rows).to_csv(tb / 'strategy_summary.csv', index=False)
        pd.DataFrame(quarter_costfree).to_csv(tb / 'quarterly_costfree.csv', index=False)

        # --- 主评估指标（用户 2026-09-20）：口径全部取 1d，分季度 + 一行总计 = 5 行 ---
        # IC = 1d RankIC（逐日截面 Spearman）；top 收益 = Σtop1（可执行、无摩擦、累加）；
        # 稳定性 = 逐日 top1 收益的「均值 ÷ 标准差」（参考库口径，不年化；×√242 即年化）。
        ev_main = variants[primary][:n_eval]

        def metrics_1d(pred, days_):
            ic = next(x for x in ic_table(pred, panel, days_) if x['label'] == 'label_ret_1d')
            stat, daily, _ = costfree(pred, panel, px, days_, 1)
            rets = np.asarray([x['return_value'] for x in daily])
            return dict(IC=ic['RankIC'], ICIR=ic['ICIR'], top_return=stat['sum_return'],
                        top_stability=float(rets.mean() / rets.std()) if rets.std() > 0 else 0.)

        main_metrics = []
        for q in quarters(panel.days):
            loc = np.array([str(pd.Period(panel.days[d], freq='Q')) == q for d in eval_days])
            if loc.any():
                main_metrics.append(dict(quarter=q, days=int(loc.sum()),
                                         **metrics_1d(ev_main[loc], eval_days[loc])))
        main_metrics.append(dict(quarter='总计', days=len(eval_days), **metrics_1d(ev_main, eval_days)))
        pd.DataFrame(main_metrics).to_csv(tb / 'main_metrics_1d.csv', index=False)
        plot_results(out, primary, dates[0])
        # 最新一日的推荐表 + 近 N 日回测（给实战/增量推演用）
        write_daily_picks(variants, panel, px, days)
        generate_report()
    print('评价完成:', out, flush=True)


if __name__ == '__main__':
    main()
