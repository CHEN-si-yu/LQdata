# 标准回测与可选近期冻结信号诊断。
"""V50: flow_temporal; seed 3253; fresh independent research unit, fixed project contract."""
import json
import os
import sys
sys.dont_write_bytecode = True
import time
import inspect
import unicodedata
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata
from model import PROJECT_ROOT, ROOT, RECIPE, Panel, Prices, atomic_json, axis, corr, score_of, blend_linear, hot_filter, predict_heads, predict_absolute, quarters, PredictModel, predict, torch, run_root, analysis_input_root, info_root, run_name, output_file, validate_layout, stock_list_path, calendar_path, splits
RUN_ROOT = run_root()
INPUT_ROOT = analysis_input_root()

def ic_table(pred, panel, days):
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
        std = float(np.std(ic, ddof=1)) if len(ic) > 1 else 0.0
        results.append(dict(label=label, days=len(ic), RankIC=float(np.mean(ic)), ICIR=float(np.mean(ic) / std) if std else 0.0, Pearson_IC=float(np.mean(pearson))))
    return results
TOP_PORTFOLIO = dict(topn=500, money=1500000000.0, label='label_ret_1d')

def rank_ic_horizons():
    return (1, 5)

def min_test_zscore(pred):
    out = np.asarray(pred, dtype=np.float64).copy()
    for i, row in enumerate(out):
        known = np.isfinite(row)
        if known.sum() < 2:
            continue
        std = row[known].std(ddof=1)
        if std > 0:
            out[i, known] = (row[known] - row[known].mean()) / std
    return out

def min_test_combinations(pred, panel, days):
    result = []
    for topn, money in [(500, 1500000000.0), (50, 100000000.0), (5, 10000000.0), (5, 100000.0)]:
        spec = dict(topn=topn, money=money, label='label_ret_1d')
        daily, ic, rank = top_portfolio_daily(pred, panel, days, spec)
        result.append(dict(params=spec, metrics=top_metrics(daily, ic, rank), quarterly=[dict(quarter=q, top_return=float(daily[take].mean()), IC=float(ic[1][take].mean())) for q in ['2025Q3', '2025Q4', '2026Q1', '2026Q2'] for take in [np.asarray([str(pd.Period(str(panel.days[d]), freq='Q')) == q for d in days])]]))
    return result

def top_portfolio_daily(pred, panel, days, spec=TOP_PORTFOLIO):
    pred = min_test_zscore(pred)
    topn, money = (spec['topn'], spec['money'])
    label = spec['label']
    horizons = rank_ic_horizons()
    daily_ret = np.zeros(len(days))
    daily_ic = {h: np.zeros(len(days)) for h in horizons}
    daily_rank_ic = {h: np.zeros(len(days)) for h in horizons}
    for i, d in enumerate(days):
        score = pred[i]
        known = np.isfinite(score)
        for h in horizons:
            yh = panel.Y[f'label_ret_{h}d'][d]
            ok = known & np.isfinite(yh)
            daily_rank_ic[h][i] = corr(rankdata(score[ok]), rankdata(yh[ok]))
            daily_ic[h][i] = corr(score[known], np.where(np.isfinite(yh[known]), yh[known], 0.0) * 100)
        y = panel.Y[label][d]
        order = pd.Series(score).sort_values(ascending=False).index.to_numpy()
        ret = np.where(np.isfinite(y), y, 0.0) * 100
        total_hold = 0
        total_earned = 0
        for num, c in enumerate(order):
            if num >= topn or money - total_hold < 1:
                break
            liquid = np.float64(panel.amount[d, c] if np.isfinite(panel.amount[d, c]) else 0.0)
            hold_money = min(money - total_hold, liquid)
            total_hold += hold_money
            total_earned += ret[c] * hold_money
        daily_ret[i] = total_earned / money
    return (daily_ret, daily_ic, daily_rank_ic)

def top_metrics(daily_ret, daily_ic, daily_rank_ic):
    h1, h5 = rank_ic_horizons()
    ic_1d = daily_ic[h1]
    ic_std = float(np.std(ic_1d, ddof=1)) if len(ic_1d) > 1 else 0.0
    ret_std = float(np.std(daily_ret, ddof=1)) if len(daily_ret) > 1 else 0.0
    return dict(IC=float(np.mean(ic_1d)), ICIR=float(np.mean(ic_1d) / ic_std) if ic_std else 0.0, top_return=float(np.mean(daily_ret)), top_return_stability=float(np.mean(daily_ret) / ret_std) if ret_std else 0.0, total_return=float((np.prod(1.0 + daily_ret / 100) - 1) * 100), **{f'IC_{h}d': float(np.mean(daily_ic[h])) for h in (h1, h5)}, **{f'RankIC_{h}d': float(np.mean(daily_rank_ic[h])) for h in (h1, h5)})
REALIZABLE_K = 5

def realizable_topk(pred, panel, prices, days, k=REALIZABLE_K, label=None):
    label = label or RECIPE['score_label']
    entry = prices.next_entry_ok()
    ideal, real = ([], [])
    for i, d in enumerate(days):
        d = int(d)
        p, y = (pred[i], panel.Y[label][d])
        known = np.isfinite(p)
        ranked = np.flatnonzero(known)[np.argsort(-p[known], kind='stable')][:k]
        if len(ranked) < k:
            ideal.append(np.nan)
            real.append(np.nan)
            continue
        ideal.append(float(np.nanmean(y[ranked])))
        ok = entry[d][ranked] & np.isfinite(y[ranked])
        real.append(float(np.nanmean(y[ranked][ok])) if ok.sum() else np.nan)
    return (np.asarray(ideal), np.asarray(real))

def realizable_summary(pred, panel, prices, eval_days, rows):
    ideal, real = realizable_topk(pred, panel, prices, eval_days)
    out = dict(topk=REALIZABLE_K, label=RECIPE['score_label'], ideal_mean_pct=float(np.nanmean(ideal)) * 100, realizable_mean_pct=float(np.nanmean(real)) * 100, ratio=float(np.nanmean(real) / np.nanmean(ideal)), days=int(len(ideal)), days_with_position=int(np.isfinite(real).sum()))
    diff = real - ideal
    se = float(np.nanstd(diff, ddof=1) / np.sqrt(np.isfinite(diff).sum()))
    out['gap_pct'] = float(np.nanmean(diff)) * 100
    out['gap_t'] = float(np.nanmean(diff) / se) if se else 0.0
    text = f"===== 可实现口径（前 {REALIZABLE_K} 名，{RECIPE['score_label']}，{out['days']} 天，**评价窗**）=====\n  ★ 只能当参照：242 天对 top5 没有统计功效（要 80% 功效检出 +0.27% 需约 37 年）。\n    有功效的版本在 **1893 天样本外**上，见 docs/iterations.md：那里可实现的 top5 只有理想的约 65%。\n  理想（不看可买性） {out['ideal_mean_pct']:+.4f} %/天\n  可实现（只留买得进） {out['realizable_mean_pct']:+.4f} %/天（{out['ratio']:.1%}，有持仓 {out['days_with_position']} 天）\n  差（买不到的票的贡献） {out['gap_pct']:+.4f} %/天  t={out['gap_t']:+.2f}"
    print('\n' + text, flush=True)
    return (out, text)

def display_width(text):
    return sum((2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in str(text)))

def summary_table_text(rows, keys, headers):
    body = [[r['quarter'], *[f'{r[k]:.4f}' for k in keys]] for r in rows]
    table = [['季度', *headers], *body]
    widths = [max((display_width(cell) for cell in col)) for col in zip(*table)]

    def line(cells):
        return '  '.join((c + ' ' * (w - display_width(c)) for c, w in zip(cells, widths))).rstrip()
    return '\n'.join([line(table[0]), '-' * (sum(widths) + 2 * (len(widths) - 1)), *[line(row) for row in table[1:]]])

def metric_blocks_text(rows, valid_rows=None):
    h1, h5 = rank_ic_horizons()
    headers = [f'IC_{h1}d', f'IC_{h5}d', f'RankIC_{h1}d', f'RankIC_{h5}d']
    keys = [f'IC_{h}d' for h in (h1, h5)] + [f'RankIC_{h}d' for h in (h1, h5)]
    text = f"---单一模型评估---\n\n===== test 评价窗（{RECIPE['test_start']}～{RECIPE['test_end']}，{rows[-1]['days']} 天）=====\n{summary_table_text(rows, ['total_return', *keys], ['收益(%)', *headers])}\n"
    if valid_rows:
        text += f"\n\n===== valid 训练验证集（各折 best epoch；季度=4 折均值，总计=16 折均值）=====\n{summary_table_text(valid_rows, ['total_return', *keys], ['收益(%)', *headers])}\n"
    return text

def print_metric_blocks(rows, valid_rows=None):
    print('\n' + metric_blocks_text(rows, valid_rows), flush=True)

def evaluation_slice(pred, panel, days):
    selected = (panel.days[days] >= RECIPE['test_start']) & (panel.days[days] <= RECIPE['test_end'])
    eval_days = days[selected]
    expected = np.flatnonzero((panel.days >= RECIPE['test_start']) & (panel.days <= RECIPE['test_end']))
    if not np.array_equal(eval_days, expected):
        raise ValueError('打分未完整覆盖固定四季度评价窗')
    return (pred[selected], eval_days)

def evaluation_metrics(ev, panel, eval_days):
    daily_ret, daily_ic, daily_rank_ic = top_portfolio_daily(ev, panel, eval_days)
    rows = []
    for q in [*quarters(panel.days), '总计']:
        loc = np.ones(len(eval_days), dtype=bool) if q == '总计' else np.array([str(pd.Period(panel.days[d], freq='Q')) == q for d in eval_days])
        rows.append(dict(quarter=q, days=int(loc.sum()), **top_metrics(daily_ret[loc], {h: s[loc] for h, s in daily_ic.items()}, {h: s[loc] for h, s in daily_rank_ic.items()})))
    return dict(rows=rows, daily_ret=daily_ret, daily_ic=daily_ic, daily_rank_ic=daily_rank_ic)

def fold_completions():
    rows = []
    for q in quarters(axis()[0]):
        for f in range(1, 5):
            path = RUN_ROOT / 'model_train' / q / f'fold{f}' / 'complete.json'
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(json.loads(path.read_text()))
    return rows

def fold_valid_days():
    days = {}
    for q in quarters(axis()[0]):
        for f in range(1, 5):
            path = RUN_ROOT / 'model_train' / q / f'fold{f}' / 'split.json'
            days[q, f] = sum((span['days'] for span in json.loads(path.read_text())['valid']))
    return days

def bag_vs_argmax(records):
    rows = [r for r in records if r.get('argmax_validation')]
    if not rows:
        return None
    fields = ['val_wei', 'IC_1d', 'IC_5d', 'RankIC_1d', 'RankIC_5d', 'val_return_proxy']
    out = dict(folds=len(rows), topk=rows[0].get('bag', {}).get('topk'), bag_epochs=[r.get('bag', {}).get('epochs') for r in rows], metrics={})
    for name in fields:
        bagged = np.array([float(r['best_validation'][name]) for r in rows])
        argmax = np.array([float(r['argmax_validation'][name]) for r in rows])
        diff = bagged - argmax
        se = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
        out['metrics'][name] = dict(bagged=float(bagged.mean()), argmax=float(argmax.mean()), delta=float(diff.mean()), t=float(diff.mean() / se) if se else 0.0, wins=int((diff > 0).sum()))
    return out

def bag_comparison_text(summary):
    if not summary:
        return ''
    lines = [f"===== V12 epoch-bagging 对照（前 {summary['topk']} 名权重平均 vs argmax；{summary['folds']} 折，配对）=====", f"{'指标':<20}{'bagged':>12}{'argmax':>12}{'差':>12}{'t':>8}{'bag 胜':>8}"]
    for name, m in summary['metrics'].items():
        lines.append(f"{name:<20}{m['bagged']:>12.5f}{m['argmax']:>12.5f}{m['delta']:>+12.5f}{m['t']:>+8.2f}{m['wins']:>5}/{summary['folds']}")
    return '\n'.join(lines)

def training_validation_rows(records):
    expected = {(q, f) for q in quarters(axis()[0]) for f in range(1, 5)}
    seen = [(r['quarter'], r['fold']) for r in records]
    if len(set(seen)) != len(seen) or set(seen) != expected:
        raise ValueError(f'训练验证集汇总需要四季度×四折共 {len(expected)} 条不重复记录，实际 {len(seen)} 条；缺 {sorted(expected - set(seen))}')
    h1, h5 = rank_ic_horizons()
    valid_days = fold_valid_days()
    fields = [f'IC_{h}d' for h in (h1, h5)] + [f'RankIC_{h}d' for h in (h1, h5)] + ['val_wei', 'return_proxy_pct', 'total_return']
    collected = []
    for r in records:
        v = r['best_validation']
        proxy = float(v['val_return_proxy'])
        segments = valid_days[r['quarter'], r['fold']] / horizon_of(RECIPE['score_label'])
        collected.append(dict(quarter=r['quarter'], **{k: float(v[k]) for k in fields if k not in ('return_proxy_pct', 'total_return')}, return_proxy_pct=100 * proxy, total_return=100 * ((1.0 + proxy) ** segments - 1)))
    frame = pd.DataFrame(collected)
    rows = []
    for q in [*quarters(axis()[0]), '总计']:
        part = frame if q == '总计' else frame[frame.quarter == q]
        mean = part[fields].mean()
        rows.append(dict(quarter=q, **{k: float(mean[k]) for k in fields}))
    return rows

def fees(notional, sell=False):
    return max(notional * 0.00025, 5) + notional * 1e-05 + (notional * 0.0005 if sell else 0)
STRATEGIES = [dict(name='top5_5d', topn=5, period=5, band=RECIPE['hold_band']), dict(name='top1_1d', topn=1, period=1, band=0)]

def cash_backtest(pred, panel, px, days, n=5, period=5, band=0, money=None, forecast_members=4, cost_gate_multiplier=None, gate_scores=None):
    money = float(RECIPE['backtest_money'] if money is None else money)
    if forecast_members < 1:
        raise ValueError('invalid forecast member count')
    gate = RECIPE['cost_gate_multiplier'] if cost_gate_multiplier is None else cost_gate_multiplier
    cost_reference = 2 * max(0.00025, 5 / (money / n)) + 2e-05 + 0.0005 + 0.0006
    score_threshold = forecast_members * gate * cost_reference
    if n > 1 and gate > 0:
        if gate_scores is None:
            if not hasattr(panel, 'absolute_forecast'):
                raise ValueError('composite gate forecast not supplied')
            gate_scores = panel.absolute_forecast[days]
        if gate_scores.shape != pred.shape:
            raise ValueError('gate forecast axes differ')
    gate_by_day = {} if gate_scores is None else {int(d): p for d, p in zip(days, gate_scores)}
    cash = money
    holdings = {}
    buys = {}
    desired = []
    curves = []
    trades = []
    corp_events = 0
    cumulative_fees = 0.0
    cumulative_slippage = 0.0
    first, last = (int(days[0]) + 1, int(days[-1]) + 2)
    pred_by_day = {int(d): p for d, p in zip(days, pred)}
    spec_band = band
    entry_ok = px.entry_ok() if px is not None else None
    for d in range(first, last + 1):
        for c in list(holdings):
            ratio = px.adj[d, c] / px.adj[d - 1, c]
            if not np.isfinite(ratio) or ratio <= 0:
                raise ValueError('持仓复权因子缺失')
            if abs(ratio - 1) > 1e-08:
                holdings[c] *= ratio
                corp_events += 1
        signal = d - 1
        p = pred_by_day.get(signal)
        if p is not None:
            candidates = np.flatnonzero(np.isfinite(p))
            utility = p.copy()
            if n > 1 and gate > 0:
                admission = gate_by_day[signal].copy() - score_threshold
                for held in holdings:
                    admission[held] = gate_by_day[signal][held]
                candidates = candidates[np.isfinite(admission[candidates]) & (admission[candidates] >= 0.0)]
            ranked = candidates[np.argsort(-utility[candidates], kind='stable')]
            if (signal - int(days[0])) % period == 0:
                band = spec_band
                keep = set((int(c) for c in ranked[:band])) & set(holdings) if band > n else set()
                fresh = [int(c) for c in ranked if entry_ok[d, c] and c not in keep]
                desired = list(keep) + fresh[:max(0, n - len(keep))]
        if d == last:
            desired = []
        blocked = []
        for c in list(holdings):
            if c in desired:
                continue
            if not px.exit[d, c] or buys[c] >= d:
                blocked.append(c)
                continue
            quantity = holdings[c]
            price = px.open[d, c] * (1 - 0.0003)
            gross = quantity * price
            fee = fees(gross, True)
            cumulative_fees += fee
            cumulative_slippage += quantity * px.open[d, c] - gross
            cash += gross - fee
            trades.append(dict(date=str(panel.days[d]), code=str(panel.codes[c]), side='sell', quantity=quantity, price=float(price), fee=float(fee)))
            del holdings[c]
            del buys[c]
        if p is not None and d < last and ((signal - int(days[0])) % period == 0):
            equity = cash + sum((q * px.open[d, c] if np.isfinite(px.open[d, c]) else q * px.close[d - 1, c] for c, q in holdings.items()))
            for c in desired:
                if c in holdings or len(holdings) >= n:
                    continue
                price = px.open[d, c] * (1 + 0.0003)
                amount = panel.amount[signal, c]
                cap = amount * 0.01 if np.isfinite(amount) else 0.0
                if RECIPE['budget_rule'] == 'divide':
                    k = max(1, n - len(holdings))
                    purse = (equity - sum((q * px.open[d, cc] if np.isfinite(px.open[d, cc]) else q * px.close[d - 1, cc] for cc, q in holdings.items()))) / k
                    budget = min(cash, purse, cap)
                else:
                    budget = min(cash, equity / n, cap)
                quantity = int(max(0, budget - 5) / (price * (1 + 0.00025 + 1e-05)) / 100) * 100
                while quantity > 0 and quantity * price + fees(quantity * price) > budget:
                    quantity -= 100
                if quantity <= 0:
                    continue
                gross = quantity * price
                fee = fees(gross)
                cumulative_fees += fee
                cumulative_slippage += gross - quantity * px.open[d, c]
                cash -= gross + fee
                holdings[c] = float(quantity)
                buys[c] = d
                trades.append(dict(date=str(panel.days[d]), code=str(panel.codes[c]), side='buy', quantity=quantity, price=float(price), fee=float(fee)))
        if cash < -1e-06 or len(holdings) > n:
            raise AssertionError('资金/持仓上限被突破')
        equity = cash + sum((q * px.close[d, c] for c, q in holdings.items()))
        if not np.isfinite(equity):
            raise ValueError('持仓估值缺失')
        curves.append(dict(date=str(panel.days[d]), equity=float(equity), cash=float(cash), positions=len(holdings), blocked_exits=len(blocked), gross_equity=float(equity + cumulative_fees + cumulative_slippage), cumulative_fees=cumulative_fees, cumulative_slippage=cumulative_slippage, exposure=float((equity - cash) / equity)))
    values = np.r_[money, [r['equity'] for r in curves]]
    returns = values[1:] / values[:-1] - 1
    dd = values / np.maximum.accumulate(values) - 1
    return (dict(topn=n, rebalance_days=period, return_value=float(values[-1] / money - 1), max_drawdown=float(dd.min()), sharpe=float(np.mean(returns) / np.std(returns) * np.sqrt(242)) if np.std(returns) > 0 else 0.0, total_fees=float(sum((t['fee'] for t in trades))), gross_return_same_positions=float((values[-1] + cumulative_fees + cumulative_slippage) / money - 1), fee_drag=float(cumulative_fees / money), slippage_drag=float(cumulative_slippage / money), gross_definition='same actual shares and trade dates; costs added back as cash, no reinvestment', avg_exposure=float(np.mean([c['exposure'] for c in curves])), trades=len(trades), ending_cash=cash, ending_positions=len(holdings), corporate_action_adjustments=corp_events, start=str(panel.days[first]), end=str(panel.days[last]), corporate_action_assumption='复权比折算经济股数，分红立即再投资近似'), curves, trades)

def write_scores(pred, panel, days, out):
    for year in sorted(set((str(panel.days[d])[:4] for d in days))):
        chunks = []
        for p, d in zip(pred, days):
            if not str(panel.days[d]).startswith(year):
                continue
            ok = np.isfinite(p)
            forecast = panel.absolute_forecast[int(d)]
            if not np.isfinite(forecast[ok]).all():
                raise ValueError('missing gate forecast')
            chunks.append(pd.DataFrame(dict(trade_date=str(panel.days[d]), stock_code=panel.codes[ok], value=p[ok].astype(np.float32), rank=(rankdata(p[ok]) / ok.sum()).astype(np.float32), absolute_forecast_1d=forecast[ok].astype(np.float32))))
        df = pd.concat(chunks, ignore_index=True)
        df['trade_date'] = df['trade_date'].astype('string')
        df['stock_code'] = df['stock_code'].astype('string')
        folder = out / f'year={year}'
        folder.mkdir(parents=True, exist_ok=True)
        target = output_file(folder / 'data.parquet')
        temp = folder / 'data.tmp.parquet'
        df.to_parquet(temp, index=False)
        os.replace(temp, target)

def quarter_metrics(source, strategy, curve, trades, pred, panel, days):
    rows = []
    previous = 100000.0
    cursor = 0
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
        ric = next((r for r in ic_table(pred[pos], panel, days[pos]) if r['label'] == RECIPE['label']))
        rows.append(dict(source=source, quarter=q, strategy=strategy, signal_days=len(qdays), execution_start=segment[0]['date'], execution_end=segment[-1]['date'], start_equity=float(previous), end_equity=float(eq[-1]), return_value=float(eq[-1] / previous - 1), max_drawdown=float(dd.min()), daily_sharpe=float(ret.mean() / ret.std() * np.sqrt(242)) if ret.std() > 0 else 0.0, daily_win_rate=float(np.mean(ret > 0)), fees=float(sum((t['fee'] for t in tx))), trades=len(tx), turnover=float(sum((t['quantity'] * t['price'] for t in tx)) / eq.mean()), RankIC_5d=ric['RankIC']))
        previous = float(eq[-1])
    assert cursor == len(curve), '季度汇总遗漏现金曲线日期'
    if rows and (not np.isclose(np.prod([1 + r['return_value'] for r in rows]), curve[-1]['equity'] / 100000, atol=1e-10)):
        raise AssertionError('季度收益无法复合回全年收益')
    return rows

def plot_results(out, source, first_day):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = out / source / 'CSV'
    plots = out
    plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.6), layout='constrained')
    for spec, color in zip(STRATEGIES, ['#17917e', '#c33f3f']):
        frame = pd.read_csv(folder / f"cash_{spec['name']}.csv")
        equity = np.r_[100000.0, frame.equity.to_numpy()]
        dates = pd.to_datetime([first_day] + frame.date.tolist())
        label = f"Top{spec['topn']} | " + ('daily baseline' if spec['topn'] == 1 else f"rebalance every {spec['period']} day(s)")
        ax.plot(dates, 100 * (equity / 100000 - 1), color=color, linewidth=2, label=f'{label} | {100 * (equity[-1] / 100000 - 1):+.2f}%')
    ax.set_title('Baseline model | Fixed strategy vs daily baseline', fontsize=14)
    ax.set_ylabel('Cumulative net return (%)')
    ax.set_xlabel('Execution / valuation date | fees and slippage included')
    ax.axhline(0, color='#555555', linewidth=0.7)
    ax.grid(alpha=0.2)
    ax.legend(loc='upper left')
    ticks = ['2025-07-01', '2025-10-01', '2026-01-01', '2026-04-01', '2026-07-02']
    ax.set_xticks(pd.to_datetime(ticks), labels=ticks)
    fig.savefig(output_file(plots / f'{source}_curves.png'), dpi=160)
    plt.close(fig)

def markdown_table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n' + ''.join(('| ' + ' | '.join(map(str, r)) + ' |\n' for r in rows))

def stock_names(codes):
    path = stock_list_path()
    try:
        frame = pd.read_parquet(path, columns=['stock_code', 'name'])
    except (OSError, ValueError, KeyError) as exc:
        print(f'警告：读不到股票名称表 {path}（{type(exc).__name__}: {exc}），picks 只显示代码', flush=True)
        return {}
    table = dict(zip(frame.stock_code.astype(str), frame.name.astype(str)))
    if len(table) != len(frame):
        raise ValueError(f'{path}: stock_code 有重复')
    return {str(c): table.get(str(c), '') for c in codes}

def horizon_of(label):
    return int(label.split('_ret_')[1].rstrip('d'))

def trade_axis(panel):
    days = [str(d) for d in panel.days]
    try:
        frame = pd.read_parquet(calendar_path(), columns=['date', 'is_open'])
    except (OSError, ValueError, KeyError) as exc:
        print(f'警告：读不到交易日历 {calendar_path()}（{type(exc).__name__}: {exc}），picks 的买卖日改用相对说法', flush=True)
        return days
    future = sorted((str(d) for d in frame[frame.is_open == 1].date if str(d) > days[-1]))
    return days + future

def write_daily_picks(pred, panel, days, metrics=None, valid_rows=None):
    if len(days) == 0:
        raise ValueError('没有可展示的打分日期')
    if pred.shape != (len(days), len(panel.codes)):
        raise ValueError('打分维度与日期、股票轴不一致')
    out = RUN_ROOT / 'model_pred'
    out.mkdir(parents=True, exist_ok=True)
    if metrics is None:
        ev, eval_days = evaluation_slice(pred, panel, days)
        metrics = evaluation_metrics(ev, panel, eval_days)['rows']
    if valid_rows is None:
        valid_rows = training_validation_rows(fold_completions())
    names = stock_names(panel.codes)
    axis = trade_axis(panel)
    position = {d: i for i, d in enumerate(axis)}
    horizon = RECIPE['prediction_horizon']
    holding = RECIPE.get('picks_holding', horizon) + 1
    topn = RECIPE['picks_topn']
    first = max(0, len(days) - 10)
    text = '```text\n' + metric_blocks_text(metrics, valid_rows) + '```\n\n'
    for index in range(len(days) - 1, first - 1, -1):
        signal_day = str(panel.days[int(days[index])])
        score = pred[index]
        candidates = np.flatnonzero(np.isfinite(score))
        cost_ref = 2 * max(0.00025, 5 / (RECIPE['backtest_money'] / 5)) + 2e-05 + 0.0005 + 0.0006
        forecast = panel.absolute_forecast[int(days[index])]
        candidates = candidates[np.isfinite(forecast[candidates]) & (forecast[candidates] >= 4 * RECIPE['cost_gate_multiplier'] * cost_ref)]
        picks = candidates[np.argsort(-score[candidates], kind='stable')][:topn]
        at = position.get(signal_day)
        buy = axis[at + 1] if at is not None and at + 1 < len(axis) else None
        sell = axis[at + holding] if at is not None and at + holding < len(axis) else None
        text += f'## {signal_day} 收盘\n\n'
        signal_description = f"模型输出 {RECIPE['neural_score_horizon']} 日神经排序分数，并以独立技术面1日收益预测检查新买入费用门槛" if RECIPE.get('score_transform') == 'raw_neural' else '模型融合资金流与技术面排名，并以技术面1日收益预测检查新买入费用门槛'
        text += f"{signal_description}；现金策略每 {STRATEGIES[0]['period']} 个交易日检查排名，持仓跌出前 {STRATEGIES[0]['band']} 名时尝试退出。下一可执行开盘：**{buy}**；卖出日期由后续排名与可成交性决定。\n\n" if buy else f'模型输出 {horizon} 日收益排序分数；下一交易日开盘可执行，卖出日期由后续排名与可成交性决定。\n\n'
        if not len(picks):
            text += '当日没有有效打分。\n\n'
            continue
        rows = [[i, str(panel.codes[c]).split('.')[0], names.get(str(panel.codes[c]), ''), f'{float(score[c]):.4f}'] for i, c in enumerate(picks, 1)]
        text += markdown_table(['排名', '代码', '名称', '打分'], rows) + '\n'
    output_file(out / 'picks.md').write_text(text, encoding='utf-8')
    print('最近10个交易日排名已生成:', out / 'picks.md', flush=True)

def save_score_metadata(folder, report, dates):
    metadata = {k: v for k, v in report.items() if k not in ('variants', 'evaluation_mode')}
    metadata['score_dates'] = list(dates)
    metadata['columns'] = ['trade_date', 'stock_code', 'value', 'rank', 'absolute_forecast_1d']
    metadata['absolute_forecast_1d'] = 'raw sum of four fresh technical components, return units; rank score is separate'
    atomic_json(folder / 'score_meta.json', metadata)

def read_saved_scores(folder, panel):
    meta = json.loads((folder / 'score_meta.json').read_text())
    dates = meta['score_dates']
    if dates != sorted(set(dates)):
        raise ValueError('duplicate or unordered score dates')
    date_axis = pd.Index(dates)
    codes = pd.Index(panel.codes)
    days = pd.Index(panel.days).get_indexer(dates)
    if (days < 0).any():
        raise ValueError('score dates absent from snapshot')
    pred = np.full((len(dates), len(codes)), np.nan, np.float32)
    forecast = np.full_like(pred, np.nan)
    files = sorted(folder.glob('year=*/data.parquet'))
    seen = set()
    if not files:
        raise FileNotFoundError('saved composite scores missing')
    for path in files:
        frame = pd.read_parquet(path, columns=['trade_date', 'stock_code', 'value', 'absolute_forecast_1d'])
        if frame.duplicated(['trade_date', 'stock_code']).any():
            raise ValueError('duplicate score keys')
        t = date_axis.get_indexer(frame.trade_date.astype(str))
        c = codes.get_indexer(frame.stock_code.astype(str))
        if (t < 0).any() or (c < 0).any():
            raise ValueError('mismatched score coordinates')
        keys = set(zip(t.tolist(), c.tolist()))
        if seen.intersection(keys):
            raise ValueError('overlapping score partitions')
        seen.update(keys)
        value = frame.value.to_numpy(dtype=np.float32)
        gate = frame.absolute_forecast_1d.to_numpy(dtype=np.float32)
        if not np.isfinite(value).all() or not np.isfinite(gate).all():
            raise ValueError('nonfinite saved score/gate')
        pred[t, c] = value
        forecast[t, c] = gate
    panel.absolute_forecast = np.full((len(panel.days), len(panel.codes)), np.nan, np.float32)
    panel.absolute_forecast[days] = forecast
    return (pred, days, meta)

def lookahead_free_blocks(days):
    out = []
    for q in quarters(days):
        for sp in splits(days, RECIPE['purge_horizon'], RECIPE['folds'], q):
            tr, va = (sp['train'], sp['valid'])
            if len(tr) and len(va) and (int(tr.max()) < int(va[0])):
                out.append(dict(quarter=q, fold=sp['fold'], valid=va))
    return out

def clean_block_summary(panel, px, spec):
    blocks = lookahead_free_blocks(panel.days)
    if not blocks:
        return dict(blocks=[], note='推导不出无前视的块')
    device = torch.device('cpu')
    panel_x = Panel(RECIPE['features'], load_x=True)
    rows = []
    for b in blocks:
        q, fold, va = (b['quarter'], b['fold'], b['valid'])
        ck = torch.load(INPUT_ROOT / 'model_train' / q / f'fold{fold}' / 'best.pt', map_location=device, weights_only=False)
        net = PredictModel(len(panel_x.features), panel_x.market_dim).to(device)
        net.load_state_dict(ck['model'])
        net.ridge = ck.get('ridge')
        n_args = len(inspect.signature(predict).parameters)
        scores = predict(net, panel_x, va, device) if n_args < 5 else predict(net, panel_x, px, va, device)
        ideal, real = realizable_topk(scores, panel, px, va, k=REALIZABLE_K)
        row = dict(quarter=q, fold=fold, start=str(panel.days[int(va[0])]), end=str(panel.days[int(va[-1])]), days=int(len(va)), ideal_mean_pct=float(np.nanmean(ideal)) * 100, realizable_mean_pct=float(np.nanmean(real)) * 100, tradable_ratio=float(np.nanmean(real) / np.nanmean(ideal)))
        if px is not None:
            st, curve, trades = cash_backtest(scores, panel, px, va, n=spec['topn'], period=spec['period'], band=spec['band'], forecast_members=1, gate_scores=predict_absolute(net, panel_x, va, device))
            eq = np.asarray([c['equity'] for c in curve], float)
            prev = np.r_[RECIPE['backtest_money'], eq[:-1]]
            day = (eq - prev) / prev
            row.update(strategy_return_value=st['return_value'], strategy_net_annual_pct=day.mean() * 242 * 100, strategy_max_drawdown=st['max_drawdown'], idle_cash_pct=float(np.mean([c['cash'] / c['equity'] for c in curve])) * 100, trades=st['trades'], gross_return_same_positions=st['gross_return_same_positions'], total_fees=st['total_fees'], fee_drag=st['fee_drag'], slippage_drag=st['slippage_drag'], net_annual_definition='actual net daily mean x 242; fees already included')
        rows.append(row)
    tot_days = sum((r['days'] for r in rows))
    return dict(strategy=spec['name'], blocks=rows, total_days=tot_days, note='这些 (季度, 折) 的训练集全部早于其验证块 ⇒ **无前视**；块与块之间日历不连续，**不可拼接**，只并列看。')

def evaluate_fixed_strategies(pred, panel, px, days, report, mode):
    out = RUN_ROOT / 'model_pred'
    csv_root = out / 'ensemble/CSV'
    csv_root.mkdir(parents=True, exist_ok=True)
    ev, eval_days = evaluation_slice(pred, panel, days)
    report = {k: v for k, v in report.items() if k not in ('score_dates', 'columns', 'variants')}
    report['evaluation_mode'] = mode
    report['evaluation_window'] = [str(panel.days[eval_days[0]]), str(panel.days[eval_days[-1]])]
    result = {'IC': ic_table(ev, panel, eval_days), 'strategies': []}
    summaries = []
    quarterly = []
    if px is not None:
        report['label_price_check'] = px.verify_labels(panel, days)
        for spec in STRATEGIES:
            stat, curve, trades = cash_backtest(ev, panel, px, eval_days, n=spec['topn'], period=spec['period'], band=spec['band'])
            result['strategies'].append(dict(strategy=spec['name'], **stat))
            summaries.append(dict(source='ensemble', strategy=spec['name'], **stat))
            quarterly.extend(quarter_metrics('ensemble', spec['name'], curve, trades, ev, panel, eval_days))
            pd.DataFrame(curve).to_csv(output_file(csv_root / f"cash_{spec['name']}.csv"), index=False)
            pd.DataFrame(trades).to_csv(output_file(csv_root / f"trades_{spec['name']}.csv"), index=False)
    report['variants'] = {'ensemble': result}
    measured = evaluation_metrics(ev, panel, eval_days)
    metrics = measured['rows']
    valid_rows = training_validation_rows(report.get('folds') or fold_completions())
    h1, h5 = rank_ic_horizons()
    result['eval_portfolio'] = dict(params=TOP_PORTFOLIO, total=metrics[-1], daily=dict(date=[str(panel.days[d]) for d in eval_days], return_pct=[round(float(x), 6) for x in measured['daily_ret']], **{f'ic_{h}d': [round(float(x), 6) for x in measured['daily_ic'][h]] for h in (h1, h5)}, **{f'rank_ic_{h}d': [round(float(x), 6) for x in measured['daily_rank_ic'][h]] for h in (h1, h5)}))
    print_metric_blocks(metrics, valid_rows)
    bag_summary = bag_vs_argmax(report.get('folds') or fold_completions())
    if bag_summary:
        print(bag_comparison_text(bag_summary), flush=True)
    write_daily_picks(pred, panel, days, metrics, valid_rows)
    if px is None:
        return report
    plot_results(out, 'ensemble', report['evaluation_window'][0])
    from_saved = mode == 'saved_ensemble_scores'
    audit = [] if from_saved else report.get('audit') or []
    real_block = realizable_summary(ev, panel, px, eval_days, metrics) if px is not None else (None, '')
    clean_block, clean_block_error = (None, None)
    try:
        clean_block = report.get('clean_blocks') if from_saved else clean_block_summary(panel, px, STRATEGIES[0])
    except Exception as exc:
        clean_block_error = f'{type(exc).__name__}: {exc}'
        print(f'★ V27 干净块实测失败（不影响主流程）：{clean_block_error}', flush=True)
    report['clean_blocks'] = clean_block
    atomic_json(info_root() / 'final_audit.json', dict(run_name=run_name(), checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), score_source=mode, evaluation_window=report['evaluation_window'], score_window=report.get('score_window'), model_train_read=bool(audit), weights_reloaded=bool(audit), folds_checked=len(audit), max_cpu_reload_error=max([a['max_cpu_reload_error'] for a in audit], default=None), audit=audit, metrics_total=metrics[-1], min_test_combinations=min_test_combinations(ev, panel, eval_days), snapshot=dict(built_at=panel.meta['built_at'], stock_feature_count=len(panel.meta['columns']['features']), input_feature_count=len(panel.features), meta_sha256=RECIPE['snapshot_meta_sha256']), clean_blocks_are_epoch_selection_data=True, min_test_algorithm='reference pandas tie order; literal sequential scalar allocation; labels multiplied by100 before Pearson IC', replica_status='single screening replica; no promotion', realizable=real_block[0], clean_blocks=clean_block if clean_block is not None else dict(blocks=[], note=clean_block_error), bag_vs_argmax=bag_summary, fusion=report.get('fusion'), auxiliary_forecast='separate raw1d technical return; existing yearly parquet column; same fixed159 paths'))
    atomic_json(info_root() / 'two_strategy_validation.json', dict(run_name=run_name(), checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), score_source=mode, reads_model_train=not from_saved, loads_model_weights=not from_saved, strategies=[s['name'] for s in STRATEGIES], strategy_rows=len(summaries), quarterly_rows=len(quarterly), evaluation_days=int(len(measured['daily_ret'])), evaluation_window=report['evaluation_window'], label_price_check=report.get('label_price_check'), quarterly_returns_compound=True, date_axis_matches_price_axis=True, realized={s['strategy']: dict(return_value=s['return_value'], max_drawdown=s['max_drawdown'], sharpe=s['sharpe'], trades=s['trades'], total_fees=s['total_fees'], gross_return_same_positions=s['gross_return_same_positions'], fee_drag=s['fee_drag'], slippage_drag=s['slippage_drag'], avg_exposure=s['avg_exposure'], gross_definition=s['gross_definition']) for s in summaries}))
    print('两策略回测完成:', json.dumps(summaries, ensure_ascii=False), flush=True)
    return report

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--from-scores', action='store_true', help='直接读取集成打分，只计算本单元两策略')
    parser.add_argument('--no-backtest', action='store_true', help='明确只计算 IC；不输出现金策略')
    parser.add_argument('--audit', action='store_true', help='最佳模型逐日截断/重载推理对拍')
    parser.add_argument('--picks-only', action='store_true', help='只出最近10个交易日排名（不跑评价/回测），给增量推演用')
    args = parser.parse_args()
    validate_layout(allow_temporary=True, allow_missing=True)
    torch.set_num_threads(int(os.environ.get('MX_THREADS', '3')))
    if args.from_scores and args.audit:
        parser.error('--from-scores不加载权重，不能同时使用--audit')
    panel = Panel(RECIPE['features'], load_x=not args.from_scores)
    out = RUN_ROOT / 'model_pred'
    out.mkdir(parents=True, exist_ok=True)
    if args.from_scores:
        pred, days, report = read_saved_scores(out / 'ensemble', panel)
        if args.picks_only:
            return write_daily_picks(pred, panel, days)
        px = None if args.no_backtest else Prices(panel)
        return evaluate_fixed_strategies(pred, panel, px, days, report, 'saved_ensemble_scores')
    px = Prices(panel) if args.audit or not args.no_backtest else None
    variants = {}
    absolute_variants = {}
    completions = []
    dates = None
    audit = []
    fold_ids = [1, 2, 3, 4]
    qs = quarters(panel.days)
    for fold in fold_ids:
        chunks = []
        absolute_chunks = []
        fold_dates = []
        for quarter in qs:
            folder = INPUT_ROOT / 'model_train' / quarter / f'fold{fold}'
            done = json.loads((folder / 'complete.json').read_text())
            st = np.load(folder / 'score_predictions.npy')
            n_test = len(done['test_dates'])
            p = st[:n_test]
            qdays = np.searchsorted(panel.days, done['test_dates'])
            chunks.append(p)
            fold_dates.extend(done['test_dates'])
            if quarter == qs[-1]:
                chunks.append(st[n_test:])
                fold_dates.extend(done['score_dates'][n_test:])
            completions.append(done)
            checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
            net = PredictModel(len(panel.features), panel.market_dim)
            net.load_state_dict(checkpoint['model'])
            net.ridge = checkpoint.get('ridge')
            net.eval()
            absolute_days = np.searchsorted(panel.days, done['score_dates'] if quarter == qs[-1] else done['test_dates'])
            absolute_chunks.append(predict_absolute(net, panel, absolute_days, torch.device('cpu')))
            if args.audit:
                checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
                net = PredictModel(len(panel.features), panel.market_dim)
                net.load_state_dict(checkpoint['model'])
                net.ridge = checkpoint.get('ridge')
                net.eval()
                check_ix = np.unique([0, len(qdays) // 2, len(qdays) - 1])
                check_days = qdays[check_ix]
                actual = predict(net, panel, px, check_days, torch.device('cpu'))
                raw = predict_heads(net, panel, check_days, torch.device('cpu'))
                expect = hot_filter(blend_linear(score_of(raw), net.ridge, panel, check_days), px, check_days)
                if not np.allclose(actual, expect, atol=2e-05, rtol=2e-05, equal_nan=True):
                    raise AssertionError('predict() 与 blend_linear(score_of(predict_heads())) 不一致')
                expected = p[check_ix]
                finite = np.isfinite(actual) & np.isfinite(expected)
                error = float(np.max(np.abs(actual[finite] - expected[finite])))
                if not np.array_equal(np.isfinite(actual), np.isfinite(expected)) or error > 0.0002:
                    raise AssertionError(f'推理重载对拍失败 {error}')
                for j, d in enumerate(check_days):
                    mask = panel.mask(d)
                    with torch.no_grad():
                        m = torch.from_numpy(panel.M[d]) if panel.market_dim else None
                        one = net(torch.from_numpy(panel.inputs(d, mask)[:10]), m).numpy()
                    if not np.allclose(one, raw[j, mask][:10], atol=2e-05, rtol=2e-05):
                        raise AssertionError('跨行/跨日推理依赖')
                audit.append(dict(quarter=quarter, fold=fold, max_cpu_reload_error=error, days=panel.days[check_days].tolist()))
        if dates is not None and dates != fold_dates:
            raise ValueError('各折测试日期不一致')
        dates = fold_dates
        if len(set(dates)) != len(dates) or dates != sorted(dates):
            raise ValueError('季度拼接重叠/乱序')
        variants[f'fold{fold}'] = np.concatenate(chunks)
        absolute_variants[f'fold{fold}'] = np.concatenate(absolute_chunks)
    days = np.searchsorted(panel.days, dates)
    if len(variants) > 1:
        variants['ensemble'] = sum(variants.values()).astype(np.float32)
    panel.absolute_forecast = np.full((len(panel.days), len(panel.codes)), np.nan, np.float32)
    panel.absolute_forecast[days] = sum(absolute_variants.values()).astype(np.float32)
    pred = variants['ensemble']
    if args.picks_only:
        return write_daily_picks(pred, panel, days)
    report = dict(evaluation_window=[RECIPE['test_start'], RECIPE['test_end']], score_window=[dates[0], dates[-1]], folds=completions, analysis_mode='IC_only' if args.no_backtest else 'full', variants={}, audit=audit, ensemble='四折打分直接相加（不归一化）；辅助1日预测也原值相加，门槛乘4', fusion=dict(flow_weight=RECIPE.get('flow_fusion_weight') if not RECIPE.get('fusion_weight_selection') else None, fold_weights={f"{c['quarter']}/fold{c['fold']}": c.get('flow_fusion_weight', RECIPE.get('flow_fusion_weight')) for c in completions}, rank_scale='raw neural output; raw fourfold sum; daily z-score for min_test evaluation' if RECIPE.get('score_transform') == 'raw_neural' else 'within submodel family percentiles; raw fourfold sum; daily z-score for min_test evaluation', absolute_gate='separate training-fitted1d return, never percentile as return'), inference_rule='每天由「最新一个没见过它的季度模型组」推演；季度之后的尾段由最后一个季度那组给', caveats=['固定存续且从未ST股票池含幸存者偏差', '四折共用测试窗，不是四段独立样本外收益', '验证收益是5日标签代理；策略净值另算', '不据测试结果调参或晋级best', '评价只用2025Q3～2026Q2四季度；之后的尾段仅供增量推演'])
    first = completions[0]
    info = json.loads((INPUT_ROOT / 'model_train' / first['quarter'] / f"fold{first['fold']}" / 'training_info.json').read_text())
    report['training'] = dict(recipe=info['recipe'], feature_count=len(info['feature_columns']), feature_source=info['feature_source'])
    write_scores(pred, panel, days, out / 'ensemble')
    result = evaluate_fixed_strategies(pred, panel, px, days, report, 'training_predictions')
    save_score_metadata(out / 'ensemble', result, dates)
from model import metadata, data_root, recipe_signature, predict_auxiliary, predict_market_risk, LAYOUT_QUARTERS
import hashlib
RULES = ('baseline', 'band20', 'band60', 'market_cash', 'cost_cash', 'defensive', 'open_close_t')
STANDARD_QUARTERS = ('2025Q3', '2025Q4', '2026Q1', '2026Q2')

def load_market_context(panel):
    cols = ['mkt_above_ma20_ratio', 'mkt_idx_csi300_ret1', 'mkt_limit_down_ratio']
    frames = [pd.read_parquet(data_root() / 'market_factors' / f'year={y}' / 'data.parquet', columns=['trade_date', *cols]) for y in panel.meta['built_years']]
    frame = pd.concat(frames).set_index('trade_date').reindex(panel.days)
    return {c: frame[c].to_numpy() for c in cols}

def market_budget(probability, signal, context, state, recovery):
    """减仓/清仓与恢复有不同阈值，避免风险边界附近每天来回交易。"""
    breadth = context['mkt_above_ma20_ratio'][signal]
    index_return = context['mkt_idx_csi300_ret1'][signal]
    limit_down = context['mkt_limit_down_ratio'][signal]
    panic = np.isfinite(index_return) and index_return <= -0.035 or (np.isfinite(limit_down) and limit_down >= 0.08)
    if panic or probability >= 0.35:
        return (0.0, 0)
    if state == 0.0:
        recovery = recovery + 1 if probability < 0.15 and np.isfinite(breadth) and (breadth >= 0.45) else 0
        return (1.0, 0) if recovery >= 2 else (0.0, recovery)
    if probability >= 0.2:
        return (0.4, 0)
    return (1.0, 0)

def strategy_backtest(pred, auxiliary, probability, panel, px, days, rule='baseline', schedule=None, context=None, slip=0.0003, liquidate=True):
    """与模板同一含费现金口径；常规订单下一开盘，T仅开盘卖隔夜底仓并收盘买回。"""
    if not len(days) or np.any(np.diff(days) != 1):
        raise ValueError('现金回测必须使用连续交易日')
    if pred.shape != (len(days), len(panel.codes)) or auxiliary.shape != (*pred.shape, 2):
        raise ValueError('现金回测输入轴不一致')
    money, n = (100000.0, 5)
    cash, cumulative_fees, cumulative_slippage = (money, 0.0, 0.0)
    holdings, buys, peaks = ({}, {}, {})
    curves, trades = ([], [])
    risk_state, recovery = (1.0, 0)
    corp_events = blocked_total = t_attempts = t_failed_buybacks = 0
    by_day = {int(d): i for i, d in enumerate(days)}
    first, last = (int(days[0]) + 1, int(days[-1]) + (2 if liquidate else 1))
    if last >= len(panel.days):
        raise ValueError('回测末尾没有完整的执行和清仓尾部')
    for d in range(first, last + 1):
        for c in list(holdings):
            ratio = px.adj[d, c] / px.adj[d - 1, c]
            if not np.isfinite(ratio) or ratio <= 0:
                raise ValueError('持仓复权因子缺失')
            if abs(ratio - 1) > 1e-08:
                holdings[c] *= ratio
                corp_events += 1
        sellable = {c: q if buys[c] < d else 0.0 for c, q in holdings.items()}

        def execute(c, side, quantity, raw_price, clock, reason):
            nonlocal cash, cumulative_fees, cumulative_slippage
            if quantity <= 0 or not np.isfinite(raw_price) or raw_price <= 0:
                return False
            if not np.isfinite(px.limit_down[d, c]) or not np.isfinite(px.limit_up[d, c]) or raw_price < px.limit_down[d, c] - 1e-08 or (raw_price > px.limit_up[d, c] + 1e-08):
                return False
            if side == 'sell' and (raw_price <= px.limit_down[d, c] + 1e-08 or quantity > sellable.get(c, 0.0) + 1e-06):
                return False
            if side == 'buy' and raw_price >= px.limit_up[d, c] - 1e-08:
                return False
            price = raw_price * (1 + slip if side == 'buy' else 1 - slip)
            gross = quantity * price
            fee = fees(gross, side == 'sell')
            if side == 'buy':
                if gross + fee > cash + 1e-06 or (c not in holdings and len(holdings) >= n):
                    return False
                cash -= gross + fee
                if c not in holdings:
                    buys[c] = d
                    peaks[c] = raw_price * px.adj[d, c]
                holdings[c] = holdings.get(c, 0.0) + quantity
            else:
                cash += gross - fee
                holdings[c] -= quantity
                sellable[c] -= quantity
                if holdings[c] < 1e-06:
                    del holdings[c]
                    del buys[c]
                    peaks.pop(c, None)
            cumulative_fees += fee
            cumulative_slippage += abs(quantity * raw_price - gross)
            trades.append(dict(date=str(panel.days[d]), time=clock, signal_date=str(panel.days[d - 1]), code=str(panel.codes[c]), side=side, quantity=float(quantity), price=float(price), fee=float(fee), reason=reason))
            return True
        signal = d - 1
        index = by_day.get(signal)
        active = rule
        if schedule and index is not None:
            active = schedule[str(pd.Period(str(panel.days[signal]), freq='Q'))]
        band = 20 if active == 'band20' else 60 if active == 'band60' else 40
        risk_rule = active in ('market_cash', 'defensive', 'open_close_t')
        cost_rule = active in ('cost_cash', 'defensive', 'open_close_t')
        target_fraction = 1.0
        desired = list(holdings)
        if index is not None:
            p = pred[index]
            candidates = np.flatnonzero(np.isfinite(p))
            if risk_rule:
                risk_state, recovery = market_budget(float(probability[index]), signal, context, risk_state, recovery)
                target_fraction = risk_state
            else:
                risk_state, recovery = (1.0, 0)
            if cost_rule:
                hurdle = 2 * max(0.00025, 5 / (money / n)) + 2e-05 + 0.0005 + 2 * slip
                accepted = auxiliary[index, :, 1] >= hurdle
                for c in holdings:
                    accepted[c] = auxiliary[index, c, 1] >= -hurdle
                candidates = candidates[accepted[candidates]]
            ranked = candidates[np.argsort(-p[candidates], kind='stable')]
            target_n = int(round(n * target_fraction))
            keep_set = set(map(int, ranked[:band])) & set(holdings)
            if target_n < n:
                keep = [int(c) for c in ranked if int(c) in keep_set][:target_n]
            else:
                keep = list(keep_set)
            fresh = [int(c) for c in ranked if px.entry[d, c] and int(c) not in keep]
            desired = keep + fresh[:max(0, target_n - len(keep))]
        if liquidate and d == last:
            desired, target_fraction = ([], 0.0)
        blocked = 0
        equity_open = cash + sum((q * (px.open[d, c] if np.isfinite(px.open[d, c]) else px.close[d - 1, c]) for c, q in holdings.items()))
        for c in list(holdings):
            quantity = holdings[c] if c not in desired else 0.0
            if c in desired and target_fraction < 1.0 and np.isfinite(px.open[d, c]):
                cap_quantity = np.floor(equity_open / n / px.open[d, c] / 100) * 100
                quantity = min(holdings[c], max(0.0, np.floor((holdings[c] - cap_quantity) / 100) * 100))
            if quantity > 0:
                if not px.exit[d, c] or not execute(c, 'sell', quantity, px.open[d, c], '09:30', 'risk_or_rank_exit'):
                    blocked += 1
        blocked_total += blocked
        buybacks = []
        reserved = 0.0
        if index is not None and active == 'open_close_t' and (target_fraction == 1.0) and (d < last or not liquidate):
            for c in list(holdings):
                if c not in desired or buys[c] >= d or sellable.get(c, 0.0) <= 0:
                    continue
                previous_close = px.raw['close'][signal, c]
                previous_base = px.raw['pre_close'][signal, c]
                if not np.isfinite(previous_close) or not np.isfinite(previous_base) or previous_base <= 0:
                    continue
                if previous_close / previous_base - 1 < 0.03:
                    continue
                price = px.open[d, c]
                capacity = panel.amount[signal, c] * 0.01
                if not px.exit[d, c] or not np.isfinite(capacity) or (not np.isfinite(price)) or (price <= 0):
                    continue
                quantity = min(int(sellable[c] / 3 / 100) * 100, int(capacity / price / 100) * 100)
                if quantity < 100:
                    continue
                t_attempts += 1
                before = cash
                if execute(c, 'sell', quantity, price, '09:30', 't_sell_overnight'):
                    reserved += cash - before
                    buybacks.append((c, quantity))
        if index is not None and (d < last or not liquidate):
            equity_open = cash + sum((q * (px.open[d, c] if np.isfinite(px.open[d, c]) else px.close[d - 1, c]) for c, q in holdings.items()))
            for c in desired:
                if c in holdings or len(holdings) >= n or (not px.entry[d, c]):
                    continue
                held_value = sum((q * (px.open[d, cc] if np.isfinite(px.open[d, cc]) else px.close[d - 1, cc]) for cc, q in holdings.items()))
                purse = (equity_open * target_fraction - held_value) / max(1, len(desired) - len(holdings))
                amount = panel.amount[signal, c]
                budget = min(max(0.0, cash - reserved), max(0.0, purse), amount * 0.01 if np.isfinite(amount) else 0.0)
                price = px.open[d, c] * (1 + slip)
                quantity = int(max(0.0, budget - 5) / (price * (1 + 0.00025 + 1e-05)) / 100) * 100
                while quantity > 0 and quantity * price + fees(quantity * price) > budget:
                    quantity -= 100
                execute(c, 'buy', quantity, px.open[d, c], '09:30', 'rank_entry')
        for c, quantity in buybacks:
            buy_price = px.raw['close'][d, c]
            buy_cap = panel.amount[signal, c] * 0.01
            if not np.isfinite(buy_price) or buy_price <= 0 or (not np.isfinite(buy_cap)) or (px.raw['vol'][d, c] <= 0):
                t_failed_buybacks += 1
                continue
            purse = min(cash, buy_cap)
            buy_quantity = min(quantity, int(max(0.0, purse - 5) / (buy_price * (1 + slip) * (1 + 0.00026)) / 100) * 100)
            failed = buy_quantity < quantity
            if buy_quantity <= 0 or not execute(c, 'buy', buy_quantity, buy_price, '15:00', 't_buyback'):
                failed = True
            if failed:
                t_failed_buybacks += 1
        if cash < -1e-06 or len(holdings) > n or any((v < -1e-06 for v in sellable.values())):
            raise AssertionError('资金、五只上限或 T+1 约束被突破')
        equity = cash + sum((q * px.close[d, c] for c, q in holdings.items()))
        if not np.isfinite(equity):
            raise ValueError('持仓估值缺失')
        curves.append(dict(date=str(panel.days[d]), equity=float(equity), cash=float(cash), positions=len(holdings), blocked_exits=blocked, cumulative_fees=cumulative_fees, cumulative_slippage=cumulative_slippage, gross_equity=equity + cumulative_fees + cumulative_slippage, exposure=float((equity - cash) / equity), rule=active, target_exposure=target_fraction, risk_probability=float(probability[index]) if index is not None else None, holding_codes=','.join((str(panel.codes[c]) for c in holdings))))
    values = np.r_[money, [row['equity'] for row in curves]]
    returns = values[1:] / values[:-1] - 1
    dd = values / np.maximum.accumulate(values) - 1
    stat = dict(topn=5, rebalance_days=1, return_value=float(values[-1] / money - 1), max_drawdown=float(dd.min()), sharpe=float(returns.mean() / returns.std() * np.sqrt(242)) if returns.std() else 0.0, total_fees=cumulative_fees, trades=len(trades), avg_exposure=float(np.mean([r['exposure'] for r in curves])), ending_cash=float(cash), ending_positions=len(holdings), corporate_action_adjustments=corp_events, start=str(panel.days[first]), end=str(panel.days[last]), blocked_exits=blocked_total, empty_days=sum((r['positions'] == 0 for r in curves)), reduced_days=sum((0 < r['positions'] < 5 for r in curves)), t_attempts=t_attempts, t_failed_buybacks=t_failed_buybacks, gross_return_same_positions=float((values[-1] + cumulative_fees + cumulative_slippage) / money - 1), fee_drag=cumulative_fees / money, slippage_drag=cumulative_slippage / money, gross_definition='same actual shares and trade dates; costs added back as cash, no reinvestment', corporate_action_assumption='复权比折算经济股数，分红立即再投资近似', execution_times=['09:30', '15:00'] if rule == 'open_close_t' or (schedule and 'open_close_t' in schedule.values()) else ['09:30'], max_positions_observed=max((r['positions'] for r in curves)))
    if abs(stat['gross_return_same_positions'] - stat['return_value'] - stat['fee_drag'] - stat['slippage_drag']) > 1e-10:
        raise AssertionError('毛收益、费用和净收益不能对账')
    return (stat, curves, trades)

def validation_selection(panel, px, context):
    schedule, diagnostics = ({}, {})
    for q in LAYOUT_QUARTERS:
        scores = {name: [] for name in RULES}
        details = []
        for fold in (1, 2, 3, 4):
            folder = INPUT_ROOT / 'model_train' / q / f'fold{fold}'
            sp = splits(panel.days, quarter=q)[fold - 1]
            pred = np.load(folder / 'valid_predictions.npy')
            fit = selector_market_risk(folder)
            probability = predict_market_risk(fit, panel.M[sp['valid']])
            for rule in RULES:
                stat, curve, _ = strategy_backtest(pred[..., 0], pred[..., 1:], probability, panel, px, sp['valid'], rule=rule, context=context)
                objective = 242 * np.log1p(stat['return_value']) / len(curve) + 0.5 * stat['max_drawdown']
                scores[rule].append(float(objective))
                details.append(dict(fold=fold, rule=rule, objective=float(objective), **stat))
        means = {name: float(np.mean(value)) for name, value in scores.items()}
        schedule[q] = max(RULES, key=lambda name: means[name])
        diagnostics[q] = dict(selected=schedule[q], objective_by_rule=means, blocks=details)
        print('VALIDATION_STRATEGY', q, schedule[q], means, flush=True)
    return (schedule, diagnostics)

def selector_market_risk(folder):
    # 严禁回退到已经看过旧验证标签的最终重训风险模型。
    selected = json.loads((folder / 'selector_complete.json').read_text())
    if selected['recipe_signature'] != recipe_signature():
        raise ValueError('验证风险预测的选择配方变化')
    return torch.load(folder / 'selector_market_risk.pt', map_location='cpu', weights_only=False)

def ensemble_predictions(panel, px, audit=True):
    score = np.full((len(panel.days), len(panel.codes)), np.nan, np.float32)
    aux = np.full((*score.shape, 2), np.nan, np.float32)
    probability = np.full(len(panel.days), np.nan)
    audits, completions = ([], [])
    for qi, q in enumerate(LAYOUT_QUARTERS):
        sp0 = splits(panel.days, quarter=q)[0]
        next_start = str(pd.Period(LAYOUT_QUARTERS[qi + 1], freq='Q').start_time.date()) if qi + 1 < len(LAYOUT_QUARTERS) else '9999'
        days = sp0['score'][panel.days[sp0['score']] < next_start]
        members, auxiliary, risks = ([], [], [])
        for fold in (1, 2, 3, 4):
            folder = INPUT_ROOT / 'model_train' / q / f'fold{fold}'
            done = json.loads((folder / 'complete.json').read_text())
            if done['recipe_signature'] != recipe_signature() or not done.get('research_extension_complete'):
                raise ValueError('折的配方或完成状态不匹配')
            saved = np.load(folder / 'score_predictions.npy')
            members.append(saved[:len(days)])
            auxiliary.append(np.load(folder / 'auxiliary_predictions.npy')[:len(days)])
            fit = torch.load(folder / 'market_risk.pt', map_location='cpu', weights_only=False)
            risks.append(predict_market_risk(fit, panel.M[days]))
            completions.append({**done, 'quarter': q, 'fold': fold})
            if audit:
                ck = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
                if ck['features'] != panel.features:
                    raise ValueError('冻结输入列不匹配')
                net = PredictModel(len(panel.features), panel.market_dim)
                net.load_state_dict(ck['model'], strict=True)
                net.ridge = ck.get('ridge')
                positions = sorted(set([0, len(days) // 2, len(days) - 1]))
                actual = predict(net, panel, px, days[positions], torch.device('cpu'))
                expected = saved[positions]
                if not np.array_equal(np.isfinite(actual), np.isfinite(expected)):
                    raise ValueError('重载推理掩码不同')
                error = float(np.nanmax(np.abs(actual - expected)))
                if error > 1e-06:
                    raise ValueError(f'重载推理不一致: {q}/{fold} {error}')
                sp = splits(panel.days, quarter=q)[fold - 1]
                selector = torch.load(folder / 'selector_best.pt', map_location='cpu', weights_only=False)
                if selector['recipe_signature'] != recipe_signature() or selector['features'] != panel.features:
                    raise ValueError('验证选择权重的配方或输入不匹配')
                net.load_state_dict(selector['model'], strict=True)
                net.ridge = selector.get('ridge')
                vp = sorted(set([0, len(sp['valid']) // 2, len(sp['valid']) - 1]))
                actual_valid = predict(net, panel, px, sp['valid'][vp], torch.device('cpu'))
                expected_valid = np.load(folder / 'valid_predictions.npy')[vp, :, 0]
                if not np.array_equal(np.isfinite(actual_valid), np.isfinite(expected_valid)):
                    raise ValueError('验证选择权重的重载掩码不同')
                valid_error = float(np.nanmax(np.abs(actual_valid - expected_valid)))
                if valid_error > 1e-6:
                    raise ValueError('验证分数不是冻结选择模型的预测')
                selector_risk = selector_market_risk(folder)
                if selector_risk['train_end'] >= str(panel.days[sp['valid'][0]]):
                    raise ValueError('验证风险模型包含未来样本')
                if not done.get('delivery_refit_complete'):
                    raise ValueError('最终交付重训尚未完成')
                audits.append(dict(quarter=q, fold=fold, max_cpu_reload_error=max(error, valid_error),
                    delivery_reload_error=error, selector_reload_error=valid_error,
                    selector_train_end=done['closed_form_fit']['selector_train_end'],
                    delivery_train_end=done['closed_form_fit']['train_end'],
                    validation_risk_source='selector_market_risk.pt',
                    days=[str(panel.days[days[p]]) for p in positions]))
        score[days] = np.sum(members, axis=0, dtype=np.float32)
        aux[days] = np.mean(auxiliary, axis=0, dtype=np.float32)
        probability[days] = np.mean(risks, axis=0)
    return (score, aux, probability, audits, completions)

def window_metrics(pred, panel, days, qs):
    daily, ic, rank = top_portfolio_daily(pred, panel, days)
    rows = []
    for q in (*qs, '总计'):
        take = np.ones(len(days), bool) if q == '总计' else np.asarray([str(pd.Period(str(panel.days[d]), freq='Q')) == q for d in days])
        rows.append(dict(quarter=q, days=int(take.sum()), **top_metrics(daily[take], {h: v[take] for h, v in ic.items()}, {h: v[take] for h, v in rank.items()})))
    return rows

def risk_diagnostics(panel, days, probability):
    y1 = np.nanmean(panel.Y['label_ret_1d'][days], axis=1)
    y5 = np.nanmean(panel.Y['label_ret_5d'][days], axis=1)
    target = (y1 <= -0.02) | (y5 <= -0.04)
    pred = probability >= 0.35
    return dict(target='pool-mean1d<=-2% OR5d<=-4%; overlapping events are not independent', days=len(days), event_days=int(target.sum()), clear_signals=int(pred.sum()), true_positive=int((target & pred).sum()), false_positive=int((~target & pred).sum()), missed_event_days=int((target & ~pred).sum()), brier=float(np.mean((probability - target) ** 2)), prior_brier=float(np.mean((target.mean() - target) ** 2)), positive_next1d_when_clear=int((pred & (y1 > 0)).sum()), mean_next1d_when_clear=float(y1[pred].mean()) if pred.any() else None)


def _relocated_reference_path(value):
    """Interpret historical reference paths after documentation-only relocation."""
    value = str(value)
    value = value.replace('experiments/v263_seed_review_20261025', 'experiments/_archive/20261024_20261025/replicas/v263_seed_review_20261025')
    value = value.replace('experiments/v287_seed_review_20261025', 'experiments/_archive/20261024_20261025/replicas/v287_seed_review_20261025')
    return value

def diversity_report(score, panel, px, days, forward, context):
    """固定参照集；相关性只作开发证据，组合权重事先固定为 50/50。"""
    references = ['V188', 'V224', 'V237', 'V244', 'V246', 'V247', 'V253', 'V254', 'V255', 'V258', 'V259', 'V261', 'V262', 'V264', 'V266', 'V267', 'V268', 'V260', 'V269', 'V270', 'V271', 'V272', 'V273', 'V265', 'V274', 'V275', 'V276', 'V277', 'V278', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V263', 'V286', 'V287']  # Frozen before this round.
    all_days = np.r_[days, forward]
    index = pd.MultiIndex.from_product([panel.days[all_days], panel.codes], names=['trade_date', 'stock_code'])
    blank_aux = np.zeros((len(all_days), len(panel.codes), 2), np.float32)
    rows = []
    for name in references:
        design=json.loads((PROJECT_ROOT/'experiments/_archive/20261028_20261029/model_info/v288_design.json').read_text())
        folder=PROJECT_ROOT/_relocated_reference_path(design['reference_evaluation_units'][name])
        frozen=design['reference_score_files'][name]
        sources=[PROJECT_ROOT/_relocated_reference_path(item['path']) for item in frozen]
        if len(sources)!=2 or any(hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256'] for path,item in zip(sources,frozen)):
            raise ValueError(f'{name}: frozen reference score fingerprint changed')
        if not all((path.is_file() for path in sources)):
            raise FileNotFoundError(f'{name}: 登记的研究参照缺少冻结分数')
        chunks = [pd.read_parquet(path, columns=['trade_date', 'stock_code', 'value']) for path in sources]
        frame = pd.concat(chunks, ignore_index=True)
        frame['trade_date'] = frame.trade_date.astype(str)
        frame['stock_code'] = frame.stock_code.astype(str)
        if frame.duplicated(['trade_date', 'stock_code']).any():
            raise ValueError(f'{name}: 历史参照主键重复')
        ref = frame.set_index(['trade_date', 'stock_code']).reindex(index).value.to_numpy(np.float32).reshape(len(all_days), len(panel.codes))
        periods = {}
        for label, selected in [('standard', days), ('forward', forward)]:
            offset = 0 if label == 'standard' else len(days)
            ref_part = ref[offset:offset + len(selected)]
            own = score[selected]
            correlations, names, overlaps = ([], [], [])
            blend = np.full_like(own, np.nan)
            for i, (a, b) in enumerate(zip(own, ref_part)):
                good = np.isfinite(a) & np.isfinite(b)
                if good.sum() < 100:
                    raise ValueError(f'{name}: {panel.days[selected[i]]} 共同有效股票不足')
                ra, rb = (rankdata(a[good]), rankdata(b[good]))
                correlations.append(corr(ra, rb))
                names.append(int(good.sum()))
                blend[i, good] = 0.5 * (ra - 0.5) / good.sum() + 0.5 * (rb - 0.5) / good.sum()
                ca = np.flatnonzero(np.isfinite(a))
                cb = np.flatnonzero(np.isfinite(b))
                topa = set(ca[np.argsort(-a[ca], kind='stable')[:5]])
                topb = set(cb[np.argsort(-b[cb], kind='stable')[:5]])
                overlaps.append(len(topa & topb) / 5.0)
            aux = blank_aux[:len(selected)]
            prob = np.zeros(len(selected))
            own_stat, own_curve, _ = strategy_backtest(own, aux, prob, panel, px, selected, context=context)
            ref_stat, ref_curve, _ = strategy_backtest(ref_part, aux, prob, panel, px, selected, context=context)
            combo, _, _ = strategy_backtest(blend, aux, prob, panel, px, selected, context=context)
            qs = STANDARD_QUARTERS if label == 'standard' else ('2026Q3',)
            own_large = window_metrics(own, panel, selected, qs)[-1]
            reference_large = window_metrics(ref_part, panel, selected, qs)[-1]
            fusion_large = window_metrics(blend, panel, selected, qs)[-1]

            def returns(curve):
                value = np.asarray([r['equity'] for r in curve])
                return value / np.r_[100000.0, value[:-1]] - 1
            periods[label] = dict(days=len(selected), daily_score_spearman=float(np.mean(correlations)), minimum_common_stocks=min(names), mean_common_stocks=float(np.mean(names)), score_spearman_by_quarter={q: float(np.mean([v for v, d in zip(correlations, selected) if str(pd.Period(str(panel.days[d]), freq='Q')) == q])) for q in (STANDARD_QUARTERS if label == 'standard' else ('2026Q3',))}, account_daily_return_pearson=corr(returns(own_curve), returns(ref_curve)), signal_top5_overlap=float(np.mean(overlaps)), own_uniform_baseline=own_stat, reference_uniform_baseline=ref_stat, fixed_half_rank_fusion=combo, fusion_return_delta_vs_reference=combo['return_value'] - ref_stat['return_value'], fusion_drawdown_delta_vs_reference=combo['max_drawdown'] - ref_stat['max_drawdown'])
            periods[label]['large_capital'] = dict(own=own_large, reference=reference_large, fixed_half_rank_fusion=fusion_large, fusion_IC_delta_vs_reference=fusion_large['IC'] - reference_large['IC'], fusion_stability_delta_vs_reference=fusion_large['top_return_stability'] - reference_large['top_return_stability'])
        audit_path = folder / 'model_info/final_audit.json'
        ref_audit = json.loads(audit_path.read_text())
        rows.append(dict(reference=name, snapshot=ref_audit.get('snapshot'), score_files=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources], **periods))
        print('DIVERSITY_STANDARD', name, periods['standard']['daily_score_spearman'], periods['standard']['account_daily_return_pearson'], flush=True)
    return dict(frozen_top_references=RECIPE['frozen_top_references'], reference_design='experiments/_archive/20261028_20261029/model_info/v288_design.json', method='daily Spearman on common finite stocks, equal day weighting; account Pearson uses same n5 band40 cash engine', fusion='50/50 within-day common-stock percentile ranks; both original fixed large-capital and band40 cash metrics; weights never chosen on test', references=rows, unavailable_release_history=dict(models=['V220', 'V230', 'V240', 'V252'], reason='only2026Q4 release checkpoints remain; no frozen four-quarter historical scores; never backfill with future-trained weights'), interpretation='low correlation alone is insufficient; compare own return/drawdown and fixed-weight fusion; forward references keep their own frozen training schedule')

def main():
    parser = ArgumentParser(description='新因子研究；标准四季度、原指标与独立 Q3 检验')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--recent30', action='store_true')
    args = parser.parse_args()
    if args.recent30:
        return recent30_review()
    validate_layout(allow_temporary=True, allow_missing=True)
    torch.set_num_threads(int(os.environ.get('MX_THREADS', '2')))
    panel = Panel(RECIPE['features'])
    px = Prices(panel)
    context = load_market_context(panel)
    score, auxiliary, probability, audits, completions = ensemble_predictions(panel, px, args.audit)
    schedule, validation = validation_selection(panel, px, context)
    days = np.flatnonzero((panel.days >= '2025-07-01') & (panel.days <= '2026-06-30'))
    forward = np.flatnonzero((panel.days >= '2026-07-01') & (panel.days <= '2026-09-30'))
    if len(days) != 242:
        raise ValueError('标准评价窗必须有 242 个信号日')
    checks = px.verify_labels(panel, np.r_[days, forward])
    rows = window_metrics(score[days], panel, days, STANDARD_QUARTERS)
    forward_rows = window_metrics(score[forward], panel, forward, ('2026Q3',))
    out = RUN_ROOT / 'model_pred'
    csv = out / 'ensemble/CSV'
    csv.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, d, rule, choices in [('baseline', days, 'baseline', None), ('adaptive', days, 'baseline', schedule), ('market_cash', days, 'market_cash', None), ('open_close_t', days, 'open_close_t', None), ('forward', forward, 'baseline', schedule)]:
        stat, curve, trades = strategy_backtest(score[d], auxiliary[d], probability[d], panel, px, d, rule=rule, schedule=choices, context=context)
        pd.DataFrame(curve).to_csv(output_file(csv / f'cash_{name}.csv'), index=False)
        pd.DataFrame(trades, columns=['date', 'time', 'signal_date', 'code', 'side', 'quantity', 'price', 'fee', 'reason']).to_csv(output_file(csv / f'trades_{name}.csv'), index=False)
        quarterly = quarter_metrics('ensemble', name, curve, trades, score[d], panel, d)
        result[name] = dict(metrics=stat, quarterly=quarterly)
        print('CASH_RESULT', name, json.dumps(stat, ensure_ascii=False), flush=True)
    old, _, _ = cash_backtest(score[days], panel, px, days, n=5, period=1, band=40, cost_gate_multiplier=0.0)
    for key in ('return_value', 'max_drawdown', 'total_fees', 'trades', 'gross_return_same_positions'):
        if not np.isclose(old[key], result['baseline']['metrics'][key], atol=1e-08, rtol=1e-10):
            raise AssertionError(f"统一基线对拍失败 {key}: {old[key]} / {result['baseline']['metrics'][key]}")
    sensitivity = {}
    for rule in RULES:
        stat, _, _ = strategy_backtest(score[days], auxiliary[days], probability[days], panel, px, days, rule=rule, context=context)
        sensitivity[rule] = stat
    forward_controls = {}
    for rule in ('baseline', 'market_cash', 'defensive', 'open_close_t'):
        stat, _, _ = strategy_backtest(score[forward], auxiliary[forward], probability[forward], panel, px, forward, rule=rule, context=context)
        forward_controls[rule] = stat
    ideal, real = realizable_topk(score[days], panel, px, days, k=5, label='label_ret_5d')
    realizable = dict(topk=5, label='label_ret_5d', days=len(days), ideal_mean_pct=float(np.nanmean(ideal)) * 100, realizable_mean_pct=float(np.nanmean(real)) * 100, days_with_position=int(np.isfinite(real).sum()), gap_pct=float(np.nanmean(real - ideal)) * 100)
    baseline_curve = pd.read_csv(csv / 'cash_baseline.csv')
    market_curve = pd.read_csv(csv / 'cash_market_cash.csv')
    base_returns = baseline_curve.equity.to_numpy() / np.r_[100000.0, baseline_curve.equity.to_numpy()[:-1]] - 1
    de_risked = market_curve.exposure.to_numpy() < baseline_curve.exposure.to_numpy() - 0.05
    risk_effect = dict(return_delta=result['market_cash']['metrics']['return_value'] - result['baseline']['metrics']['return_value'], drawdown_delta=result['market_cash']['metrics']['max_drawdown'] - result['baseline']['metrics']['max_drawdown'], fee_delta=result['market_cash']['metrics']['total_fees'] - result['baseline']['metrics']['total_fees'], actual_de_risked_days=int(de_risked.sum()), de_risked_on_positive_baseline_days=int((de_risked & (base_returns > 0)).sum()), de_risked_on_negative_baseline_days=int((de_risked & (base_returns < 0)).sum()))
    stressed, _, _ = strategy_backtest(score[days], auxiliary[days], probability[days], panel, px, days, schedule=schedule, context=context, slip=0.0006)
    report = dict(run_name=run_name(), checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), score_source='training_predictions', evaluation_window=['2025-07-01', '2026-06-30'], forward_window=['2026-07-01', '2026-09-30'], model_train_read=True, weights_reloaded=bool(audits), folds_checked=len(audits), standard_folds_checked=sum((r['quarter'] in STANDARD_QUARTERS for r in audits)), forward_folds_checked=sum((r['quarter'] == '2026Q3' for r in audits)), max_cpu_reload_error=max([r['max_cpu_reload_error'] for r in audits], default=None), audit=audits, metrics_total=rows[-1], quarterly_metrics=rows[:-1], min_test_combinations=min_test_combinations(score[days], panel, days), realizable=realizable, snapshot=dict(built_at=panel.meta['built_at'], stock_feature_count=len(panel.meta['columns']['features']), input_feature_count=len(panel.features), meta_sha256=RECIPE['snapshot_meta_sha256']), replica_status='single screening replica; no promotion', small_capital=result['adaptive'], uniform_baseline=result['baseline'], strategies=sensitivity, strategy_schedule=schedule, strategy_validation=validation, forward=dict(metrics_total=forward_rows[-1], small_capital=result['forward'], controls=forward_controls), market_risk=dict(standard=risk_diagnostics(panel, days, probability[days]), forward=risk_diagnostics(panel, forward, probability[forward])), risk_ablation=dict(no_gate=result['baseline'], market_gate=result['market_cash']), risk_effect=risk_effect, t_ablation=dict(without_t=sensitivity['defensive'], with_t=sensitivity['open_close_t']), slippage_stress=stressed, label_price_check=checks, baseline_engine_parity=True, clean_blocks_are_epoch_and_strategy_selection_data=True, large_capital_parameters=TOP_PORTFOLIO, small_capital_parameters=dict(money=100000, max_positions=5, regular_execution_time='09:30', t_sell_time='09:30', t_buyback_time='15:00'), execution_data=[dict(path=str(data_root() / block / f'year={y}/data.parquet'), sha256=hashlib.sha256((data_root() / block / f'year={y}/data.parquet').read_bytes()).hexdigest()) for block in ('prices', 'amount') for y in (2025, 2026)], caveats=['固定存续股票池与修订因子含回顾偏差', '验证块有重叠，四折不是独立市场样本', 'Q3 是时序测试，但已参与历史研究选择，属于开发证据', '单种子筛选，不自动发布'])
    report['diversity'] = diversity_report(score, panel, px, days, forward, context)
    report['resources'] = None
    resource_log = RUN_ROOT / 'model_logs/analysis_0.log'
    if resource_log.exists():
        for resource_line in resource_log.read_text().splitlines():
            if resource_line.startswith('RESOURCE_SUMMARY_JSON '):
                resource_run = json.loads(resource_line.split(' ',1)[1])
                scope = resource_run.get('task_scope')
                expected_scope = {(q,f) for q in LAYOUT_QUARTERS for f in (1,2,3,4)}
                if scope is None or set(map(tuple,scope)) == expected_scope:
                    report['resources'] = resource_run
    report['epoch_selection_used'] = RECIPE.get('neural_kind') != 'ridge_closed_form'
    report['relative_score'] = dict(formula='fixed closed-form ridge on daily standardized 1d return', penalty=RECIPE['ridge_penalty'], neural_correction=False)
    report['selection_refit_protocol'] = dict(enabled=True,
        validation_source='strict pre-validation selector model and risk fit; never final-refit validation',
        delivery_source='same fixed recipe refit through mature pretest history with21day isolation',
        independent_replicas=False)

    atomic_json(info_root() / 'final_audit.json', report)
    atomic_json(info_root() / 'two_strategy_validation.json', dict(evaluation_days=len(days), quarterly_returns_compound=True, date_axis_matches_price_axis=True, label_price_check=checks, baseline_engine_parity=True, strategy_schedule=schedule, realized={k: v['metrics'] for k, v in result.items()}, sensitivity=sensitivity))
    dates = np.flatnonzero(panel.days >= '2025-07-01')
    for year in (2025, 2026):
        chunks = []
        for d in dates[np.char.startswith(panel.days[dates], str(year))]:
            known = np.isfinite(score[d])
            chunks.append(pd.DataFrame(dict(trade_date=str(panel.days[d]), stock_code=panel.codes[known], value=score[d, known], rank=rankdata(score[d, known]) / known.sum(), absolute_forecast_1d=auxiliary[d, known, 0], absolute_forecast_5d=auxiliary[d, known, 1], market_risk_probability=probability[d])))
        path = output_file(out / f'ensemble/year={year}/data.parquet')
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(chunks, ignore_index=True).to_parquet(path, index=False)
    atomic_json(out / 'ensemble/score_meta.json', dict(run_name=run_name(), snapshot=report['snapshot'], recipe_signature=recipe_signature(), score_dates=[str(panel.days[d]) for d in dates], strategy_schedule=schedule, rank_scale='raw fourfold sum; daily zscore for standard min_test', absolute_scale='mean of four training-only absolute return fits; return units', market_risk_scale='mean of four training-only logistic probabilities', evaluation_window=report['evaluation_window'], forward_window=report['forward_window']))
    valid_rows = training_validation_rows(completions)
    text = '```text\n' + metric_blocks_text(rows, valid_rows) + '```\n\n'
    names = stock_names(panel.codes)
    calendar = trade_axis(panel)
    calendar_index = {day: i for i, day in enumerate(calendar)}
    for d in dates[-10:][::-1]:
        ranked = np.flatnonzero(np.isfinite(score[d]))
        ranked = ranked[np.argsort(-score[d, ranked], kind='stable')[:5]]
        at = calendar_index[str(panel.days[d])]
        buy = calendar[at + 1] if at + 1 < len(calendar) else '下一交易日'
        text += f'## {panel.days[d]} 收盘\n\n模型输出{"一/五日联合" if len(RECIPE["score_horizon_weights"])>1 else str(next(iter(RECIPE["score_horizon_weights"])))+"日"}收益排序分数；市场大跌风险概率：{probability[d]:.2%}。下一可执行早盘：**{buy}**；实际仓位由账户策略与可成交性决定。\n\n'
        text += markdown_table(['排名', '代码', '名称', '打分'], [[i + 1, panel.codes[c], names.get(str(panel.codes[c]), ''), f'{score[d, c]:.4f}'] for i, c in enumerate(ranked)]) + '\n'
    output_file(out / 'picks.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from zoneinfo import ZoneInfo
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for name in ('baseline', 'adaptive', 'market_cash', 'open_close_t'):
        frame = pd.read_csv(csv / f'cash_{name}.csv')
        axes[0].plot(frame.date.to_numpy(dtype='datetime64[D]').astype(np.int64), (frame.equity / 100000 - 1) * 100, label=name)
    frame = pd.read_csv(csv / 'cash_forward.csv')
    axes[1].plot(frame.date.to_numpy(dtype='datetime64[D]').astype(np.int64), (frame.equity / 100000 - 1) * 100, label='Q3 chronological development')
    for ax in axes:
        x=np.asarray(ax.get_lines()[0].get_xdata(),dtype=np.int64)
        ticks=np.unique(np.linspace(0,len(x)-1,min(6,len(x)),dtype=int))
        ax.set_xticks(x[ticks],labels=x[ticks].astype('datetime64[D]').astype(str))
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.grid(alpha=0.2)
        ax.legend()
        ax.set_ylabel('Net return (%)')
        ax.tick_params(axis='x', rotation=25)
    axes[0].set_title('Standard 242 signal days | costs included')
    axes[1].set_title('Separate forward window')
    fig.savefig(output_file(out / 'ensemble_curves.png'), dpi=160)
    plt.close(fig)
    validate_layout(allow_missing=False)

def recent30_review(optional_current=None):
    """最近30个执行日的冻结信号诊断；在仓按末日收盘估值，不借用未来清仓。"""
    panel = Panel(RECIPE['features'], load_x=False)
    px = Prices(panel)
    context = load_market_context(panel)
    shape = (len(panel.days), len(panel.codes))
    score = np.full(shape, np.nan, np.float32)
    aux = np.full((*shape, 2), np.nan, np.float32)
    risk = np.full(len(panel.days), np.nan)
    ti, ci = (pd.Index(panel.days), pd.Index(panel.codes))
    root = RUN_ROOT / 'model_pred/ensemble'
    meta = json.loads((root / 'score_meta.json').read_text())
    if meta['recipe_signature'] != recipe_signature() or meta['snapshot']['meta_sha256'] != RECIPE['snapshot_meta_sha256']:
        raise ValueError('冻结分数的配方或快照不匹配')
    for year in (2025, 2026):
        frame = pd.read_parquet(root / f'year={year}/data.parquet')
        t = ti.get_indexer(frame.trade_date.astype(str))
        c = ci.get_indexer(frame.stock_code.astype(str))
        if (t < 0).any() or (c < 0).any():
            raise ValueError('近期分数轴不匹配')
        score[t, c] = frame.value.to_numpy(np.float32)
        aux[t, c, 0] = frame.absolute_forecast_1d.to_numpy(np.float32)
        aux[t, c, 1] = frame.absolute_forecast_5d.to_numpy(np.float32)
        daily = frame.groupby('trade_date').market_risk_probability.agg(['first', 'nunique'])
        if (daily['nunique'] != 1).any():
            raise ValueError('同日市场风险概率不一致')
        risk[ti.get_indexer(daily.index.astype(str))] = daily['first']
    suffix = '_q4' if optional_current is not None else ''
    if optional_current is not None:
        ix = optional_current['days']
        score[ix] = optional_current['score']
        aux[ix] = optional_current['auxiliary']
        risk[ix] = optional_current['probability']
    days = np.arange(max(0, len(panel.days) - 31), len(panel.days) - 1)
    if len(days) != 30 or not np.isfinite(score[days]).any(axis=1).all():
        raise ValueError('没有完整的30个执行日冻结信号')
    schedule = dict(meta['strategy_schedule'])
    if optional_current is not None:
        schedule.update(optional_current['schedule'])
    carried = {}
    for q in sorted({str(pd.Period(str(panel.days[d]), freq='Q')) for d in days}):
        if q not in schedule:
            earlier = sorted((k for k in schedule if k < q))
            if not earlier:
                raise ValueError('近期区间没有已冻结策略')
            schedule[q] = schedule[earlier[-1]]
            carried[q] = earlier[-1]
    accounts = {}
    curves = {}
    transactions = {}
    for name, choices in [('recent30', schedule), ('recent30_baseline', None)]:
        stat, curve, trades = strategy_backtest(score[days], aux[days], risk[days], panel, px, days, schedule=choices, context=context, liquidate=False)
        accounts[name] = stat
        curves[name] = curve
        transactions[name] = trades
        pd.DataFrame(curve).to_csv(output_file(root / f'CSV/cash_{name}{suffix}.csv'), index=False)
        pd.DataFrame(trades, columns=['date', 'time', 'signal_date', 'code', 'side', 'quantity', 'price', 'fee', 'reason']).to_csv(output_file(root / f'CSV/trades_{name}{suffix}.csv'), index=False)
    horizon = {}
    for h in (1, 5):
        values = []
        ranks = []
        dates = []
        for d in days:
            if d + h + 1 >= len(panel.days):
                continue
            y = panel.Y[f'label_ret_{h}d'][d]
            p = score[d]
            good = np.isfinite(p) & np.isfinite(y)
            if good.sum() < 100:
                continue
            values.append(corr(p[good], y[good]))
            ranks.append(corr(rankdata(p[good]), rankdata(y[good])))
            dates.append(str(panel.days[d]))
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        horizon[str(h)] = dict(mature_days=len(values), IC=float(np.mean(values)), ICIR=float(np.mean(values) / sd) if sd else None, RankIC=float(np.mean(ranks)), daily_IC_std=sd, first_signal=dates[0], last_signal=dates[-1], worst_5day_mean_IC=min((float(np.mean(values[i:i + 5])) for i in range(max(1, len(values) - 4)))))
    mature = days[days + 2 < len(panel.days)]
    daily, ic, rank = top_portfolio_daily(score[mature], panel, mature)
    large = top_metrics(daily, ic, rank)
    for h in (1, 5):
        large[f'IC_{h}d'] = horizon[str(h)]['IC']
        large[f'RankIC_{h}d'] = horizon[str(h)]['RankIC']
    large['IC'] = horizon['1']['IC']
    large['ICIR'] = horizon['1']['ICIR']
    concentration = []
    for d in mature:
        p = score[d]
        ids = pd.Series(p).sort_values(ascending=False).index.to_numpy()[:500]
        capacity = np.maximum(np.nan_to_num(panel.amount[d, ids], nan=0.0), 0.0)
        held = np.minimum(1500000000.0, np.cumsum(capacity))
        w = np.diff(np.r_[0.0, held]) / 1500000000.0
        concentration.append((1 / float(w @ w) if w @ w > 0 else 0.0, float(w.max()), int((w > 0).sum())))
    large.update(mature_1d_days=len(mature), median_effective_names=float(np.median([v[0] for v in concentration])), median_max_weight=float(np.median([v[1] for v in concentration])), median_funded_names=float(np.median([v[2] for v in concentration])))
    curve = curves['recent30']
    values = np.asarray([r['equity'] for r in curve])
    returns = values / np.r_[100000.0, values[:-1]] - 1
    cases = []
    for i in np.argsort(returns)[:3]:
        row = curve[int(i)]
        date = row['date']
        d = int(ti.get_indexer([date])[0])
        trades = [t for t in transactions['recent30'] if t['date'] == date]
        held = set(row.get('holding_codes', '').split(',')) - {''}
        if i:
            held |= set(curve[int(i) - 1].get('holding_codes', '').split(',')) - {''}
        held |= {t['code'] for t in trades}
        stocks = []
        for code in sorted(held):
            c = int(ci.get_indexer([code])[0])
            o, cl, pre = (px.open[d, c], px.raw['close'][d, c], px.raw['pre_close'][d, c])
            stocks.append(dict(code=code, open=float(o) if np.isfinite(o) else None, close=float(cl) if np.isfinite(cl) else None, opening_gap=float(o / pre - 1) if np.isfinite(o) and pre > 0 else None, intraday_return=float(cl / o - 1) if np.isfinite(cl) and o > 0 else None))
        cases.append(dict(date=date, account_return=float(returns[i]), positions=row['positions'], rule=row['rule'], exposure=row['exposure'], trades=trades, stocks=stocks))
    original = json.loads((info_root() / 'final_audit.json').read_text())
    report = dict(model=ROOT.name, kind='frozen-signal paper account; not actual live fills', execution_window=[curve[0]['date'], curve[-1]['date']], execution_days=30, signal_window=[str(panel.days[days[0]]), str(panel.days[days[-1]])], snapshot=meta['snapshot'], account_start='100000 cash; no inherited actual holdings', liquidated=False, final_signal_not_yet_executed=str(panel.days[-1]), strategy_carried_from=carried, small_capital=accounts['recent30'], uniform_baseline=accounts['recent30_baseline'], large_capital=large, horizon_metrics=horizon, five_day_account_segments=[float(np.prod(1 + returns[i:i + 5]) - 1) for i in range(0, 30, 5)], worst_account_days=cases, full_window_reference=dict(IC=original['metrics_total']['IC'], ICIR=original['metrics_total']['ICIR'], small_net=original['small_capital']['metrics']['return_value']), evidence_boundary='short diagnostic with sampling/path randomness; idea generation only, not independent confirmation or posthoc strategy replacement', manual_review=None)
    if optional_current is not None:
        report['optional_current_quarter'] = optional_current['provenance']
    atomic_json(info_root() / f'recent30_review{suffix}.json', report)
    text = f"# {ROOT.name} 最近30个交易日诊断\n\n冻结信号纸面账户，非真实成交；{report['execution_window'][0]} 至 {report['execution_window'][1]}，10万元现金起步，末日在仓按收盘价估值，不借用未来清仓价。\n\n"
    text += '| 口径 | 净收益 | 最大回撤 | 费用/元 | 期末持仓数 |\n|---|---:|---:|---:|---:|\n'
    for label, key in [('自身策略', 'small_capital'), ('统一band40', 'uniform_baseline')]:
        s = report[key]
        text += f"| {label} | {s['return_value']:.2%} | {s['max_drawdown']:.2%} | {s['total_fees']:.2f} | {s['ending_positions']} |\n"
    text += f"\n大资金固定Top500/15亿元：IC={large['IC']:.5f}，ICIR={large['ICIR']:.5f}，理想区间收益={large['total_return']:.2f}%。一日标签成熟{horizon['1']['mature_days']}日，五日成熟{horizon['5']['mature_days']}日。\n\n"
    text += f"容量分配的中位有效持仓数={large['median_effective_names']:.2f}、中位最大单股权重={large['median_max_weight']:.2%}，需与全截面IC分开判断。\n\n"
    text += '| 最差账户日 | 日净收益 | 规则 | 持仓数 |\n|---|---:|---|---:|\n' + ''.join((f"| {c['date']} | {c['account_return']:.2%} | {c['rule']} | {c['positions']} |\n" for c in cases))
    text += '\n逐笔成交与每日持仓代码保留在CSV，逐股开盘跳空/日内收益保留在JSON。短窗结果受随机性与持仓路径影响，仅作为人工诊断和新Idea的证据，与长窗及季度表现并看。\n'
    output_file(RUN_ROOT / f'model_pred/recent30_review{suffix}.md').write_text(text)
    print('RECENT30_READY', ROOT.name, json.dumps(dict(large=large, small=accounts['recent30'], baseline=accounts['recent30_baseline'], maturity={h: v['mature_days'] for h, v in horizon.items()}), ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
