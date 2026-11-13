"""Frozen next-open B40 event-risk execution and causal daily budget."""

from pathlib import Path

import json, numpy as np, pandas as pd

def fees(notional, sell=False):
    return max(notional * 0.00025, 5) + notional * 1e-05 + (notional * 0.0005 if sell else 0)

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

def strategy_backtest(pred, auxiliary, probability, panel, px, days, rule='market_cash', schedule=None, context=None, slip=0.0003, liquidate=True):
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

def predict_market_risk(fit, market):
    if fit['constant'] is not None:
        return np.full(len(market), fit['constant'])
    logit = (market.astype(np.float64) - fit['mean']) / fit['scale'] @ fit['coef'] + fit['intercept']
    return 1. / (1. + np.exp(-np.clip(logit, -40, 40)))


def budget_trace(data, end, bundle=None):
    import torch
    bundle=Path(bundle or Path(__file__).resolve().parent)
    if not '2026-10-01' <= end <= '2026-12-31':raise ValueError('Risk weights are for2026Q4 only')
    fits=[torch.load(bundle/'strategy_data'/f'fold{f}.pt',map_location='cpu',weights_only=False) for f in range(1,5)]
    cols=fits[0]['columns'];raw=['mkt_above_ma20_ratio','mkt_idx_csi300_ret1','mkt_limit_down_ratio']
    frame=pd.read_parquet(Path(data)/'market_factors/year=2026/data.parquet',columns=['trade_date',*cols,*raw],filters=[('trade_date','>=','2026-10-01'),('trade_date','<=',end)]).sort_values('trade_date').reset_index(drop=True)
    assert len(frame) and frame.trade_date.iloc[-1]==end
    assert all(f['columns']==cols and f['train_end']<'2026-10-01' for f in fits)
    x=np.clip(np.nan_to_num(frame[cols].to_numpy(np.float32),nan=0.),-10,10)
    probability=np.mean([predict_market_risk(f,x) for f in fits],axis=0)
    context={k:frame[k].to_numpy(float) for k in raw};state,recovery=1.,0;rows={}
    for i,day in enumerate(frame.trade_date):
        state,recovery=market_budget(float(probability[i]),i,context,state,recovery)
        rows[str(day)]=dict(risk_probability=float(probability[i]),target_exposure=state,recovery_count=recovery,target_names=int(round(5*state)),execution='next_open',hold_band=40,reference_start='2026-10-01 cash account; budget state initialized to1, recovery0',note='Target only; actual exits obeyT+1/limits/suspension/capacity. Holdings needed to form executable orders.')
    return rows

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--asof',required=True);parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parent.parent/'trainingdata');args=parser.parse_args()
    print(json.dumps(budget_trace(args.data,args.asof)[args.asof],ensure_ascii=False,indent=2))

