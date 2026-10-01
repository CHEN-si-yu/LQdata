"""Model-side trading overlay only; original cash engine remains unchanged.
Risk-off forces empty targets before the existing sell/buy loops. No new input factors.
"""
import numpy as np
from evaluation_core import fees


def gated_cash_backtest(pred, panel, px, days, n=5, period=1, exit_rank=None, min_hold=1,
                  stop_loss=None, trailing=None, skip=0, sell_at='open', slippage=.0003, risk_on=None):
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
    if sell_at != 'open':
        raise ValueError('This research only supports next-open execution')
    if risk_on is None:
        risk_on = np.ones(len(panel.days), dtype=bool)
    risk_on = np.asarray(risk_on)
    if risk_on.dtype != np.dtype(bool) or risk_on.shape != (len(panel.days),):
        raise ValueError('risk_on must be a full-axis boolean series, known at each signal close')
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
        # The prior close's risk-off signal overrides period/min_hold targets.
        # The original sell loop still enforces T+1 and executable opening prices.
        # Fixed-period re-entry waits for its original calendar rebalance day.
        if d == last or not risk_on[signal]:
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
            price = sell_raw[d, c] * (1 - slippage)
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
                price = px.open[d, c] * (1 + slippage)
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
