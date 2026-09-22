"""由本版本model.py和analysis.py原样抽取；只用于轻量策略比较。"""

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent

# ★ 与 model.py 同源：本版锁定 trainingdata 快照，避免工具链读回旧快照而静默错配。
os.environ.setdefault('MX_DATA', str(ROOT.parent / 'trainingdata'))

RECIPE = dict(
    name='V32',
    time_half_life=0,
    runtime_policy='torch-cu128-explicit-device-v1',
    epoch_agg='mean_all',
    architecture='MLP-4linear',
    label='label_ret_5d',
    features='all',
    folds=4,
    seed=3253,
    batch_days=4,
    optimizer='adamw',
    lr=.001,
    weight_decay=.02,
    max_epochs=30,
    early_stop_patience=6,
    lr_patience=3,
    lr_factor=.5,
    lr_cooldown=2,
    min_lr=5e-6,
    min_feature_coverage=.2,
    device='auto',
    threads=3,
    valid_money=100000,
    valid_topn=5,
    valid_metric='不选轮：打分取全部轮次预测的平均（epoch_agg=mean_all）；val_wei 只用来决定何时早停',
    preprocessing='daily_rank_centered_fixed_missing_0.5',
    test_start='2025-07-01',
    test_end='2026-06-30',
    training_window='2018起扩展窗，每季度初更新；季度边界purge h+1交易日',
)

def data_root():
    """按 MX_DATA > V16/trainingdata > ../trainingdata 的顺序找数据根，以 meta.json 是否存在判断。"""
    explicit = os.environ.get('MX_DATA')
    candidates = [Path(explicit)] if explicit else [ROOT / 'trainingdata', ROOT.parent / 'trainingdata']
    for p in candidates:
        if (p / 'meta.json').is_file():
            return p.resolve()
    raise FileNotFoundError('找不到 trainingdata/meta.json，请设置 MX_DATA')

def metadata():
    return json.loads((data_root() / 'meta.json').read_text())

def frame(root, kind, year, columns):
    """读一个数据块的某一年，统一主键类型并断言无重复。"""
    p = root / kind / f'year={year}' / 'data.parquet'
    df = pq.ParquetFile(p).read(columns=['trade_date', 'stock_code'] + list(columns)).to_pandas()
    df['trade_date'] = df['trade_date'].astype(str).str[:10]
    df['stock_code'] = df['stock_code'].astype(str)
    if df.duplicated(['trade_date', 'stock_code']).any():
        raise ValueError(f'{p}: 重复主键')
    return df.set_index(['trade_date', 'stock_code']).sort_index()

def atomic_json(path, value):
    """先写 .tmp 再 os.replace，保证读到的永远是完整 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)

def axis():
    """交易日轴与股票轴：年份从 meta 的 built_years 来，顺序与 factors 块一致。"""
    root = data_root()
    meta = metadata()
    days = []
    for y in sorted(meta['built_years']):
        d = pq.ParquetFile(root / 'factors' / f'year={y}' / 'data.parquet').read(columns=['trade_date'])
        days.extend(sorted(set(d.column(0).to_pylist())))
    return np.asarray(days), np.asarray(meta['axis']['codes'])

def quarters(days):
    """评价窗口固定为 2025-07-01 ~ 2026-06-30（用户指定，V16 及以后版本都遵守）。

    入参 days 只为保持调用签名一致，不影响返回值——不能自行扩展或缩成部分季度。
    """
    return ['2025Q3', '2025Q4', '2026Q1', '2026Q2']

class Panel:
    """全历史面板，按 (交易日 × 股票) 的二维网格存放。

    load_x=False 时只读标签与成交额（评价阶段用），省掉最大的一块内存。
    """

    def __init__(self, features='all', load_x=True):
        self.root = data_root()
        self.meta = metadata()
        self.days, self.codes = axis()
        self.features = (list(features) if isinstance(features, (list, tuple)) else
                         self.meta['columns']['features'] if features == 'all' else self.meta['feature_sets'][features])
        self.labels = self.meta['columns']['labels']

        T, C = len(self.days), len(self.codes)
        self.X = np.empty((T, C, len(self.features)), np.float32) if load_x else None
        self.coverage = np.empty((T, C), np.float32) if load_x else None
        self.Y = {name: np.full((T, C), np.nan, np.float32) for name in self.labels}
        self.amount = np.full((T, C), np.nan, np.float64)

        # 逐年读取并写进预分配好的网格；每年都断言轴一致，避免静默错位。
        t_load = time.time()
        years = list(self.meta['built_years'])
        for yi, y in enumerate(years, 1):
            rows = np.flatnonzero(np.char.startswith(self.days, str(y)))
            index = pd.MultiIndex.from_product([self.days[rows], self.codes], names=['trade_date', 'stock_code'])
            if load_x:
                df = frame(self.root, 'factors', y, self.features)
                if len(df) != len(index) or not df.index.equals(index):
                    raise ValueError(f'{y} factors 轴不一致')
                x = df.to_numpy(np.float32).reshape(len(rows), C, -1)
                self.coverage[rows] = np.isfinite(x).mean(axis=2)
                # 输入已经是当日截面 rank；固定中心 0.5，缺失中性填充，无全样本拟合。
                self.X[rows] = (np.nan_to_num(x, nan=.5) - .5) * 2
                del df, x
            df = frame(self.root, 'target', y, self.labels).reindex(index)
            for name in self.labels:
                self.Y[name][rows] = df[name].to_numpy(np.float32).reshape(len(rows), C)
            self.amount[rows] = frame(self.root, 'amount', y, ['amount']).reindex(index)['amount'].to_numpy().reshape(len(rows), C)
            print(f'已读取 {y} 年，{len(rows)} 个交易日'
                  f'（{yi}/{len(years)}，已用 {time.time() - t_load:.0f}s，'
                  f'预计还需 {(time.time() - t_load) / yi * (len(years) - yi):.0f}s）', flush=True)

        if load_x and not np.isfinite(self.X).all():
            raise ValueError('特征含无穷值')
        # 读盘是整条流水线里最长的静默段（约 1 分钟），逐行给进度与 ETA。
        print(f'面板读取完成：{len(self.meta["built_years"])} 年，'
              f'已用 {time.time() - t_load:.0f}s', flush=True)

    def mask(self, day, label=None):
        """当日可用的股票掩码：覆盖率达标，且（给定标签时）标签已知。"""
        out = self.coverage[day] >= RECIPE['min_feature_coverage']
        if label is not None:
            out = out & np.isfinite(self.Y[label][day])
        return out

class Prices:
    """原始价 + 停牌/限价的可交易性判定。

    只前向填充状态量（复权因子、收盘价）；停牌无开盘价仍不可交易，不前填成交量。
    """

    def __init__(self, panel):
        names = ['open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'vol', 'adj_factor']
        T, C = len(panel.days), len(panel.codes)
        self.raw = {n: np.full((T, C), np.nan, np.float64) for n in names}

        # 逐年读原始价；轴必须与 factors 完全一致，否则回测会错位。
        for year in panel.meta['built_years']:
            ix = np.flatnonzero(np.char.startswith(panel.days, str(year)))
            index = pd.MultiIndex.from_product([panel.days[ix], panel.codes], names=['trade_date', 'stock_code'])
            df = frame(panel.root, 'prices', year, names)
            if not df.index.equals(index):
                raise ValueError(f'{year} prices 轴与 factors 不一致')
            for n in names:
                self.raw[n][ix] = df[n].to_numpy().reshape(len(ix), C)

        self.adj = pd.DataFrame(self.raw['adj_factor']).ffill().to_numpy()
        self.close = pd.DataFrame(self.raw['close']).ffill().to_numpy()
        self.open = self.raw['open']

        # 冻结池均为非ST主板；按分四舍五入。开盘触及限价即保守拒单，
        # 不读取当天 high/low 决定开盘订单，避免用未来全天行情挑选成交。
        pre = self.raw['pre_close']
        upper = np.floor(pre * 1.10 * 100 + .5) / 100
        lower = np.floor(pre * .90 * 100 + .5) / 100
        traded = (np.isfinite(self.open) & (self.open > 0) & np.isfinite(pre) & (pre > 0)
                  & np.isfinite(self.adj) & (self.adj > 0) & (self.raw['vol'] > 0))
        self.entry = traded & (self.open < upper - 1e-8)
        self.exit = traded & (self.open > lower + 1e-8)

        # next_entry[d] = 第 d 日开盘能不能买进「T-1 日选出的票」。
        self.next_entry = np.zeros_like(self.entry)
        self.next_entry[:-1] = self.entry[1:]

        # 收盘卖的可执行性：判据与开盘卖同法，但用当日**收盘价**判跌停 —— 收盘单就在那个
        # 时点成交，不含任何超过当日收盘的信息（不是"用当天行情挑成交"）。停牌/无量同样拒单。
        raw_close = self.raw['close']
        self.exit_close = (np.isfinite(raw_close) & (raw_close > 0) & np.isfinite(pre) & (pre > 0)
                           & np.isfinite(self.adj) & (self.adj > 0) & (self.raw['vol'] > 0)
                           & (raw_close > lower + 1e-8))

    def verify_labels(self, panel, days):
        """价格轴/复权口径对拍：若标签与价格不是同一口径，回测不能继续。"""
        px = pd.DataFrame(self.open).ffill().to_numpy() * self.adj
        errors = []
        for h in [1, 3, 5, 10, 20]:
            ix = np.asarray(days)
            ix = ix[ix + h + 1 < len(px)]
            calc = px[ix + h + 1] / px[ix + 1] - 1
            target = panel.Y[f'label_ret_{h}d'][ix]
            good = np.isfinite(calc) & np.isfinite(target)
            diff = np.abs(calc[good] - target[good])
            errors.append({'horizon': h, 'checked': int(good.sum()),
                           'max_abs_error': float(diff.max()) if len(diff) else None})
            if not len(diff) or np.mean(diff > 1e-5) > .001:
                raise ValueError(f'价格与 {h}d 标签口径不一致: {errors[-1]}')
        return errors

def fees(notional, sell=False):
    """原参考工程/项目约定的固定费率；这是研究假设，不是历史费率复原。"""
    return max(notional * .00025, 5) + notional * .00001 + (notional * .0005 if sell else 0)

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

def stress_cash_backtest(pred, panel, px, days, n=5, period=1, exit_rank=None, min_hold=1,
                  stop_loss=None, trailing=None, skip=0, sell_at='open', slippage=.0003):
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

def capital_cash_backtest(pred, panel, px, days, n=5, period=1, exit_rank=None, min_hold=1, stop_loss=None, trailing=None, skip=0, sell_at='open', slippage=0.0003, initial_cash=100000.0):
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
    assert sell_at == 'open', 'capital sleeves only implement opening execution'
    assert np.isfinite(initial_cash) and initial_cash > 0
    cash = initial_cash
    holdings = {}
    buys = {}
    entry = {}
    peak = {}
    desired = []
    curves = []
    trades = []
    corp_events = 0
    first, last = (int(days[0]) + 1, int(days[-1]) + 2)
    pred_by_day = {int(d): p for d, p in zip(days, pred)}
    adj_open = px.open * px.adj
    adj_close = px.close * px.adj
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
            ranked = candidates[np.argsort(-p[candidates], kind='stable')]
            if exit_rank is None:
                if (signal - int(days[0])) % period == 0:
                    desired = [int(c) for c in ranked[skip:] if px.entry[d, c]][:n]
            else:
                head = set((int(c) for c in ranked[skip:skip + exit_rank]))
                exits = set()
                for c in list(holdings):
                    if c not in head and d - buys[c] >= min_hold:
                        exits.add(c)
                        continue
                    if stop_loss is None and trailing is None:
                        continue
                    price = adj_close[signal, c]
                    if not np.isfinite(price):
                        continue
                    peak[c] = max(peak.get(c, entry[c]), price)
                    if stop_loss is not None and price <= entry[c] * (1 - stop_loss):
                        exits.add(c)
                    elif trailing is not None and price <= peak[c] * (1 - trailing):
                        exits.add(c)
                desired = [c for c in holdings if c not in exits]
                for c in ranked[skip:]:
                    if len(desired) >= n:
                        break
                    c = int(c)
                    if c in desired or c in exits:
                        continue
                    if px.entry[d, c]:
                        desired.append(c)
        if d == last:
            desired = []
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
            trades.append(dict(date=str(panel.days[d]), code=str(panel.codes[c]), side='sell', quantity=quantity, price=float(price), fee=float(fee)))
            del holdings[c]
            del buys[c]
            entry.pop(c, None)
            peak.pop(c, None)
        if p is not None and d < last and (exit_rank is not None or (signal - int(days[0])) % period == 0):
            equity = cash + sum((q * px.open[d, c] if np.isfinite(px.open[d, c]) else q * px.close[d - 1, c] for c, q in holdings.items()))
            for c in desired:
                if c in holdings or len(holdings) >= n:
                    continue
                price = px.open[d, c] * (1 + slippage)
                amount = panel.amount[signal, c]
                budget = min(cash, equity / n, amount * 0.01 if np.isfinite(amount) else 0.0)
                quantity = int(max(0, budget - 5) / (price * (1 + 0.00025 + 1e-05)) / 100) * 100
                while quantity > 0 and quantity * price + fees(quantity * price) > budget:
                    quantity -= 100
                if quantity <= 0:
                    continue
                gross = quantity * price
                fee = fees(gross)
                cash -= gross + fee
                holdings[c] = float(quantity)
                buys[c] = d
                entry[c] = price * px.adj[d, c]
                peak[c] = entry[c]
                trades.append(dict(date=str(panel.days[d]), code=str(panel.codes[c]), side='buy', quantity=quantity, price=float(price), fee=float(fee)))
        if cash < -1e-06 or len(holdings) > n:
            raise AssertionError('资金/持仓上限被突破')
        equity = cash + sum((q * px.close[d, c] for c, q in holdings.items()))
        if not np.isfinite(equity):
            raise ValueError('持仓估值缺失')
        curves.append(dict(date=str(panel.days[d]), equity=float(equity), cash=float(cash), positions=len(holdings), blocked_exits=len(blocked)))
    values = np.r_[initial_cash, [r['equity'] for r in curves]]
    returns = values[1:] / values[:-1] - 1
    dd = values / np.maximum.accumulate(values) - 1
    return (dict(topn=n, rebalance_days=period, exit_rank=exit_rank, min_hold=min_hold, stop_loss=stop_loss, trailing=trailing, skip=skip, sell_at=sell_at, return_value=float(values[-1] / initial_cash - 1), max_drawdown=float(dd.min()), sharpe=float(np.mean(returns) / np.std(returns) * np.sqrt(242)) if np.std(returns) > 0 else 0.0, total_fees=float(sum((t['fee'] for t in trades))), trades=len(trades), ending_cash=cash, ending_positions=len(holdings), corporate_action_adjustments=corp_events, start=str(panel.days[first]), end=str(panel.days[last]), corporate_action_assumption='复权比折算经济股数，分红立即再投资近似'), curves, trades)

def markdown_table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)

STRATEGIES=[dict(name=f'buffer_r{r}_t{n}',topn=n,exit_rank=r,min_hold=1) for n,rs in [(1,(2,5,10)),(5,(10,20))] for r in rs]
