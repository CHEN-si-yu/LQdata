"""V1 · analysis（市场门控 + 多目标）。约定见项目 docs/architecture.md。

独立单元约束：除 `trainingdata` 数据块外不依赖任何外部环境——不读 /sys、/proc，
不调用 nvidia-smi 之类的外部命令，也不引用单元目录之外的路径。
"""

# =============================================================================
# ① 导入
#
# matplotlib 只在下标曲线的函数里按需 import：IC-only 的运行不该为它付加载开销。
# =============================================================================
import json
import os
import subprocess
import sys
sys.dont_write_bytecode = True
import time
from argparse import ArgumentParser
from pathlib import Path

# 单元内自带路径帮助文件；代码和配置无需依赖其他实验或公共模块。
from run import (PROJECT_ROOT, UNIT, run_root, analysis_input_root, feature_block, feature_columns,
                 data_root as project_data_root, info_root, run_name, output_file, validate_layout,
                 inspect_inputs, stock_list_path, calendar_path)
RUN_ROOT = run_root()
INPUT_ROOT = analysis_input_root()


import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

from model import (ROOT, RECIPE, Panel, Prices, atomic_json, axis, corr, fingerprint, head_of,
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
# ②b 四指标（口径出处：tmp/min_test.ipynb 的 get_ret_ic / get_metrics）
#
# 用户 2026-09-24 定：每版实验在 docs/iterations.md 里记录 IC / ICIR /
# top_return / top_return_stability，另附季度表现表。这一节只做「逐日序列 →
# 四指标」的换算，组合与 IC 的定义照抄 min_test：
#   · 组合：逐日调仓，按打分降序建仓，每只票建仓 min(剩余资金, 当日成交额)；
#   · 收益：Σ(1日标签 × 仓位) / 资金，单位 %，不连乘；
#   · IC：逐日截面 Pearson(打分, 1日标签)。
# ★ 这里的 IC 是 Pearson，训练侧验证与 report.json 里的 RankIC 是逐日
# Spearman —— 同名不同源，报告里不能只写「IC」。
# =============================================================================

TOP_PORTFOLIO = dict(topn=500, money=1.5e9, label='label_ret_1d')


def top_portfolio_daily(pred, panel, days, spec=TOP_PORTFOLIO):
    """评价窗内逐日的组合日收益（%）与 IC 序列。

    只读标签和成交额：不碰价格层、不套涨跌停/停牌掩码，也不计费用——这正是
    min_test 的无摩擦口径，成交额在这里只当资金容量上限（缺失或为 0 的票拿
    不到仓位）。标签缺失按 0 计，照抄 min_test 的 fillna(0)：本快照评价窗内
    逐日缺失率 ≤0.05%，对结果无实质影响。
    """
    topn, money = spec['topn'], spec['money']
    label = spec['label']
    daily_ret = np.zeros(len(days))
    daily_ic = np.zeros(len(days))
    for i, d in enumerate(days):
        score = pred[i]
        y = panel.Y[label][d]
        known = np.isfinite(score)          # 打分缺失的票既排不了序、也进不了 IC
        # IC：全截面 Pearson。标签缺失补 0 而不是剔除，保持与 min_test 同式。
        daily_ic[i] = corr(score[known], np.where(np.isfinite(y[known]), y[known], 0.))
        # 组合：打分降序取前 topn 只，仓位按「先到先得、单只不超过当日成交额」分配，
        # 也就是累计建仓额取 min(资金, 累计成交额) 后的逐只差分；资金用尽后其余为 0。
        order = np.flatnonzero(known)[np.argsort(-score[known], kind='stable')][:topn]
        cap = np.where(np.isfinite(panel.amount[d, order]), panel.amount[d, order], 0.)
        held = np.minimum(money, np.cumsum(cap))
        weight = np.diff(np.r_[0., held]) / money
        ret = np.where(np.isfinite(y[order]), y[order], 0.) * 100
        daily_ret[i] = float(weight @ ret)
    return daily_ret, daily_ic


def top_metrics(daily_ret, daily_ic):
    """四指标：IC / ICIR / top_return / top_return_stability（min_test 的 get_metrics）。

    两个「稳定性」都是均值 ÷ 标准差，未年化；标准差用样本口径（ddof=1），
    与 min_test 里的 pandas std 一致。top_return 是日均收益（%），不是区间累计。
    """
    ic_std = float(np.std(daily_ic, ddof=1)) if len(daily_ic) > 1 else 0.
    ret_std = float(np.std(daily_ret, ddof=1)) if len(daily_ret) > 1 else 0.
    return dict(IC=float(np.mean(daily_ic)),
                ICIR=float(np.mean(daily_ic) / ic_std) if ic_std else 0.,
                top_return=float(np.mean(daily_ret)),
                top_return_stability=float(np.mean(daily_ret) / ret_std) if ret_std else 0.)


def metric_blocks_text(rows):
    """min_test 版式的评价文字块：季度表现表 + 四指标。

    日志、picks.md 与报告用同一段文字，避免三处各写一份格式化代码。
    季度表只列四个季度（总计是另一块的四个数），与 min_test 的打印一致。
    """
    quarters_only = [r for r in rows if r['quarter'] != '总计']
    quarterly = pd.DataFrame(
        {'季度平均收益(%)': [round(r['top_return'], 4) for r in quarters_only],
         '季度平均IC': [round(r['IC'], 4) for r in quarters_only]},
        index=[r['quarter'] for r in quarters_only])
    keys = ['IC', 'ICIR', 'top_return', 'top_return_stability']
    total = rows[-1]
    metrics = pd.DataFrame({'模型值': [round(total[k], 4) for k in keys]}, index=keys)
    metrics.index.name = '指标'
    return ('---单一模型评估---\n\n'
            '===== 季度表现指标 =====\n'
            f'{quarterly.to_string()}\n'
            '=======================\n\n\n'
            '模型评估结果\n'
            f'{metrics.to_string()}\n')


def print_metric_blocks(rows):
    """把上面那段评价文字打到日志（数字以 main_metrics_1d.csv 为准）。"""
    print('\n' + metric_blocks_text(rows), flush=True)


def evaluation_slice(pred, panel, days):
    """固定四季度评价窗的（打分切片, 日期索引）；打分没覆盖满窗就直接报错。

    入参 days 是全部打分日（含评价窗之后的尾段），这里只取评价窗那一段。
    """
    selected = (panel.days[days] >= RECIPE['test_start']) & (panel.days[days] <= RECIPE['test_end'])
    eval_days = days[selected]
    expected = np.flatnonzero((panel.days >= RECIPE['test_start']) & (panel.days <= RECIPE['test_end']))
    if not np.array_equal(eval_days, expected):
        raise ValueError('打分未完整覆盖固定四季度评价窗')
    return pred[selected], eval_days


def evaluation_metrics(ev, panel, eval_days):
    """评价窗的逐日序列与四指标行（口径见 §②b），供回测与 picks 共用。"""
    daily_ret, daily_ic = top_portfolio_daily(ev, panel, eval_days)
    rows = []
    for q in [*quarters(panel.days), '总计']:
        loc = (np.ones(len(eval_days), dtype=bool) if q == '总计'
               else np.array([str(pd.Period(panel.days[d], freq='Q')) == q for d in eval_days]))
        rows.append(dict(quarter=q, days=int(loc.sum()), **top_metrics(daily_ret[loc], daily_ic[loc])))
    return dict(rows=rows, daily_ret=daily_ret, daily_ic=daily_ic)

# =============================================================================
# ③ 模型质量与固定策略分开评价
# =============================================================================


# =============================================================================
# ④ 费用与现金回测
# =============================================================================


def fees(notional, sell=False):
    """原参考工程/项目约定的固定费率；这是研究假设，不是历史费率复原。"""
    return max(notional * .00025, 5) + notional * .00001 + (notional * .0005 if sell else 0)


# 本单元只有一个固定交易策略及一个每日换手基线。其他策略研究使用导出的打分。
STRATEGIES = [
    dict(name='top5_5d', topn=5, period=5),
    dict(name='top1_1d', topn=1, period=1),
]


def cash_backtest(pred, panel, px, days, n=5, period=5):
    """固定周期TopN现金账户：T日排序，T+1开盘执行，保留费用、拒单和资金约束。

    每period日重选目标；目标相同则继续持有。仅服务本单元的两种固定策略。
    """
    cash = 100000.
    holdings = {}
    buys = {}
    desired = []
    curves = []
    trades = []
    corp_events = 0
    first, last = int(days[0]) + 1, int(days[-1]) + 2
    pred_by_day = {int(d): p for d, p in zip(days, pred)}
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
            if (signal - int(days[0])) % period == 0:
                desired = [int(c) for c in ranked if px.entry[d, c]][:n]
        if d == last:
            desired = []                 # 末尾清仓，把持仓换成现金后再估值

        # --- 先卖：不在目标里就卖；卖不出去或当日刚买的顺延到下一个开盘 ---
        blocked = []
        for c in list(holdings):
            if c in desired:
                continue
            if not px.exit[d, c] or buys[c] >= d:
                blocked.append(c)
                continue
            quantity = holdings[c]
            price = px.open[d, c] * (1 - .0003)
            gross = quantity * price
            fee = fees(gross, True)
            cash += gross - fee
            trades.append(dict(date=str(panel.days[d]), code=str(panel.codes[c]), side='sell',
                               quantity=quantity, price=float(price), fee=float(fee)))
            del holdings[c]
            del buys[c]

        # --- 再买：按 1/N 预算、100 股整手，单笔不超过信号日成交额的 1% ---
        # 只在固定调仓日买入。
        if p is not None and d < last and (signal - int(days[0])) % period == 0:
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
    return dict(topn=n, rebalance_days=period,
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
        target = output_file(folder / 'data.parquet')
        temp = folder / 'data.tmp.parquet'
        # 新单元第一次运行时年度目录还不存在（建目录的 mkdir 在回测那一步，比这里晚）。
        folder.mkdir(parents=True, exist_ok=True)
        df.to_parquet(temp, index=False)
        os.replace(temp, target)

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
    """独立单元仅展示固定策略和Top1日换手基线的含费收益。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = out / source / 'CSV'
    plots = out
    plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.6), layout='constrained')
    for spec, color in zip(STRATEGIES, ['#17917e', '#c33f3f']):
        frame = pd.read_csv(folder / f'cash_{spec["name"]}.csv')
        equity = np.r_[100000., frame.equity.to_numpy()]
        dates = pd.to_datetime([first_day] + frame.date.tolist())
        label = ('Top5 | rebalance every 5 days' if spec['name'] == 'top5_5d'
                 else 'Top1 | daily baseline')
        ax.plot(dates, 100 * (equity / 100000 - 1), color=color, linewidth=2,
                label=f'{label} | {100*(equity[-1]/100000-1):+.2f}%')
    ax.set_title('Baseline model | Fixed strategy vs daily baseline', fontsize=14)
    ax.set_ylabel('Cumulative net return (%)')
    ax.set_xlabel('Execution / valuation date | fees and slippage included')
    ax.axhline(0, color='#555555', linewidth=.7)
    ax.grid(alpha=.2)
    ax.legend(loc='upper left')
    ticks = ['2025-07-01', '2025-10-01', '2026-01-01', '2026-04-01', '2026-07-02']
    ax.set_xticks(pd.to_datetime(ticks), labels=ticks)
    fig.savefig(output_file(plots / f'{source}_curves.png'), dpi=160)
    plt.close(fig)

# =============================================================================
# ⑧ 全流程调度器
# =============================================================================


def pending_folds(device):
    """检查固定目录中的已完成模型，返回尚待训练的季度折。

    已完成的折要同时过「产物齐全」和「指纹一致」两关：只判文件存在的话，另一版模型
    留下的 complete.json 会让这一折被静默跳过，训练和打分都停在旧口径上。
    """
    fp = fingerprint(RECIPE)
    tasks = []
    for q in quarters(axis()[0]):
        for f in range(1, 5):
            folder = RUN_ROOT / 'model_train' / q / f'fold{f}'
            if (folder / 'complete.json').exists():
                if not all((folder / n).is_file() for n in ['best.pt', 'test_predictions.npy', 'score_predictions.npy']):
                    raise RuntimeError(f'{q}/fold{f} 已完成记录的产物不完整')
                if json.loads((folder / 'complete.json').read_text()).get('fingerprint') != fp:
                    raise RuntimeError(f'{q}/fold{f} 的产物属于另一个口径（指纹不符），'
                                       '删除该折产物后重训，不能静默复用')
            else:
                tasks.append((q, f))
    return tasks


def run_pipeline(jobs=2, device='cpu'):
    """本单元内调度全部16个季度折、记录每个任务用时、跑完整评价。

    只记**本单元自己的**事实（子进程 PID、起止时间、退出码、并发度、CPU 亲和）。
    不探测宿主状态：独立单元除 `trainingdata` 外不依赖外部环境，`/sys`、`/proc`、
    `nvidia-smi` 这类读数一律不碰 —— 它们既不是本单元的数据契约，还会因为换机器
    （如 cgroup v1/v2 差异、共享宿主的别家进程）让整条流水线在中途或起步时倒掉。
    """
    logs = RUN_ROOT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log_lines = []
    def note(text):
        """流水线自己的记录：既打到 stdout，也留一份到 model_info/pipeline.log。"""
        print(text, flush=True)
        log_lines.append(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {text}')
    tasks = pending_folds(device)
    all_tasks = [(q, f) for q in quarters(axis()[0]) for f in range(1, 5)]
    reused = [t for t in all_tasks if t not in tasks]
    # 复用的折也要留下属于自己的日志：固定文件树里每折都有 logs/<季度>_<折>.log，
    # 不能因为「没重新训练」就让这个文件缺席。
    for q, f in reused:
        with open(output_file(logs/f'{q}_{f}.log'), 'a', buffering=1) as handle:
            handle.write(f'{q} fold{f} 已完成且指纹一致，本次复用，未重新训练\n')
    launch_path = info_root() / 'launch.json'
    # 新单元第一次运行时 launch.json 还不存在（写入就在下面几行）。
    launch_info = json.loads(launch_path.read_text()) if launch_path.is_file() else {}
    launch_info.update(run_name=run_name(), layout='fixed-unit-v1',
                       output_root=str(RUN_ROOT), feature_block=feature_block())
    atomic_json(launch_path, launch_info)
    inputs = inspect_inputs()
    layout = validate_layout(allow_temporary=True, allow_missing=True)
    atomic_json(info_root()/'preflight.json', dict(
        checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), unit=UNIT, run_name=run_name(),
        data_root=inputs['data_root'], feature_block=inputs['feature_block'],
        feature_count=len(inputs['features']), market_count=len(inputs['market_features']),
        input_files=len(inputs['files']), epochs=RECIPE['max_epochs'],
        quarters=quarters(axis()[0]), folds=list(range(1, 5)),
        pending_folds=len(tasks), reused_folds=len(reused),
        layout_files=layout['files'], layout_missing=len(layout['missing'])))
    cap = min(jobs, max(1, len(tasks)))
    pending=tasks.copy();active={};records=[];analysis_started=False;failure=None;max_active=0
    aff=sorted(os.sched_getaffinity(0));cores_per_job=max(1,min(int(os.environ.get('MX_THREADS','2')),len(aff)//cap))
    slots=list(range(cap)); completed=0
    note(f'特征块={RECIPE["feature_block"]}；每折{RECIPE["max_epochs"]}轮；四季度×四折；'
         f'并发{cap}；本次复用{len(reused)}折、待训{len(tasks)}折')
    def launch(task, command, slot):
        handle=open(output_file(logs/('_'.join(map(str,task))+'.log')),'a',buffering=1)
        try:
            child=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
            if task[0]!='analysis':os.sched_setaffinity(child.pid,aff[slot*cores_per_job:(slot+1)*cores_per_job])
        except BaseException:
            handle.close();raise
        active[task]=dict(child=child,handle=handle,slot=slot,started=time.time())
        note(f'启动 {task} PID={child.pid}')
    try:
        while True:
            while pending and len(active)<cap:
                q,f=pending.pop(0);slot=slots.pop(0)
                launch((q,f),[sys.executable,'-B','-u',str(ROOT/'model.py'),'--quarter',q,'--fold',str(f),'--device',device],slot)
            max_active=max(max_active,len(active))
            for task,run in list(active.items()):
                code=run['child'].poll()
                if code is None:continue
                run['handle'].write(f'\nEXIT:{code}\n');run['handle'].close()
                del active[task];slots.append(run['slot'])
                records.append(dict(task='_'.join(map(str,task)),quarter=task[0],fold=task[1],
                                    pid=run['child'].pid,seconds=round(time.time()-run['started'],2),exit_code=code))
                if code:raise RuntimeError(f'{task} 失败，退出码 {code}，见本次logs')
                if task[0]!='analysis':completed+=1
                note(f'完成 {task}；本次训练完成 {completed}/{len(tasks)}')
            if not active and not pending:
                if analysis_started:break
                launch(('analysis',0),[sys.executable,'-B','-u',str(ROOT/'analysis.py'),'--audit'],slots.pop(0))
                analysis_started=True
            time.sleep(2)
    except BaseException as exc:
        failure=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        for run in active.values():
            if run['child'].poll() is None:run['child'].terminate()
        for run in active.values():
            try:run['child'].wait(timeout=10)
            except subprocess.TimeoutExpired:run['child'].kill();run['child'].wait()
            run['handle'].close()
        wall = round(time.time()-started,2)
        result=dict(exit_code=1 if failure else 0,failure=failure,wall_seconds=wall,
                    tasks_total=len(tasks),tasks_completed=completed,reused_folds=len(reused),
                    max_parallel=cap,worker_cpu_affinity=cores_per_job,newly_completed_models=completed,
                    seconds_by_task={r['task']:r['seconds'] for r in records})
        atomic_json(logs/'resource_summary.json',result)
        pd.DataFrame(records,columns=['task','quarter','fold','pid','seconds','exit_code']).to_csv(
            output_file(logs/'resource_usage.csv'),index=False)
        atomic_json(info_root()/'finished.json', dict(
            unit=UNIT, run_name=run_name(), started_at=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started)),
            finished_at=time.strftime('%Y-%m-%d %H:%M:%S'), wall_seconds=wall,
            exit_code=1 if failure else 0, failure=failure, device=device, epochs=RECIPE['max_epochs'],
            pending_folds=len(tasks), newly_completed_models=completed, reused_folds=len(reused),
            max_parallel=cap, seconds_by_task=result['seconds_by_task']))
        note(f'{"FAILED" if failure else "DONE"} wall={wall:.1f}s 待训{len(tasks)}折/复用{len(reused)}折')
        output_file(info_root()/'pipeline.log').write_text('\n'.join(log_lines)+'\n', encoding='utf-8')

# =============================================================================
# ⑧b 表目录说明（model_pred/tables/README.md）
#
# 用户在 2026-09-24 定：docs 下的两份 Markdown 报告（_REPORT.md / _picks.md）
# 以后不再生成，相关的报告生成代码与 run.py 的 report 命令已删除。
# 脚本只落 CSV / JSON / 图；实验结果的叙述由 Agent 写进 docs/iterations.md。
# 单元内这份表目录说明属于固定产物，继续原样维护。
# =============================================================================


def markdown_table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)


def write_tables_guide(out):
    text = '''# 固定策略与基线的结果

本单元只计算四折集成的两种策略，不计算其他策略组合或逐折策略矩阵。

- 固定策略top5_5d：买入Top5，每5个交易日按排名重新选股；仍在目标中的股票继续持有。
- 对照基线top1_1d：每日按排名选择Top1，下一交易日开盘执行；相同目标继续持有。

| 文件 | 内容 |
| --- | --- |
| [strategy_summary.csv](strategy_summary.csv) | 2行：两种策略完整四季度的含费收益、回撤、夏普、费用 |
| [quarterly_summary.csv](quarterly_summary.csv) | 8行：两种策略各4个季度，资金连续 |
| [main_metrics_1d.csv](main_metrics_1d.csv) | 四个指标，4个季度及总计；IC是逐日截面Pearson，ICIR与top_return_stability未年化，top_return是日均收益(%) |
| [report.json](report.json) | 本次两策略评价、四指标的逐日序列、数据来源和训练阶段的核验记录 |

return_value和max_drawdown用小数保存，0.1245表示12.45%。季度收益需要复合，不能直接相加。

四个指标（IC / ICIR / top_return / top_return_stability）的口径出处是`tmp/min_test.ipynb`：
IC = 逐日截面Pearson(打分, 1日标签)的日等权平均，ICIR = IC均值÷IC标准差；
top_return = 逐日调仓top500组合的日收益均值(%)，top_return_stability = 其均值÷标准差。
组合按打分降序遍历、每只建仓不超过当日成交额、资金15亿元、不计费用，也不套涨跌停/停牌掩码——
是无摩擦的信号口径，与上面两套现金账户策略不是一回事。
季度行就是季度表现表：`top_return`列即季度平均收益(%)，`IC`列即季度平均IC；
总计行的四个数写进`docs/iterations.md`。注意IC是Pearson，与训练侧和report.json里的RankIC不同源。

逐日资金与成交只有两组，位于../ensemble/CSV/。[对比图](../ensemble_curves.png)只有两条含费收益曲线；[最近10个交易日排名](../picks.md)直接放在model_pred根目录，**文件开头先给本版的四指标与季度表现**（版式同`tmp/min_test.ipynb`）。每个交易日一节：标题是「<信号日> 收盘」，第一行写明这个打分预测的是几日收益率、以及具体到日期的买卖时点，再列前10名的`排名/代码/名称/打分`。名称来自只读对照表、买卖日来自交易日历，都只做展示，不参与排序与任何指标。

模型打分保存在../ensemble/year=年份/data.parquet，包含trade_date、stock_code、value、rank；score_meta.json记录日期范围、训练参数和输入位置。value是四折分数相加，rank是当日分位排名，分数越大越靠前。

其他策略研究应在独立研究单元中读取这些打分和价格/成交额数据。无需重新训练，也不向本单元写入额外策略结果。
'''
    output_file(Path(out) / 'tables/README.md').write_text(text, encoding='utf-8')


# =============================================================================
# ⑧b 最新一日推荐表（给实战 / 增量推演用，不参与任何评价指标）
# =============================================================================


def stock_names(codes):
    """股票代码 → 中文名。查不到、表读不到都返回空串（只影响显示，不影响任何数值）。

    外部读取之一（用户 2026-09-25 授权）：`datadownload/data/stock_list/data.parquet`
    的 `stock_code` + `name` 两列。退市/更名的票查不到就留空，不报错——picks.md 是常驻
    产物，不能因为一张对照表而缺席。
    """
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
    """`label_ret_5d` → 5。"""
    return int(label.split('_ret_')[1].rstrip('d'))


def trade_axis(panel):
    """信号日 → 交易日轴：快照内用 `panel.days`，快照之后接外部交易日历。

    标签口径是 `px[T+1+h] / px[T+1] - 1`，所以「T+1 开盘买入、T+h+1 开盘卖出」里的日期
    必须落在真实交易日上。快照内直接用 `panel.days`（与标签同源）；快照之后的日期快照里
    没有、但交易所有（日历含未来交易日），否则最新一天的买卖日只能写成「次日」这种
    相对说法。日历读不到就降级为相对说法，不中断。
    """
    days = [str(d) for d in panel.days]
    try:
        frame = pd.read_parquet(calendar_path(), columns=['date', 'is_open'])
    except (OSError, ValueError, KeyError) as exc:
        print(f'警告：读不到交易日历 {calendar_path()}（{type(exc).__name__}: {exc}），'
              'picks 的买卖日改用相对说法', flush=True)
        return days
    future = sorted(str(d) for d in frame[frame.is_open == 1].date if str(d) > days[-1])
    return days + future


def write_daily_picks(pred, panel, days, metrics=None):
    """最近10个交易日、每天前 `RECIPE['picks_topn']` 名的打分排名。

    版式（用户 2026-09-25 定）：每日标题是「<信号日> 收盘」，正文第一行写清这个打分预测
    的是什么、以及**具体到日期的**买卖时点，再列 排名/代码/名称/打分 四列。
    文件开头先放本次评价窗的四指标与季度表现（用户 2026-09-24 定，版式同 min_test）。
    回测路径已算好指标，用 metrics 传入即可，不重算。

    打分口径沿用 `RECIPE['score_label']`（5d 头），与集成打分、四指标、固定策略同源；
    名称与买卖日期只做展示，不参与排序或任何指标。
    """
    if len(days) == 0:
        raise ValueError('没有可展示的打分日期')
    if pred.shape != (len(days), len(panel.codes)):
        raise ValueError('打分维度与日期、股票轴不一致')
    out = RUN_ROOT / 'model_pred'
    out.mkdir(parents=True, exist_ok=True)
    if metrics is None:
        ev, eval_days = evaluation_slice(pred, panel, days)
        metrics = evaluation_metrics(ev, panel, eval_days)['rows']
    names = stock_names(panel.codes)
    axis = trade_axis(panel)
    position = {d: i for i, d in enumerate(axis)}
    horizon = horizon_of(RECIPE['score_label'])
    holding, topn = horizon + 1, RECIPE['picks_topn']
    first = max(0, len(days) - 10)
    count = len(days) - first
    text = ('```text\n' + metric_blocks_text(metrics) + '```\n\n'
            f'上表是评价窗**{RECIPE["test_start"]}～{RECIPE["test_end"]}**的四指标与季度表现'
            '（口径见[tables/README.md](tables/README.md)）；下面是最近10个交易日的打分排名。\n\n'
            f'# 最近10个交易日集成打分排名\n\n'
            f'数据截至：**{panel.days[int(days[-1])]} 收盘**。'
            f'本页展示 **{panel.days[int(days[first])]} ～ {panel.days[int(days[-1])]}**，'
            f'共 **{count} 个交易日**，按日期从新到旧排列。\n\n'
            f'打分是四折直接相加的`{RECIPE["score_label"]}`预测值，用于排序，不是收益百分比。\n\n'
            f'每天列前 **{topn}** 名。前 5 名是 Top5固定策略（每 5 个交易日调仓）的候选目标，'
            '第 1 名同时是 Top1基线（每日调仓）候选；实际交易仍按各自调仓日、'
            '下一交易日的可成交条件和资金约束执行。\n\n'
            f'名称来自只读对照表（`{stock_list_path()}`），买卖日来自交易日历'
            f'（`{calendar_path()}`）；两者查不到的票/日期留空或降级为相对说法，'
            '都不参与排序与任何指标。\n\n')
    for index in range(len(days) - 1, first - 1, -1):
        signal_day = str(panel.days[int(days[index])])
        score = pred[index]
        candidates = np.flatnonzero(np.isfinite(score))
        picks = candidates[np.argsort(-score[candidates], kind='stable')][:topn]
        at = position.get(signal_day)
        buy = axis[at + 1] if at is not None and at + 1 < len(axis) else None
        sell = axis[at + holding] if at is not None and at + holding < len(axis) else None
        text += f'## {signal_day} 收盘\n\n'
        text += (f'模型预测 {horizon} 日收益率：**{buy} 开盘买入 → {sell} 开盘卖出**\n\n'
                 if buy and sell else
                 f'模型预测 {horizon} 日收益率：次一交易日（T+1）开盘买入 → '
                 f'第 {holding} 个交易日（T+{holding}）开盘卖出'
                 '（买卖日超出已知交易日历，故用相对说法）\n\n')
        if not len(picks):
            text += '当日没有有效打分。\n\n'
            continue
        rows = [[i, str(panel.codes[c]).split('.')[0], names.get(str(panel.codes[c]), ''),
                 f'{float(score[c]):.4f}'] for i, c in enumerate(picks, 1)]
        text += markdown_table(['排名', '代码', '名称', '打分'], rows) + '\n'
    output_file(out / 'picks.md').write_text(text, encoding='utf-8')
    print('最近10个交易日排名已生成:', out / 'picks.md', flush=True)


# =============================================================================
# ⑨ 命令行入口
# =============================================================================


def save_score_metadata(folder, report, dates):
    metadata = {k:v for k,v in report.items() if k not in ('variants','evaluation_mode')}
    metadata['score_dates'] = list(dates)
    metadata['columns'] = ['trade_date','stock_code','value','rank']
    atomic_json(folder/'score_meta.json',metadata)


def read_saved_scores(folder, panel):
    """独立读集成打分；不访问model_train或模型权重。"""
    meta = json.loads((folder/'score_meta.json').read_text())
    dates = meta['score_dates']
    if dates != sorted(set(dates)):
        raise ValueError('打分日期重复或乱序')
    date_axis = pd.Index(dates)
    codes = pd.Index(panel.codes)
    days = pd.Index(panel.days).get_indexer(dates)
    if (days<0).any():raise ValueError('打分日期不在当前价格数据中')
    pred = np.full((len(dates),len(codes)),np.nan,np.float32)
    files = sorted(folder.glob('year=*/data.parquet'))
    if not files:
        raise FileNotFoundError(f'{folder}: 缺年度打分文件')
    seen = set()
    for path in files:
        frame = pd.read_parquet(path,columns=['trade_date','stock_code','value'])
        if frame.duplicated(['trade_date','stock_code']).any():raise ValueError('打分主键重复')
        t=date_axis.get_indexer(frame.trade_date.astype(str));c=codes.get_indexer(frame.stock_code.astype(str))
        if (t<0).any() or (c<0).any():raise ValueError('打分坐标不匹配')
        keys=set(zip(t.tolist(),c.tolist()))
        if seen.intersection(keys):raise ValueError('年度文件含重复打分')
        seen.update(keys)
        values=frame.value.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():raise ValueError('导出打分含无效值')
        pred[t,c]=values
    return pred,days,meta


def evaluate_fixed_strategies(pred, panel, px, days, report, mode):
    out=RUN_ROOT/'model_pred';tb=out/'tables';csv_root=out/'ensemble/CSV'
    tb.mkdir(parents=True,exist_ok=True);csv_root.mkdir(parents=True,exist_ok=True)
    ev,eval_days=evaluation_slice(pred,panel,days)
    report={k:v for k,v in report.items() if k not in ('score_dates','columns','variants')}
    report['evaluation_mode']=mode
    report['evaluation_window']=[str(panel.days[eval_days[0]]),str(panel.days[eval_days[-1]])]
    result={'IC':ic_table(ev,panel,eval_days),'strategies':[]}
    summaries=[];quarterly=[]
    if px is not None:
        report['label_price_check']=px.verify_labels(panel,days)
        for spec in STRATEGIES:
            stat,curve,trades=cash_backtest(ev,panel,px,eval_days,n=spec['topn'],period=spec['period'])
            result['strategies'].append(dict(strategy=spec['name'],**stat))
            summaries.append(dict(source='ensemble',strategy=spec['name'],**stat))
            quarterly.extend(quarter_metrics('ensemble',spec['name'],curve,trades,ev,panel,eval_days))
            pd.DataFrame(curve).to_csv(output_file(csv_root/f'cash_{spec["name"]}.csv'),index=False)
            pd.DataFrame(trades).to_csv(output_file(csv_root/f'trades_{spec["name"]}.csv'),index=False)
    report['variants']={'ensemble':result}
    # --- 四指标与季度表现：评价窗 242 天的逐日调仓组合（口径见 §②b）---
    # 这一块不依赖价格层：只读标签与成交额，所以 --no-backtest 也能出。
    measured=evaluation_metrics(ev,panel,eval_days)
    metrics=measured['rows']
    pd.DataFrame(metrics).to_csv(output_file(tb/'main_metrics_1d.csv'),index=False)
    # 逐日序列随报告一并留档，便于复核四指标之外的时点（不另建文件）。
    result['eval_portfolio']=dict(params=TOP_PORTFOLIO,total=metrics[-1],
        daily=dict(date=[str(panel.days[d]) for d in eval_days],
                   return_pct=[round(float(x),6) for x in measured['daily_ret']],
                   ic=[round(float(x),6) for x in measured['daily_ic']]))
    print_metric_blocks(metrics)
    write_daily_picks(pred,panel,days,metrics)
    if px is None:
        # 首次 --no-backtest 时 report.json 还不存在（全量路径才写它）。
        combined = json.loads((tb/'report.json').read_text()) if (tb/'report.json').is_file() else {}
        combined['ic_only'] = report
        atomic_json(tb/'report.json', combined)
        return report
    atomic_json(tb/'report.json',report)
    pd.DataFrame(summaries).to_csv(output_file(tb/'strategy_summary.csv'),index=False)
    pd.DataFrame(quarterly).to_csv(output_file(tb/'quarterly_summary.csv'),index=False)
    plot_results(out,'ensemble',report['evaluation_window'][0])
    write_tables_guide(out)
    # 两个核验记录：推理对拍的逐折结果，以及两策略口径的自检。
    # `--from-scores` 不加载权重，此时 report 里带的 audit 是上一次训练运行的存档，
    # 不能当成本次验过，所以按来源显式区分。
    from_saved = mode == 'saved_ensemble_scores'
    audit = [] if from_saved else (report.get('audit') or [])
    atomic_json(info_root()/'final_audit.json', dict(
        run_name=run_name(), checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), score_source=mode,
        evaluation_window=report['evaluation_window'], score_window=report.get('score_window'),
        model_train_read=bool(audit), weights_reloaded=bool(audit), folds_checked=len(audit),
        max_cpu_reload_error=max([a['max_cpu_reload_error'] for a in audit], default=None),
        audit=audit,
        metrics_total=metrics[-1]))
    # 每一项都是本次真正量到的事实，不写没验过的断言。
    atomic_json(info_root()/'two_strategy_validation.json', dict(
        run_name=run_name(), checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), score_source=mode,
        reads_model_train=not from_saved, loads_model_weights=not from_saved,
        strategies=[s['name'] for s in STRATEGIES], strategy_rows=len(summaries),
        quarterly_rows=len(quarterly), evaluation_days=int(len(measured['daily_ret'])),
        evaluation_window=report['evaluation_window'], label_price_check=report.get('label_price_check'),
        # quarter_metrics 里已经断言「四段季度收益能复合回全年」，跑到这里就是通过。
        quarterly_returns_compound=True,
        date_axis_matches_price_axis=True,
        realized={s['strategy']: dict(return_value=s['return_value'], max_drawdown=s['max_drawdown'],
                                      sharpe=s['sharpe'], trades=s['trades'], total_fees=s['total_fees'])
                  for s in summaries}))
    print('两策略回测完成:',json.dumps(summaries,ensure_ascii=False),flush=True)
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', action='store_true', help='训练调度、任务用时记录、完整回测')
    parser.add_argument('--jobs', type=int, choices=range(1, 9), default=8, help='pipeline 的并发任务上限')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--from-scores', action='store_true', help='直接读取集成打分，只计算本单元两策略')
    parser.add_argument('--no-backtest', action='store_true', help='明确只计算 IC；不输出现金策略')
    parser.add_argument('--audit', action='store_true', help='最佳模型逐日截断/重载推理对拍')
    parser.add_argument('--picks-only', action='store_true',
                        help='只出最近10个交易日排名（不跑评价/回测），给增量推演用')
    args = parser.parse_args()
    # 新单元的常驻产物由运行生成，所以起点允许缺失；多余路径仍然一律拒绝。
    validate_layout(allow_temporary=True, allow_missing=True)
    # 进程内线程数与模型侧用同一个旋钮（train.sh 导出 MX_THREADS），不写死主机假设。
    torch.set_num_threads(int(os.environ.get('MX_THREADS', '3')))

    if args.pipeline:
        return run_pipeline(args.jobs, args.device)

    if args.from_scores and args.audit:
        parser.error('--from-scores不加载权重，不能同时使用--audit')
    panel = Panel(RECIPE['features'], load_x=args.audit)
    out = RUN_ROOT / 'model_pred'
    out.mkdir(parents=True, exist_ok=True)
    if args.from_scores:
        pred,days,report=read_saved_scores(out/'ensemble',panel)
        if args.picks_only:return write_daily_picks(pred,panel,days)
        px=None if args.no_backtest else Prices(panel)
        return evaluate_fixed_strategies(pred,panel,px,days,report,'saved_ensemble_scores')

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
            folder = INPUT_ROOT / 'model_train' / quarter / f'fold{fold}'
            done = json.loads((folder / 'complete.json').read_text())
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
                net = PredictModel(len(panel.features), panel.market_dim)
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
                # 多头输出必须显式取打分头：.squeeze(-1) 在 (n,5) 上是 no-op，会和
                # (k,) 的 actual 靠广播比较而假通过。
                score_head = head_of(RECIPE['score_label'])
                for j, d in enumerate(check_days):
                    mask = panel.mask(d)
                    with torch.no_grad():
                        m = torch.from_numpy(panel.M[d]) if panel.market_dim else None
                        one = net(torch.from_numpy(panel.X[d, mask][:10]), m)[:, score_head].numpy()
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

    # --- 集成：四折打分**直接相加**，不做逐日标准化（用户 2026-09-20）---
    # 用 NaN 传播的加法：某票某天没有分数时，集成结果仍是"没有分数"，
    # 不会变成 0 分混进排序（np.nansum 会犯这个错）。
    if len(variants) > 1:
        variants['ensemble'] = sum(variants.values()).astype(np.float32)

    pred=variants['ensemble']
    if args.picks_only:return write_daily_picks(pred,panel,days)
    report = dict(evaluation_window=[RECIPE["test_start"], RECIPE["test_end"]],
                  score_window=[dates[0], dates[-1]],
                  folds=completions, analysis_mode='IC_only' if args.no_backtest else 'full',
                  variants={}, audit=audit,
                  ensemble='四折打分直接相加（不归一化）',
                  inference_rule='每天由「最新一个没见过它的季度模型组」推演；季度之后的尾段由最后一个季度那组给',
                  caveats=['固定存续且从未ST股票池含幸存者偏差', '四折共用测试窗，不是四段独立样本外收益',
                           '验证收益是5日标签代理；策略净值另算', '不据测试结果调参或晋级best',
                           '评价只用2025Q3～2026Q2四季度；之后的尾段仅供增量推演'])
    first=completions[0]
    info=json.loads((INPUT_ROOT/'model_train'/first['quarter']/f'fold{first["fold"]}'/'training_info.json').read_text())
    report['training']=dict(recipe=info['recipe'],feature_count=len(info['feature_columns']),feature_source=info['feature_source'])
    write_scores(pred,panel,days,out/'ensemble')
    px=None if args.no_backtest else Prices(panel)
    result=evaluate_fixed_strategies(pred,panel,px,days,report,'training_predictions')
    save_score_metadata(out/'ensemble',result,dates)


if __name__ == '__main__':
    main()
