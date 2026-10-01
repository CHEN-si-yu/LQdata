"""Frozen, validation-only rank fusion with a separate absolute-return gate."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import torch
from scipy.stats import rankdata

ROOT = Path('/root/autodl-fs/model')
A = ROOT/'history/flow_absolute_fusion_20260930'
PRIMARY = ROOT/'history/diversification_20260930'
ABSOLUTE = ROOT/'history/absolute_ridge_20260930'
PLAN = json.loads((A/'protocol.json').read_text())
args = argparse.ArgumentParser()
args.add_argument('--worker', type=int, required=True)
worker = args.parse_args().worker
assert worker in (0, 1)
os.nice(12)
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
torch.set_num_threads(1)
sys.path.insert(0, str(ROOT/'experiments/V62'))
import model as m
import analysis as a


def stopped():
    return (A/'STOP').exists() or (PRIMARY/'STOP').exists()


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def percentile(scores):
    out = np.full(scores.shape, np.nan, dtype=np.float64)
    for i, row in enumerate(scores):
        known = np.isfinite(row)
        if known.sum() > 1:
            out[i, known] = (rankdata(row[known])-.5)/known.sum()
    return out


# Only the research ranking/admission choice changes; cash/fees/fills remain native.
source = inspect.getsource(a.cash_backtest)
assert source.count('cost_gate_multiplier=None):') == 1
source = source.replace('cost_gate_multiplier=None):',
    'cost_gate_multiplier=None, gate_scores=None, phase=0):', 1)
source = source.replace('    money = float(',
    "    if period<1 or not 0<=phase<period:raise ValueError('invalid phase')\n    money = float(", 1)
source = source.replace('% period == 0', '% period == phase')
source = source.replace('    spec_band = band',
    '    gate_by_day = {} if gate_scores is None else {int(d): p for d,p in zip(days,gate_scores)}\n    spec_band = band', 1)
old = '''            if n>1 and gate>0:
                utility-=score_threshold
                for held in holdings:utility[held]=p[held]
                candidates=candidates[utility[candidates]>=0.]'''
new = '''            if n>1 and gate>0:
                if gate_scores is None:
                    utility-=score_threshold
                    for held in holdings:utility[held]=p[held]
                    candidates=candidates[utility[candidates]>=0.]
                else:
                    # Percentile ranks are never represented as expected returns.
                    admission=gate_by_day[signal].copy()-score_threshold
                    for held in holdings:admission[held]=gate_by_day[signal][held]
                    candidates=candidates[np.isfinite(admission[candidates]) & (admission[candidates]>=0.)]'''
assert source.count(old) == 1
source = source.replace(old, new, 1)
space = dict(a.__dict__)
exec(compile(source, 'fusion_cash_backtest', 'exec'), space)
backtest = space['cash_backtest']
panel = m.Panel(load_x=False)
px = m.Prices(panel)
assert m.data_root().resolve() == (PRIMARY/'trainingdata_frozen').resolve()
out = A/'clean_cash'
out.mkdir(exist_ok=True)
quarters = PLAN['quarters'][worker::2]
completed = []

for flow, absolute in PLAN['seed_pairs']:
    for quarter in quarters:
        if stopped():
            raise SystemExit('stopped by user')
        key = f'{flow}_{absolute}_{quarter}'
        target = out/f'{key}.json'
        if target.exists():
            completed.append(key)
            continue
        flow_path = PRIMARY/'clean_innovations'/f'{flow}_{quarter}_fold4.npz'
        absolute_path = ABSOLUTE/'clean_absolute'/f'{absolute}_{quarter}_fold4.npz'
        vf = np.load(flow_path)
        va = np.load(absolute_path)
        assert np.array_equal(vf['days'], va['days'])
        assert np.array_equal(vf['dates'], va['dates'])
        days = va['days']
        raw_absolute = va['scores']
        ranks_flow = percentile(vf['scores'])
        ranks_absolute = percentile(raw_absolute)
        split = m.splits(panel.days, quarter=quarter)[3]
        assert np.array_equal(days, split['valid'])
        assert split['train'].max()+m.RECIPE['purge_horizon']+1 < days.min()
        started = time.time()
        mixed = .5*ranks_flow+.5*ranks_absolute
        short = days[:12]
        native = a.cash_backtest(mixed[:12], panel, px, short, n=10, period=1, band=40,
            forecast_members=1, cost_gate_multiplier=0.)
        zero = backtest(mixed[:12], panel, px, short, n=10, period=1, band=40,
            forecast_members=1, cost_gate_multiplier=0., gate_scores=raw_absolute[:12])
        assert native == zero
        gated = backtest(mixed[:12], panel, px, short, n=10, period=1, band=40,
            forecast_members=1, cost_gate_multiplier=1., gate_scores=raw_absolute[:12])
        scaled = backtest(mixed[:12], panel, px, short, n=10, period=1, band=40,
            forecast_members=4, cost_gate_multiplier=1., gate_scores=raw_absolute[:12]*4)
        assert gated == scaled
        empty = backtest(mixed[:12], panel, px, short, n=10, period=1, band=40,
            forecast_members=1, cost_gate_multiplier=1., gate_scores=np.zeros_like(raw_absolute[:12]))
        assert empty[0]['trades'] == 0 and empty[0]['return_value'] == 0.
        one_native = a.cash_backtest(mixed[:12], panel, px, short, n=1, period=1, band=0,
            forecast_members=1, cost_gate_multiplier=1.)
        one_new = backtest(mixed[:12], panel, px, short, n=1, period=1, band=0,
            forecast_members=1, cost_gate_multiplier=1., gate_scores=raw_absolute[:12])
        assert one_native == one_new
        rows = []
        for weight in PLAN['flow_weight_candidates']:
            score = weight*ranks_flow+(1-weight)*ranks_absolute
            for n in PLAN['target_size_candidates']:
                for period in PLAN['period_candidates']:
                    for phase in range(period):
                        for gate in PLAN['cost_gate_candidates']:
                            if stopped():
                                raise SystemExit('stopped by user')
                            stats, _, _ = backtest(score, panel, px, days, n=n, period=period,
                                phase=phase, band=40, forecast_members=1,
                                cost_gate_multiplier=gate, gate_scores=raw_absolute)
                            assert abs(stats['gross_return_same_positions']-stats['fee_drag']-
                                stats['slippage_drag']-stats['return_value']) < 1e-10
                            rows.append(dict(flow_weight=weight, topn=n, period=period, phase=phase,
                                band=40, cost_gate_multiplier=gate, net_return=stats['return_value'],
                                max_drawdown=stats['max_drawdown'], sharpe=stats['sharpe'],
                                fee_drag=stats['fee_drag'], slippage_drag=stats['slippage_drag'],
                                gross_return_same_positions=stats['gross_return_same_positions'],
                                trades=stats['trades'], avg_exposure=stats['avg_exposure']))
        expected = len(PLAN['flow_weight_candidates'])*len(PLAN['target_size_candidates'])*sum(PLAN['period_candidates'])*len(PLAN['cost_gate_candidates'])
        assert len(rows) == expected == 180
        value = dict(flow=flow, absolute=absolute, quarter=quarter, days=len(days),
            start=str(panel.days[days[0]]), end=str(panel.days[days[-1]]), rows=rows,
            preflight=dict(zero_gate_exact_native=True, fourfold_gate_scaling_exact=True,
                nonpositive_forecast_exact_cash=True, top1_exact_native=True),
            source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [flow_path, absolute_path]},
            seconds=round(time.time()-started, 2),
            inference='same-seed clean fold4 saved predictions; within-model raw folds never normalized; cross-family percentile fusion explicit; absolute returns only determine admission; no new training; no fixed-test parameter selection')
        atomic(target, value)
        completed.append(key)
        atomic(A/f'worker{worker}_status.json', dict(worker=worker, pid=os.getpid(),
            checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), completed=completed,
            planned_blocks=8, cases_per_block=180))
        print('FUSION_BLOCK_COMPLETE', key, value['seconds'], flush=True)
print('FUSION_WORKER_COMPLETE', worker, flush=True)
