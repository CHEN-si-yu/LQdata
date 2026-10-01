"""Read-only baseline inference on the exact frozen snapshot used by new experiments."""
from pathlib import Path
import gc
import hashlib
import inspect
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT = Path('/root/autodl-fs/model')
AUDIT = PROJECT / 'history/diversification_20260930'
OUT = AUDIT / 'clean_references'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
rows = []


def account_row(stat, curve, trades):
    money = 100000.0
    equity = np.asarray([c['equity'] for c in curve])
    daily = equity / np.r_[money, equity[:-1]] - 1.0
    # The actual shares/trade dates are identical in the counterfactual; freed costs remain cash.
    slip = sum(float(t['quantity'] * t['price'] * (0.0003 / (1.0003 if t['side'] == 'buy' else 0.9997))) for t in trades)
    fee = sum(float(t['fee']) for t in trades)
    return dict(net_return=stat['return_value'], max_drawdown=stat['max_drawdown'],
                sharpe=stat['sharpe'], trades=stat['trades'], fees=fee,
                slippage=slip, gross_return_same_positions=stat['return_value']+(fee+slip)/money,
                net_arithmetic_annual_pct=float(daily.mean()*242*100),
                net_geometric_annual_pct=float(((1+stat['return_value'])**(242/len(curve))-1)*100),
                gross_definition='same actual shares and trade dates; costs added back as idle cash',
                execution_start=curve[0]['date'], execution_end=curve[-1]['date'])


if __name__ == '__main__':
    for unit in ['V31', 'V36', 'V38']:
        if (AUDIT / 'STOP').exists():
            raise SystemExit('stopped by user')
        d = PROJECT / 'experiments' / unit
        for name in ['model', 'analysis']:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(d))
        import model as m
        import analysis as a
        print('REFERENCE_LOAD', unit, flush=True)
        panel = m.Panel(m.RECIPE['features'])
        prices = m.Prices(panel)
        for block in a.lookahead_free_blocks(panel.days):
            if (AUDIT / 'STOP').exists():
                raise SystemExit('stopped by user')
            q, f, va = block['quarter'], block['fold'], block['valid']
            cp = d / 'model_train' / q / f'fold{f}' / 'best.pt'
            ck = torch.load(cp, map_location='cpu', weights_only=False)
            assert ck['features'] == panel.features, 'checkpoint feature order changed'
            net = m.PredictModel(len(panel.features), panel.market_dim)
            net.load_state_dict(ck['model'])
            net.ridge = ck.get('ridge')
            scores = m.predict(net, panel, prices, va, torch.device('cpu')) if len(inspect.signature(m.predict).parameters) == 5 else m.predict(net, panel, va, torch.device('cpu'))
            np.savez_compressed(OUT / f'{unit}_{q}_fold{f}.npz', scores=scores, days=va, dates=panel.days[va])
            entry = dict(unit=unit, quarter=q, fold=f, start=str(panel.days[va[0]]),
                         end=str(panel.days[va[-1]]), days=len(va),
                         checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
                         snapshot=str(panel.root), accounts={})
            for period in [1, 5]:
                st, curve, trades = a.cash_backtest(scores, panel, prices, va, n=5, period=period, band=40)
                entry['accounts'][f'top5_{period}d'] = account_row(st, curve, trades)
                np.savez_compressed(OUT / f'{unit}_{q}_cash_{period}d.npz',
                                    dates=np.asarray([c['date'] for c in curve]),
                                    equity=np.asarray([c['equity'] for c in curve]))
            rows.append(entry)
            (AUDIT / 'clean_reference_results.json').write_text(json.dumps(dict(
                checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), completed_blocks=len(rows),
                rows=rows, note='existing fixed weights, current frozen snapshot; validation blocks are diagnostics, not independent tests; actual net returns without double fee deduction'), ensure_ascii=False, indent=2))
            print('REFERENCE_BLOCK', unit, q, entry['accounts'], flush=True)
        sys.path.pop(0)
        del panel, prices, net, scores, ck, m, a
        gc.collect()
    print('REFERENCES_COMPLETE', flush=True)
