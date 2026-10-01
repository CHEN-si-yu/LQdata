import ast
import json
from pathlib import Path

BASE = Path('/root/autodl-fs/model/experiments/V38')
OUT = Path('/root/autodl-fs/model/experiments')


def replace_node(source, name, replacement):
    tree = ast.parse(source)
    for node in tree.body:
        if getattr(node, 'name', None) == name or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        ):
            lines = source.splitlines(keepends=True)
            first = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])]) - 1
            return ''.join(lines[:first]) + replacement.rstrip() + '\n' + ''.join(lines[node.end_lineno:])
    raise KeyError(name)


QUALITY_NAMES = '''ar_ap_to_revenue asset_growth_qoq bp cash_conversion_cycle cash_sales_ratio cfcr cfp_ttm dividend_yield_3y_avg dp_ttm dv_stability_4q ebitda_to_market efx_annual_earnings_yield efx_dividend_gap ep_ttm etp5 forecast_profit_midpoint_change forecast_profit_range_uncertainty forecast_type_score free_share_ratio fv_gain_share growth_stability holder_number_chg ind_bps_yoy ind_currentdebt_to_debt ind_dt_netprofit_yoy ind_int_to_talcap interest_coverage inventory_turnover invest_income_share np_to_opex_yoy ocf_to_profit ocf_to_revenue opm_npm_spread quality_composite rd_intensity revenue_cagr_3y roa_ttm roe_ttm share_issuance_yoy sp_ttm total_leverage_ratio yoy_revenue'''.split()

QUALITY_FEATURES = f'''def feature_columns(meta):
    """Frozen financial/valuation view; no price momentum, size or microstructure inputs."""
    names = set({QUALITY_NAMES!r})
    columns = [c for c in meta['columns']['features']
               if c.startswith(('afx_', 'qf_', 'bs_', 'cf_')) or c in names]
    if len(columns) < 100:
        raise ValueError('financial view unexpectedly small')
    return columns
'''

FLOW_FEATURES = '''def feature_columns(meta):
    """Frozen chip, money-flow, margin and intraday view; no annual financial fields."""
    prefixes = ('chip_', 'cyqp_', 'cost_', 'mf_', 'margin_', 'short_', 'idt_', 'id2_', 'close30_')
    names = {'big_vs_small_divergence_5d', 'fundflow_retail_inst_divergence',
             'order_size_concentration', 'small_order_crowding',
             'super_large_order_intensity', 'winner_rate_acceleration',
             'winner_rate_reversal_signal', 'dragon_tiger_org_net_20',
             'elg_net_60d_to_mv', 'top_list_net_rate_20'}
    columns = [c for c in meta['columns']['features'] if c.startswith(prefixes) or c in names]
    if len(columns) < 70:
        raise ValueError('flow view unexpectedly small')
    return columns
'''

ADDITIVE_MODEL = '''class PredictModel(nn.Module):
    """Small additive spline model with fixed causal daily standardized inputs."""
    def __init__(self, input_dim, market_dim, horizons=None):
        super().__init__()
        if market_dim:
            raise ValueError('financial additive family has no shared market gate')
        self.horizons = list(horizons or RECIPE['label_horizons'])
        self.output_layer = nn.Linear(input_dim * 4, len(self.horizons))
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, tsdata, market=None):
        x = tsdata.float().clamp(-5.0, 5.0)
        basis = torch.cat([x, torch.tanh(x), torch.relu(x - 0.75),
                           torch.relu(-x - 0.75)], dim=-1)
        return self.output_layer(basis)
'''

TEMPORAL_MODEL = '''class PredictModel(nn.Module):
    """Eight causal sessions of flow inputs; shared GRU, no common MLP/ridge/market gate."""
    def __init__(self, input_dim, market_dim, horizons=None):
        super().__init__()
        if market_dim:
            raise ValueError('flow temporal family has no shared market gate')
        self.horizons = list(horizons or RECIPE['label_horizons'])
        self.encoder = nn.Sequential(nn.Linear(input_dim, 24), nn.LayerNorm(24), nn.GELU())
        self.recurrent = nn.GRU(24, 32, batch_first=True)
        self.output_layer = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, len(self.horizons)))

    def forward(self, tsdata, market=None):
        if tsdata.ndim != 3 or tsdata.shape[1] != RECIPE['sequence_days']:
            raise ValueError('expected stock x causal-session x feature input')
        hidden, _ = self.recurrent(self.encoder(tsdata.float().clamp(-5.0, 5.0)))
        # The final hidden state only consumes dates <= the signal date.
        return self.output_layer(hidden[:, -1])
'''


def make(unit, seed, family):
    model = (BASE / 'model.py').read_text(encoding='utf8')
    analysis = (BASE / 'analysis.py').read_text(encoding='utf8')
    run = (BASE / 'run.py').read_text(encoding='utf8')
    tree = ast.parse(model)
    recipe_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'RECIPE' for t in n.targets))
    recipe = {kw.arg: ast.unparse(kw.value) for kw in recipe_node.value.keywords}
    horizons = [10, 20] if family == 'quality_additive' else [3, 5, 10]
    changes = {
        'name': repr(unit), 'architecture': repr(family), 'seed': str(seed),
        'label_horizons': repr(horizons),
        'label_weights': repr({10: 0.4, 20: 0.6} if family == 'quality_additive' else {3: 0.2, 5: 0.5, 10: 0.3}),
        'valid_ic_weights': repr({10: 0.4, 20: 0.6} if family == 'quality_additive' else {3: 0.2, 5: 0.5, 10: 0.3}),
        'score_blend': repr('head:label_ret_20d' if family == 'quality_additive' else 'head:label_ret_5d'),
        # Standard project reporting keeps the raw 1d and 5d labels for comparisons.
        'label': repr('label_ret_5d'), 'score_label': repr('label_ret_5d'),
        'market_gate': 'False', 'linear_weight': '0.0', 'train_pool_frac': '1.0',
        'lr': '0.001', 'weight_decay': '0.02', 'max_epochs': '30',
        'valid_criterion': repr('rankic'), 'valid_metric': repr('weighted raw-label RankIC of delivered score'),
        'sequence_days': '1' if family == 'quality_additive' else '8',
        'picks_holding': '5' if family == 'quality_additive' else '1',
        'family': repr(family),
        'training_target': repr('daily return percentile; smooth-L1' if family == 'quality_additive' else 'daily standardized return; WPCC'),
        'weights_origin': repr('fresh; no copied checkpoint'),
        'research_protocol': repr('four seeds 3253/4253/5253/6253; 4 quarters x 4 folds; all train-before-validation blocks; frozen configurations before test; test display only; no releases promotion'),
        'strategy_candidates': repr({'topn': [5, 10, 20], 'period': [1, 5, 10], 'band': [0, 20, 40], 'phase': 'all phases for each period'}),
        'snapshot_cutoff': repr('2026-09-29'),
        'prediction_horizon': '20' if family == 'quality_additive' else '5',
    }
    recipe.update(changes)
    model = replace_node(model, 'RECIPE', 'RECIPE = dict(\n' + ''.join(f'    {k}={v},\n' for k, v in recipe.items()) + ')\n')
    model = replace_node(model, 'feature_columns', QUALITY_FEATURES if family == 'quality_additive' else FLOW_FEATURES)
    model = replace_node(model, 'PredictModel', ADDITIVE_MODEL if family == 'quality_additive' else TEMPORAL_MODEL)
    # Reporting always measures the fixed raw 1d/5d targets, even with different trained heads.
    validation = next(n for n in ast.parse(model).body if isinstance(n, ast.FunctionDef) and n.name == 'validation')
    section = ast.get_source_segment(model, validation)
    section = section.replace("for h in RECIPE['label_horizons']:", "for h in (1, 3, 5, 10, 20):", 1)
    section = section.replace("for h in RECIPE['label_horizons']})", "for h in (1, 3, 5, 10, 20)})")
    model = replace_node(model, 'validation', section)
    model = model.replace("self.days, self.codes = axis()", "self.days, self.codes = axis()\n        if str(self.days[-1]) != RECIPE['snapshot_cutoff']:\n            raise ValueError('snapshot changed: create a new independent version')", 1)
    analysis = analysis.replace("horizon = horizon_of(RECIPE['score_label'])", "horizon = RECIPE['prediction_horizon']", 1)
    # Skip the common 5d ridge entirely, rather than keeping its style exposure in a new family.
    marker = '    D = len(panel.features)\n    G = np.zeros((D + 1, D + 1))'
    assert marker in model
    model = model.replace(marker, "    if not RECIPE['linear_weight']:\n        return None\n" + marker, 1)
    if family == 'quality_additive':
        model = replace_node(model, 'wpcc', '''def wpcc(preds, y):
    """Robust pointwise rank regression, not the historical Pearson loss."""
    return torch.nn.functional.smooth_l1_loss(preds.reshape(-1), y.reshape(-1), beta=0.25)
''')
        old = 'yt = (yt - yt.mean()) / yt.std()'
        assert model.count(old) == 1
        model = model.replace(old, "yt = torch.from_numpy(((rankdata(y) - 0.5) / len(y) * 2 - 1).astype(np.float32)).to(dev)")
    else:
        # Gather only one day's causal sequence, never materialize an 8x full-history tensor.
        old = 'torch.from_numpy(panel.X[d, mask])'
        assert model.count(old) == 3
        model = model.replace(old, 'torch.from_numpy(panel.inputs(d, mask))')
        pos = '    def mask(self, day, label=None):'
        model = model.replace(pos, '''    def inputs(self, day, mask):
        width = RECIPE['sequence_days']
        if day < width - 1:
            raise ValueError('insufficient causal history')
        return np.ascontiguousarray(self.X[day - width + 1:day + 1, mask].transpose(1, 0, 2))

''' + pos, 1)
        old = "out = self.coverage[day] >= RECIPE['min_feature_coverage']"
        new = """width = RECIPE['sequence_days']
        if day < width - 1:
            return np.zeros(len(self.codes), dtype=bool)
        out = (self.coverage[day-width+1:day+1] >= RECIPE['min_feature_coverage']).all(axis=0)"""
        assert model.count(old) == 1
        model = model.replace(old, new)
        # Existing inference audit still compares a stock subset with the full daily batch.
        old = "torch.from_numpy(panel.X[d, mask][:10])"
        assert analysis.count(old) == 1
        analysis = analysis.replace(old, "torch.from_numpy(panel.inputs(d, mask)[:10])")
    if family == 'quality_additive':
        # Rename one strategy, preserving the project's 159-file / 28-directory contract.
        model = model.replace("('top5_1d', 'top1_1d')", "('top5_5d', 'top1_1d')")
        analysis = analysis.replace("dict(name='top5_1d', topn=5, period=1, band=RECIPE['hold_band'])", "dict(name='top5_5d', topn=5, period=5, band=RECIPE['hold_band'])")
    # Audited cost accounting: gross is computed on exactly the actual positions/trade quantities.
    analysis = analysis.replace('    corp_events = 0\n', '    corp_events = 0\n    cumulative_fees = 0.0\n    cumulative_slippage = 0.0\n', 1)
    analysis = analysis.replace('            fee = fees(gross, True)\n', '            fee = fees(gross, True)\n            cumulative_fees += fee\n            cumulative_slippage += quantity * px.open[d, c] - gross\n', 1)
    analysis = analysis.replace('                fee = fees(gross)\n', '                fee = fees(gross)\n                cumulative_fees += fee\n                cumulative_slippage += gross - quantity * px.open[d, c]\n', 1)
    old = "positions=len(holdings), blocked_exits=len(blocked)))"
    assert analysis.count(old) == 1
    analysis = analysis.replace(old, "positions=len(holdings), blocked_exits=len(blocked), gross_equity=float(equity+cumulative_fees+cumulative_slippage), cumulative_fees=cumulative_fees, cumulative_slippage=cumulative_slippage, exposure=float((equity-cash)/equity)))")
    old = 'total_fees=float(sum(t[\'fee\'] for t in trades)), trades=len(trades)'
    assert old in analysis
    analysis = analysis.replace(old, 'total_fees=float(sum(t[\'fee\'] for t in trades)), gross_return_same_positions=float((values[-1]+cumulative_fees+cumulative_slippage)/money-1), fee_drag=float(cumulative_fees/money), slippage_drag=float(cumulative_slippage/money), gross_definition="same actual shares and trade dates; costs added back as cash, no reinvestment", avg_exposure=float(np.mean([c["exposure"] for c in curves])), trades=len(trades)', 1)
    # The historical proxy subtracted fees twice. New versions report the actual net daily return.
    analysis = analysis.replace("prev = np.r_[100000., eq[:-1]]", "prev = np.r_[RECIPE['backtest_money'], eq[:-1]]")
    analysis = analysis.replace("(day.mean() - st['total_fees'] / 100000. / len(day)) * 242 * 100", "day.mean() * 242 * 100")
    analysis = analysis.replace("trades=st['trades'])", "trades=st['trades'], gross_return_same_positions=st['gross_return_same_positions'], total_fees=st['total_fees'], fee_drag=st['fee_drag'], slippage_drag=st['slippage_drag'], net_annual_definition='actual net daily mean x 242; fees already included')", 1)
    analysis = analysis.replace("sharpe=s['sharpe'], trades=s['trades'], total_fees=s['total_fees'])", "sharpe=s['sharpe'], trades=s['trades'], total_fees=s['total_fees'], gross_return_same_positions=s['gross_return_same_positions'], fee_drag=s['fee_drag'], slippage_drag=s['slippage_drag'], avg_exposure=s['avg_exposure'], gross_definition=s['gross_definition'])")
    # Replace stale source docstrings while keeping copied, independent implementation functions.
    for name, source in [('model.py', model), ('analysis.py', analysis), ('run.py', run)]:
        tree = ast.parse(source)
        first = tree.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            lines = source.splitlines(keepends=True)
            source = ''.join(lines[:first.lineno-1]) + repr(f'{unit}: {family}; seed {seed}; fresh independent research unit, fixed project contract.') + '\n' + ''.join(lines[first.end_lineno:])
        compile(source, name, 'exec')
        d = OUT / unit
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(source, encoding='utf8')


if __name__ == '__main__':
    for family, first in [('quality_additive', 46), ('flow_temporal', 50)]:
        for i, seed in enumerate([3253, 4253, 5253, 6253]):
            make(f'V{first+i}', seed, family)
    print(json.dumps({'units': [p.name for p in OUT.iterdir()], 'scripts_each': 3}))
