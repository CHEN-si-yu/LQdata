import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

root = Path('/root/autodl-fs/model')
audit = Path(__file__).resolve().parent
stage = audit / 'new_trainingdata'
sys.path.insert(0, str(root))
import preparingdata as prep

started = time.time()
result = {'ok': False, 'phase': 'verifying', 'checks': [], 'errors': []}


def save():
    result['seconds'] = round(time.time() - started, 1)
    (audit / 'alignment_verification.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2))


def checked(name, count):
    result['checks'].append({'check': name, 'count': count})
    print(name, count, flush=True)
    save()


def source_values(path, dates, codes):
    tab = pq.ParquetFile(path).read(columns=['trade_date', 'stock_code', 'value'])
    tab = tab.filter(pc.and_(pc.greater_equal(tab['trade_date'], dates[0]),
                            pc.less_equal(tab['trade_date'], dates[-1])))
    assert tab.num_rows == len(dates) * len(codes), str(path) + ': source row count'
    assert np.array_equal(tab['trade_date'].to_numpy(), np.repeat(dates, len(codes))), str(path) + ': source dates'
    assert np.array_equal(tab['stock_code'].to_numpy(), np.tile(codes, len(dates))), str(path) + ': source codes'
    return tab['value'].to_numpy(zero_copy_only=False)


try:
    cfg = prep.Cfg(trainingdata=stage)
    meta = prep.load_meta(stage)
    cutoff = prep.last_upstream_day(cfg)
    specs = prep.upstream_specs(cfg)
    features = sorted(n for n, s in specs.items() if not s['is_label'])
    labels = sorted(n for n, s in specs.items() if s['is_label'])
    market_names = sorted(prep.list_market_factor_names(cfg))
    assert meta['columns']['features'] == features
    assert meta['columns']['labels'] == labels
    assert meta['semantics'] == prep.VALUES_SEMANTICS
    assert meta['axis']['end'] == cutoff
    assert not meta.get('partial')
    days, codes = prep.build_axis(cfg, meta['axis']['start'], cutoff)
    assert meta['axis']['n_days'] == len(days)
    assert meta['axis']['codes'] == list(codes)
    assert meta['axis']['n_codes'] == len(codes)
    assert meta['market_factors']['n_factors'] == len(market_names)
    assert sorted(c for c in meta['market_factors']['columns'] if not c.endswith(prep.MK_Z_SUFFIX)) == market_names
    for f in features:
        assert meta['columns']['direction'][f] == (1 if specs[f]['higher_is_better'] else -1)
    checked('current_registration_axis_and_semantics', len(features) + len(labels) + len(market_names))

    samples = ['adx_14', 'roe_ttm', 'cp_value_quality', 'ep_ttm',
               'rel_turnover_ind_20d', 'idt_rv_daily',
               'open5_capacity_floor_ratio_20', 'efx_limit_first',
               'chip_concentration', 'mf_net_inflow_ratio',
               'margin_balance_5d', 'bw_session_follow_20']
    samples = [n for n in samples if n in features]
    label_checks = feature_checks = sample_checks = market_checks = 0
    date_issues = []
    missing_latest = []
    for year in map(int, meta['built_years']):
        dates = np.asarray([d for d in days if int(d[:4]) == year], dtype=object)
        N = len(dates) * len(codes)
        fp = prep.year_file(stage, prep.FK, year)
        coord = pq.ParquetFile(fp).read(columns=['trade_date', 'stock_code'])
        assert np.array_equal(coord['trade_date'].to_numpy(), np.repeat(dates, len(codes)))
        assert np.array_equal(coord['stock_code'].to_numpy(), np.tile(codes, len(dates)))
        assert pq.ParquetFile(fp).schema_arrow.names == ['trade_date', 'stock_code'] + features
        for block in prep.ALL_KINDS:
            p = prep.year_file(stage, block, year)
            pf = pq.ParquetFile(p)
            assert pf.metadata.num_rows == N, str(p)
            assert p.stat().st_size == meta['years'][str(year)]['files'][block]['bytes'], str(p)
        ytab = pq.ParquetFile(prep.year_file(stage, prep.TK, year)).read(columns=labels)
        for label in labels:
            expected = source_values(prep.year_file(cfg.factors_dir, label, year), dates, codes)
            actual = ytab[label].to_numpy(zero_copy_only=False)
            assert np.array_equal(actual, expected, equal_nan=True), str(year) + '/' + label
            label_checks += 1

        xtab = pq.ParquetFile(fp).read(columns=samples)
        for name in samples:
            raw = source_values(prep.year_file(cfg.factors_dir, name, year), dates, codes)
            expected = prep._standardize(raw, len(dates), len(codes), meta['columns']['direction'][name] < 0)
            actual = xtab[name].to_numpy(zero_copy_only=False)
            assert np.array_equal(actual, expected, equal_nan=True), str(year) + '/' + name
            feature_checks += 1

        sample_names = meta['fac_sample']['columns']
        sample_tab = pq.ParquetFile(prep.year_file(stage, prep.SK, year)).read(columns=sample_names)
        source_tab = pq.ParquetFile(fp).read(columns=sample_names)
        for name in sample_names:
            assert np.array_equal(sample_tab[name].to_numpy(zero_copy_only=False),
                                  source_tab[name].to_numpy(zero_copy_only=False), equal_nan=True)
            sample_checks += 1

        market = pq.ParquetFile(prep.year_file(stage, prep.MK, year)).read()
        assert market.num_rows == len(dates)
        assert np.array_equal(market['trade_date'].to_numpy(), dates)
        for name in market_names:
            source = prep._market_series(cfg, name)
            expected = source.reindex(dates).to_numpy(dtype=np.float32)
            expected_z = prep._rolling_z(source).reindex(dates).to_numpy(dtype=np.float32)
            assert np.array_equal(market[name].to_numpy(zero_copy_only=False), expected, equal_nan=True), str(year) + '/' + name
            assert np.array_equal(market[name + prep.MK_Z_SUFFIX].to_numpy(zero_copy_only=False), expected_z, equal_nan=True), str(year) + '/' + name + prep.MK_Z_SUFFIX
            market_checks += 2
            if year == int(cutoff[:4]) and not np.isfinite(expected[-1]):
                missing_latest.append(name)
        print('VERIFIED_YEAR', year, 'days', len(dates), 'rows', N, flush=True)
        result['last_verified_year'] = year
        save()
        del coord, ytab, xtab, sample_tab, source_tab, market

    checked('all_years_all_labels_exact_values', label_checks)
    checked('all_years_representative_stock_features_exact_values', feature_checks)
    checked('all_years_fac_sample_exact_values', sample_checks)
    checked('all_years_all_market_raw_and_rolling_z_exact_values', market_checks)
    basic = json.loads((audit / 'check_result.json').read_text())
    inherited = basic.get('problems', [])
    assert all('/market_factors/' in p and '首个有效值之后还有' in p for p in inherited), inherited
    result.update({'ok': True, 'phase': 'verified', 'cutoff': cutoff,
                   'start': days[0], 'n_days': len(days), 'n_codes': len(codes),
                   'stock_feature_count': len(features), 'label_count': len(labels),
                   'market_factor_count': len(market_names),
                   'rows_per_stock_block_total': len(days) * len(codes),
                   'representative_stock_features': samples,
                   'inherited_upstream_market_gap_warnings': inherited,
                   'market_factors_missing_latest': missing_latest})
    save()
except BaseException as exc:
    traceback.print_exc()
    result.update({'ok': False, 'phase': 'failed', 'errors': [str(exc)]})
    save()
    sys.exit(1)
