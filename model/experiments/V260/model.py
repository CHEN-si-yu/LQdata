# 新快照研究版本；继承现有训练、权重和输入合同。
import os
DEFAULT_THREADS = '2'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = os.environ.get('MX_THREADS', DEFAULT_THREADS)
import copy
import json
import random
import resource
import sys
sys.dont_write_bytecode = True
import time
import re
import hashlib
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.stats import rankdata
from torch import nn
ROOT = Path(__file__).resolve().parent
UNIT_ROOT = ROOT
PROJECT_ROOT = UNIT_ROOT.parents[1]
UNIT = UNIT_ROOT.name
DEFAULT_FEATURE_BLOCK = 'factors'
SOURCE_FILES = ('run.py', 'model.py', 'analysis.py')
LAYOUT_QUARTERS = ('2025Q3', '2025Q4', '2026Q1', '2026Q2')
LAYOUT_FOLDS = (1, 2, 3, 4)
LAYOUT_YEARS = (2025, 2026)
LAYOUT_LOGS = 'model_logs'
FOLD_FILES = ('best.pt', 'last.pt', 'complete.json', 'history.json', 'score_predictions.npy', 'test_predictions.npy', 'split.json', 'training_info.json')

def fixed_files():
    files = set(SOURCE_FILES)
    files.update((f'model_train/{q}/fold{f}/{name}' for q in LAYOUT_QUARTERS for f in LAYOUT_FOLDS for name in FOLD_FILES))
    files.update((f'{LAYOUT_LOGS}/{q}_{f}.log' for q in LAYOUT_QUARTERS for f in LAYOUT_FOLDS))
    files.update((f'{LAYOUT_LOGS}/analysis_0.log' for _ in (0,)))
    files.update(('model_info/' + name for name in ('final_audit.json', 'two_strategy_validation.json')))
    files.update(('model_pred/ensemble_curves.png', 'model_pred/picks.md', 'model_pred/ensemble/score_meta.json'))
    files.update((f'model_pred/ensemble/year={y}/data.parquet' for y in LAYOUT_YEARS))
    files.update((f'model_pred/ensemble/CSV/{kind}_{strategy}.csv' for kind in ('cash', 'trades') for strategy in ('top5_5d', 'top1_1d')))
    return files
TOOL_DIRS = ('.ipynb_checkpoints', '__pycache__')

def unit_paths():
    files, dirs = (set(), set())
    for p in UNIT_ROOT.rglob('*'):
        rel = p.relative_to(UNIT_ROOT)
        if any((part in TOOL_DIRS for part in rel.parts)):
            continue
        (files if p.is_file() else dirs).add(rel.as_posix())
    return (files, dirs)

def validate_layout(*, allow_temporary=False, allow_missing=False):
    expected = fixed_files()
    actual, actual_dirs = unit_paths()
    if allow_temporary:
        temporary = set()
        for name in expected:
            path = Path(name)
            if path.suffix == '.json':
                temporary.add(path.with_suffix('.json.tmp').as_posix())
            elif path.suffix in ('.pt', '.npy'):
                temporary.add(path.with_suffix('.tmp').as_posix())
            elif path.suffix == '.parquet':
                temporary.add(path.with_name('data.tmp.parquet').as_posix())
        actual -= temporary
    allowed_dirs = {parent.as_posix() for name in expected for parent in Path(name).parents if str(parent) != '.'}
    extra = sorted(actual - expected | actual_dirs - allowed_dirs)
    missing = sorted((expected - actual) - globals().get('OPTIONAL_FILES',set()))
    if extra or (missing and (not allow_missing)):
        raise ValueError(f'独立单元文件树必须保持固定；额外路径={extra}；缺失文件={missing}；规则见docs/architecture.md')
    return dict(files=len(actual), directories=len(actual_dirs), source_files=list(SOURCE_FILES), missing=missing)

def output_file(path):
    path = Path(path).resolve()
    if not path.is_relative_to(UNIT_ROOT) or path.relative_to(UNIT_ROOT).as_posix() not in fixed_files():
        raise ValueError(f'禁止增加或移动独立单元文件: {path}')
    if path.name in SOURCE_FILES:
        raise ValueError(f'源码文件只能由开发者修改，不能当产物覆写: {path}')
    return path

def checked_name(value):
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'非法运行名称: {value}')
    return value

def run_name():
    name = os.environ.get('MODEL_RUN_NAME')
    if not name:
        for filename, key in [('launch.json', 'run_name'), ('final_audit.json', 'run')]:
            path = info_root() / filename
            if path.is_file():
                name = json.loads(path.read_text()).get(key)
                if name:
                    break
    name = checked_name(name or 'current')
    if name in ('legacy', 'before_restructure'):
        raise ValueError('请选择新的运行名称')
    return name

def run_root():
    return UNIT_ROOT

def info_root():
    return UNIT_ROOT / 'model_info'

def data_root():
    root = Path(os.environ.get('MX_DATA', PROJECT_ROOT / 'trainingdata' / 'factors')).expanduser().resolve()
    if root.name in ('fac_sample', 'factors') and (root.parent / 'meta.json').is_file():
        root = root.parent
    if not (root / 'meta.json').is_file():
        raise FileNotFoundError(f'{root}/meta.json 不存在；传入trainingdata根目录或其 factors 子目录')
    return root

def feature_block():
    explicit = Path(os.environ.get('MX_DATA', DEFAULT_FEATURE_BLOCK)).name
    block = explicit if explicit in ('fac_sample', 'factors') else os.environ.get('MODEL_FEATURE_BLOCK', DEFAULT_FEATURE_BLOCK)
    if block not in ('fac_sample', 'factors'):
        raise ValueError('特征块仅支持 fac_sample 或 factors')
    if block != DEFAULT_FEATURE_BLOCK:
        raise ValueError(f"{UNIT} 是**全因子正式训练单元**，特征块必须是 {DEFAULT_FEATURE_BLOCK}（当前 ={block}）。\n  · MX_DATA={os.environ.get('MX_DATA', '(未设)')}  MODEL_FEATURE_BLOCK={os.environ.get('MODEL_FEATURE_BLOCK', '(未设)')}\n  · 想跑 20 列的样例：用 experiments/V1，不要在 {UNIT} 里降级。\n  · 确实要改本单元的特征块：先改 run.py 的 DEFAULT_FEATURE_BLOCK，并同步 docs/iterations.md 的配置摘要（属科研决定，不是运行参数）。")
    return block

def price_root():
    return data_root() / 'prices'

def stock_list_path():
    return Path(os.environ.get('MX_STOCK_LIST', PROJECT_ROOT.parent / 'datadownload' / 'data' / 'stock_list' / 'data.parquet'))

def calendar_path():
    return Path(os.environ.get('MX_CALENDAR', PROJECT_ROOT.parent / 'datadownload' / 'data' / 'basic_calendar' / 'data.parquet'))

def feature_columns(meta):
    columns = list(RECIPE['feature_names'])
    if not set(columns).issubset(meta['columns']['features']):
        raise ValueError('pinned rebuilt feature view changed; create another unit')
    return columns

def market_columns(meta):
    block = meta['market_factors']
    columns = [c for c in block['columns'] if c.endswith(f"_z{block['z_window']}")]
    if len(columns) != block['n_factors']:
        raise ValueError(f"市场因子 z 窗口列数 {len(columns)} 与 n_factors {block['n_factors']} 不一致")
    return columns

def analysis_input_root():
    requested = os.environ.get('MODEL_INPUT_RUN')
    if requested and requested != run_name():
        raise ValueError('本单元使用固定目录，不再支持另一个 --input-run；直接读取 model_train')
    return UNIT_ROOT
RUN_ROOT = run_root()
INPUT_ROOT = analysis_input_root()
RECIPE = {'name': 'V260', 'architecture': 'same fixed ridge asV258; strictly earlier selector validation then mature pretest final refit', 'label': 'label_ret_5d', 'label_horizons': [1], 'score_blend': 'flow_technical_percentile', 'score_label': 'label_ret_5d', 'label_weights': {1: 1.0}, 'valid_ic_weights': {1: 1.0}, 'valid_ic_scale': 1.0, 'market_gate': True, 'market_block': 'market_factors', 'market_clip': 10.0, 'picks_topn': 5, 'backtest_money': 100000.0, 'picks_holding': 5, 'valid_criterion': 'macro_stable', 'loss_rank_decay': 0.5, 'train_pool_frac': 1.0, 'hot_pool': {'frac': 1.0, 'amount_window': 20, 'amp_window': 5, 'limit_window': 20, 'min_names': 20}, 'entry_rule': 'open', 'hold_band': 40, 'budget_rule': 'divide', 'bag_topk': 1, 'bag_recalibrate': False, 'features': 'all', 'feature_block': 'factors', 'folds': 4, 'seed': 35253, 'batch_days': 16, 'optimizer': 'closed_form_ridge', 'lr': 0.0, 'weight_decay': 0.0, 'dropout': (0.2, 0.1), 'hidden': (256, 128, 32), 'max_epochs': 1, 'early_stop_patience': 0, 'lr_patience': 2, 'lr_factor': 0.5, 'lr_cooldown': 2, 'min_lr': 5e-06, 'min_feature_coverage': 0.2, 'linear_weight': 0.0, 'device': 'auto', 'threads': 1, 'valid_topn': 500, 'valid_metric': '0.7IC1+0.3IC5-0.2stdIC1+0.2worst21dIC1+2capacityMean-0.1capacityVol; no Top5 cash selection', 'preprocessing': 'daily_zscore_win1_99_fixed_missing_0', 'test_start': '2025-07-01', 'test_end': '2026-06-30', 'purge_horizon': 20, 'training_window': 'selection train before valid; delivery expanding history through testfirst-minus21; valid scores never from delivered refit', 'sequence_days': 1, 'family': 'fixed_ridge_causal_selection_then_refit', 'training_target': 'date-equal full cross-section standardized1d return squared loss plus fixed L2; no tail rank reweighting', 'weights_origin': 'V228 three scripts only, no copied weights or fold products', 'research_protocol': '20261024 frozen screening; 4 standard quarters x 4 strictly forward validation folds; separate frozen 2026Q3; no test-driven strategy/threshold choice; no release promotion', 'snapshot_cutoff': '2026-10-23', 'prediction_horizon': 1, 'parent_sources': ['V258'], 'feature_names': ['afx_cf_im_net_cashflow_oper_act', 'afx_fi_fcfe', 'afx_fi_fcff', 'afx_fi_ocf_to_interestdebt', 'afx_fi_ocf_to_opincome', 'afx_fi_ocf_to_profit', 'afx_fi_ocfps', 'afx_fi_q_gr_qoq', 'afx_fi_q_gsprofit_margin', 'afx_fi_q_netprofit_qoq', 'afx_fi_q_ocf_to_or', 'afx_is_basic_eps', 'afx_is_ebitda', 'amihud_daily_5', 'amount_mom_accel', 'amount_ratio_20', 'annual_credit_impairment_adjustment_assets_v1', 'annual_report_lag_median3_v1', 'asset_growth_qoq', 'atr_14_ratio', 'beta_60', 'bp', 'bs_other_receiv_to_assets', 'bw_strength_sensitivity_20', 'ca_range_excess_5_v1', 'ca_tail_pressure_5_v1', 'calendar_gap_idio_variance_slope_60', 'cash_conversion_cycle', 'cash_sales_ratio', 'cf_purchase_cash_intensity', 'cfcr', 'cfp_ttm', 'close30_ret', 'corwin_schultz_overnight_adjusted_hl_proxy_20', 'cp_momentum_lowvol_20', 'cp_value_momentum_div', 'disposal_proceeds_intensity_yoy_v1', 'dividend_yield_3y_avg', 'dp_ttm', 'dpo_20', 'dragon_tiger_institution_net_pressure_20_v1', 'ebitda_to_market', 'efx_annual_earnings_yield', 'efx_volume_ratio', 'ep_ttm', 'etp5', 'extreme_move_event', 'holder_number_chg', 'id2_close_vs_pm_vwap', 'id2_close_vs_pm_vwap_20', 'id2_hi_lo_pos_gap', 'id2_parkinson_vol', 'id2_ret_concentration', 'idio_vol_60', 'idt_overnight_minus_intraday', 'idt_rv_daily', 'idt_rv_term_structure_slope', 'idt_vol_stability', 'ind_bps_yoy', 'ind_dt_netprofit_yoy', 'ind_int_to_talcap', 'interest_coverage', 'intraday_vol_ratio_5d', 'inventory_turnover', 'lf_margin_dip_demand_corr_60', 'limit_broken_ratio_20', 'limit_days_since_up_20', 'limit_down_event_5', 'log_mv', 'luna_minute_return_volume_corr_5d_v1', 'luna_minute_return_volume_lead_lag_5d_v1', 'luna_minute_unexpected_clock_volume_return_5d_v1', 'luna_roll_bounce_proxy_20d_v1', 'luna_turnover_weighted_intraday_return_path_20d_v1', 'margin_repay_deceleration', 'margin_repay_shock', 'mf_amount_weighted_direction', 'mf_big_order_net_ac1_20', 'mf_net_inflow_5d', 'mf_net_vol_surprise_20', 'mf_small_order_avg_price_dev', 'momentum_120', 'momentum_20', 'momentum_250', 'momentum_60', 'ocf_to_profit', 'ocf_to_revenue', 'open5_capacity_floor_ratio_20', 'overnight_ma_20d', 'overnight_ma_5d', 'overnight_ma_60d', 'overnight_std_5d', 'pe_own_history_percentile_252_20', 'price_delay_r2_60', 'price_distance_from_52w_low', 'price_to_52w_high', 'qf_cash_margin_floor_4q', 'qf_cash_margin_yoy_change', 'qf_core_roe_floor_4q', 'qf_core_roe_vol_4q', 'qf_core_roe_yoy_change', 'qf_roa_yoy_change', 'qf_sales_growth_accel', 'qf_sales_growth_vol_4q', 'r10gc_bias_std_turn_21_252_v1_bcd82eb2bc4c4fe7', 'r10gc_bias_turn_126_252_v1_a9656f4fececd030', 'r10gc_bias_turn_21_252_v1_f541ccc9f9fa2581', 'r10gc_bias_turn_63_252_v1_27c60f4ea79e6742', 'r10gc_high_low_21_v1_ca60e882d8b5c22f', 'r10gc_high_low_252_v1_facb1d1f647f48ca', 'r10gc_high_low_63_v1_34a74707a9f916a3', 'r10gc_income_tax_yoy_v1_c051c303608f0a17', 'r10gc_ret_std_21_v1_51f3cc1a8b65d126', 'r10gc_ret_std_252_v1_db7166018e1a0511', 'r10gc_ret_std_63_v1_19b9e9fa082c7cec', 'r10gc_turn_lvl_21_v1_2bab2360d6cb7099', 'r10gc_turn_lvl_252_v1_2dd70c23d17694f4', 'r10gc_turn_lvl_63_v1_7d51db9d4ff08793', 'r10gc_turn_std_126_v1_e96f93f78ed0e472', 'r10gc_turn_std_252_v1_9a4d3d577535e360', 'r10gc_turn_std_42_v1_21349ae0e32c1db7', 'r10gc_yoy_net_asset_v1_2a57af1e5c40492f', 'r10gc_yoy_net_profit_v1_bdc315b5d4b02fac', 'r10gc_yoy_ocf_v1_066e6046572baf9c', 'r10gc_yoy_total_asset_v1_3b5f72f283c0e3ca', 'r11gc_amount_ma_20_v1_1858415feaa5dbe6', 'r11gc_bias_std_turn_126_252_v1_5be1af21b3af2732', 'r11gc_bias_std_turn_126_504_v1_ec735b581d5c6134', 'r11gc_bias_std_turn_21_504_v1_0ec972aa142a07aa', 'r11gc_bias_std_turn_42_252_v1_f671e4d8cc39efd1', 'r11gc_bias_std_turn_42_504_v1_405ae54d3c3ed62f', 'r11gc_bias_std_turn_63_252_v1_d03f938d9f4b2cfe', 'r11gc_bias_std_turn_63_504_v1_54007b4942cea4c1', 'r11gc_bias_turn_126_504_v1_e3ccc91522792d65', 'r11gc_bias_turn_21_504_v1_111533df505215cb', 'r11gc_bias_turn_42_252_v1_fa9e6caf1c763260', 'r11gc_bias_turn_42_504_v1_c66e9ee701bdb2ad', 'r11gc_bias_turn_63_504_v1_387613367cbd4dd9', 'r11gc_high_low_126_v1_2666509099db4a47', 'r11gc_high_low_42_v1_36fd67d17bea68ac', 'r11gc_ret_std_42_v1_23474362384bc399', 'r11gc_turn_lvl_10_v1_c377944b650ec037', 'r11gc_turn_lvl_126_v1_ef48ac322c8d0929', 'r11gc_turn_lvl_20_v1_2f60dce5b6aba8a4', 'r11gc_turn_lvl_42_v1_24d04a044ce607a0', 'r11gc_turn_lvl_5_v1_332d53f4bb907a79', 'r11gc_turn_std_63_v1_5ef776ad40ef77f6', 'r11u2_peak_timing_mktstate_gate_fix1_83f5c455e951bbb3', 'r11u4_flow_intensity_x_surprise_v1_1cec9f0321fe52c1', 'r12gc_asset_turnover_v1_40d616e07fd4a097', 'r12gc_cash_ratio_v1_7ef2829865ad2cc2', 'r12gc_de_v1_d4cd33b978d63b0a', 'r12gc_eps_q_v1_8f3703e8eb218668', 'r12gc_eps_ttm_v1_2d84c57b9ea5f85d', 'r12gc_eps_y_v1_6f0a67c91531a732', 'r12gc_equity_turnover_v1_4cba36560ca472e5', 'r12gc_fcf_to_market_v1_49c0b5d214b35db0', 'r12gc_fixed_asset_turnover_v1_f5bd2c641032b1a2', 'r12gc_float_size_v1_8b275acb0cb20dec', 'r12gc_gpm_q_v1_842e57e4d0d033e5', 'r12gc_gpm_ttm_v1_7e99428574d400ea', 'r12gc_gpm_y_v1_102eddddb8fac4c4', 'r12gc_log_price_v1_0a22efbc7b6bb1f7', 'r12gc_market_value_leverage_v1_9f3705df7636e0cc', 'r12gc_ncf_to_market_v1_71cf1e0deddc7070', 'r12gc_npm_q_v1_a1e5e8bb0a48de33', 'r12gc_npm_ttm_v1_4e4381d67f1d7391', 'r12gc_npm_y_v1_f78b237b3b7c1b18', 'r12gc_opm_ttm_v1_2827ff5d605bdeeb', 'r12gc_opm_y_v1_6e2f1f9aaf3c6106', 'r12gc_opt_tpro_v1_d832d118e2414259', 'r12gc_quick_ratio_v1_ea13af873ce572fc', 'r12gc_receivable_turnover_v1_458e2aaecafa62b9', 'r12gc_return_42_v1_0032bf23a28de8f7', 'r12gc_roa_q_v1_4b4ae2280384ebde', 'r12gc_roa_y_v1_6e585294932485e4', 'r12gc_roe_y_v1_bca9de9f86ce36dd', 'r12gc_sales_to_market_v1_ff344107e3f2c925', 'r13f_gpm_dispersion_v1_d45cc0c6c766d353', 'r13f_ocf_seasonal_trend_innovation_v1_4c66203abbadc485', 'r13gc_daily_range_tail_spread_63_v1_51e696671ccfee43', 'r13gc_daily_return_quantile_skew_63_v1_42a906e8b444a54e', 'r13gc_daily_turnover_ac1_63_v1_e42ca3d25ca965e6', 'r13gc_fb05_deferred_tax_asset_share_v1_d3599ffb97a55de4', 'r13gc_fb06_long_equity_investment_asset_mix_v1_c051b7080293a254', 'r13gc_fb07_supplier_prepayment_asset_share_v1_8a49e274b2af4e41', 'r13gc_fc01_payroll_vs_goods_cash_mix_v1_009c5259e3988576', 'r13gc_fc02_financial_investment_outflow_share_v1_950a52c23c5ce39a', 'r13gc_fi01_selling_admin_mix_v1_8c8039b595a7f2ab', 'r13gc_fi06_revenue_seasonality_v1_8a12e8686b06f0e2', 'r13gc_financial_quality_cash_tax_paid_asinh_ttm_v2_648e7409d04795dd', 'r13gc_financial_quality_income_tax_revenue_ttm_v2_6bdc1290078a0504', 'r13gc_financial_quality_minority_interest_assets_point_v2_9fead6bc31654eb9', 'r13gc_financial_quality_non_oper_exp_revenue_ttm_v2_0e83ec40a51ad981', 'r13gc_overnight_gap_intraday_response_beta_60_v1_789a747483612449', 'r13gc_same_calendar_month_turnover_profile_v2_a64fbb3e9b93a2dd', 'r13gc_turnover_commonality_beta_126_v1_8be87d88402aac70', 'r13gc_turnover_to_nextday_absret_corr_126_v2_ed7738ff0c879711', 'r13p_fip_return_discreteness_v1_94875f42e1e5efcf', 'r14f15_minority_external_equity_cash_share_v2_2f46ba1f387a53c4', 'r14f1_ccc_seasonal_trend_innovation_v2_0a98f6e1fc8b0970', 'r14f27_supplier_prepayment_direction_v1_348e7f359c7fd769', 'r14f28_tax_cash_refund_share_v1_3904a84026422c3e', 'r14f29_intangible_amort_share_v1_476979a530c60728', 'r14f7_positive_contract_rights_share_v2_3cab6f4117f47708', 'r14gc_session_illiquidity_asymmetry_v1_93978c1ec1b77ea8', 'r14p11_market_lower_tail_coincidence_v2_d02a562b871b1976', 'r14p6_raw_monthly_tk_v1_dee5c4eeffeaaa9f', 'r14p7_signed_amount_reversal_v2_7dd5dbc630f5495e', 'r14p9_negative_shock_variance_feedback_v2_d4ad4585f4444283', 'r15f1_bs_pool_composition_v1_longterm_fix_assets_0e39009c5c8f2db5', 'r15f1_bs_pool_composition_v1_longterm_intangibles_8431a77bc492ef3f', 'r15f1_bs_pool_composition_v1_operating_inventories_0bf01e7fe618a7e8', 'r15f1_bs_pool_composition_v1_operating_receivables_444f672a458e0d1f', 'r15gc_cf01_op_inflow_sales_collection_share_v1_6a13add0e7ebbb79', 'r15gc_cf02_op_outflow_tax_payment_share_v1_1434eaa634455e8c', 'r15gc_cf03_investing_outflow_capex_share_v1_8b4aa1df901e47bb', 'r15gc_cf04_financing_inflow_borrow_share_v1_e4398f1eb4148501', 'r15gc_cf05_financing_outflow_distribution_share_v1_e3e31e9803a73394', 'r15gc_cf06_collection_share_seasonal_deviation_v1_674354f5ce32e8bf', 'r15gc_is01_selling_expense_ratio_v1_d05f1e4dd2122eb4', 'r15gc_is02_admin_expense_ratio_v1_c7c097590031f7e3', 'r15gc_is03_financial_expense_ratio_v1_4b977de0274a7b08', 'r15gc_is04_gross_profit_opex_coverage_v1_410dabf3b40ac2b4', 'r15n14_crossfit_pca_ou_displacement_v1_6de328942e44733b', 'r15net_peer_flow_diffusion_v1_0ea2820cafbe9177', 'r15p13_restricted_first_passage_v1_6734bafa6d4f2ab8', 'r15p14_monthly_component_range_memory_v1_6fe135138d05893d', 'r15t3_overnight_variance_share_v1_09b9ae4f22c98618', 'r15t3_spillover_v1_432027233601b658', 'r16c_ml1_price_volume_monthly_h5_mlp_replay_v2_d9ac78cf139d8d43', 'r16gc_is05_selling_ratio_seasonal_deviation_replay_v1_371a3b541117bf5e', 'r17cg_chip_gini_38d02c6cc01fbd71', 'r17cg_p90_gap_chg20_c91115e3f691152c', 'r17cg_peak_vs_top3_d62c93f9f9b52f89', 'r17cg_skew_chg20_4c9f6345c8034e61', 'r2own_v2_trend_persistence_v1_464e80a3ee6ef0bb', 'r3b_intraday_rv_skewness_v2_44ef55d83a759fba', 'r4a_forecast_landing_gap_v3_972c0653571ab5d8', 'r4a_sue_accel_v1_4b443eb2d4fda83c', 'r4a_sue_seasonal_q_v1_f1def0b01eb0b979', 'r4b_intraday_rv_kurtosis_20_4fdea35833870ebb', 'r4b_intraday_volume_hhi_20_2462d9ca21d9adf8', 'r4c_accrual_ratio_q_v1_8bef7aa2caaf33d9', 'r4c_accrual_ratio_ttm_v1_80e27bcbce7fddb5', 'r4e_idio_max_20_v1_d0d201efd7755e86', 'r4j_same_month_consistency_v1_be7f8ede9c4b5afe', 'r4q_adv_recv_cl_growth_yoy_v1_066cb6dd0e26b3e1', 'r4t_preann_turnover_shock_v1_3a95800106c754f2', 'r4t_preann_vol_expansion_v1_bebf97b2c5260706', 'r7a_tax_expense_seasonal_change_v2_371da87d5055df86', 'r7e_context_salience20_v1_f4efb8789772d003', 'r8e_book_wc_cash_realization_noise20q_v2_3b7ffc1a4a997035', 'r9a_sec_own_rv_beta_v2_ef77e079dd4478fe', 'r9c_abnormal_dispersion_shift_k4_v2_cb653c1abc1fb728', 'r9f_breadth_intraday_slope_x_beta_v1_51445f98a0907e8f', 'raw_price_tick_clustering_20', 'ret_skew_20', 'revenue_cagr_3y', 'reversal_2d', 'roa_ttm', 'roc_12', 'roe_ttm', 'rsi_14', 'rsrs_beta_18', 'rv_term_structure', 's26_liq_up_fragility_v1_b43c747ff9090bc2', 's26_structure_side_hhi_v1_bb77e106a875308a', 'short_term_reversal_5', 'sp_ttm', 'td_setup_count', 'three_black_crows', 'toplist_multi_reason_breadth_20_v1', 'turnover_anomaly_20', 'turnover_f_delta_5', 'turnover_std_20', 'vol_120', 'vol_clustering_20', 'vol_of_rv', 'vol_ratio_ma5_ma20', 'volume_ratio', 'yoy_revenue'], 'flow_feature_names': ['afx_cf_im_net_cashflow_oper_act', 'afx_fi_fcfe', 'afx_fi_fcff', 'afx_fi_ocf_to_interestdebt', 'afx_fi_ocf_to_opincome', 'afx_fi_ocf_to_profit', 'afx_fi_ocfps', 'afx_fi_q_gr_qoq', 'afx_fi_q_gsprofit_margin', 'afx_fi_q_netprofit_qoq', 'afx_fi_q_ocf_to_or', 'afx_is_basic_eps', 'afx_is_ebitda', 'amihud_daily_5', 'amount_mom_accel', 'amount_ratio_20', 'annual_credit_impairment_adjustment_assets_v1', 'annual_report_lag_median3_v1', 'asset_growth_qoq', 'atr_14_ratio', 'beta_60', 'bp', 'bs_other_receiv_to_assets', 'bw_strength_sensitivity_20', 'ca_range_excess_5_v1', 'ca_tail_pressure_5_v1', 'calendar_gap_idio_variance_slope_60', 'cash_conversion_cycle', 'cash_sales_ratio', 'cf_purchase_cash_intensity', 'cfcr', 'cfp_ttm', 'close30_ret', 'corwin_schultz_overnight_adjusted_hl_proxy_20', 'cp_momentum_lowvol_20', 'cp_value_momentum_div', 'disposal_proceeds_intensity_yoy_v1', 'dividend_yield_3y_avg', 'dp_ttm', 'dpo_20', 'dragon_tiger_institution_net_pressure_20_v1', 'ebitda_to_market', 'efx_annual_earnings_yield', 'efx_volume_ratio', 'ep_ttm', 'etp5', 'extreme_move_event', 'holder_number_chg', 'id2_close_vs_pm_vwap', 'id2_close_vs_pm_vwap_20', 'id2_hi_lo_pos_gap', 'id2_parkinson_vol', 'id2_ret_concentration', 'idio_vol_60', 'idt_overnight_minus_intraday', 'idt_rv_daily', 'idt_rv_term_structure_slope', 'idt_vol_stability', 'ind_bps_yoy', 'ind_dt_netprofit_yoy', 'ind_int_to_talcap', 'interest_coverage', 'intraday_vol_ratio_5d', 'inventory_turnover', 'lf_margin_dip_demand_corr_60', 'limit_broken_ratio_20', 'limit_days_since_up_20', 'limit_down_event_5', 'log_mv', 'luna_minute_return_volume_corr_5d_v1', 'luna_minute_return_volume_lead_lag_5d_v1', 'luna_minute_unexpected_clock_volume_return_5d_v1', 'luna_roll_bounce_proxy_20d_v1', 'luna_turnover_weighted_intraday_return_path_20d_v1', 'margin_repay_deceleration', 'margin_repay_shock', 'mf_amount_weighted_direction', 'mf_big_order_net_ac1_20', 'mf_net_inflow_5d', 'mf_net_vol_surprise_20', 'mf_small_order_avg_price_dev', 'momentum_120', 'momentum_20', 'momentum_250', 'momentum_60', 'ocf_to_profit', 'ocf_to_revenue', 'open5_capacity_floor_ratio_20', 'overnight_ma_20d', 'overnight_ma_5d', 'overnight_ma_60d', 'overnight_std_5d', 'pe_own_history_percentile_252_20', 'price_delay_r2_60', 'price_distance_from_52w_low', 'price_to_52w_high', 'qf_cash_margin_floor_4q', 'qf_cash_margin_yoy_change', 'qf_core_roe_floor_4q', 'qf_core_roe_vol_4q', 'qf_core_roe_yoy_change', 'qf_roa_yoy_change', 'qf_sales_growth_accel', 'qf_sales_growth_vol_4q', 'r10gc_bias_std_turn_21_252_v1_bcd82eb2bc4c4fe7', 'r10gc_bias_turn_126_252_v1_a9656f4fececd030', 'r10gc_bias_turn_21_252_v1_f541ccc9f9fa2581', 'r10gc_bias_turn_63_252_v1_27c60f4ea79e6742', 'r10gc_high_low_21_v1_ca60e882d8b5c22f', 'r10gc_high_low_252_v1_facb1d1f647f48ca', 'r10gc_high_low_63_v1_34a74707a9f916a3', 'r10gc_income_tax_yoy_v1_c051c303608f0a17', 'r10gc_ret_std_21_v1_51f3cc1a8b65d126', 'r10gc_ret_std_252_v1_db7166018e1a0511', 'r10gc_ret_std_63_v1_19b9e9fa082c7cec', 'r10gc_turn_lvl_21_v1_2bab2360d6cb7099', 'r10gc_turn_lvl_252_v1_2dd70c23d17694f4', 'r10gc_turn_lvl_63_v1_7d51db9d4ff08793', 'r10gc_turn_std_126_v1_e96f93f78ed0e472', 'r10gc_turn_std_252_v1_9a4d3d577535e360', 'r10gc_turn_std_42_v1_21349ae0e32c1db7', 'r10gc_yoy_net_asset_v1_2a57af1e5c40492f', 'r10gc_yoy_net_profit_v1_bdc315b5d4b02fac', 'r10gc_yoy_ocf_v1_066e6046572baf9c', 'r10gc_yoy_total_asset_v1_3b5f72f283c0e3ca', 'r11gc_amount_ma_20_v1_1858415feaa5dbe6', 'r11gc_bias_std_turn_126_252_v1_5be1af21b3af2732', 'r11gc_bias_std_turn_126_504_v1_ec735b581d5c6134', 'r11gc_bias_std_turn_21_504_v1_0ec972aa142a07aa', 'r11gc_bias_std_turn_42_252_v1_f671e4d8cc39efd1', 'r11gc_bias_std_turn_42_504_v1_405ae54d3c3ed62f', 'r11gc_bias_std_turn_63_252_v1_d03f938d9f4b2cfe', 'r11gc_bias_std_turn_63_504_v1_54007b4942cea4c1', 'r11gc_bias_turn_126_504_v1_e3ccc91522792d65', 'r11gc_bias_turn_21_504_v1_111533df505215cb', 'r11gc_bias_turn_42_252_v1_fa9e6caf1c763260', 'r11gc_bias_turn_42_504_v1_c66e9ee701bdb2ad', 'r11gc_bias_turn_63_504_v1_387613367cbd4dd9', 'r11gc_high_low_126_v1_2666509099db4a47', 'r11gc_high_low_42_v1_36fd67d17bea68ac', 'r11gc_ret_std_42_v1_23474362384bc399', 'r11gc_turn_lvl_10_v1_c377944b650ec037', 'r11gc_turn_lvl_126_v1_ef48ac322c8d0929', 'r11gc_turn_lvl_20_v1_2f60dce5b6aba8a4', 'r11gc_turn_lvl_42_v1_24d04a044ce607a0', 'r11gc_turn_lvl_5_v1_332d53f4bb907a79', 'r11gc_turn_std_63_v1_5ef776ad40ef77f6', 'r11u2_peak_timing_mktstate_gate_fix1_83f5c455e951bbb3', 'r11u4_flow_intensity_x_surprise_v1_1cec9f0321fe52c1', 'r12gc_asset_turnover_v1_40d616e07fd4a097', 'r12gc_cash_ratio_v1_7ef2829865ad2cc2', 'r12gc_de_v1_d4cd33b978d63b0a', 'r12gc_eps_q_v1_8f3703e8eb218668', 'r12gc_eps_ttm_v1_2d84c57b9ea5f85d', 'r12gc_eps_y_v1_6f0a67c91531a732', 'r12gc_equity_turnover_v1_4cba36560ca472e5', 'r12gc_fcf_to_market_v1_49c0b5d214b35db0', 'r12gc_fixed_asset_turnover_v1_f5bd2c641032b1a2', 'r12gc_float_size_v1_8b275acb0cb20dec', 'r12gc_gpm_q_v1_842e57e4d0d033e5', 'r12gc_gpm_ttm_v1_7e99428574d400ea', 'r12gc_gpm_y_v1_102eddddb8fac4c4', 'r12gc_log_price_v1_0a22efbc7b6bb1f7', 'r12gc_market_value_leverage_v1_9f3705df7636e0cc', 'r12gc_ncf_to_market_v1_71cf1e0deddc7070', 'r12gc_npm_q_v1_a1e5e8bb0a48de33', 'r12gc_npm_ttm_v1_4e4381d67f1d7391', 'r12gc_npm_y_v1_f78b237b3b7c1b18', 'r12gc_opm_ttm_v1_2827ff5d605bdeeb', 'r12gc_opm_y_v1_6e2f1f9aaf3c6106', 'r12gc_opt_tpro_v1_d832d118e2414259', 'r12gc_quick_ratio_v1_ea13af873ce572fc', 'r12gc_receivable_turnover_v1_458e2aaecafa62b9', 'r12gc_return_42_v1_0032bf23a28de8f7', 'r12gc_roa_q_v1_4b4ae2280384ebde', 'r12gc_roa_y_v1_6e585294932485e4', 'r12gc_roe_y_v1_bca9de9f86ce36dd', 'r12gc_sales_to_market_v1_ff344107e3f2c925', 'r13f_gpm_dispersion_v1_d45cc0c6c766d353', 'r13f_ocf_seasonal_trend_innovation_v1_4c66203abbadc485', 'r13gc_daily_range_tail_spread_63_v1_51e696671ccfee43', 'r13gc_daily_return_quantile_skew_63_v1_42a906e8b444a54e', 'r13gc_daily_turnover_ac1_63_v1_e42ca3d25ca965e6', 'r13gc_fb05_deferred_tax_asset_share_v1_d3599ffb97a55de4', 'r13gc_fb06_long_equity_investment_asset_mix_v1_c051b7080293a254', 'r13gc_fb07_supplier_prepayment_asset_share_v1_8a49e274b2af4e41', 'r13gc_fc01_payroll_vs_goods_cash_mix_v1_009c5259e3988576', 'r13gc_fc02_financial_investment_outflow_share_v1_950a52c23c5ce39a', 'r13gc_fi01_selling_admin_mix_v1_8c8039b595a7f2ab', 'r13gc_fi06_revenue_seasonality_v1_8a12e8686b06f0e2', 'r13gc_financial_quality_cash_tax_paid_asinh_ttm_v2_648e7409d04795dd', 'r13gc_financial_quality_income_tax_revenue_ttm_v2_6bdc1290078a0504', 'r13gc_financial_quality_minority_interest_assets_point_v2_9fead6bc31654eb9', 'r13gc_financial_quality_non_oper_exp_revenue_ttm_v2_0e83ec40a51ad981', 'r13gc_overnight_gap_intraday_response_beta_60_v1_789a747483612449', 'r13gc_same_calendar_month_turnover_profile_v2_a64fbb3e9b93a2dd', 'r13gc_turnover_commonality_beta_126_v1_8be87d88402aac70', 'r13gc_turnover_to_nextday_absret_corr_126_v2_ed7738ff0c879711', 'r13p_fip_return_discreteness_v1_94875f42e1e5efcf', 'r14f15_minority_external_equity_cash_share_v2_2f46ba1f387a53c4', 'r14f1_ccc_seasonal_trend_innovation_v2_0a98f6e1fc8b0970', 'r14f27_supplier_prepayment_direction_v1_348e7f359c7fd769', 'r14f28_tax_cash_refund_share_v1_3904a84026422c3e', 'r14f29_intangible_amort_share_v1_476979a530c60728', 'r14f7_positive_contract_rights_share_v2_3cab6f4117f47708', 'r14gc_session_illiquidity_asymmetry_v1_93978c1ec1b77ea8', 'r14p11_market_lower_tail_coincidence_v2_d02a562b871b1976', 'r14p6_raw_monthly_tk_v1_dee5c4eeffeaaa9f', 'r14p7_signed_amount_reversal_v2_7dd5dbc630f5495e', 'r14p9_negative_shock_variance_feedback_v2_d4ad4585f4444283', 'r15f1_bs_pool_composition_v1_longterm_fix_assets_0e39009c5c8f2db5', 'r15f1_bs_pool_composition_v1_longterm_intangibles_8431a77bc492ef3f', 'r15f1_bs_pool_composition_v1_operating_inventories_0bf01e7fe618a7e8', 'r15f1_bs_pool_composition_v1_operating_receivables_444f672a458e0d1f', 'r15gc_cf01_op_inflow_sales_collection_share_v1_6a13add0e7ebbb79', 'r15gc_cf02_op_outflow_tax_payment_share_v1_1434eaa634455e8c', 'r15gc_cf03_investing_outflow_capex_share_v1_8b4aa1df901e47bb', 'r15gc_cf04_financing_inflow_borrow_share_v1_e4398f1eb4148501', 'r15gc_cf05_financing_outflow_distribution_share_v1_e3e31e9803a73394', 'r15gc_cf06_collection_share_seasonal_deviation_v1_674354f5ce32e8bf', 'r15gc_is01_selling_expense_ratio_v1_d05f1e4dd2122eb4', 'r15gc_is02_admin_expense_ratio_v1_c7c097590031f7e3', 'r15gc_is03_financial_expense_ratio_v1_4b977de0274a7b08', 'r15gc_is04_gross_profit_opex_coverage_v1_410dabf3b40ac2b4', 'r15n14_crossfit_pca_ou_displacement_v1_6de328942e44733b', 'r15net_peer_flow_diffusion_v1_0ea2820cafbe9177', 'r15p13_restricted_first_passage_v1_6734bafa6d4f2ab8', 'r15p14_monthly_component_range_memory_v1_6fe135138d05893d', 'r15t3_overnight_variance_share_v1_09b9ae4f22c98618', 'r15t3_spillover_v1_432027233601b658', 'r16c_ml1_price_volume_monthly_h5_mlp_replay_v2_d9ac78cf139d8d43', 'r16gc_is05_selling_ratio_seasonal_deviation_replay_v1_371a3b541117bf5e', 'r17cg_chip_gini_38d02c6cc01fbd71', 'r17cg_p90_gap_chg20_c91115e3f691152c', 'r17cg_peak_vs_top3_d62c93f9f9b52f89', 'r17cg_skew_chg20_4c9f6345c8034e61', 'r2own_v2_trend_persistence_v1_464e80a3ee6ef0bb', 'r3b_intraday_rv_skewness_v2_44ef55d83a759fba', 'r4a_forecast_landing_gap_v3_972c0653571ab5d8', 'r4a_sue_accel_v1_4b443eb2d4fda83c', 'r4a_sue_seasonal_q_v1_f1def0b01eb0b979', 'r4b_intraday_rv_kurtosis_20_4fdea35833870ebb', 'r4b_intraday_volume_hhi_20_2462d9ca21d9adf8', 'r4c_accrual_ratio_q_v1_8bef7aa2caaf33d9', 'r4c_accrual_ratio_ttm_v1_80e27bcbce7fddb5', 'r4e_idio_max_20_v1_d0d201efd7755e86', 'r4j_same_month_consistency_v1_be7f8ede9c4b5afe', 'r4q_adv_recv_cl_growth_yoy_v1_066cb6dd0e26b3e1', 'r4t_preann_turnover_shock_v1_3a95800106c754f2', 'r4t_preann_vol_expansion_v1_bebf97b2c5260706', 'r7a_tax_expense_seasonal_change_v2_371da87d5055df86', 'r7e_context_salience20_v1_f4efb8789772d003', 'r8e_book_wc_cash_realization_noise20q_v2_3b7ffc1a4a997035', 'r9a_sec_own_rv_beta_v2_ef77e079dd4478fe', 'r9c_abnormal_dispersion_shift_k4_v2_cb653c1abc1fb728', 'r9f_breadth_intraday_slope_x_beta_v1_51445f98a0907e8f', 'raw_price_tick_clustering_20', 'ret_skew_20', 'revenue_cagr_3y', 'reversal_2d', 'roa_ttm', 'roc_12', 'roe_ttm', 'rsi_14', 'rsrs_beta_18', 'rv_term_structure', 's26_liq_up_fragility_v1_b43c747ff9090bc2', 's26_structure_side_hhi_v1_bb77e106a875308a', 'short_term_reversal_5', 'sp_ttm', 'td_setup_count', 'three_black_crows', 'toplist_multi_reason_breadth_20_v1', 'turnover_anomaly_20', 'turnover_f_delta_5', 'turnover_std_20', 'vol_120', 'vol_clustering_20', 'vol_of_rv', 'vol_ratio_ma5_ma20', 'volume_ratio', 'yoy_revenue'], 'technical_feature_names': ['momentum_20', 'momentum_60', 'momentum_120', 'reversal_2d', 'short_term_reversal_5', 'ret_skew_20', 'vol_120', 'beta_60', 'idio_vol_60', 'log_mv', 'price_to_52w_high', 'atr_14_ratio', 'rsi_14', 'roc_12', 'dpo_20', 'amihud_daily_5', 'overnight_std_5d', 'holder_number_chg', 'interest_coverage'], 'flow_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281], 'technical_indices': [82, 84, 81, 259, 268, 257, 276, 20, 53, 68, 95, 19, 263, 261, 39, 13, 91, 47, 61], 'output_horizons': [1, 1], 'flow_fusion_weight': 1.0, 'absolute_sample_caps': [512], 'absolute_stock_ridge': 0.05, 'absolute_market_ridge': 0.25, 'absolute_intercept_ridge': 1e-08, 'absolute_target_clip': {1: 0.12, 5: 0.3}, 'absolute_loss_scale': {1: 0.03, 5: 0.08}, 'absolute_label_weights': {1: 0.75, 5: 0.25}, 'absolute_stage_bag': 'one fixed training-only ridge fit', 'cost_gate_multiplier': 1.0, 'ensemble_sum_members': 4, 'final_prediction_backend': 'CPU canonical rank outputs; CUDA train/validation raw heads numerically checked', 'retrain_parent': 'V247', 'snapshot_meta_sha256': 'd4c3dea169ea9c4305aeb04226c39bc473853e284eeb2d4ec5cdc18ceaa0c8e0', 'fund_feature_names': ['afx_cf_im_net_cashflow_oper_act', 'afx_fi_fcfe', 'afx_fi_fcff', 'afx_fi_ocf_to_interestdebt', 'afx_fi_ocf_to_opincome', 'afx_fi_ocf_to_profit', 'afx_fi_ocfps', 'afx_fi_q_gr_qoq', 'afx_fi_q_gsprofit_margin', 'afx_fi_q_netprofit_qoq', 'afx_fi_q_ocf_to_or', 'afx_is_basic_eps', 'afx_is_ebitda', 'amihud_daily_5', 'amount_mom_accel', 'amount_ratio_20', 'annual_credit_impairment_adjustment_assets_v1', 'annual_report_lag_median3_v1', 'asset_growth_qoq', 'atr_14_ratio', 'beta_60', 'bp', 'bs_other_receiv_to_assets', 'bw_strength_sensitivity_20', 'ca_range_excess_5_v1', 'ca_tail_pressure_5_v1', 'calendar_gap_idio_variance_slope_60', 'cash_conversion_cycle', 'cash_sales_ratio', 'cf_purchase_cash_intensity', 'cfcr', 'cfp_ttm', 'close30_ret', 'corwin_schultz_overnight_adjusted_hl_proxy_20', 'cp_momentum_lowvol_20', 'cp_value_momentum_div', 'disposal_proceeds_intensity_yoy_v1', 'dividend_yield_3y_avg', 'dp_ttm', 'dpo_20', 'dragon_tiger_institution_net_pressure_20_v1', 'ebitda_to_market', 'efx_annual_earnings_yield', 'efx_volume_ratio', 'ep_ttm', 'etp5', 'extreme_move_event', 'holder_number_chg', 'id2_close_vs_pm_vwap', 'id2_close_vs_pm_vwap_20', 'id2_hi_lo_pos_gap', 'id2_parkinson_vol', 'id2_ret_concentration', 'idio_vol_60', 'idt_overnight_minus_intraday', 'idt_rv_daily', 'idt_rv_term_structure_slope', 'idt_vol_stability', 'ind_bps_yoy', 'ind_dt_netprofit_yoy', 'ind_int_to_talcap', 'interest_coverage', 'intraday_vol_ratio_5d', 'inventory_turnover', 'lf_margin_dip_demand_corr_60', 'limit_broken_ratio_20', 'limit_days_since_up_20', 'limit_down_event_5', 'log_mv', 'luna_minute_return_volume_corr_5d_v1', 'luna_minute_return_volume_lead_lag_5d_v1', 'luna_minute_unexpected_clock_volume_return_5d_v1', 'luna_roll_bounce_proxy_20d_v1', 'luna_turnover_weighted_intraday_return_path_20d_v1', 'margin_repay_deceleration', 'margin_repay_shock', 'mf_amount_weighted_direction', 'mf_big_order_net_ac1_20', 'mf_net_inflow_5d', 'mf_net_vol_surprise_20', 'mf_small_order_avg_price_dev', 'momentum_120', 'momentum_20', 'momentum_250', 'momentum_60', 'ocf_to_profit', 'ocf_to_revenue', 'open5_capacity_floor_ratio_20', 'overnight_ma_20d', 'overnight_ma_5d', 'overnight_ma_60d', 'overnight_std_5d', 'pe_own_history_percentile_252_20', 'price_delay_r2_60', 'price_distance_from_52w_low', 'price_to_52w_high', 'qf_cash_margin_floor_4q', 'qf_cash_margin_yoy_change', 'qf_core_roe_floor_4q', 'qf_core_roe_vol_4q', 'qf_core_roe_yoy_change', 'qf_roa_yoy_change', 'qf_sales_growth_accel', 'qf_sales_growth_vol_4q', 'r10gc_bias_std_turn_21_252_v1_bcd82eb2bc4c4fe7', 'r10gc_bias_turn_126_252_v1_a9656f4fececd030', 'r10gc_bias_turn_21_252_v1_f541ccc9f9fa2581', 'r10gc_bias_turn_63_252_v1_27c60f4ea79e6742', 'r10gc_high_low_21_v1_ca60e882d8b5c22f', 'r10gc_high_low_252_v1_facb1d1f647f48ca', 'r10gc_high_low_63_v1_34a74707a9f916a3', 'r10gc_income_tax_yoy_v1_c051c303608f0a17', 'r10gc_ret_std_21_v1_51f3cc1a8b65d126', 'r10gc_ret_std_252_v1_db7166018e1a0511', 'r10gc_ret_std_63_v1_19b9e9fa082c7cec', 'r10gc_turn_lvl_21_v1_2bab2360d6cb7099', 'r10gc_turn_lvl_252_v1_2dd70c23d17694f4', 'r10gc_turn_lvl_63_v1_7d51db9d4ff08793', 'r10gc_turn_std_126_v1_e96f93f78ed0e472', 'r10gc_turn_std_252_v1_9a4d3d577535e360', 'r10gc_turn_std_42_v1_21349ae0e32c1db7', 'r10gc_yoy_net_asset_v1_2a57af1e5c40492f', 'r10gc_yoy_net_profit_v1_bdc315b5d4b02fac', 'r10gc_yoy_ocf_v1_066e6046572baf9c', 'r10gc_yoy_total_asset_v1_3b5f72f283c0e3ca', 'r11gc_amount_ma_20_v1_1858415feaa5dbe6', 'r11gc_bias_std_turn_126_252_v1_5be1af21b3af2732', 'r11gc_bias_std_turn_126_504_v1_ec735b581d5c6134', 'r11gc_bias_std_turn_21_504_v1_0ec972aa142a07aa', 'r11gc_bias_std_turn_42_252_v1_f671e4d8cc39efd1', 'r11gc_bias_std_turn_42_504_v1_405ae54d3c3ed62f', 'r11gc_bias_std_turn_63_252_v1_d03f938d9f4b2cfe', 'r11gc_bias_std_turn_63_504_v1_54007b4942cea4c1', 'r11gc_bias_turn_126_504_v1_e3ccc91522792d65', 'r11gc_bias_turn_21_504_v1_111533df505215cb', 'r11gc_bias_turn_42_252_v1_fa9e6caf1c763260', 'r11gc_bias_turn_42_504_v1_c66e9ee701bdb2ad', 'r11gc_bias_turn_63_504_v1_387613367cbd4dd9', 'r11gc_high_low_126_v1_2666509099db4a47', 'r11gc_high_low_42_v1_36fd67d17bea68ac', 'r11gc_ret_std_42_v1_23474362384bc399', 'r11gc_turn_lvl_10_v1_c377944b650ec037', 'r11gc_turn_lvl_126_v1_ef48ac322c8d0929', 'r11gc_turn_lvl_20_v1_2f60dce5b6aba8a4', 'r11gc_turn_lvl_42_v1_24d04a044ce607a0', 'r11gc_turn_lvl_5_v1_332d53f4bb907a79', 'r11gc_turn_std_63_v1_5ef776ad40ef77f6', 'r11u2_peak_timing_mktstate_gate_fix1_83f5c455e951bbb3', 'r11u4_flow_intensity_x_surprise_v1_1cec9f0321fe52c1', 'r12gc_asset_turnover_v1_40d616e07fd4a097', 'r12gc_cash_ratio_v1_7ef2829865ad2cc2', 'r12gc_de_v1_d4cd33b978d63b0a', 'r12gc_eps_q_v1_8f3703e8eb218668', 'r12gc_eps_ttm_v1_2d84c57b9ea5f85d', 'r12gc_eps_y_v1_6f0a67c91531a732', 'r12gc_equity_turnover_v1_4cba36560ca472e5', 'r12gc_fcf_to_market_v1_49c0b5d214b35db0', 'r12gc_fixed_asset_turnover_v1_f5bd2c641032b1a2', 'r12gc_float_size_v1_8b275acb0cb20dec', 'r12gc_gpm_q_v1_842e57e4d0d033e5', 'r12gc_gpm_ttm_v1_7e99428574d400ea', 'r12gc_gpm_y_v1_102eddddb8fac4c4', 'r12gc_log_price_v1_0a22efbc7b6bb1f7', 'r12gc_market_value_leverage_v1_9f3705df7636e0cc', 'r12gc_ncf_to_market_v1_71cf1e0deddc7070', 'r12gc_npm_q_v1_a1e5e8bb0a48de33', 'r12gc_npm_ttm_v1_4e4381d67f1d7391', 'r12gc_npm_y_v1_f78b237b3b7c1b18', 'r12gc_opm_ttm_v1_2827ff5d605bdeeb', 'r12gc_opm_y_v1_6e2f1f9aaf3c6106', 'r12gc_opt_tpro_v1_d832d118e2414259', 'r12gc_quick_ratio_v1_ea13af873ce572fc', 'r12gc_receivable_turnover_v1_458e2aaecafa62b9', 'r12gc_return_42_v1_0032bf23a28de8f7', 'r12gc_roa_q_v1_4b4ae2280384ebde', 'r12gc_roa_y_v1_6e585294932485e4', 'r12gc_roe_y_v1_bca9de9f86ce36dd', 'r12gc_sales_to_market_v1_ff344107e3f2c925', 'r13f_gpm_dispersion_v1_d45cc0c6c766d353', 'r13f_ocf_seasonal_trend_innovation_v1_4c66203abbadc485', 'r13gc_daily_range_tail_spread_63_v1_51e696671ccfee43', 'r13gc_daily_return_quantile_skew_63_v1_42a906e8b444a54e', 'r13gc_daily_turnover_ac1_63_v1_e42ca3d25ca965e6', 'r13gc_fb05_deferred_tax_asset_share_v1_d3599ffb97a55de4', 'r13gc_fb06_long_equity_investment_asset_mix_v1_c051b7080293a254', 'r13gc_fb07_supplier_prepayment_asset_share_v1_8a49e274b2af4e41', 'r13gc_fc01_payroll_vs_goods_cash_mix_v1_009c5259e3988576', 'r13gc_fc02_financial_investment_outflow_share_v1_950a52c23c5ce39a', 'r13gc_fi01_selling_admin_mix_v1_8c8039b595a7f2ab', 'r13gc_fi06_revenue_seasonality_v1_8a12e8686b06f0e2', 'r13gc_financial_quality_cash_tax_paid_asinh_ttm_v2_648e7409d04795dd', 'r13gc_financial_quality_income_tax_revenue_ttm_v2_6bdc1290078a0504', 'r13gc_financial_quality_minority_interest_assets_point_v2_9fead6bc31654eb9', 'r13gc_financial_quality_non_oper_exp_revenue_ttm_v2_0e83ec40a51ad981', 'r13gc_overnight_gap_intraday_response_beta_60_v1_789a747483612449', 'r13gc_same_calendar_month_turnover_profile_v2_a64fbb3e9b93a2dd', 'r13gc_turnover_commonality_beta_126_v1_8be87d88402aac70', 'r13gc_turnover_to_nextday_absret_corr_126_v2_ed7738ff0c879711', 'r13p_fip_return_discreteness_v1_94875f42e1e5efcf', 'r14f15_minority_external_equity_cash_share_v2_2f46ba1f387a53c4', 'r14f1_ccc_seasonal_trend_innovation_v2_0a98f6e1fc8b0970', 'r14f27_supplier_prepayment_direction_v1_348e7f359c7fd769', 'r14f28_tax_cash_refund_share_v1_3904a84026422c3e', 'r14f29_intangible_amort_share_v1_476979a530c60728', 'r14f7_positive_contract_rights_share_v2_3cab6f4117f47708', 'r14gc_session_illiquidity_asymmetry_v1_93978c1ec1b77ea8', 'r14p11_market_lower_tail_coincidence_v2_d02a562b871b1976', 'r14p6_raw_monthly_tk_v1_dee5c4eeffeaaa9f', 'r14p7_signed_amount_reversal_v2_7dd5dbc630f5495e', 'r14p9_negative_shock_variance_feedback_v2_d4ad4585f4444283', 'r15f1_bs_pool_composition_v1_longterm_fix_assets_0e39009c5c8f2db5', 'r15f1_bs_pool_composition_v1_longterm_intangibles_8431a77bc492ef3f', 'r15f1_bs_pool_composition_v1_operating_inventories_0bf01e7fe618a7e8', 'r15f1_bs_pool_composition_v1_operating_receivables_444f672a458e0d1f', 'r15gc_cf01_op_inflow_sales_collection_share_v1_6a13add0e7ebbb79', 'r15gc_cf02_op_outflow_tax_payment_share_v1_1434eaa634455e8c', 'r15gc_cf03_investing_outflow_capex_share_v1_8b4aa1df901e47bb', 'r15gc_cf04_financing_inflow_borrow_share_v1_e4398f1eb4148501', 'r15gc_cf05_financing_outflow_distribution_share_v1_e3e31e9803a73394', 'r15gc_cf06_collection_share_seasonal_deviation_v1_674354f5ce32e8bf', 'r15gc_is01_selling_expense_ratio_v1_d05f1e4dd2122eb4', 'r15gc_is02_admin_expense_ratio_v1_c7c097590031f7e3', 'r15gc_is03_financial_expense_ratio_v1_4b977de0274a7b08', 'r15gc_is04_gross_profit_opex_coverage_v1_410dabf3b40ac2b4', 'r15n14_crossfit_pca_ou_displacement_v1_6de328942e44733b', 'r15net_peer_flow_diffusion_v1_0ea2820cafbe9177', 'r15p13_restricted_first_passage_v1_6734bafa6d4f2ab8', 'r15p14_monthly_component_range_memory_v1_6fe135138d05893d', 'r15t3_overnight_variance_share_v1_09b9ae4f22c98618', 'r15t3_spillover_v1_432027233601b658', 'r16c_ml1_price_volume_monthly_h5_mlp_replay_v2_d9ac78cf139d8d43', 'r16gc_is05_selling_ratio_seasonal_deviation_replay_v1_371a3b541117bf5e', 'r17cg_chip_gini_38d02c6cc01fbd71', 'r17cg_p90_gap_chg20_c91115e3f691152c', 'r17cg_peak_vs_top3_d62c93f9f9b52f89', 'r17cg_skew_chg20_4c9f6345c8034e61', 'r2own_v2_trend_persistence_v1_464e80a3ee6ef0bb', 'r3b_intraday_rv_skewness_v2_44ef55d83a759fba', 'r4a_forecast_landing_gap_v3_972c0653571ab5d8', 'r4a_sue_accel_v1_4b443eb2d4fda83c', 'r4a_sue_seasonal_q_v1_f1def0b01eb0b979', 'r4b_intraday_rv_kurtosis_20_4fdea35833870ebb', 'r4b_intraday_volume_hhi_20_2462d9ca21d9adf8', 'r4c_accrual_ratio_q_v1_8bef7aa2caaf33d9', 'r4c_accrual_ratio_ttm_v1_80e27bcbce7fddb5', 'r4e_idio_max_20_v1_d0d201efd7755e86', 'r4j_same_month_consistency_v1_be7f8ede9c4b5afe', 'r4q_adv_recv_cl_growth_yoy_v1_066cb6dd0e26b3e1', 'r4t_preann_turnover_shock_v1_3a95800106c754f2', 'r4t_preann_vol_expansion_v1_bebf97b2c5260706', 'r7a_tax_expense_seasonal_change_v2_371da87d5055df86', 'r7e_context_salience20_v1_f4efb8789772d003', 'r8e_book_wc_cash_realization_noise20q_v2_3b7ffc1a4a997035', 'r9a_sec_own_rv_beta_v2_ef77e079dd4478fe', 'r9c_abnormal_dispersion_shift_k4_v2_cb653c1abc1fb728', 'r9f_breadth_intraday_slope_x_beta_v1_51445f98a0907e8f', 'raw_price_tick_clustering_20', 'ret_skew_20', 'revenue_cagr_3y', 'reversal_2d', 'roa_ttm', 'roc_12', 'roe_ttm', 'rsi_14', 'rsrs_beta_18', 'rv_term_structure', 's26_liq_up_fragility_v1_b43c747ff9090bc2', 's26_structure_side_hhi_v1_bb77e106a875308a', 'short_term_reversal_5', 'sp_ttm', 'td_setup_count', 'three_black_crows', 'toplist_multi_reason_breadth_20_v1', 'turnover_anomaly_20', 'turnover_f_delta_5', 'turnover_std_20', 'vol_120', 'vol_clustering_20', 'vol_of_rv', 'vol_ratio_ma5_ma20', 'volume_ratio', 'yoy_revenue'], 'fund_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281], 'script_parent': 'V247', 'input_migration': {'snapshot_built_at': '2026-10-24 20:55:35', 'snapshot_stock_feature_count': 282, 'snapshot_market_factor_count': 47, 'parent_unit': 'V247', 'added_features': ['annual_credit_impairment_adjustment_assets_v1', 'annual_report_lag_median3_v1', 'ca_range_excess_5_v1', 'ca_tail_pressure_5_v1', 'calendar_gap_idio_variance_slope_60', 'corwin_schultz_overnight_adjusted_hl_proxy_20', 'disposal_proceeds_intensity_yoy_v1', 'dragon_tiger_institution_net_pressure_20_v1', 'lf_margin_dip_demand_corr_60', 'luna_minute_return_volume_corr_5d_v1', 'luna_minute_return_volume_lead_lag_5d_v1', 'luna_minute_unexpected_clock_volume_return_5d_v1', 'luna_roll_bounce_proxy_20d_v1', 'luna_turnover_weighted_intraday_return_path_20d_v1', 'pe_own_history_percentile_252_20', 'price_delay_r2_60', 'r13f_gpm_dispersion_v1_d45cc0c6c766d353', 'r13f_ocf_seasonal_trend_innovation_v1_4c66203abbadc485', 'r13gc_daily_range_tail_spread_63_v1_51e696671ccfee43', 'r13gc_daily_return_quantile_skew_63_v1_42a906e8b444a54e', 'r13gc_daily_turnover_ac1_63_v1_e42ca3d25ca965e6', 'r13gc_fb05_deferred_tax_asset_share_v1_d3599ffb97a55de4', 'r13gc_fb06_long_equity_investment_asset_mix_v1_c051b7080293a254', 'r13gc_fb07_supplier_prepayment_asset_share_v1_8a49e274b2af4e41', 'r13gc_fc01_payroll_vs_goods_cash_mix_v1_009c5259e3988576', 'r13gc_fc02_financial_investment_outflow_share_v1_950a52c23c5ce39a', 'r13gc_fi01_selling_admin_mix_v1_8c8039b595a7f2ab', 'r13gc_fi06_revenue_seasonality_v1_8a12e8686b06f0e2', 'r13gc_financial_quality_cash_tax_paid_asinh_ttm_v2_648e7409d04795dd', 'r13gc_financial_quality_income_tax_revenue_ttm_v2_6bdc1290078a0504', 'r13gc_financial_quality_minority_interest_assets_point_v2_9fead6bc31654eb9', 'r13gc_financial_quality_non_oper_exp_revenue_ttm_v2_0e83ec40a51ad981', 'r13gc_overnight_gap_intraday_response_beta_60_v1_789a747483612449', 'r13gc_same_calendar_month_turnover_profile_v2_a64fbb3e9b93a2dd', 'r13gc_turnover_commonality_beta_126_v1_8be87d88402aac70', 'r13gc_turnover_to_nextday_absret_corr_126_v2_ed7738ff0c879711', 'r13p_fip_return_discreteness_v1_94875f42e1e5efcf', 'r14f15_minority_external_equity_cash_share_v2_2f46ba1f387a53c4', 'r14f1_ccc_seasonal_trend_innovation_v2_0a98f6e1fc8b0970', 'r14f27_supplier_prepayment_direction_v1_348e7f359c7fd769', 'r14f28_tax_cash_refund_share_v1_3904a84026422c3e', 'r14f29_intangible_amort_share_v1_476979a530c60728', 'r14f7_positive_contract_rights_share_v2_3cab6f4117f47708', 'r14gc_session_illiquidity_asymmetry_v1_93978c1ec1b77ea8', 'r14p11_market_lower_tail_coincidence_v2_d02a562b871b1976', 'r14p6_raw_monthly_tk_v1_dee5c4eeffeaaa9f', 'r14p7_signed_amount_reversal_v2_7dd5dbc630f5495e', 'r14p9_negative_shock_variance_feedback_v2_d4ad4585f4444283', 'r15f1_bs_pool_composition_v1_longterm_fix_assets_0e39009c5c8f2db5', 'r15f1_bs_pool_composition_v1_longterm_intangibles_8431a77bc492ef3f', 'r15f1_bs_pool_composition_v1_operating_inventories_0bf01e7fe618a7e8', 'r15f1_bs_pool_composition_v1_operating_receivables_444f672a458e0d1f', 'r15gc_cf01_op_inflow_sales_collection_share_v1_6a13add0e7ebbb79', 'r15gc_cf02_op_outflow_tax_payment_share_v1_1434eaa634455e8c', 'r15gc_cf03_investing_outflow_capex_share_v1_8b4aa1df901e47bb', 'r15gc_cf04_financing_inflow_borrow_share_v1_e4398f1eb4148501', 'r15gc_cf05_financing_outflow_distribution_share_v1_e3e31e9803a73394', 'r15gc_cf06_collection_share_seasonal_deviation_v1_674354f5ce32e8bf', 'r15gc_is01_selling_expense_ratio_v1_d05f1e4dd2122eb4', 'r15gc_is02_admin_expense_ratio_v1_c7c097590031f7e3', 'r15gc_is03_financial_expense_ratio_v1_4b977de0274a7b08', 'r15gc_is04_gross_profit_opex_coverage_v1_410dabf3b40ac2b4', 'r15n14_crossfit_pca_ou_displacement_v1_6de328942e44733b', 'r15net_peer_flow_diffusion_v1_0ea2820cafbe9177', 'r15p13_restricted_first_passage_v1_6734bafa6d4f2ab8', 'r15p14_monthly_component_range_memory_v1_6fe135138d05893d', 'r15t3_overnight_variance_share_v1_09b9ae4f22c98618', 'r15t3_spillover_v1_432027233601b658', 'r16c_ml1_price_volume_monthly_h5_mlp_replay_v2_d9ac78cf139d8d43', 'r16gc_is05_selling_ratio_seasonal_deviation_replay_v1_371a3b541117bf5e', 'r17cg_chip_gini_38d02c6cc01fbd71', 'r17cg_p90_gap_chg20_c91115e3f691152c', 'r17cg_peak_vs_top3_d62c93f9f9b52f89', 'r17cg_skew_chg20_4c9f6345c8034e61', 'raw_price_tick_clustering_20', 'toplist_multi_reason_breadth_20_v1'], 'removed_features': [], 'active_stock_features': 282}, 'target_anchor': {'topn': 500, 'money': 1500000000.0, 'ic_weight': 0.7}, 'neural_kind': 'ridge_closed_form', 'neural_score_horizon': 1, 'score_transform': 'raw_neural', 'personal_validation': {'money': 100000, 'n': 5, 'period': 1, 'band': 40, 'ic_weight': 0.7, 'drawdown_weight': 0.1, 'absolute_gate': False}, 'train_window': 0, 'neural_market_input': True, 'mlp_layernorm': True, 'huber_aux_weight': 0.1, 'score_horizon_weights': {1: 1.0}, 'legacy_feature_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 32, 34, 35, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 273, 274, 275, 276, 277, 278, 279, 280, 281], 'new_feature_indices': [16, 17, 24, 25, 26, 33, 36, 40, 64, 69, 70, 71, 72, 73, 92, 93, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 256, 272], 'macro_selection': 'fixed train-only ridge; no epoch or regularization selection; macro validation diagnostics', 'ridge_penalty': 0.1, 'ridge_block_days': 32, 'research_idea': 'strict forward selection left delivered fits5-8months stale; refit fixed recipe on all mature pretest history while keeping validation predictions selector-only', 'selection_refit_protocol': True}
if feature_block() != 'factors':
    raise ValueError(f'V4 是全因子正式训练单元，特征块必须是 factors（当前 ={feature_block()}）。\n  · 若想跑 20 列的样例：用 experiments/V1，不要在 V4 里降级。\n  · 若确实要改本单元的特征块：先改 run.py 的 DEFAULT_FEATURE_BLOCK 与本闸，并同步更新 docs/iterations.md 的配置摘要（这属于科研决定，不是运行参数）。')

def head_of(label):
    return RECIPE['output_horizons'].index(int(label.split('_ret_')[1].rstrip('d')))

class MarketGate(nn.Module):

    def __init__(self, market_dim, widths):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(market_dim, 64), nn.GELU())
        self.heads = nn.ModuleList((nn.Linear(64, width) for width in widths))

    def zero_heads(self):
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, market):
        z = self.encoder(market.float())
        return [1.0 + torch.tanh(head(z)) for head in self.heads]

class PredictModel(nn.Module):

    def __init__(self, input_dim, market_dim, horizons=None):
        super().__init__()
        self.horizons = list(horizons or RECIPE['output_horizons'])
        self.market_dim = market_dim
        self.flow_indices = RECIPE['flow_indices']
        self.technical_indices = RECIPE['technical_indices']
        if input_dim != len(RECIPE['feature_names']) or self.horizons != RECIPE['output_horizons']:
            raise ValueError('frozen composite feature/output contract changed')
        self.encoder = nn.Sequential(nn.Linear(len(self.flow_indices), 24), nn.LayerNorm(24), nn.GELU())
        self.recurrent = nn.GRU(24, 32, batch_first=True)
        self.fund_indices = RECIPE['fund_indices']
        self.encoder_fund = nn.Sequential(nn.Linear(len(self.fund_indices), 24), nn.LayerNorm(24), nn.GELU())
        self.recurrent_fund = nn.GRU(24, 32, batch_first=True)
        self.output_layer = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, len(RECIPE['label_horizons'])))
        self.absolute_basis_dim = 3 * len(self.technical_indices) + market_dim + 1
        self.register_buffer('absolute_coef', torch.zeros(2, self.absolute_basis_dim))

    def absolute_basis_numpy(self, current, market=None):
        x = np.clip(np.asarray(current, np.float64)[:, self.technical_indices], -5.0, 5.0)
        parts = [x, np.tanh(x), x * x / (1 + x * x)]
        if self.market_dim:
            if market is None:
                raise ValueError('absolute branch requires current causal market inputs')
            parts.append(np.broadcast_to(np.asarray(market, np.float64), (len(x), self.market_dim)))
        parts.append(np.ones((len(x), 1), np.float64))
        return np.concatenate(parts, axis=1)

    def absolute_forward(self, current, market=None):
        x = current[:, self.technical_indices].float().clamp(-5.0, 5.0)
        parts = [x, torch.tanh(x), x * x / (1 + x * x)]
        if self.market_dim:
            if market is None:
                raise ValueError('absolute branch requires current causal market inputs')
            parts.append(market.float().expand(len(x), -1))
        parts.append(torch.ones((len(x), 1), dtype=x.dtype, device=x.device))
        return torch.cat(parts, dim=1) @ self.absolute_coef.T

    def forward(self, tsdata, market=None):
        if tsdata.ndim != 3 or tsdata.shape[1] != RECIPE['sequence_days']:
            raise ValueError('expected causal eight-session stock input')
        flow = tsdata[:, :, self.flow_indices].float().clamp(-5.0, 5.0)
        hidden, _ = self.recurrent(self.encoder(flow))
        fund = tsdata[:, :, self.fund_indices].float().clamp(-5.0, 5.0)
        hidden_fund, _ = self.recurrent_fund(self.encoder_fund(fund))
        both = torch.cat([hidden[:, -1], hidden_fund[:, -1]], dim=1)
        rank_heads = self.output_layer(both)
        absolute = self.absolute_forward(tsdata[:, -1], market)
        return torch.cat([rank_heads, absolute[:, :1]], dim=1)

def fit_absolute_component(model, panel, train_days, seed):
    caps = RECIPE['absolute_sample_caps']
    dim = model.absolute_basis_dim
    xx = np.zeros((len(caps), dim, dim), np.float64)
    xy = np.zeros((len(caps), dim, 2), np.float64)
    yy = np.zeros((len(caps), 2), np.float64)
    dates = np.zeros(len(caps), np.int64)
    samples = np.zeros(len(caps), np.int64)
    for day in train_days:
        d = int(day)
        known = panel.technical_mask(d)
        for h in [1, 5]:
            known &= np.isfinite(panel.Y[f'label_ret_{h}d'][d])
        candidates = np.flatnonzero(known)
        if len(candidates) < 3:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), d]))
        selected = candidates[rng.permutation(len(candidates))[:caps[-1]]]
        phi = model.absolute_basis_numpy(panel.X[d, selected], panel.M[d] if model.market_dim else None)
        y = np.column_stack([np.clip(panel.Y[f'label_ret_{h}d'][d, selected], -RECIPE['absolute_target_clip'][h], RECIPE['absolute_target_clip'][h]) for h in [1, 5]])
        gram = np.zeros((dim, dim))
        cross = np.zeros((dim, 2))
        sq = np.zeros(2)
        previous = 0
        for stage, cap in enumerate(caps):
            n = min(cap, len(selected))
            xblock = phi[previous:n]
            yblock = y[previous:n]
            gram += xblock.T @ xblock
            cross += xblock.T @ yblock
            sq += (yblock * yblock).sum(axis=0)
            xx[stage] += gram / n
            xy[stage] += cross / n
            yy[stage] += sq / n
            dates[stage] += 1
            samples[stage] += n
            previous = n
    if np.any(dates == 0):
        raise ValueError('absolute branch has no valid training dates')
    penalty = np.full(dim, RECIPE['absolute_stock_ridge'], np.float64)
    penalty[3 * len(model.technical_indices):-1] = RECIPE['absolute_market_ridge']
    penalty[-1] = RECIPE['absolute_intercept_ridge']
    stages = []
    for i, cap in enumerate(caps):
        gram = xx[i] / dates[i]
        cross = xy[i] / dates[i]
        normal = gram + np.diag(penalty)
        coef = np.linalg.solve(normal, cross)
        residual = float(np.linalg.norm(normal @ coef - cross) / (np.linalg.norm(cross) + 1e-20))
        if not np.isfinite(coef).all() or residual > 1e-08:
            raise FloatingPointError('absolute normal equation failure')
        mse = np.maximum(np.einsum('dh,de,eh->h', coef, gram, coef) - 2 * (coef * cross).sum(axis=0) + yy[i] / dates[i], 0.0)
        normalized_loss = sum((RECIPE['absolute_label_weights'][h] * mse[j] / RECIPE['absolute_loss_scale'][h] ** 2 for j, h in enumerate([1, 5])))
        normalized_loss /= sum(RECIPE['absolute_label_weights'].values())
        stages.append(dict(coef=coef.T.astype(np.float32), fit_dates=int(dates[i]), sample_rows=int(samples[i]), stock_sample_cap=int(cap), normal_equation_relative_residual=residual, train_loss=float(normalized_loss)))
    averaged = np.mean(np.stack([x['coef'] for x in stages]), axis=0, dtype=np.float32)
    with torch.no_grad():
        model.absolute_coef.copy_(torch.from_numpy(averaged).to(model.absolute_coef.device))
    return dict(seed=int(seed), horizons=[1, 5], sample_caps=list(caps), stages=stages, bag_rule='all fixed training fits equally; no validation/test stage choice', origin='fresh fits from this unit training partition; no other checkpoint read')

def recipe_signature():
    return hashlib.sha256(json.dumps(RECIPE, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

@torch.no_grad()
def predict_absolute(model, panel, days, device):
    model.eval()
    out = np.full((len(days), len(panel.codes)), np.nan, np.float32)
    for i, day in enumerate(days):
        d = int(day)
        known = panel.mask(d)
        if known.any():
            x = torch.from_numpy(panel.X[d, known]).to(device)
            market = torch.from_numpy(panel.M[d]).to(device) if panel.market_dim else None
            out[i, known] = model.absolute_forward(x, market)[:, 0].cpu().numpy()
    return out

def wpcc(preds, y):
    p = preds.reshape(-1)
    y = y.reshape(-1)
    n = p.numel()
    if n < 2:
        return p.sum() * 0
    order = torch.argsort(p, descending=True, stable=True)
    w = torch.empty_like(p)
    w[order] = torch.pow(RECIPE['loss_rank_decay'], torch.arange(n, device=p.device, dtype=p.dtype) / (n - 1))
    w_sum = w.sum()
    cov = (p * y * w).sum() / w_sum - (p * w).sum() * (y * w).sum() / w_sum.square()
    vp = ((p - p.mean()).square() * w).sum() / w_sum
    vy = ((y - y.mean()).square() * w).sum() / w_sum
    return -cov / (vp * vy).clamp_min(1e-16).sqrt()

def metadata():
    return json.loads((data_root() / 'meta.json').read_text())

def training_info(recipe):
    return dict(recipe=recipe, feature_source=str(data_root() / recipe['feature_block']), feature_columns=feature_columns(metadata()), market_columns=market_columns(metadata()) if recipe['market_gate'] else [])

def frame(root, kind, year, columns):
    p = root / kind / f'year={year}' / 'data.parquet'
    df = pq.ParquetFile(p).read(columns=['trade_date', 'stock_code'] + list(columns)).to_pandas()
    df['trade_date'] = df['trade_date'].astype(str).str[:10]
    df['stock_code'] = df['stock_code'].astype(str)
    if df.duplicated(['trade_date', 'stock_code']).any():
        raise ValueError(f'{p}: 重复主键')
    return df.set_index(['trade_date', 'stock_code']).sort_index()
PRICE_COLUMNS = ('open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'vol', 'adj_factor')

def price_frame(root, year, index):
    p = Path(root) / f'year={year}' / 'data.parquet'
    part = pq.ParquetFile(p).read(columns=['trade_date', 'stock_code', *PRICE_COLUMNS]).to_pandas()
    part['trade_date'] = part['trade_date'].astype(str).str[:10]
    part['stock_code'] = part['stock_code'].astype(str)
    if part.duplicated(['trade_date', 'stock_code']).any():
        raise ValueError(f'{p}: 重复主键')
    part = part.set_index(['trade_date', 'stock_code'])
    want = set(index.get_level_values(0))
    days_seen = set(part.index.get_level_values(0))
    if days_seen != want:
        raise ValueError(f'{year} 价格块日线轴与 factors 不一致（缺 {sorted(want - days_seen)[:3]} / 多 {sorted(days_seen - want)[:3]}）—— 回测会错位；跑 preparingdata.py --prices-only 重建')
    return part.reindex(index)

def market_frame(root, year, columns):
    p = root / RECIPE['market_block'] / f'year={year}' / 'data.parquet'
    df = pq.ParquetFile(p).read(columns=['trade_date'] + list(columns)).to_pandas()
    df['trade_date'] = df['trade_date'].astype(str).str[:10]
    if df['trade_date'].duplicated().any():
        raise ValueError(f'{p}: 市场块 trade_date 重复')
    return df.set_index('trade_date').sort_index()

def atomic_json(path, value):
    path = output_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)

def save_torch(path, value):
    path = output_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)

def save_npy(path, value):
    path = output_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        np.save(f, value)
    os.replace(tmp, path)

def axis():
    root = data_root()
    meta = metadata()
    days = []
    for y in sorted(meta['built_years']):
        d = pq.ParquetFile(root / feature_block() / f'year={y}' / 'data.parquet').read(columns=['trade_date'])
        days.extend(sorted(set(d.column(0).to_pylist())))
    return (np.asarray(days), np.asarray(meta['axis']['codes']))

def quarters(days):
    return ['2025Q3', '2025Q4', '2026Q1', '2026Q2']

def splits(days, purge_horizon=RECIPE['purge_horizon'], folds=4, quarter=None):
    quarter = quarter or quarters(days)[0]
    q = pd.Period(quarter, freq='Q')
    start = str(q.start_time.date())
    end = str(q.end_time.date())
    ix = np.arange(len(days))
    first = int(np.searchsorted(days, start))
    cutoff = first - purge_horizon - 1
    eligible = ix[:cutoff]
    test = ix[(days >= max(start, '2025-07-01')) & (days <= min(end, '2026-06-30'))]
    score = ix[first:]
    out = []
    for k, block in enumerate(np.array_split(eligible, folds), 1):
        a, b = (int(block[0]), int(block[-1]) + 1)
        valid = block[block + purge_horizon + 1 < b]
        train = eligible[(eligible + purge_horizon + 1 < a) | (eligible >= b + purge_horizon + 1)]
        if not len(train) or not len(valid) or (not len(test)) or (not len(score)):
            raise ValueError('训练/验证/测试/推演窗为空')
        assert max(train.max(), valid.max()) + purge_horizon + 1 < first
        assert not np.intersect1d(train, valid).size
        assert test[0] == score[0] and test[-1] == score[len(test) - 1]
        out.append(dict(quarter=quarter, fold=k, train=train, valid=valid, test=test, score=score))
    return out

def split_summary(days, split):

    def spans(ix):
        runs = np.split(ix, np.flatnonzero(np.diff(ix) > 1) + 1)
        return [{'start': str(days[r[0]]), 'end': str(days[r[-1]]), 'days': len(r)} for r in runs if len(r)]
    return {'quarter': split['quarter'], 'fold': split['fold'], **{k: spans(split[k]) for k in ['train', 'valid', 'test', 'score']}}

class Panel:

    def __init__(self, features='all', load_x=True):
        self.root = data_root()
        self.meta = metadata()
        self.days, self.codes = axis()
        if str(self.days[-1]) != RECIPE['snapshot_cutoff']:
            raise ValueError('snapshot changed: create a new independent version')
        self.features = feature_columns(self.meta) if features in ('all', RECIPE['features']) else self.meta['feature_sets'][features]
        if not set(self.features).issubset(feature_columns(self.meta)):
            raise ValueError('请求的特征不在选定数据块中')
        self.labels = self.meta['columns']['labels']
        self.market_features = market_columns(self.meta) if RECIPE['market_gate'] else []
        self.market_dim = len(self.market_features)
        T, C = (len(self.days), len(self.codes))
        self.X = np.empty((T, C, len(self.features)), np.float32) if load_x else None
        self.coverage = np.empty((T, C), np.float32) if load_x else None
        self.flow_coverage = np.empty((T, C), np.float32) if load_x else None
        self.fund_coverage = np.empty((T, C), np.float32) if load_x else None
        self.technical_coverage = np.empty((T, C), np.float32) if load_x else None
        self.Y = {name: np.full((T, C), np.nan, np.float32) for name in self.labels}
        self.amount = np.full((T, C), np.nan, np.float64)
        self.M = np.zeros((T, self.market_dim), np.float32)
        t_load = time.time()
        years = list(self.meta['built_years'])
        filled = 0
        for yi, y in enumerate(years, 1):
            rows = np.flatnonzero(np.char.startswith(self.days, str(y)))
            filled += len(rows)
            index = pd.MultiIndex.from_product([self.days[rows], self.codes], names=['trade_date', 'stock_code'])
            if load_x:
                df = frame(self.root, feature_block(), y, self.features)
                if len(df) != len(index) or not df.index.equals(index):
                    raise ValueError(f'{y} {feature_block()} 轴不一致')
                x = df.to_numpy(np.float32).reshape(len(rows), C, -1)
                self.coverage[rows] = np.isfinite(x).mean(axis=2)
                self.flow_coverage[rows] = np.isfinite(x[:, :, RECIPE['flow_indices']]).mean(axis=2)
                self.fund_coverage[rows] = np.isfinite(x[:, :, RECIPE['fund_indices']]).mean(axis=2)
                self.technical_coverage[rows] = np.isfinite(x[:, :, RECIPE['technical_indices']]).mean(axis=2)
                x = np.nan_to_num(x, nan=0.0)
                if not np.isfinite(x).all():
                    raise ValueError(f'{y} 年特征含无穷值')
                self.X[rows] = x
                del df, x
            if self.market_dim:
                mkt = market_frame(self.root, y, self.market_features)
                if not mkt.index.equals(pd.Index(self.days[rows])):
                    raise ValueError(f'{y} 市场块日期轴与 {feature_block()} 不一致')
                self.M[rows] = np.clip(np.nan_to_num(mkt.to_numpy(np.float32), nan=0.0), -RECIPE['market_clip'], RECIPE['market_clip'])
            df = frame(self.root, 'target', y, self.labels).reindex(index)
            for name in self.labels:
                self.Y[name][rows] = df[name].to_numpy(np.float32).reshape(len(rows), C)
            self.amount[rows] = frame(self.root, 'amount', y, ['amount']).reindex(index)['amount'].to_numpy().reshape(len(rows), C)
            print(f'已读取 {y} 年，{len(rows)} 个交易日（{yi}/{len(years)}，已用 {time.time() - t_load:.0f}s，预计还需 {(time.time() - t_load) / yi * (len(years) - yi):.0f}s）', flush=True)
        if load_x and filled != T:
            raise ValueError(f'特征面板只填了 {filled}/{T} 个交易日 —— 年份分区没覆盖整条日期轴')
        if not np.isfinite(self.M).all():
            raise ValueError('市场块含无穷值')
        print(f"面板读取完成：{len(self.meta['built_years'])} 年，市场列 {self.market_dim}，已用 {time.time() - t_load:.0f}s", flush=True)

    def inputs(self, day, mask):
        width = RECIPE['sequence_days']
        if day < width - 1:
            raise ValueError('insufficient causal history')
        return np.ascontiguousarray(self.X[day - width + 1:day + 1, mask].transpose(1, 0, 2))

    def technical_mask(self, day):
        if self.technical_coverage is None:
            raise ValueError('feature coverage not loaded')
        return self.technical_coverage[int(day)] >= RECIPE['min_feature_coverage']

    def mask(self, day, label=None):
        width = RECIPE['sequence_days']
        day = int(day)
        if day < width - 1:
            return np.zeros(len(self.codes), dtype=bool)
        if self.flow_coverage is None:
            raise ValueError('feature coverage not loaded')
        out = (self.flow_coverage[day - width + 1:day + 1] >= RECIPE['min_feature_coverage']).all(axis=0)
        out &= (self.fund_coverage[day - width + 1:day + 1] >= RECIPE['min_feature_coverage']).all(axis=0)
        out &= self.technical_mask(day)
        if label is not None:
            out &= np.isfinite(self.Y[label][day])
        return out

class Prices:

    def __init__(self, panel):
        names = list(PRICE_COLUMNS)
        T, C = (len(panel.days), len(panel.codes))
        self.raw = {n: np.full((T, C), np.nan, np.float64) for n in names}
        for year in panel.meta['built_years']:
            ix = np.flatnonzero(np.char.startswith(panel.days, str(year)))
            index = pd.MultiIndex.from_product([panel.days[ix], panel.codes], names=['trade_date', 'stock_code'])
            df = price_frame(price_root(), year, index)
            for n in names:
                self.raw[n][ix] = df[n].to_numpy().reshape(len(ix), C)
        self.adj = pd.DataFrame(self.raw['adj_factor']).ffill().to_numpy()
        self.close = pd.DataFrame(self.raw['close']).ffill().to_numpy()
        self.open = self.raw['open']
        pre = self.raw['pre_close']
        upper = np.floor(pre * 1.1 * 100 + 0.5) / 100
        lower = np.floor(pre * 0.9 * 100 + 0.5) / 100
        self.limit_up = upper
        self.limit_down = lower
        traded = np.isfinite(self.open) & (self.open > 0) & np.isfinite(pre) & (pre > 0) & np.isfinite(self.adj) & (self.adj > 0) & (self.raw['vol'] > 0)
        self.entry = traded & (self.open < upper - 1e-08)
        self.exit = traded & (self.open > lower + 1e-08)
        self.board_entry = traded & (self.raw['low'] < upper - 1e-08)
        self.next_entry = np.zeros_like(self.entry)
        self.next_entry[:-1] = self.entry[1:]
        self.next_board_entry = np.zeros_like(self.board_entry)
        self.next_board_entry[:-1] = self.board_entry[1:]
        raw_close = self.raw['close']
        self.exit_close = np.isfinite(raw_close) & (raw_close > 0) & np.isfinite(pre) & (pre > 0) & np.isfinite(self.adj) & (self.adj > 0) & (self.raw['vol'] > 0) & (raw_close > lower + 1e-08)
        self._hot_pool(panel)

    def _hot_pool(self, panel):
        spec = RECIPE['hot_pool']
        frac, aw, lw = (spec['frac'], spec['amount_window'], spec['limit_window'])
        T, C = (len(panel.days), len(panel.codes))
        self.hot_frac = frac
        self.hot = np.ones((T, C), bool)
        self.hot_score = np.full((T, C), np.nan, np.float32)
        if frac >= 1.0:
            return
        amount = panel.amount
        base = pd.DataFrame(amount).rolling(aw, min_periods=5).mean().shift(1).to_numpy()
        with np.errstate(invalid='ignore', divide='ignore'):
            activity = np.where((base > 0) & np.isfinite(amount), amount / base, np.nan)
        pre = self.raw['pre_close']
        with np.errstate(invalid='ignore', divide='ignore'):
            amp = np.where((pre > 0) & np.isfinite(self.raw['high']) & np.isfinite(self.raw['low']), (self.raw['high'] - self.raw['low']) / pre, np.nan)
        amplitude = pd.DataFrame(amp).rolling(spec['amp_window'], min_periods=2).mean().to_numpy()
        raw_close = self.raw['close']
        touch = np.isfinite(raw_close) & (raw_close > 0) & (pre > 0) & (self.raw['vol'] > 0) & (raw_close >= self.limit_up - 1e-08)
        boards = pd.DataFrame(touch.astype(np.float64)).rolling(lw, min_periods=1).sum().to_numpy()
        bare = []
        for d in range(T):
            ranked = []
            for values in (activity[d], amplitude[d], boards[d]):
                ok = np.isfinite(values)
                if ok.sum() < spec['min_names']:
                    continue
                r = np.full(C, np.nan)
                r[ok] = (rankdata(values[ok]) - 1) / (ok.sum() - 1)
                ranked.append(r)
            if len(ranked) < 2:
                bare.append(d)
                continue
            score = np.nanmean(np.stack(ranked), axis=0)
            self.hot_score[d] = score.astype(np.float32)
            ok = np.isfinite(score)
            if ok.sum() < spec['min_names'] or float(np.ptp(score[ok])) < 1e-12:
                bare.append(d)
                continue
            self.hot[d] = ok & (score >= np.quantile(score[ok], 1.0 - frac))
        if bare:
            print(f"""热门池：{len(bare)} 天算不出来（预热期，按"不设池"处理，池内=全截面）：{[str(panel.days[d]) for d in bare[:5]]}{(' …' if len(bare) > 5 else '')}""", flush=True)

    def entry_ok(self):
        rule = RECIPE['entry_rule']
        if rule == 'open':
            return self.entry
        if rule == 'board':
            return self.board_entry
        raise ValueError(f"未知 entry_rule={rule!r}；见 RECIPE['entry_rule']")

    def next_entry_ok(self):
        rule = RECIPE['entry_rule']
        if rule == 'open':
            return self.next_entry
        if rule == 'board':
            return self.next_board_entry
        raise ValueError(f"未知 entry_rule={rule!r}；见 RECIPE['entry_rule']")

    def verify_labels(self, panel, days):
        px = pd.DataFrame(self.open).ffill().to_numpy() * self.adj
        errors = []
        for h in [1, 3, 5, 10, 20]:
            ix = np.asarray(days)
            ix = ix[ix + h + 1 < len(px)]
            calc = px[ix + h + 1] / px[ix + 1] - 1
            target = panel.Y[f'label_ret_{h}d'][ix]
            good = np.isfinite(calc) & np.isfinite(target)
            diff = np.abs(calc[good] - target[good])
            errors.append({'horizon': h, 'checked': int(good.sum()), 'max_abs_error': float(diff.max()) if len(diff) else None})
            if not len(diff) or np.mean(diff > 1e-05) > 0.001:
                raise ValueError(f'价格与 {h}d 标签口径不一致: {errors[-1]}')
        return errors

def corr(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def device_for(request):
    if request == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if request == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('指定 CUDA 但不可用')
    return torch.device(request)

@torch.no_grad()
def predict_heads(model, panel, days, device):
    model.eval()
    out = np.full((len(days), len(panel.codes), len(model.horizons)), np.nan, np.float32)
    for i, d in enumerate(days):
        mask = panel.mask(d)
        if mask.sum():
            x = torch.from_numpy(panel.inputs(d, mask)).to(device)
            m = torch.from_numpy(panel.M[d]).to(device) if panel.market_dim else None
            out[i, mask] = model(x, m).cpu().numpy()
    return out

def score_of(heads):
    if RECIPE['score_blend'] != 'flow_technical_percentile' or heads.shape[-1] != 4:
        raise ValueError('unexpected composite head schema')
    out = np.full(heads.shape[:2], np.nan, np.float32)
    weight = RECIPE['flow_fusion_weight']
    for i, row in enumerate(heads):
        known = np.isfinite(row).all(axis=1)
        n = known.sum()
        if n > 1:
            flow = (rankdata(row[known, 1]) - 0.5) / n
            technical = (rankdata(row[known, 3]) - 0.5) / n
            out[i, known] = weight * flow + (1 - weight) * technical
    return out

def fit_ridge(panel, train_days, lam=1000.0, pool=None):
    if not RECIPE['linear_weight']:
        return None
    D = len(panel.features)
    G = np.zeros((D + 1, D + 1))
    b = np.zeros(D + 1)
    for d in train_days:
        m = panel.mask(d, 'label_ret_5d')
        if pool is not None:
            m = m & pool[d]
        if m.sum() < 50:
            continue
        X = panel.X[d, m].astype(np.float64)
        y = panel.Y['label_ret_5d'][d, m].astype(np.float64)
        sd = y.std()
        if sd < 1e-08:
            continue
        y = (y - y.mean()) / sd
        A = np.c_[X, np.ones(len(X))]
        G += A.T @ A
        b += A.T @ y
    G[np.arange(D), np.arange(D)] += lam
    return np.linalg.solve(G, b)

def blend_linear(score, w, panel, days):
    lw = RECIPE['linear_weight']
    if not lw or w is None:
        return score
    out = score.copy()
    for i, d in enumerate(days):
        s = score[i]
        known = np.isfinite(s)
        if known.sum() < 20:
            continue
        r = panel.X[d, known].astype(np.float64) @ w[:-1] + w[-1]
        sd = r.std()
        if sd < 1e-12:
            continue
        r = (r - r.mean()) * (s[known].std() / sd) + s[known].mean()
        out[i, known] = (1.0 - lw) * s[known] + lw * r
    return out

def average_state_dicts(states):
    averaged = {}
    for key, first in states[0].items():
        if key == 'absolute_coef':
            if not all((torch.equal(first, s[key]) for s in states)):
                raise ValueError('static absolute fit changed across epochs')
            averaged[key] = first.clone()
            continue
        if not first.is_floating_point():
            averaged[key] = first.clone()
            continue
        averaged[key] = torch.stack([state[key].float() for state in states]).mean(dim=0).to(first.dtype)
    return averaged

def recalibrate_bn(model, panel, train_days, device):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.reset_running_stats()
            module.momentum = None
    with torch.no_grad():
        for d in train_days:
            mask = panel.mask(d)
            if mask.sum() < 3:
                continue
            x = torch.from_numpy(panel.inputs(d, mask)).to(device)
            m = torch.from_numpy(panel.M[d]).to(device) if panel.market_dim else None
            model(x, m)
    model.eval()

def hot_filter(score, prices, days):
    frac = RECIPE['hot_pool']['frac']
    if frac != prices.hot_frac:
        raise ValueError(f'热门池掩码是按 frac={prices.hot_frac} 算的，当前 RECIPE 写的是 {frac} —— 掩码不会自动重算，直接跑会静默沿用旧掩码。\n  · 要换 frac：重建 Prices（`Prices(panel)` 会在构造时按新 frac 重算掩码）。\n  · 要做 A/B：改完 frac 必须重建，否则两边拿到的是同一份掩码。')
    if frac >= 1.0:
        return score
    out = score.copy()
    for i, d in enumerate(days):
        out[i][~prices.hot[d]] = np.nan
    return out

def predict(model, panel, prices, days, device):
    return hot_filter(blend_linear(score_of(predict_heads(model, panel, days, device)), getattr(model, 'ridge', None), panel, days), prices, days)

def validation(model, panel, prices, days, device):
    heads = predict_heads(model, panel, days, device)
    score = hot_filter(blend_linear(score_of(heads), getattr(model, 'ridge', None), panel, days), prices, days)
    ic, rank = ({}, {})
    for h in (1, 3, 5, 10, 20):
        label = f'label_ret_{h}d'
        pe, sp = ([], [])
        for i, d in enumerate(days):
            y = panel.Y[label][d]
            p = score[i]
            known = np.isfinite(p)
            ok = known & np.isfinite(y)
            sp.append(corr(rankdata(p[ok]), rankdata(y[ok])))
            pe.append(corr(p[known], np.where(np.isfinite(y[known]), y[known], 0.0)))
        ic[label] = float(np.mean(pe)) if pe else 0.0
        rank[label] = float(np.mean(sp)) if sp else 0.0
    ic_term = sum((RECIPE['valid_ic_weights'][h] * rank[f'label_ret_{h}d'] for h in RECIPE['label_horizons']))
    criterion = RECIPE['valid_criterion']
    cash_info = {}
    if criterion not in ('rankic', 'tradable_topn', 'tradable_topn_net', 'capacity_wei', 'pearson1d', 'personal_cash', 'macro_stable'):
        raise ValueError(f"未知 valid_criterion={criterion!r}；见 RECIPE['valid_criterion']")
    next_entry = prices.next_entry_ok()
    preds = score
    rets = []
    unknown_picks = 0
    for p, d in zip(preds, days):
        y = panel.Y[RECIPE['score_label']][d]
        can = np.flatnonzero(np.isfinite(p) & next_entry[d])
        picks = can[np.argsort(-p[can], kind='stable')[:RECIPE['valid_topn']]]
        miss = ~np.isfinite(y[picks])
        unknown_picks += int(miss.sum())
        rets.append(float(np.where(miss, 0.0, y[picks]).sum() / RECIPE['valid_topn']) if len(picks) else 0.0)
    ret = float(np.mean(rets))
    label5 = RECIPE['score_label']
    if label5 not in ic:
        label5 = f"label_ret_{max(RECIPE['label_horizons'])}d"
    primary = ret if criterion == 'tradable_topn' else RECIPE['valid_ic_scale'] * ic_term
    if criterion == 'pearson1d':
        primary = ic['label_ret_1d']
    elif criterion == 'capacity_wei':
        anchor = RECIPE['target_anchor']
        cap_returns = []
        for p, d in zip(score, days):
            ids = np.flatnonzero(np.isfinite(p))
            ids = ids[np.argsort(-p[ids], kind='stable')[:anchor['topn']]]
            cap = np.maximum(np.nan_to_num(panel.amount[d, ids], nan=0.0), 0.0)
            held = np.minimum(anchor['money'], np.cumsum(cap))
            w = np.diff(np.r_[0.0, held]) / anchor['money']
            y = np.nan_to_num(panel.Y['label_ret_1d'][d, ids], nan=0.0)
            cap_returns.append(float(w @ y))
        primary = 100.0 * np.mean(cap_returns) + anchor['ic_weight'] * ic['label_ret_1d']
    if criterion == 'personal_cash':
        spec = RECIPE['personal_validation']
        stat, curve, trades = personal_validation_cash(score, panel, prices, days, n=spec['n'], period=spec['period'], band=spec['band'], money=spec['money'], cost_gate_multiplier=0.0)
        growth = 100.0 * np.log1p(stat['return_value']) / len(curve)
        primary = growth + spec['ic_weight'] * ic['label_ret_1d'] + 100.0 * spec['drawdown_weight'] * stat['max_drawdown']
        cash_info = {'validation_cash': stat, 'validation_cash_params': spec, 'validation_cash_daily_log_pct': growth, 'validation_cash_role': 'epoch selection only; n5 diversified execution; labels and prices within valid plus reserved exit embargo'}
    return dict(**cash_info, val_wei=primary, val_criterion=criterion, val_return_proxy=ret, val_ic_weighted=ic_term, val_unknown_label_picks=unknown_picks, Pearson_IC=ic[label5], RankIC=rank[label5], **{f'IC_{h}d': ic[f'label_ret_{h}d'] for h in (1, 3, 5, 10, 20)}, **{f'RankIC_{h}d': rank[f'label_ret_{h}d'] for h in (1, 3, 5, 10, 20)})

def train_fold(fold, device='auto', panel=None, prices=None, quarter=None, absolute_only=False):
    recipe = copy.deepcopy(RECIPE)
    recipe['device'] = device
    quarter = quarter or quarters(axis()[0])[0]
    recipe['quarter'] = quarter
    info = training_info(recipe)
    signature = recipe_signature()
    out = RUN_ROOT / 'model_train' / quarter / f'fold{fold}'
    output_file(out / 'complete.json')
    out.mkdir(parents=True, exist_ok=True)
    done = out / 'complete.json'
    if done.exists():
        old = json.loads(done.read_text())
        if old.get('recipe_signature') != signature:
            raise ValueError('completed recipe fingerprint changed; never reuse stale folds')
        if not all(((out / name).is_file() for name in ('best.pt', 'test_predictions.npy', 'score_predictions.npy'))):
            raise RuntimeError('完成记录存在但产物缺失')
        print(f'fold{fold} 已完成，跳过', flush=True)
        return old
    torch.set_num_threads(recipe['threads'])
    dev = device_for(device)
    seed = recipe['seed'] + fold - 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if dev.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    panel = panel or Panel(recipe['features'])
    prices = prices or Prices(panel)
    split = splits(panel.days, recipe['purge_horizon'], recipe['folds'], quarter)[fold - 1]
    atomic_json(out / 'training_info.json', info)
    atomic_json(out / 'split.json', split_summary(panel.days, split))
    model = PredictModel(len(panel.features), panel.market_dim).to(dev)
    if (out / 'last.pt').exists():
        prepared = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        if prepared.get('recipe_signature') != signature:
            raise ValueError('resume recipe fingerprint changed')
        absolute_fit_info = prepared['absolute_fit']
    else:
        absolute_fit_info = fit_absolute_component(model, panel, split['train'], seed)
    absolute_fit_summary = {k: v for k, v in absolute_fit_info.items() if k != 'stages'}
    absolute_fit_summary['stages'] = [{k: v for k, v in s.items() if k != 'coef'} for s in absolute_fit_info['stages']]
    print('ABSOLUTE_COMPONENT_READY', len(absolute_fit_info['stages']), 'fresh training fits', flush=True)
    model.ridge = fit_ridge(panel, split['train'], pool=prices.hot if recipe['train_pool_frac'] < 1.0 else None)
    print('fold%d linear component %s (training %d days)' % (fold, 'disabled by recipe' if model.ridge is None else 'fitted', len(split['train'])), flush=True)
    if recipe['optimizer'] != 'adamw':
        raise ValueError(f"暂只实现 adamw，收到 {recipe['optimizer']!r}")
    opt = torch.optim.AdamW(model.parameters(), lr=recipe['lr'], weight_decay=recipe['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=recipe['lr_factor'], patience=recipe['lr_patience'], cooldown=recipe['lr_cooldown'], min_lr=recipe['min_lr'])
    start = 0
    best = -float('inf')
    stale = 0
    history = []
    bank = []
    last = out / 'last.pt'
    if last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(dev)
        start = ck['epoch'] + 1
        best = ck['best']
        stale = ck['stale']
        history = ck['history']
        bank = ck.get('bank', [])
        np.random.set_state(ck['numpy_rng'])
        random.setstate(ck['python_rng'])
        torch.set_rng_state(ck['torch_rng'])
        if dev.type == 'cuda':
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        print(f'fold{fold} 恢复 epoch {start}', flush=True)
    print(f"fold{fold} 设备 {dev}，训练 {len(split['train'])} / 验证 {len(split['valid'])} / 测试 {len(split['test'])} 天", flush=True)
    if absolute_only:
        if not last.exists():
            save_torch(last, dict(recipe_signature=signature, absolute_fit=absolute_fit_info, model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), epoch=-1, best=-float('inf'), stale=0, history=[], bank=[], numpy_rng=np.random.get_state(), python_rng=random.getstate(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if dev.type == 'cuda' else []))
            atomic_json(out / 'history.json', [])
        print('ABSOLUTE_PREPARED_ONLY', quarter, fold, 'neural updates', start, flush=True)
        return dict(quarter=quarter, fold=fold, absolute_prepared=True, neural_epochs=start, recipe_signature=signature, absolute_fit=absolute_fit_summary)
    t0 = time.time()
    for epoch in range(start, recipe['max_epochs']):
        if stale >= recipe['early_stop_patience']:
            break
        model.train()
        order = np.random.permutation(split['train'])
        losses = []
        t = time.time()
        for pos in range(0, len(order), recipe['batch_days']):
            opt.zero_grad(set_to_none=True)
            batch = []
            batch_days=order[pos:pos+recipe['batch_days']]
            inputs=[];markets=[];sizes=[];effective=[]
            for bd in batch_days:
                bm=panel.mask(bd)
                if bm.sum()<3:continue
                inputs.append(panel.inputs(bd,bm))
                markets.append(np.broadcast_to(panel.M[bd],(int(bm.sum()),panel.market_dim)))
                sizes.append(int(bm.sum()));effective.append(int(bd))
            if not effective:continue
            all_pred=model(torch.from_numpy(np.concatenate(inputs)).to(dev),torch.from_numpy(np.concatenate(markets)).to(dev))
            pred_by_day=dict(zip(effective,torch.split(all_pred,sizes)))
            for d in order[pos:pos + recipe['batch_days']]:
                mask = panel.mask(d)
                if recipe['train_pool_frac'] < 1.0:
                    mask = mask & prices.hot[d]
                if mask.sum() < 3:
                    continue
                targets = {}
                for h in recipe['label_horizons']:
                    y = panel.Y[f'label_ret_{h}d'][d, mask]
                    ok = np.isfinite(y)
                    if ok.sum() < 3 or np.std(y[ok]) < 1e-08:
                        continue
                    targets[h] = (ok, y[ok])
                if not targets:
                    continue
                pred = pred_by_day[int(d)]
                terms, weights = ([], [])
                for h, (ok, y) in targets.items():
                    j = recipe['label_horizons'].index(h)
                    yt = torch.from_numpy(y).to(dev)
                    yt = (yt - yt.mean()) / yt.std()
                    rows = torch.from_numpy(np.flatnonzero(ok)).to(dev)
                    prediction = pred[rows, j]
                    if recipe.get('liquidity_loss_weight'):
                        amount = torch.from_numpy(panel.amount[d, mask][ok].astype(np.float32)).to(dev)
                        importance = torch.log1p(torch.nan_to_num(amount).clamp_min(0.0))
                        importance = importance / importance.mean().clamp_min(1e-08)
                        importance = importance.clamp(0.25, 2.0)
                        prediction = prediction * importance.sqrt()
                        yt = yt * importance.sqrt()
                    terms.append(recipe['label_weights'][h] * wpcc(prediction, yt))
                    weights.append(recipe['label_weights'][h])
                batch.append(torch.stack(terms).sum() / sum(weights))
            if not batch:
                continue
            loss = torch.stack(batch).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('WPCC 非有限')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            losses.append(float(loss.detach()))
        metrics = validation(model, panel, prices, split['valid'], dev)
        score = metrics['val_wei']
        scheduler.step(score)
        improved = score > best + 1e-10
        if improved:
            best = score
            stale = 0
            save_torch(out / 'best.pt', dict(recipe_signature=signature, absolute_fit=absolute_fit_info, model=model.state_dict(), input_dim=len(panel.features), market_dim=panel.market_dim, horizons=list(model.horizons), features=panel.features, epoch=epoch, metrics=metrics, ridge=getattr(model, 'ridge', None)))
        else:
            stale += 1
        if recipe['bag_topk']:
            bank.append((score, epoch, {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}))
            bank.sort(key=lambda item: -item[0])
            del bank[recipe['bag_topk']:]
        record = dict(epoch=epoch, train_loss=float(np.mean(losses)), lr=opt.param_groups[0]['lr'], seconds=round(time.time() - t, 2), **metrics)
        history.append(record)
        atomic_json(out / 'history.json', history)
        save_torch(last, dict(recipe_signature=signature, absolute_fit=absolute_fit_info, model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch, best=best, stale=stale, history=history, bank=bank, numpy_rng=np.random.get_state(), python_rng=random.getstate(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if dev.type == 'cuda' else []))
        elapsed = time.time() - t0
        done_n = epoch - start + 1
        eta = elapsed / done_n * (recipe['max_epochs'] - done_n)
        print(f"fold{fold} epoch {epoch + 1}/{recipe['max_epochs']}: loss={record['train_loss']:.5f} val_wei={score:.5f} (ret={metrics['val_return_proxy']:.5f} ICw={metrics['val_ic_weighted']:.5f}) IC in recipe horizon order=" + '/'.join((f"{metrics[f'IC_{h}d']:.4f}" for h in recipe['label_horizons'])) + f" lr={record['lr']:.6g} {record['seconds']:.1f}s best={improved} | {done_n}/{recipe['max_epochs']} 轮，已用 {elapsed / 60:.1f}m，ETA≤{eta / 60:.1f}m（早停提前）", flush=True)
    ck = torch.load(out / 'best.pt', map_location=dev, weights_only=False)
    argmax_epoch, argmax_metrics = (ck['epoch'], ck['metrics'])
    if recipe['bag_topk'] and len(bank) > 1:
        model.load_state_dict(average_state_dicts([state for _, _, state in bank]))
        if recipe['bag_recalibrate']:
            recalibrate_bn(model, panel, split['train'], dev)
        bagged_metrics = validation(model, panel, prices, split['valid'], dev)
        save_torch(out / 'best.pt', dict(recipe_signature=signature, absolute_fit=absolute_fit_info, model=model.state_dict(), input_dim=len(panel.features), market_dim=panel.market_dim, horizons=list(model.horizons), features=panel.features, epoch=argmax_epoch, metrics=bagged_metrics, ridge=ck.get('ridge'), model_argmax=ck['model'], metrics_argmax=argmax_metrics))
    else:
        model.load_state_dict(ck['model'])
        bagged_metrics = argmax_metrics
    model.eval()
    model = model.to('cpu')
    pred = predict(model, panel, prices, split['score'], torch.device('cpu'))
    save_npy(out / 'score_predictions.npy', pred)
    save_npy(out / 'test_predictions.npy', pred[:len(split['test'])])
    result = dict(recipe_signature=signature, absolute_fit=absolute_fit_summary, quarter=quarter, fold=fold, best_epoch=argmax_epoch + 1, epochs=len(history), test_dates=panel.days[split['test']].tolist(), score_dates=panel.days[split['score']].tolist(), best_validation=bagged_metrics, argmax_validation=argmax_metrics, bag=dict(topk=recipe['bag_topk'], used=len(bank), epochs=[int(e) + 1 for _, e, _ in bank], scores=[float(s) for s, _, _ in bank]), peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, seconds=round(time.time() - t0, 2), device=str(dev))
    atomic_json(done, result)
    print(f"fold{fold} EXIT:0（bag {len(bank)}/{recipe['bag_topk']} 份，轮号 {result['bag']['epochs']}；valid val_wei argmax {argmax_metrics['val_wei']:.5f} → bagged {bagged_metrics['val_wei']:.5f}）", flush=True)
    return result

def rescore(quarter_list, fold_list, device='auto', panel=None, prices=None):
    dev = device_for(device)
    torch.set_num_threads(RECIPE['threads'])
    panel = panel or Panel(RECIPE['features'])
    prices = prices or Prices(panel)
    for quarter in quarter_list:
        for fold in fold_list:
            out = RUN_ROOT / 'model_train' / quarter / f'fold{fold}'
            done_path = out / 'complete.json'
            if not done_path.exists():
                print(f'{quarter} fold{fold} 未完成，跳过', flush=True)
                continue
            done = json.loads(done_path.read_text())
            if not (out / 'best.pt').exists():
                raise RuntimeError(f'{quarter} fold{fold} 缺 best.pt')
            split = splits(panel.days, RECIPE['purge_horizon'], RECIPE['folds'], quarter)[fold - 1]
            ck = torch.load(out / 'best.pt', map_location=dev, weights_only=False)
            model = PredictModel(len(panel.features), panel.market_dim).to(dev)
            model.load_state_dict(ck['model'])
            model.ridge = ck.get('ridge')
            pred = predict(model, panel, prices, split['score'], dev)
            save_npy(out / 'score_predictions.npy', pred)
            save_npy(out / 'test_predictions.npy', pred[:len(split['test'])])
            atomic_json(out / 'split.json', split_summary(panel.days, split))
            done['score_dates'] = panel.days[split['score']].tolist()
            done['test_dates'] = panel.days[split['test']].tolist()
            done['best_validation'] = validation(model, panel, prices, split['valid'], dev)
            atomic_json(done_path, done)
            print(f"{quarter} fold{fold} 重推演完成：推演窗 {pred.shape[0]} 天（其中评价 {len(split['test'])} 天），best_validation 已按当前口径重算", flush=True)

def main():
    parser = ArgumentParser(description='V2：全因子 + 市场门控 + 多目标（1/3/5/10/20d）四折模型，单折训练（断点续跑）')
    parser.add_argument('--fold', type=int, choices=[1, 2, 3, 4], default=None, help='只跑指定折')
    parser.add_argument('--quarter', default=None, help='例如 2025Q3；默认全部评价季度')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--split', action='store_true', help='只打印各季度切分，不训练')
    parser.add_argument('--rescore', action='store_true', help='只重推演、不重训：用已有 best.pt 按当前口径刷新打分')
    parser.add_argument('--doctor', action='store_true', help='环境与数据自检')
    parser.add_argument('--prepare-absolute-only', action='store_true', help='仅准备本单元新拟合技术分支，不形成完成折')
    args = parser.parse_args()
    validate_layout(allow_temporary=True, allow_missing=True)
    if args.doctor:
        print('Python:', sys.executable, 'PyTorch:', torch.__version__)
        print('Data:', data_root())
        print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
        print('Device:', device_for(RECIPE['device']))
        return
    days, _ = axis()
    qs = [args.quarter] if args.quarter else quarters(days)
    if any((q not in quarters(days) for q in qs)):
        raise ValueError('季度不在评价窗口内')
    if args.split:
        for q in qs:
            for s in splits(days, quarter=q):
                print(json.dumps(split_summary(days, s), ensure_ascii=False))
        return
    if args.rescore:
        panel = Panel(RECIPE['features'])
        return rescore(qs, [args.fold] if args.fold else [1, 2, 3, 4], args.device, panel)
    panel = Panel(RECIPE['features'])
    prices = Prices(panel)
    check_days = np.concatenate([splits(panel.days, quarter=q)[0]['test'] for q in qs])
    prices.verify_labels(panel, check_days)
    for q in qs:
        for fold in [args.fold] if args.fold else range(1, 5):
            train_fold(fold, args.device, panel, prices, q, absolute_only=args.prepare_absolute_only)
_ParentPredictModel = PredictModel

class PredictModel(_ParentPredictModel):

    def __init__(self, input_dim, market_dim, horizons=None):
        super().__init__(input_dim, market_dim, horizons)
        if RECIPE['neural_kind'] == 'mlp':
            norm = nn.LayerNorm if RECIPE.get('mlp_layernorm') else nn.BatchNorm1d
            self.mlp = nn.Sequential(nn.Linear(input_dim, 256), nn.LeakyReLU(), nn.Dropout(0.2), nn.Linear(256, 128), norm(128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 32), norm(32), nn.GELU(), nn.Linear(32, len(RECIPE['label_horizons'])))
            for layer in self.mlp.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.zeros_(layer.bias)

    def forward(self, tsdata, market=None):
        if RECIPE['neural_kind'] != 'mlp':
            return super().forward(tsdata, market)
        rank_heads = self.mlp(tsdata[:, -1].float().clamp(-5.0, 5.0))
        absolute = self.absolute_forward(tsdata[:, -1], market)
        return torch.cat([rank_heads, absolute[:, :1]], dim=1)

def score_of(heads):
    out = np.full(heads.shape[:2], np.nan, np.float32)
    j = RECIPE['label_horizons'].index(RECIPE['neural_score_horizon'])
    for i, row in enumerate(heads):
        known = np.isfinite(row).all(axis=1)
        n = known.sum()
        if n > 1:
            if RECIPE['score_transform'] == 'raw_neural':
                out[i, known] = row[known, j]
            else:
                neural = (rankdata(row[known, j]) - 0.5) / n
                technical = (rankdata(row[known, -1]) - 0.5) / n
                weight = RECIPE['flow_fusion_weight']
                out[i, known] = weight * neural + (1 - weight) * technical
    return out

def metadata():
    raw = (data_root() / 'meta.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != RECIPE['snapshot_meta_sha256']:
        raise ValueError('pinned snapshot changed; create new unit, never resume')
    return json.loads(raw)

def personal_validation_fees(notional, sell=False):
    return max(notional * 0.00025, 5) + notional * 1e-05 + (notional * 0.0005 if sell else 0)

def personal_validation_cash(pred, panel, px, days, n=5, period=5, band=0, money=None, forecast_members=4, cost_gate_multiplier=None, gate_scores=None):
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
            fee = personal_validation_fees(gross, True)
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
                while quantity > 0 and quantity * price + personal_validation_fees(quantity * price) > budget:
                    quantity -= 100
                if quantity <= 0:
                    continue
                gross = quantity * price
                fee = personal_validation_fees(gross)
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
_Wave2BaseSplits = splits

def splits(days, purge_horizon=RECIPE['purge_horizon'], folds=4, quarter=None):
    rows = _Wave2BaseSplits(days, purge_horizon, folds, quarter)
    if RECIPE['train_window']:
        for row in rows:
            row['train'] = row['train'][-RECIPE['train_window']:]
    return rows
_Wave2BaseModel = PredictModel

class PredictModel(_Wave2BaseModel):

    def __init__(self, input_dim, market_dim, horizons=None):
        super().__init__(input_dim, market_dim, horizons)
        if RECIPE['neural_market_input']:
            self.mlp[0] = nn.Linear(input_dim + market_dim, 256)
            nn.init.kaiming_uniform_(self.mlp[0].weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.mlp[0].bias)

    def forward(self, tsdata, market=None):
        if not RECIPE['neural_market_input']:
            return super().forward(tsdata, market)
        if market is None or market.ndim != 1:
            raise ValueError('one causal market vector per trading-day cross section required')
        x = tsdata[:, -1].float().clamp(-5.0, 5.0)
        state = market.float().clamp(-RECIPE['market_clip'], RECIPE['market_clip']).expand(len(x), -1)
        heads = self.mlp(torch.cat([x, state], dim=1))
        absolute = self.absolute_forward(x, market)
        return torch.cat([heads, absolute[:, :1]], dim=1)


# 新版本沿用原产物命名；扩展项专门保存验证分数、绝对收益及市场风险模型。
LAYOUT_QUARTERS = ('2025Q3', '2025Q4', '2026Q1', '2026Q2', '2026Q3')
FOLD_FILES = (*FOLD_FILES, 'valid_predictions.npy', 'auxiliary_predictions.npy', 'market_risk.pt')
_template_fixed_files = fixed_files
OPTIONAL_FILES = {'model_info/recent30_review.json','model_pred/recent30_review.md',*[f'model_pred/ensemble/CSV/{kind}_{name}.csv' for kind in ('cash','trades') for name in ('recent30','recent30_baseline')]}


def fixed_files():
    files = _template_fixed_files()
    files -= {f'model_pred/ensemble/CSV/{kind}_{name}.csv'
              for kind in ('cash', 'trades') for name in ('top5_5d', 'top1_1d')}
    files |= {f'model_pred/ensemble/CSV/{kind}_{name}.csv'
              for kind in ('cash', 'trades')
              for name in ('baseline', 'adaptive', 'market_cash', 'open_close_t', 'forward')}
    return files | OPTIONAL_FILES


def quarters(days):
    return list(LAYOUT_QUARTERS)


def splits(days, purge_horizon=20, folds=4, quarter=None):
    """所有验证折都严格向前；最长标签的兑现日早于下一块首日。"""
    q = pd.Period(quarter or '2025Q3', freq='Q')
    first = int(np.searchsorted(days, str(q.start_time.date())))
    test = np.flatnonzero((days >= str(q.start_time.date())) & (days <= str(q.end_time.date())))
    result = []
    for fold in range(1, folds + 1):
        end = first - purge_horizon - 1 - (folds - fold) * 21
        valid = np.arange(end - 63, end)
        train_end = int(valid[0]) - purge_horizon - 1
        start = max(0, train_end - RECIPE['train_window']) if RECIPE['train_window'] else 0
        train = np.arange(start, train_end)
        assert train[-1] + purge_horizon + 1 < valid[0]
        assert valid[-1] + purge_horizon + 1 < first
        assert len(train) > 250 and len(test) > 0
        result.append(dict(quarter=str(q), fold=fold, train=train, valid=valid,
                           test=test, score=np.arange(first, len(days))))
    return result


_template_wpcc = wpcc


def wpcc(preds, y):
    # Huber 约束绝对输出尺度，避免相关损失对异常尾部和任意缩放完全不敏感。
    loss = _template_wpcc(preds, y)
    weight = RECIPE.get('huber_aux_weight', 0.)
    return loss + weight * torch.nn.functional.smooth_l1_loss(preds, y) if weight else loss


def score_of(heads):
    result = np.full(heads.shape[:2], np.nan, np.float32)
    weights = RECIPE['score_horizon_weights']
    for i, row in enumerate(heads):
        known = np.isfinite(row).all(axis=1)
        if known.sum() < 3:
            continue
        score = np.zeros(int(known.sum()), np.float64)
        for horizon, weight in weights.items():
            value = row[known, RECIPE['label_horizons'].index(horizon)]
            # 多期限头先统一截面尺度，单期限仍保留模板的原始输出。
            if len(weights) > 1:
                value = (value - value.mean()) / max(float(value.std()), 1e-8)
            score += weight * value
        result[i, known] = score / sum(weights.values())
    return result


def fit_market_risk(panel, train_days):
    """只用市场因子预测可执行区间的大跌；标签不作为模型输入。"""
    from sklearn.linear_model import LogisticRegression
    y1 = np.nanmean(panel.Y['label_ret_1d'][train_days], axis=1)
    y5 = np.nanmean(panel.Y['label_ret_5d'][train_days], axis=1)
    good = np.isfinite(y1) & np.isfinite(y5)
    target = ((y1 <= -.02) | (y5 <= -.04))[good].astype(int)
    x = panel.M[train_days][good].astype(np.float64)
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .1)
    info = dict(mean=mean, scale=scale, columns=panel.market_features,
                events=int(target.sum()), days=len(target), train_end=str(panel.days[train_days[-1]]),
                target='next1d pool-mean open/open <= -2% OR next5d <= -4%',
                C=.1, constant=None)
    if len(np.unique(target)) < 2:
        info['constant'] = float((target.sum() + 1) / (len(target) + 2))
    else:
        estimator = LogisticRegression(C=.1, max_iter=1000, solver='lbfgs')
        estimator.fit((x - mean) / scale, target)
        info.update(coef=estimator.coef_[0], intercept=float(estimator.intercept_[0]),
                    iterations=int(estimator.n_iter_[0]))
        if info['iterations'] >= 1000:
            raise RuntimeError('市场风险模型未收敛')
    return info


def predict_market_risk(fit, market):
    if fit['constant'] is not None:
        return np.full(len(market), fit['constant'])
    logit = (market.astype(np.float64) - fit['mean']) / fit['scale'] @ fit['coef'] + fit['intercept']
    return 1. / (1. + np.exp(-np.clip(logit, -40, 40)))


@torch.no_grad()
def predict_auxiliary(net, panel, days, device):
    result = np.full((len(days), len(panel.codes), 2), np.nan, np.float32)
    net.eval()
    for i, d in enumerate(days):
        known = panel.mask(int(d))
        x = torch.from_numpy(panel.X[d, known]).to(device)
        market = torch.from_numpy(panel.M[d]).to(device)
        result[i, known] = net.absolute_forward(x, market).cpu().numpy()
    return result


_template_train_fold = train_fold


def train_fold(fold, device='auto', panel=None, prices=None, quarter=None, absolute_only=False):
    panel = panel or Panel(RECIPE['features'])
    prices = prices or Prices(panel)
    result = _template_train_fold(fold, device, panel, prices, quarter, absolute_only)
    if absolute_only:
        return result
    folder = RUN_ROOT / 'model_train' / quarter / f'fold{fold}'
    if result.get('research_extension_complete'):
        for name in ('valid_predictions.npy', 'auxiliary_predictions.npy', 'market_risk.pt'):
            if not (folder / name).is_file():
                raise RuntimeError(f'已完成折缺少 {name}')
        return result
    split = splits(panel.days, quarter=quarter)[fold - 1]
    dev = torch.device('cpu')  # 模板的最终落盘采用 CPU 作为规范后端。
    checkpoint = torch.load(folder / 'best.pt', map_location=dev, weights_only=False)
    net = PredictModel(len(panel.features), panel.market_dim).to(dev)
    net.load_state_dict(checkpoint['model'], strict=True)
    net.ridge = checkpoint.get('ridge')
    valid_score = predict(net, panel, prices, split['valid'], dev)
    valid_aux = predict_auxiliary(net, panel, split['valid'], dev)
    save_npy(folder / 'valid_predictions.npy', np.concatenate([valid_score[..., None], valid_aux], axis=2))
    save_npy(folder / 'auxiliary_predictions.npy', predict_auxiliary(net, panel, split['score'], dev))
    save_torch(folder / 'market_risk.pt', fit_market_risk(panel, split['train']))
    result.update(research_extension_complete=True, validation_is_strictly_forward=True)
    atomic_json(folder / 'complete.json', result)
    print('RESEARCH_EXTENSION_READY', quarter, fold, flush=True)
    return result


_template_Prices = Prices


class Prices(_template_Prices):
    def __init__(self, panel):
        super().__init__(panel)
        # 开盘订单只用当时存在的价格，不用尚未结束的全天成交量作筛选。
        traded = (np.isfinite(self.open) & (self.open > 0) &
                  np.isfinite(self.raw['pre_close']) & (self.raw['pre_close'] > 0) &
                  np.isfinite(self.adj) & (self.adj > 0))
        self.entry = traded & (self.open < self.limit_up - 1e-8)
        self.exit = traded & (self.open > self.limit_down + 1e-8)
        self.next_entry = np.zeros_like(self.entry)
        self.next_entry[:-1] = self.entry[1:]



# 股票因子与市场状态逐样本拼接；LayerNorm 允许在训练时合并多日计算而不混合日截面统计。
_AbsoluteMethods = PredictModel
class PredictModel(nn.Module):
    absolute_basis_numpy = _AbsoluteMethods.absolute_basis_numpy
    absolute_forward = _AbsoluteMethods.absolute_forward
    def __init__(self, input_dim, market_dim, horizons=None):
        super().__init__()
        self.horizons = list(horizons or RECIPE['output_horizons'])
        self.market_dim = market_dim
        self.technical_indices = RECIPE['technical_indices']
        self.absolute_basis_dim = 3*len(self.technical_indices)+market_dim+1
        self.register_buffer('absolute_coef',torch.zeros(2,self.absolute_basis_dim))
        mask=torch.ones(input_dim)
        if RECIPE.get('disable_new_factors'):
            mask[RECIPE['new_feature_indices']]=0.
        self.register_buffer('feature_mask',mask)
        if RECIPE.get('dual_tower'):
            self.legacy_encoder=nn.Sequential(nn.Linear(len(RECIPE['legacy_feature_indices']),64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.2))
            self.new_encoder=nn.Sequential(nn.Linear(len(RECIPE['new_feature_indices']),64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.2))
            self.market_encoder=nn.Sequential(nn.Linear(market_dim,32),nn.LayerNorm(32),nn.GELU())
            self.mlp=nn.Sequential(nn.Linear(160,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,32),nn.GELU(),nn.Linear(32,len(RECIPE['label_horizons'])))
        else:
            self.mlp=nn.Sequential(nn.Linear(input_dim+market_dim,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,32),nn.LayerNorm(32),nn.GELU(),nn.Linear(32,len(RECIPE['label_horizons'])))
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.kaiming_uniform_(layer.weight,mode='fan_in',nonlinearity='relu')
                nn.init.zeros_(layer.bias)
    def forward(self,tsdata,market=None):
        x=tsdata[:,-1].float().clamp(-5.,5.)*self.feature_mask
        if market is None:raise ValueError('缺少因果市场输入')
        state=market.float().clamp(-10.,10.)
        if state.ndim==1:state=state.expand(len(x),-1)
        if RECIPE.get('dual_tower'):
            z=torch.cat([self.legacy_encoder(x[:,RECIPE['legacy_feature_indices']]),self.new_encoder(x[:,RECIPE['new_feature_indices']]),self.market_encoder(state)],dim=1)
        else:z=torch.cat([x,state],dim=1)
        heads=self.mlp(z)
        return torch.cat([heads,self.absolute_forward(x,state)[:,:1]],dim=1)


# 宏观稳定表现是训练目标；小账户收益仅作为执行评价，不用于选轮。
_macro_base_validation = validation
def validation(model,panel,prices,days,device):
    metrics=_macro_base_validation(model,panel,prices,days,device)
    scores=predict(model,panel,prices,days,device)
    ic1=[];ic5=[];returns=[]
    for score,d in zip(scores,days):
        known=np.isfinite(score)
        for horizon,series in [(1,ic1),(5,ic5)]:
            y=panel.Y[f'label_ret_{horizon}d'][d]
            series.append(corr(score[known],np.nan_to_num(y[known],nan=0.)))
        ids=np.flatnonzero(known)
        ids=ids[np.argsort(-score[ids],kind='stable')[:500]]
        capacity=np.maximum(np.nan_to_num(panel.amount[d,ids],nan=0.),0.)
        held=np.minimum(1.5e9,np.cumsum(capacity))
        weights=np.diff(np.r_[0.,held])/1.5e9
        returns.append(float(weights@np.nan_to_num(panel.Y['label_ret_1d'][d,ids],nan=0.)))
    ic1=np.asarray(ic1);ic5=np.asarray(ic5);returns=np.asarray(returns)
    worst=min(float(block.mean()) for block in np.array_split(ic1,3))
    primary=.7*ic1.mean()+.3*ic5.mean()-.2*ic1.std(ddof=1)+.2*worst+2*returns.mean()-.1*returns.std(ddof=1)
    metrics.update(val_wei=float(primary),val_criterion='macro_stable',
        macro_validation=dict(IC1=float(ic1.mean()),IC5=float(ic5.mean()),IC1_std=float(ic1.std(ddof=1)),
            worst_21day_IC1=worst,capacity_return=float(returns.mean()),capacity_volatility=float(returns.std(ddof=1)),
            topn=500,money=1500000000,small_account_used_for_epoch_selection=False))
    return metrics


# 固定凸估计器与深网络形成容量对照；全部股票逐日等权，不做尾部择优或选轮。
_RidgeAbsoluteMethods = PredictModel


class PredictModel(nn.Module):
    absolute_basis_numpy = _RidgeAbsoluteMethods.absolute_basis_numpy
    absolute_forward = _RidgeAbsoluteMethods.absolute_forward

    def __init__(self,input_dim,market_dim,horizons=None):
        super().__init__()
        self.horizons=list(horizons or RECIPE['output_horizons'])
        self.market_dim=market_dim
        self.technical_indices=RECIPE['technical_indices']
        self.absolute_basis_dim=3*len(self.technical_indices)+market_dim+1
        self.register_buffer('absolute_coef',torch.zeros(2,self.absolute_basis_dim))
        self.register_buffer('linear_coef',torch.zeros(1,input_dim+1))

    def forward(self,tsdata,market=None):
        x=tsdata[:,-1].float().clamp(-5.,5.)
        phi=torch.cat([x,torch.ones((len(x),1),device=x.device)],dim=1)
        relative=phi@self.linear_coef.T
        return torch.cat([relative,self.absolute_forward(x,market)[:,:1]],dim=1)


def train_fold(fold,device='auto',panel=None,prices=None,quarter=None,absolute_only=False):
    if absolute_only:raise ValueError('固定岭回归没有单独的神经预训练阶段')
    panel=panel or Panel(RECIPE['features']);prices=prices or Prices(panel)
    quarter=quarter or '2025Q3';signature=recipe_signature()
    folder=RUN_ROOT/'model_train'/quarter/f'fold{fold}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        done=json.loads((folder/'complete.json').read_text())
        if done['recipe_signature']!=signature:raise ValueError('已完成配方变化')
        if not all((folder/name).is_file() for name in FOLD_FILES):raise RuntimeError('已完成折产物缺失')
        return done
    torch.set_num_threads(int(os.environ.get('MX_THREADS','2')))
    dev=device_for(device);seed=RECIPE['seed']+fold-1
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    split=splits(panel.days,quarter=quarter)[fold-1]
    atomic_json(folder/'split.json',split_summary(panel.days,split))
    atomic_json(folder/'training_info.json',training_info({**RECIPE,'quarter':quarter,'device':str(dev)}))
    width=len(panel.features)+1
    gram=np.zeros((width,width),np.float64);cross=np.zeros((width,1),np.float64)
    yy=0.;dates=rows=processed=0;t0=time.time();history=[]
    last=folder/'last.pt'
    if last.exists():
        checkpoint=torch.load(last,map_location='cpu',weights_only=False)
        if checkpoint['recipe_signature']!=signature:raise ValueError('岭回归断点配方变化')
        gram=checkpoint['gram'];cross=checkpoint['cross'];yy=checkpoint['yy']
        dates=checkpoint['fit_dates'];rows=checkpoint['fit_rows'];processed=checkpoint['processed']
        history=checkpoint['history']
    def checkpoint_fit():
        save_torch(last,dict(recipe_signature=signature,gram=gram,cross=cross,yy=yy,
            fit_dates=dates,fit_rows=rows,processed=processed,history=history,
            resume_kind='fixed ridge normal equations; no optimizer or epoch selection'))
    block_size=RECIPE['ridge_block_days']
    for start in range(processed,len(split['train']),block_size):
        xs=[];ys=[]
        for d in split['train'][start:start+block_size]:
            mask=panel.mask(int(d))&np.isfinite(panel.Y['label_ret_1d'][d])
            if mask.sum()<3:continue
            y=panel.Y['label_ret_1d'][d,mask].astype(np.float64)
            sd=y.std(ddof=1)
            if sd<1e-8:continue
            x=np.clip(panel.X[d,mask],-5.,5.)
            phi=np.column_stack([x,np.ones(len(x),np.float32)])
            weight=1/np.sqrt(len(x))
            xs.append((phi*weight).astype(np.float32));ys.append(((y-y.mean())/sd*weight).astype(np.float32)[:,None])
            dates+=1;rows+=len(x)
        if xs:
            x=torch.from_numpy(np.concatenate(xs)).to(dev)
            y=torch.from_numpy(np.concatenate(ys)).to(dev)
            gram+=(x.T@x).cpu().numpy().astype(np.float64)
            cross+=(x.T@y).cpu().numpy().astype(np.float64)
            yy+=float((y*y).sum().cpu())
            del x,y,xs,ys
        processed=min(start+block_size,len(split['train']))
        if processed%(4*block_size)==0 or processed==len(split['train']):
            history.append(dict(stage='normal_equations',processed_days=processed,fit_dates=dates,fit_rows=rows))
            checkpoint_fit();atomic_json(folder/'history.json',history)
            print('RIDGE_GRAM',quarter,fold,processed,'/',len(split['train']),flush=True)
    if not dates:raise ValueError('岭回归无有效训练截面')
    normal=gram/dates+np.diag(np.r_[np.full(width-1,RECIPE['ridge_penalty']),1e-8])
    target=cross/dates;coef=np.linalg.solve(normal,target)
    residual=float(np.linalg.norm(normal@coef-target)/(np.linalg.norm(target)+1e-20))
    if residual>1e-8 or not np.isfinite(coef).all():raise FloatingPointError('固定岭回归求解失败')
    net=PredictModel(len(panel.features),panel.market_dim)
    net.linear_coef.copy_(torch.from_numpy(coef.T.astype(np.float32)))
    absolute=fit_absolute_component(net,panel,split['train'],seed)
    net.ridge=None;cpu=torch.device('cpu')
    metrics=validation(net,panel,prices,split['valid'],cpu)
    mse=float((coef.T@(gram/dates)@coef-2*coef.T@target).item()+yy/dates)
    fit=dict(fit_dates=dates,fit_rows=rows,penalty=RECIPE['ridge_penalty'],
             normal_equation_relative_residual=residual,train_mse=mse,
             target='date-equal standardized1d return',validation_not_used_for_regularization=True)
    save_torch(folder/'best.pt',dict(recipe_signature=signature,model=net.state_dict(),
        input_dim=len(panel.features),market_dim=panel.market_dim,horizons=net.horizons,
        features=panel.features,epoch=0,metrics=metrics,ridge=None,absolute_fit=absolute,closed_form_fit=fit))
    score=predict(net,panel,prices,split['score'],cpu)
    test=score[:len(split['test'])]
    valid=predict(net,panel,prices,split['valid'],cpu)
    valid_aux=predict_auxiliary(net,panel,split['valid'],cpu)
    save_npy(folder/'score_predictions.npy',score);save_npy(folder/'test_predictions.npy',test)
    save_npy(folder/'valid_predictions.npy',np.concatenate([valid[...,None],valid_aux],axis=2))
    save_npy(folder/'auxiliary_predictions.npy',predict_auxiliary(net,panel,split['score'],cpu))
    save_torch(folder/'market_risk.pt',fit_market_risk(panel,split['train']))
    history.append(dict(epoch=0,train_loss=mse,**metrics));atomic_json(folder/'history.json',history)
    checkpoint_fit()
    done=dict(recipe_signature=signature,quarter=quarter,fold=fold,best_epoch=0,epochs=1,
        test_dates=[str(panel.days[d]) for d in split['test']],score_dates=[str(panel.days[d]) for d in split['score']],
        best_validation=metrics,argmax_validation=metrics,
        bag=dict(topk=1,used=1,epochs=[0],scores=[metrics['val_wei']]),
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        seconds=round(time.time()-t0,2),device=str(dev),closed_form_fit=fit,
        research_extension_complete=True,validation_is_strictly_forward=True,
        epoch_selection=False,regularization_search=False)
    atomic_json(folder/'complete.json',done)
    print('RIDGE_COMPLETE',quarter,fold,fit,flush=True)
    return done


_SnapshotPanel = Panel


class Panel(_SnapshotPanel):
    """只缓存源快照派生的只读面板；不同容器自动重建，不改变数值合同。"""
    def __init__(self,features='all',load_x=True):
        if not load_x or os.environ.get('MX_PANEL_CACHE','on').lower()=='off':
            super().__init__(features,load_x=load_x)
            return
        import fcntl
        import shutil
        meta=metadata();root=data_root()
        columns=feature_columns(meta) if features in ('all',RECIPE['features']) else meta['feature_sets'][features]
        source=dict(schema='postfill-panel-v1',meta_sha256=RECIPE['snapshot_meta_sha256'],
            root=str(root.resolve()),features=columns,semantics=meta['semantics'],
            flow_indices=RECIPE['flow_indices'],fund_indices=RECIPE['fund_indices'],
            technical_indices=RECIPE['technical_indices'],
            factors=[(int(y),(root/'factors'/f'year={y}/data.parquet').stat().st_size,
                      (root/'factors'/f'year={y}/data.parquet').stat().st_mtime_ns) for y in meta['built_years']])
        key=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()[:24]
        folder=Path('/dev/shm')/f'quant-model-panel-{key}'
        names=('X','coverage','flow_coverage','fund_coverage','technical_coverage')
        shape=(meta['axis']['n_days'],meta['axis']['n_codes'],len(columns))
        expected_bytes=int(np.prod(shape))*4
        def get_int(path):
            try:return int(Path(path).read_text().strip())
            except (OSError,ValueError):return None
        def stats(path):
            try:return {k:int(v) for k,v in (line.split() for line in Path(path).read_text().splitlines())}
            except (OSError,ValueError):return {}
        with (Path('/dev/shm')/f'quant-model-panel-{key}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            manifest=folder/'meta.json'
            cached=json.loads(manifest.read_text()) if manifest.exists() else None
            if cached and cached['source_key']==key and all((folder/f'{n}.npy').is_file() for n in names):
                arrays={n:np.load(folder/f'{n}.npy',mmap_mode='r',allow_pickle=False) for n in names}
                if arrays['X'].shape!=shape or any(a.dtype!=np.float32 for a in arrays.values()):
                    raise ValueError('派生面板缓存结构变化，请移除该缓存后重建')
                if any((folder/f'{n}.npy').stat().st_size!=cached['files'][n] for n in names):
                    raise ValueError('派生面板缓存文件长度变化')
                super().__init__(features,load_x=False)
                for n,a in arrays.items():setattr(self,n,a)
                print('PANEL_CACHE_HIT',key,round(expected_bytes/2**30,2),'GiB',flush=True)
                return
            budget=get_int('/sys/fs/cgroup/memory.max') or get_int('/sys/fs/cgroup/memory/memory.limit_in_bytes')
            state=stats('/sys/fs/cgroup/memory.stat') or stats('/sys/fs/cgroup/memory/memory.stat')
            nonreclaim=state.get('anon',state.get('total_rss',state.get('rss',0)))+state.get('shmem',0)
            enough_memory=budget is not None and budget-nonreclaim>3*expected_bytes+2*2**30
            enough_space=shutil.disk_usage('/dev/shm').free>expected_bytes+2*2**30
            if not enough_memory or not enough_space:
                print('PANEL_CACHE_SKIP 容器预算不足以安全建立额外副本',flush=True)
                super().__init__(features,load_x=True)
                return
            super().__init__(features,load_x=True)
            folder.mkdir(mode=0o700,exist_ok=True)
            arrays={}
            for name in names:
                value=getattr(self,name);temp=folder/f'{name}.tmp.npy';dest=folder/f'{name}.npy'
                output=np.lib.format.open_memmap(temp,mode='w+',dtype=value.dtype,shape=value.shape)
                for first in range(0,len(value),64):output[first:first+64]=value[first:first+64]
                output.flush();del output;os.replace(temp,dest)
                mapped=np.load(dest,mmap_mode='r',allow_pickle=False)
                for first in range(0,len(value),64):
                    if not np.array_equal(mapped[first:first+64],value[first:first+64]):
                        raise ValueError(f'{name}: 派生缓存与源面板不一致')
                arrays[name]=mapped
            payload=dict(source_key=key,source=source,files={n:(folder/f'{n}.npy').stat().st_size for n in names})
            temp=folder/'meta.tmp.json';temp.write_text(json.dumps(payload,ensure_ascii=False));os.replace(temp,manifest)
            for name,a in arrays.items():setattr(self,name,a)
            print('PANEL_CACHE_BUILT_VERIFIED',key,round(expected_bytes/2**30,2),'GiB',flush=True)


# 验证保持严格样本外；交付权重独立使用测试前已经成熟的全部历史。
_selection_fit = train_fold
FOLD_FILES = (*FOLD_FILES, 'selector_best.pt', 'selector_market_risk.pt', 'selector_complete.json')


def delivery_train_days(days, split):
    end = int(split['test'][0]) - RECIPE['purge_horizon'] - 1
    result = np.arange(end)
    if not len(result) or result[-1] + RECIPE['purge_horizon'] + 1 >= split['test'][0]:
        raise ValueError('交付重训标签没有与测试隔离')
    if not np.array_equal(split['train'], result[:len(split['train'])]):
        raise ValueError('只有训练前缀完全相同时才能复用岭回归充分统计量')
    return result


def train_fold(fold, device='auto', panel=None, prices=None, quarter=None, absolute_only=False):
    if absolute_only:
        raise ValueError('两阶段固定岭回归不支持单独预训练')
    panel = panel or Panel(RECIPE['features'])
    prices = prices or Prices(panel)
    quarter = quarter or '2025Q3'
    folder = RUN_ROOT / 'model_train' / quarter / f'fold{fold}'
    folder.mkdir(parents=True, exist_ok=True)
    signature = recipe_signature()
    completion = folder / 'complete.json'
    previous = json.loads(completion.read_text()) if completion.exists() else None
    if previous is not None and previous['recipe_signature'] != signature:
        raise ValueError('两阶段配方变化')
    if previous is not None and previous.get('delivery_refit_complete'):
        if not all((folder / name).is_file() for name in FOLD_FILES):
            raise RuntimeError('已完成两阶段折的产物缺失')
        return previous
    selector_marker = folder / 'selector_complete.json'
    if not selector_marker.exists():
        selected = previous or _selection_fit(fold, device, panel, prices, quarter, False)
        save_torch(folder / 'selector_best.pt', torch.load(folder / 'best.pt', map_location='cpu', weights_only=False))
        save_torch(folder / 'selector_market_risk.pt', torch.load(folder / 'market_risk.pt', map_location='cpu', weights_only=False))
        atomic_json(selector_marker, selected)
    selected = json.loads(selector_marker.read_text())
    if selected['recipe_signature'] != signature:
        raise ValueError('选择阶段的配方变化')
    # 从此之后不再写 valid_predictions，也不再用交付模型计算旧验证指标。
    split = splits(panel.days, quarter=quarter)[fold - 1]
    train_days = delivery_train_days(panel.days, split)
    summary = split_summary(panel.days, split)
    summary['delivery_train'] = dict(start=str(panel.days[train_days[0]]), end=str(panel.days[train_days[-1]]), days=len(train_days))
    summary['validation_prediction_source'] = 'selector_best.pt and selector_market_risk.pt only'
    atomic_json(folder / 'split.json', summary)
    dev = device_for(device)
    torch.set_num_threads(int(os.environ.get('MX_THREADS', '1')))
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    checkpoint = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
    if checkpoint['recipe_signature'] != signature:
        raise ValueError('交付重训的充分统计量配方变化')
    gram, cross, yy = checkpoint['gram'], checkpoint['cross'], checkpoint['yy']
    dates, rows, processed = checkpoint['fit_dates'], checkpoint['fit_rows'], checkpoint['processed']
    history = checkpoint['history']
    if processed < len(split['train']) or processed > len(train_days):
        raise ValueError('交付重训断点不是合法训练前缀')
    def checkpoint_fit():
        save_torch(folder / 'last.pt', dict(recipe_signature=signature, gram=gram, cross=cross, yy=yy,
            fit_dates=dates, fit_rows=rows, processed=processed, history=history,
            phase='delivery_refit', train_end=str(panel.days[train_days[-1]]),
            resume_kind='exact selector prefix normal equations plus mature pretest extension'))
    block_size = RECIPE['ridge_block_days']
    for start in range(processed, len(train_days), block_size):
        xs, ys = [], []
        for d in train_days[start:start + block_size]:
            mask = panel.mask(int(d)) & np.isfinite(panel.Y['label_ret_1d'][d])
            if mask.sum() < 3:
                continue
            y = panel.Y['label_ret_1d'][d, mask].astype(np.float64)
            sd = y.std(ddof=1)
            if sd < 1e-8:
                continue
            phi = np.column_stack([np.clip(panel.X[d, mask], -5., 5.), np.ones(mask.sum(), np.float32)])
            weight = 1 / np.sqrt(mask.sum())
            xs.append((phi * weight).astype(np.float32))
            ys.append(((y - y.mean()) / sd * weight).astype(np.float32)[:, None])
            dates += 1
            rows += int(mask.sum())
        if xs:
            x = torch.from_numpy(np.concatenate(xs)).to(dev)
            y = torch.from_numpy(np.concatenate(ys)).to(dev)
            gram += (x.T @ x).cpu().numpy().astype(np.float64)
            cross += (x.T @ y).cpu().numpy().astype(np.float64)
            yy += float((y * y).sum().cpu())
            del x, y, xs, ys
        processed = min(start + block_size, len(train_days))
        checkpoint_fit()
    width = len(panel.features) + 1
    normal = gram / dates + np.diag(np.r_[np.full(width - 1, RECIPE['ridge_penalty']), 1e-8])
    target = cross / dates
    coef = np.linalg.solve(normal, target)
    residual = float(np.linalg.norm(normal @ coef - target) / (np.linalg.norm(target) + 1e-20))
    if residual > 1e-8 or not np.isfinite(coef).all():
        raise FloatingPointError('最终交付岭回归求解失败')
    net = PredictModel(len(panel.features), panel.market_dim)
    net.linear_coef.copy_(torch.from_numpy(coef.T.astype(np.float32)))
    absolute = fit_absolute_component(net, panel, train_days, RECIPE['seed'] + fold - 1)
    net.ridge = None
    fit = dict(fit_dates=dates, fit_rows=rows, penalty=RECIPE['ridge_penalty'],
        normal_equation_relative_residual=residual, train_end=str(panel.days[train_days[-1]]),
        selector_train_end=str(panel.days[split['train'][-1]]), validation_not_run_on_refit=True,
        reused_selector_prefix_days=len(split['train']))
    save_torch(folder / 'best.pt', dict(recipe_signature=signature, model=net.state_dict(),
        input_dim=len(panel.features), market_dim=panel.market_dim, horizons=net.horizons,
        features=panel.features, epoch=0, metrics=selected['best_validation'],
        validation_source='selector_best.pt only; final refit has no out-of-sample old validation',
        ridge=None, absolute_fit=absolute, closed_form_fit=fit))
    cpu = torch.device('cpu')
    score = predict(net, panel, prices, split['score'], cpu)
    save_npy(folder / 'score_predictions.npy', score)
    save_npy(folder / 'test_predictions.npy', score[:len(split['test'])])
    save_npy(folder / 'auxiliary_predictions.npy', predict_auxiliary(net, panel, split['score'], cpu))
    save_torch(folder / 'market_risk.pt', fit_market_risk(panel, train_days))
    history.append(dict(stage='delivery_refit', **fit))
    atomic_json(folder / 'history.json', history)
    checkpoint_fit()
    atomic_json(folder / 'training_info.json', training_info({**RECIPE, 'quarter':quarter,
        'device':str(dev), 'selection_and_delivery_train_end':fit}))
    done = dict(selected, closed_form_fit=fit, selection_fit=selected['closed_form_fit'],
        seconds=round(selected['seconds'] + time.time() - t0, 2), delivery_refit_complete=True,
        validation_source='selector_best.pt and selector_market_risk.pt; saved valid_predictions unchanged',
        deterministic_delivery_folds='same mature data; not independent statistical replicas')
    atomic_json(completion, done)
    print('DELIVERY_REFIT_COMPLETE', quarter, fold, fit, flush=True)
    return done

# 当季纸面对照为可选产物，不扩大标准四季度+Q3的主榜审计窗口。
OPTIONAL_FILES.update(f'model_train/2026Q4/fold{f}/{n}' for f in LAYOUT_FOLDS for n in FOLD_FILES)
OPTIONAL_FILES.update(f'model_logs/2026Q4_{f}.log' for f in LAYOUT_FOLDS)
OPTIONAL_FILES.update({'model_info/recent30_review_q4.json', 'model_pred/recent30_review_q4.md',
    *[f'model_pred/ensemble/CSV/{kind}_{name}_q4.csv' for kind in ('cash','trades') for name in ('recent30','recent30_baseline')]})


if __name__ == '__main__':
    main()
