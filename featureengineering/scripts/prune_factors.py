#!/usr/bin/env python3
"""去糟粕：按分诊清单**三层删除**因子（代码注册 + 产物 + 状态）。

## 为什么必须三层一起删

只删数据与状态是没用的：`factors/*.py` 里的 `@register` 还在，下一次 `main.py run`
会把它**原样重算回来**。所以删除 = 从源码里摘掉注册与函数 + 删 `data/factors/<名>/`
+ 删 `state/<名>.json`。

## 安全约束（脚本自己强制的）

1. **保护名单只剩 5 个标签**（`label_ret_*`）—— 它们不是因子，是下游的目标变量。
   ⚠️ 用户 2026-09-15 明确「因子不存在保护名单，你觉得不应该存在就删了」：
   原先那份"第一批 10 个因子"的保护名单**已取消**，它们按同一把尺子复判
   （结果：`gpm_ttm`/`cash_profit_ratio`/`debt_asset_ratio`/`yoy_net_profit`/`yoy_roe` 被删）。
2. **先写报告再删**：`docs/FACTOR_TRIAGE.md` 里逐条记录 名称/家族/公式/证据/理由，
   删掉之后仍可据此重建（参考库 `学习资料/factors.md` 里也有原始定义）。
3. `--dry-run`（默认）：预览待删对象并将报告记入总手册，不改因子代码或数据。要真删必须显式 `--apply`。

## 用法

    PY=/autodl-fs/data/miniconda3/bin/python
    $PY scripts/prune_factors.py                 # 预览（默认 dry-run）
    $PY scripts/prune_factors.py --apply         # 真删
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# 分诊清单：{因子名: 理由}。判据只有三类（都写进报告）：
#   【重复】|ρ|≥0.90 的簇里，保留代表、删其余（`main.py dedup` 的实测结果）
#   【退化】零膨胀（>75% 取值为 0）/ 截面取值过少 / 覆盖率过低 —— 排名被并列值支配
#   【无效】|RankIC| 极小且分层无单调性（单年样本，仅作辅助证据，不单独作为理由）
# ─────────────────────────────────────────────────────────────────────────────
DELETE: dict[str, str] = {
    # ★★ 本轮（2026-09-17）清单 = `main.py eval --years 2026 2026` 的 ⚠NOISE 集合
    #   （36 个）**减去 2 个父依赖**（见下方说明）。判据：|RankIC| < 0.005 且样本 ≥60 天。
    #   无重复项可删 —— 本轮 `main.py dedup` 实测 **0 簇**（上一轮的去重已清干净）。
    "asset_turnover": "【无效】|RankIC|=0.0023（<0.005，2026 全年 166 个截面）；ac1=1.000 —— 慢变量单年只有约 4 个独立样本，IC 本身不可靠，**标注可复检**",
    "capex_to_revenue": "【无效】|RankIC|=0.0049（<0.005，2026 全年 166 个截面）；ac1=0.999 —— 慢变量单年只有约 4 个独立样本，IC 本身不可靠，**标注可复检**",
    "chip_below_momentum": "【无效】|RankIC|=0.0018（<0.005，166 个截面），与噪声不可区分",
    "chip_cost_kurtosis_20d": "【无效】|RankIC|=0.0031（<0.005，166 个截面），与噪声不可区分",
    "chip_cr3_factor": "【无效】|RankIC|=0.0050（<0.005，166 个截面），与噪声不可区分",
    "chip_loss_peak_frac": "【无效】|RankIC|=0.0001（<0.005，166 个截面），与噪声不可区分",
    "chip_tail_risk_change": "【无效】|RankIC|=0.0046（<0.005，166 个截面），与噪声不可区分",
    "corr_market_60": "【无效】|RankIC|=0.0047（<0.005，166 个截面），与噪声不可区分",
    "cp_limitup_momentum": "【无效】|RankIC|=0.0033（<0.005，166 个截面），与噪声不可区分",
    "cp_retail_reversal_5": "【无效】|RankIC|=0.0018（<0.005，166 个截面），与噪声不可区分",
    "cp_vol_turnover_20": "【无效】|RankIC|=0.0050（<0.005，164 个截面），与噪声不可区分",
    "cp_winner_momentum_60": "【无效】|RankIC|=0.0013（<0.005，166 个截面），与噪声不可区分",
    "current_ratio": "【无效】|RankIC|=0.0005（<0.005，2026 全年 166 个截面）；ac1=1.000 —— 慢变量单年只有约 4 个独立样本，IC 本身不可靠，**标注可复检**",
    "delta_gpm": "【无效】|RankIC|=0.0035（<0.005，166 个截面），与噪声不可区分",
    "delta_roa": "【无效】|RankIC|=0.0014（<0.005，166 个截面），与噪声不可区分",
    "delta_roe": "【无效】|RankIC|=0.0019（<0.005，166 个截面），与噪声不可区分",
    "doji_frequency_20": "【无效】|RankIC|=0.0037（<0.005，166 个截面），与噪声不可区分",
    "forecast_p_change_median": "【无效】|RankIC|=0.0025（<0.005，2026 全年 166 个截面）；ac1=0.995 —— 慢变量单年只有约 4 个独立样本，IC 本身不可靠，**标注可复检**",
    "gap_open_follow_ratio_20": "【无效】|RankIC|=0.0008（<0.005，166 个截面），与噪声不可区分",
    "margin_buy_momentum_5d": "【无效】|RankIC|=0.0017（<0.005，166 个截面），与噪声不可区分",
    "margin_leverage_trend_10d": "【无效】|RankIC|=0.0032（<0.005，166 个截面），与噪声不可区分",
    "margin_net_flow_ratio": "【无效】|RankIC|=0.0047（<0.005，166 个截面），与噪声不可区分",
    "mf_big_order_stability_20d": "【无效】|RankIC|=0.0037（<0.005，166 个截面），与噪声不可区分",
    "mf_flow_continuity": "【无效】|RankIC|=0.0008（<0.005，166 个截面），与噪声不可区分；截面取值卡片化",
    "mf_net_inflow_trend_5d": "【无效】|RankIC|=0.0031（<0.005，166 个截面），与噪声不可区分",
    "mf_net_inflow_volatility_20d": "【无效】|RankIC|=0.0021（<0.005，166 个截面），与噪声不可区分",
    "mf_net_persistent_5d": "【无效】|RankIC|=0.0041（<0.005，166 个截面），与噪声不可区分；截面取值卡片化",
    "mf_vol_amount_corr_20": "【无效】|RankIC|=0.0011（<0.005，166 个截面），与噪声不可区分",
    "momentum_accel_60_120": "【无效】|RankIC|=0.0007（<0.005，166 个截面），与噪声不可区分",
    "ncf_to_market": "【无效】|RankIC|=0.0045（<0.005，2026 全年 166 个截面）；ac1=0.993 —— 慢变量单年只有约 4 个独立样本，IC 本身不可靠，**标注可复检**",
    "net_turnover_rate_20": "【无效】|RankIC|=0.0047（<0.005，166 个截面），与噪声不可区分",
    "obv_divergence_20": "【无效】|RankIC|=0.0035（<0.005，166 个截面），与噪声不可区分",
    "peg_252d": "【无效】|RankIC|=0.0009（<0.005，2026 全年 166 个截面）；ac1=0.996 —— 慢变量单年只有约 4 个独立样本，IC 本身不可靠，**标注可复检**；覆盖率仅 28%",
    "quick_ratio": "【无效】|RankIC|=0.0020（<0.005，2026 全年 166 个截面）；ac1=1.000 —— 慢变量单年只有约 4 个独立样本，IC 本身不可靠，**标注可复检**",
    # ---- 以下 2 个同样被标 ⚠NOISE，但**保留**（是存活耦合因子的父依赖，删了会打断子因子）----
    #   `mf_big_order_ratio` ← fundflow_retail_inst_divergence（子因子 RankIC −0.0382 / mono −0.85）
    #   `short_term_reversal_5` ← cp_chip_support_reversal_5（子因子 RankIC 0.0339 / mono 0.80）

    # ★★ 2026-09-22 轮（用户交办：「把冗余过高的因子都删了」）：
    #   依据 = `main.py dedup` 的全量实测（**656 因子 × 2018~2026、|ρ|≥0.95**，2026-09-22 09:14 完成），
    #   37 个簇里每簇的重复项，共 **58 个**。完整簇清单与代表清单见
    #   `artifacts/audits/dedup_20260922/{REPORT.md,representatives.json}`；旧报告（337×2026、3 簇）
    #   归档在 `artifacts/audits/dedup_20260919_legacy/`。
    #   ★ 本批跨**四种注册机制**（conf 目录 / REJECTED_CANDIDATES / 数字循环 / 常规 @register），
    #     `_remove_symbol` 只覆盖后两种 —— 前两种由本脚本新增的 `_remove_from_conf()` /
    #     `_add_rejected()` 处理（不处理的话产物删了、注册还在，下一次 `main.py run` 会全部重算回来）。
    #   ★ 删除前的全量备份：`artifacts/backups/prune_20260922/`（产物 58 目录 + state 58 文件 + 源码/配置）。
    "afx_fi_roa": "【重复】与 `afx_fi_npta` |ρ|∈[0.908,1.000]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 10 个）；本簇保留代表 `afx_fi_npta`",
    "afx_fi_roa2_yearly": "【重复】与 `afx_fi_npta` |ρ|∈[0.908,1.000]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 10 个）；本簇保留代表 `afx_fi_npta`",
    "afx_fi_roa_dp": "【重复】与 `afx_fi_npta` |ρ|∈[0.908,1.000]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 10 个）；本簇保留代表 `afx_fi_npta`",
    "afx_fi_roa_yearly": "【重复】与 `afx_fi_npta` |ρ|∈[0.908,1.000]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 10 个）；本簇保留代表 `afx_fi_npta`",
    "afx_fi_roe": "【重复】与 `afx_fi_npta` |ρ|∈[0.908,1.000]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 10 个）；本簇保留代表 `afx_fi_npta`",
    "afx_fi_roe_waa": "【重复】与 `afx_fi_npta` |ρ|∈[0.908,1.000]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 10 个）；本簇保留代表 `afx_fi_npta`",
    "afx_fi_roe_yearly": "【重复】与 `afx_fi_npta` |ρ|∈[0.908,1.000]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 10 个）；本簇保留代表 `afx_fi_npta`",
    "afx_fi_roic": "【重复】与 `afx_fi_npta` |ρ|∈[0.908,1.000]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 10 个）；本簇保留代表 `afx_fi_npta`",
    "afx_fi_roic_yearly": "【重复】与 `afx_fi_npta` |ρ|∈[0.908,1.000]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 10 个）；本簇保留代表 `afx_fi_npta`",
    "afx_bs_oth_liab": "【重复】与 `afx_bs_oth_assets` |ρ|∈[0.957,1.000]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 6 个）；本簇保留代表 `afx_bs_oth_assets`",
    "afx_is_n_commis_income": "【重复】与 `afx_bs_oth_assets` |ρ|∈[0.957,1.000]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 6 个）；本簇保留代表 `afx_bs_oth_assets`",
    "afx_is_n_oth_income": "【重复】与 `afx_bs_oth_assets` |ρ|∈[0.957,1.000]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 6 个）；本簇保留代表 `afx_bs_oth_assets`",
    "afx_is_oper_exp": "【重复】与 `afx_bs_oth_assets` |ρ|∈[0.957,1.000]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 6 个）；本簇保留代表 `afx_bs_oth_assets`",
    "afx_is_oth_b_income": "【重复】与 `afx_bs_oth_assets` |ρ|∈[0.957,1.000]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 6 个）；本簇保留代表 `afx_bs_oth_assets`",
    "afx_fi_netprofit_margin": "【重复】与 `afx_fi_ebit_of_gr` |ρ|∈[0.924,0.998]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 5 个）；本簇保留代表 `afx_fi_ebit_of_gr`",
    "afx_fi_op_of_gr": "【重复】与 `afx_fi_ebit_of_gr` |ρ|∈[0.924,0.998]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 5 个）；本簇保留代表 `afx_fi_ebit_of_gr`",
    "afx_fi_profit_to_gr": "【重复】与 `afx_fi_ebit_of_gr` |ρ|∈[0.924,0.998]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 5 个）；本簇保留代表 `afx_fi_ebit_of_gr`",
    "afx_fi_profit_to_op": "【重复】与 `afx_fi_ebit_of_gr` |ρ|∈[0.924,0.998]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 5 个）；本簇保留代表 `afx_fi_ebit_of_gr`",
    "mfx_minute_pressure": "【重复】与 `mfx_dc_pressure` |ρ|∈[0.883,0.988]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 4 个）；本簇保留代表 `mfx_dc_pressure`",
    "mfx_tdx_pressure": "【重复】与 `mfx_dc_pressure` |ρ|∈[0.883,0.988]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 4 个）；本簇保留代表 `mfx_dc_pressure`",
    "mfx_ths_pressure": "【重复】与 `mfx_dc_pressure` |ρ|∈[0.883,0.988]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 4 个）；本簇保留代表 `mfx_dc_pressure`",
    "mfx_dc_swing": "【重复】与 `mfx_dc_range` |ρ|∈[0.963,0.998]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 4 个）；本簇保留代表 `mfx_dc_range`",
    "mfx_tdx_range": "【重复】与 `mfx_dc_range` |ρ|∈[0.963,0.998]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 4 个）；本簇保留代表 `mfx_dc_range`",
    "mfx_ths_range": "【重复】与 `mfx_dc_range` |ρ|∈[0.963,0.998]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 4 个）；本簇保留代表 `mfx_dc_range`",
    "afx_fi_dt_eps": "【重复】与 `afx_fi_diluted2_eps` |ρ|∈[0.961,0.997]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 3 个）；本簇保留代表 `afx_fi_diluted2_eps`",
    "afx_fi_eps": "【重复】与 `afx_fi_diluted2_eps` |ρ|∈[0.961,0.997]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 3 个）；本簇保留代表 `afx_fi_diluted2_eps`",
    "chip_resistance_distance": "【重复】与 `avg_cost_premium` |ρ|∈[0.943,0.992]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 3 个）；本簇保留代表 `avg_cost_premium`",
    "cyqp_average_cost_premium": "【重复】与 `avg_cost_premium` |ρ|∈[0.943,0.992]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 3 个）；本簇保留代表 `avg_cost_premium`",
    "afx_cf_c_cash_equ_beg_period": "【重复】与 `afx_cf_beg_bal_cash` |ρ|∈[0.976,0.976]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_cf_beg_bal_cash`",
    "afx_cf_end_bal_cash": "【重复】与 `afx_cf_c_cash_equ_end_period` |ρ|∈[0.976,0.976]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_cf_c_cash_equ_end_period`",
    "afx_cf_st_cash_out_act": "【重复】与 `afx_cf_im_net_cashflow_oper_act` |ρ|∈[1.000,1.000]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_cf_im_net_cashflow_oper_act`",
    "afx_fi_dt_eps_yoy": "【重复】与 `afx_fi_basic_eps_yoy` |ρ|∈[0.985,0.985]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_fi_basic_eps_yoy`",
    "afx_fi_grossprofit_margin": "【重复】与 `afx_fi_cogs_of_sales` |ρ|∈[0.998,0.998]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_fi_cogs_of_sales`",
    "afx_fi_profit_prefin_exp": "【重复】与 `afx_fi_ebit` |ρ|∈[0.989,0.989]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_fi_ebit`",
    "afx_fi_tangasset_to_intdebt": "【重复】与 `afx_fi_eqt_to_interestdebt` |ρ|∈[0.978,0.978]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_fi_eqt_to_interestdebt`",
    "afx_fi_fcfe_ps": "【重复】与 `afx_fi_fcfe` |ρ|∈[0.963,0.963]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_fi_fcfe`",
    "afx_fi_fcff_ps": "【重复】与 `afx_fi_fcff` |ρ|∈[0.974,0.974]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_fi_fcff`",
    "afx_fi_nop_to_ebt": "【重复】与 `afx_fi_n_op_profit_of_ebt` |ρ|∈[1.000,1.000]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_fi_n_op_profit_of_ebt`",
    "afx_fi_ocf_to_shortdebt": "【重复】与 `afx_fi_ocf_to_debt` |ρ|∈[0.952,0.952]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_fi_ocf_to_debt`",
    "afx_fi_q_sales_qoq": "【重复】与 `afx_fi_q_gr_qoq` |ρ|∈[0.997,0.997]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_fi_q_gr_qoq`",
    "afx_fi_q_profit_to_gr": "【重复】与 `afx_fi_q_netprofit_margin` |ρ|∈[0.998,0.998]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_fi_q_netprofit_margin`",
    "afx_fi_q_profit_yoy": "【重复】与 `afx_fi_q_netprofit_yoy` |ρ|∈[0.951,0.951]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_fi_q_netprofit_yoy`",
    "afx_fi_undist_profit_ps": "【重复】与 `afx_fi_retainedps` |ρ|∈[0.976,0.976]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_fi_retainedps`",
    "afx_fi_total_revenue_ps": "【重复】与 `afx_fi_revenue_ps` |ρ|∈[1.000,1.000]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_fi_revenue_ps`",
    "afx_fi_roe_dt": "【重复】与 `afx_fi_roe_avg` |ρ|∈[0.966,0.966]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_fi_roe_avg`",
    "afx_is_diluted_eps": "【重复】与 `afx_is_basic_eps` |ρ|∈[0.982,0.982]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_is_basic_eps`",
    "afx_is_t_compr_income": "【重复】与 `afx_is_compr_inc_attr_p` |ρ|∈[0.987,0.987]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `afx_is_compr_inc_attr_p`",
    "market_cap_concentration_20d": "【重复】与 `bollinger_width_20` |ρ|∈[0.981,0.981]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `bollinger_width_20`",
    "chip_support_distance": "【重复】与 `chip_median_distance` |ρ|∈[0.957,0.957]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `chip_median_distance`",
    "cyqp_price_cost_position": "【重复】与 `chip_position` |ρ|∈[0.957,0.957]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `chip_position`",
    "ret_skew_60": "【重复】与 `downside_upside_vol_60` |ρ|∈[0.952,0.952]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `downside_upside_vol_60`",
    "efx_annual_sales_yield": "【重复】与 `sp_ttm` |ρ|∈[0.974,0.974]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `sp_ttm`",
    "efx_top_participation": "【重复】与 `efx_top_amount_rate` |ρ|∈[1.000,1.000]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `efx_top_amount_rate`",
    "efx_top_net_rate": "【重复】与 `efx_top_imbalance` |ρ|∈[0.968,0.968]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `efx_top_imbalance`",
    "mfx_ths_pct_change": "【重复】与 `mfx_dc_pct_change` |ρ|∈[0.988,0.988]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `mfx_dc_pct_change`",
    "mfx_ths_turnover_rate": "【重复】与 `mfx_dc_turnover_rate` |ρ|∈[0.965,0.965]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `mfx_dc_turnover_rate`",
    "mfx_minute_realized_vol": "【重复】与 `mfx_minute_amplitude` |ρ|∈[0.959,0.959]（2018~2026 全窗 · dedup 2026-09-22 · 簇内 2 个）；本簇保留代表 `mfx_minute_amplitude`",
    # ---- 以下 1 个同样在 2026-09-22 的重复簇里，但**保留**（同 09-17 轮的道理）----
    #   `momentum_60` ← cp_momentum_highvol_60 / cp_quality_momentum（两个存活耦合因子的父依赖）。
    #   删了父因子，子因子的 `ctx.load_factor` 会**静默**返回全 NaN（只 warning）⇒
    #   表现为非空率 0% 而不是报错。本脚本的「依赖守卫」会拦下这种情况（见 main()）。
}

# ★ 保护名单：**只有 5 个标签**（label_ret_1d/3d/5d/10d/20d）。
#   它们不是因子，是下游模型的目标变量（`is_label=True`），删了模型就没标签了。
#   ⚠️ 用户 2026-09-15 明确：「因子不存在保护名单，你觉得不应该存在就删了」——
#      所以原来那份"第一批 10 个因子"的保护名单**已取消**，它们按同一把尺子复判。
PROTECTED = {
    "label_ret_1d", "label_ret_3d", "label_ret_5d", "label_ret_10d", "label_ret_20d",
}


def _specs():
    import factors                                          # noqa: F401
    from fea.spec import REGISTRY
    return REGISTRY


def _remove_symbol(path: Path, names: set[str], reg: dict) -> list[str]:
    """从源码里摘掉这些因子的 `@register(...)` 块与函数体。

    ★ 定位方式（2026-09-22 改）：**用运行时函数对象的行号**，不再解析装饰器文本。
      `spec.fn.__code__.co_firstlineno` 就是 `def` 那一行；AST 里取同一行的顶层
      FunctionDef，删 `[最上面的 decorator 行, node.end_lineno]`。
      ★ 为什么改：老实现用正则找装饰器里的 `name='...'`，而 `factors/cyq_perf.py` 写的是
        `@register(_spec('cyqp_price_cost_position', ...))` —— 名字是**位置参数**、正则匹配不到，
        实测漏了 2 个（产物删了、注册还在 ⇒ 下一次 `main.py run` 会把它们重算回来）。
      ★ 副作用正好是想要的：工厂生成的函数（`_build`/`_mk` 里那种）行号落在**别的 def 内部**，
        取不到顶层节点 ⇒ 自动跳过，留给各自的机制（conf / REJECTED_CANDIDATES / 数字循环）。
      再按行号**从后往前**删，避免行号位移。
    """
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    # ★ 装饰器函数的 `co_firstlineno` 指向**第一个装饰器行**，而 AST 的 `lineno` 是 `def` 行
    #   （实测 cyq_perf.py：co_firstlineno=88、node.lineno=89）⇒ 两个行号都要能命中。
    top: dict[int, ast.FunctionDef] = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            top[n.lineno] = n
            if n.decorator_list:
                top[min(d.lineno for d in n.decorator_list)] = n
    spans: list[tuple[int, int, str]] = []                  # (起始行, 结束行, 名字)
    for nm in sorted(names):
        s = reg.get(nm)
        if s is None or Path(s.fn.__code__.co_filename) != path:
            continue
        node = top.get(s.fn.__code__.co_firstlineno)
        if node is None or node.name != nm:                 # 工厂生成的函数：不在这条路径上
            continue
        start = min(d.lineno for d in node.decorator_list) if node.decorator_list else node.lineno
        spans.append((start, node.end_lineno, nm))
    removed = []
    for start, end, nm in sorted(spans, reverse=True):
        # 连同上方紧邻的注释块一起删（那段注释只属于这个因子）
        s = start - 1
        while s - 1 >= 0 and lines[s - 1].lstrip().startswith("#"):
            s -= 1
        del lines[s:end]
        removed.append(nm)

    # 动态注册：`for _k in (5, 10, 20, 60, 120, 250):` + `name=f"momentum_{_k}"`
    for nm in sorted(names):
        m = re.fullmatch(r"(\w+?)_(\d+)", nm)
        if not m or nm in removed:
            continue
        prefix, num = m.group(1), m.group(2)
        text = "".join(lines)
        # ★ 不能用 f-string 拼这段：f-string 里不允许 \" 转义（Python 语法限制）
        pat = re.compile(r"(for \w+ in \()([^)]*)(\):\n(?:(?!\n\S).)*?name=f\""
                         + prefix + r"_\{\w+\}\")", re.S)
        mm = pat.search(text)
        if not mm:
            continue
        kept = [x.strip() for x in mm.group(2).split(",") if x.strip() and x.strip() != num]
        text = text[:mm.start(2)] + ", ".join(kept) + text[mm.end(2):]
        lines = text.splitlines(keepends=True)
        removed.append(nm)
    if removed:
        # ★ 原子写：别的 Agent 可能正在 `import factors`，`write_text` 有一瞬间
        #   文件是空的 → 那个 Agent 会拿到 SyntaxError。先写临时文件再 os.replace。
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("".join(lines), encoding="utf-8")
        os.replace(tmp, path)
    return removed


def _remove_from_conf(names: set[str], conf: Path) -> list[str]:
    """从 `conf/field_expansion.json` 摘掉这些因子 —— 那个目录文件**就是**它们的注册源。

    `factors/annual_field_expansion.py` 是 `for _entry in CATALOG: ...`，而 CATALOG
    直接读这个 JSON ⇒ 源码里**没有**可摘的注册块（AST 一个也定位不到）。
    对这族因子，「摘注册」= 从目录里删条目。原子写（和 `_remove_symbol` 同一个理由）。
    """
    data = json.loads(conf.read_text(encoding="utf-8"))
    kept = [e for e in data if e.get("name") not in names]
    removed = sorted(e["name"] for e in data if e.get("name") in names)
    if removed:
        tmp = conf.with_suffix(conf.suffix + ".tmp")
        tmp.write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, conf)
    return removed


def _add_rejected(path: Path, names: set[str]) -> list[str]:
    """把名字并进 `field_events.py` / `field_markets.py` 自带的 `REJECTED_CANDIDATES`。

    这两个文件本来就是 `if name in REJECTED_CANDIDATES: return`（构建期直接跳过），
    所以「摘注册」= 往那个集合里加名字。原地改 `REJECTED_CANDIDATES = ...` 那一行，
    幂等（先解析出当前集合，再取并集）。
    """
    src = path.read_text(encoding="utf-8")
    m = re.search(r"^REJECTED_CANDIDATES\s*=\s*(.*)$", src, re.M)
    if not m:
        return []
    try:
        cur = set(ast.literal_eval(m.group(1)) or ())
    except Exception:                                          # noqa: BLE001
        cur = set()
    added = sorted(names - cur)
    if not added:
        return []
    block = ("REJECTED_CANDIDATES = {\n"
             + "".join(f'    "{n}",   # 2026-09-22 去冗余：|ρ|≥0.95 簇的重复项，见 scripts/prune_factors.py 清单\n'
                       for n in sorted(cur | names))
             + "}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(src[:m.start()] + block + src[m.end():], encoding="utf-8")
    os.replace(tmp, path)
    return added


def _kind(spec, root: Path) -> str:
    """按**注册机制**分派摘除路径（三类源码 + conf，见文件头的 2026-09-22 说明）。"""
    fn = Path(spec.fn.__code__.co_filename).name
    if fn == "annual_field_expansion.py":
        return "conf"                       # 目录驱动：删 conf 里的条目
    if fn in ("field_events.py", "field_markets.py"):
        return "rejected"                   # 文件自带的 REJECTED_CANDIDATES 钩子
    return "symbol"                         # 常规 @register / 数字循环：交给 _remove_symbol


def main() -> int:
    ap = argparse.ArgumentParser(description="按分诊清单三层删除因子")
    ap.add_argument("--apply", action="store_true", help="真的删（默认只预览）")
    ap.add_argument("--report", default="", help="可选独立导出路径；默认记入 README.md")
    args = ap.parse_args()

    from fea import store
    from fea.config import load as load_cfg

    cfg = load_cfg()
    reg = _specs()
    bad = [n for n in DELETE if n in PROTECTED]
    if bad:
        raise SystemExit(f"清单里有受保护的因子，拒绝执行：{bad}")
    unknown = [n for n in DELETE if n not in reg]
    missing = [n for n in DELETE if n not in reg]
    todo = {n: r for n, r in DELETE.items() if n in reg}

    # ---- ★ 依赖守卫（2026-09-22 新增）：待删的名字里有没有谁是**存活因子**的父依赖？
    #   耦合因子（group='coupling'）的 `deps` 写的是父**因子名**；父因子被删 ⇒ 子因子的
    #   `ctx.load_factor` 会**静默**返回全 NaN（只 warning）⇒ 表现为"非空率 0%"而不报错。
    #   2026-09-17 那轮是人工把 2 个父依赖从清单里挑出来的（见上方 DELETE 里的注释）；
    #   2026-09-22 这轮实测又踩到一次（`momentum_60` ← cp_momentum_highvol_60 /
    #   cp_quality_momentum）⇒ 固化成守卫，谁也别再靠人工记得。
    parents: dict[str, list[str]] = {}
    for nm, s in reg.items():
        if nm in todo:
            continue
        for d in s.deps:
            if d in todo:
                parents.setdefault(d, []).append(nm)
    if parents:
        print("\n🔴 依赖守卫拦下本轮（未改任何代码或数据）：以下待删因子是**存活因子**的父依赖")
        for d, kids in sorted(parents.items()):
            print(f"   `{d}` ← {', '.join(sorted(kids))}")
        print("   处置：① 把它从清单里去掉（同 09-17 轮的 `mf_big_order_ratio` / `short_term_reversal_5`）；")
        print("         ② 或连同子因子一起删；③ 或先改子因子的 deps（那会让子因子重算）。")
        return 2

    # ---- 报告（先写，删了也能照它重建）
    rep = [f"# 因子分诊报告（去糟粕）", "",
           f"> 由 `scripts/prune_factors.py` 生成。判据见脚本头部注释。",
           f"> 计划删除 **{len(todo)}** 个（当前注册 {len(reg)} 个因子）；此为执行前计划，不代表删除成功。", "",
           "| 因子 | 家族 | 定义 | 公式 | 删除理由 |", "|:--|:--|:--|:--|:--|"]
    for n, r in sorted(todo.items()):
        s = reg[n]
        # ★ 公式里可能有换行（多行公式），不替换会把 markdown 表格撑破
        fml = s.formula.replace("\n", " ⏎ ").replace("|", chr(92) + "|")[:120]
        rep.append(f"| `{n}` | {s.group} | {s.desc} | `{fml}` | {r} |")
    if unknown:
        rep += ["", f"> ⚠️ 清单里有 {len(unknown)} 个未注册（可能已删过）：{unknown}"]
    rep += ["", "## 保留但标注低置信", "",
            "- `ac1 > 0.995` 的财务/慢变量：2026 单年只有约 4 个独立样本（季报），IC 统计上不可信；",
            "  同时它们换手≈0，无法独立产生交易信号 —— 作为**风格暴露/控制变量**保留。",
            "- `ac1 < 0.10` 的日内/资金流瞬时因子：日频全换手，必须按**成本后**收益复核。"]
    if args.report:
        out = ROOT / args.report
        if out.resolve() == (ROOT / "README.md").resolve():
            raise ValueError("不要通过 --report 覆盖总手册；省略该参数即可追加报告")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(rep) + "\n", encoding="utf-8")
    else:
        from fea.documentation import append_report
        out = append_report(ROOT, "prune-preview", "因子分诊执行前计划", "\n".join(rep) + "\n")
    print(f"分诊报告已写入 {out}（{len(todo)} 个待删）")

    # ---- 预览
    print(f"\n{'因子':<28}{'家族':<10}{'产物':>10}{'状态':<8}")
    print("-" * 70)
    n_data = n_state = 0
    for n in sorted(todo):
        d = cfg.factors_dir / n
        st = cfg.state_dir / f"{n}.json"
        has_d = d.is_dir()
        has_s = st.exists()
        n_data += has_d
        n_state += has_s
        print(f"{n:<28}{reg[n].group:<10}{'有' if has_d else '—':>10}{'有' if has_s else '—':<8}")
    print("-" * 70)
    print(f"合计：{len(todo)} 个因子 · 产物目录 {n_data} · 状态文件 {n_state}")

    # ---- 按「注册机制 + 文件」分组预览（2026-09-22：机制不止一种，见 _kind）
    by_kind: dict[str, dict[str, list[str]]] = {"symbol": {}, "conf": {}, "rejected": {}}
    for n in todo:
        k = _kind(reg[n], ROOT)
        by_kind[k].setdefault(Path(reg[n].fn.__code__.co_filename).name, []).append(n)
    print("\n摘注册路径（按机制）：")
    for k, label in (("symbol", "AST 摘注册块"), ("conf", "删 conf 目录条目"),
                     ("rejected", "并入 REJECTED_CANDIDATES")):
        for f, ns in sorted(by_kind[k].items()):
            print(f"  [{label}] {f:<28} {len(ns):>3} 个：{', '.join(sorted(ns)[:5])}"
                  f"{' …' if len(ns) > 5 else ''}")

    if not args.apply:
        print("\n（dry-run，仅更新报告，未改动因子代码或数据。要真删请加 --apply）")
        return 0

    # ---- 真删（按机制分派）
    print("\n开始删除 …")
    misses: list[str] = []
    # ① 常规 @register / 数字循环：AST 摘注册块
    for f, ns in sorted(by_kind["symbol"].items()):
        p = ROOT / "factors" / f
        removed = _remove_symbol(p, set(ns), reg)
        miss = sorted(set(ns) - set(removed))
        misses += miss
        print(f"  ✔ {f}：摘掉 {len(removed)} 个注册块"
              + (f"   ⚠️ 未定位到：{miss}" if miss else ""))
    # ② conf 目录驱动：删 conf/field_expansion.json 里的条目
    if by_kind["conf"]:
        names = {n for ns in by_kind["conf"].values() for n in ns}
        removed = _remove_from_conf(names, ROOT / "conf" / "field_expansion.json")
        miss = sorted(names - set(removed))
        misses += miss
        print(f"  ✔ conf/field_expansion.json：摘掉 {len(removed)} 个条目"
              + (f"   ⚠️ 未定位到：{miss}" if miss else ""))
    # ③ REJECTED_CANDIDATES 钩子：并进集合
    for f, ns in sorted(by_kind["rejected"].items()):
        p = ROOT / "factors" / f
        added = _add_rejected(p, set(ns))
        miss = sorted(set(ns) - set(added))
        misses += miss
        print(f"  ✔ {f}：并入 REJECTED_CANDIDATES {len(added)} 个"
              + (f"   ⚠️ 未定位到：{miss}" if miss else ""))
    # ④ 产物目录 + 状态文件
    for n in sorted(todo):
        shutil.rmtree(cfg.factors_dir / n, ignore_errors=True)
        st = cfg.state_dir / f"{n}.json"
        if st.exists():
            st.unlink()
    print(f"  删除产物目录 {n_data} 个、状态文件 {n_state} 个")

    # ---- 复核：**另起进程** import factors，确认这些名字真的从注册表消失了。
    #   没删干净的后果不是"报错"，而是下一次 `main.py run` 把它们原样算回来 ——
    #   所以这一步必须做，且不能复用本进程已 import 的注册表。
    import subprocess
    code = ("import sys, factors; from fea.spec import REGISTRY;"
            "left=[n for n in sys.argv[1:] if n in REGISTRY]; print(','.join(left))")
    out = subprocess.run([sys.executable, "-c", code, *sorted(todo)],
                         cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    if out:
        print(f"\n🔴 复核未通过：这些名字仍在注册表里（下次 run 会重算）：{out.split(',')}")
    else:
        print(f"\n✔ 复核通过：{len(todo)} 个名字都已不在注册表（另起进程 import factors 核对）")
    if misses:
        print(f"⚠️ 有 {len(misses)} 个名字在源码里未定位到：{misses}")
    print("★ 记得核对：`python main.py list | wc -l` 与 README 因子字典（`python main.py docs`）")
    return 0 if not (misses or out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
