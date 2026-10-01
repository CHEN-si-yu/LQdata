# V24: stable-factor consensus on a frozen holdout

## Protocol

The four features were selected only from V11 model training-fold importance artifacts through 2023Q4. Feature selection stops before the 2024Q1 holdout. At each shared monthly signal close, the strategy averages the four feature ranks across the four banks and equal-weights the top two. Orders execute at the next session open.

Selected features: mf_tier_flow_agreement_20, rel_mom_ind_3d, rel_mom_ind_5d, gap_down_recover_freq_20d

Holdout: 2024-01-02 through 2026-06-30 (30 monthly signals). No refit, parameter sweep, or use of the 2024Q1-2026Q2 strategy outcomes for feature choice.

## Results

| Strategy | Net return | Annualized | Max drawdown | Ann. vol | Fees | Slippage | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| V24 stable-factor consensus Top2 | 71.36% | 25.38% | -13.42% | 18.64% | 2580.99 | 1503.17 | 81 |
| V11 LambdaRank Top2 | 76.29% | 26.89% | -18.03% | 19.25% | 2123.41 | 1212.20 | 73 |
| 60d momentum Top2 | 81.69% | 28.50% | -17.41% | 19.03% | 1101.27 | 614.50 | 48 |
| equal-weight hold | 70.44% | 25.10% | -15.42% | 18.36% | 25.91 | 29.88 | 4 |

Quarterly and annual results are in quarterly_returns.csv and annual_returns.csv. Full daily ledger, trades, frozen protocol, and membership audit are in this directory.

This is a four-bank historical holdout. Positive return alone does not demonstrate stable future alpha.
