# V25 — single-factor strict holdout test

Holdout lifecycle: 2024-01-02 to 2026-06-30; 30 monthly signal dates. All strategies use the same V11 cash, fills, split adjustment, fees, and daily marks.

The only candidate rule ranks the four banks by `rel_mom_ind_250d` at the first session's close of each month, selects the top two equally, and executes at the next session's open. Factor choice is fixed from V11 model training evidence: it appeared in the saved top-20 gain features in 13 of 16 quarters from 2020Q1 through 2023Q4. No holdout return was used to select the factor or alter the rule.

| Strategy | Net return | Same-fill gross return | Max drawdown | Annualized vol. | Fees | Slippage | Trades | Min cash |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V25 rel_mom_ind_250d Top2 | 65.50% | 66.61% | -19.27% | 19.00% | CNY 736.82 | CNY 369.28 | 47 | CNY 9.59 |
| 60d momentum Top2 | 81.69% | 83.40% | -17.41% | 19.03% | CNY 1,101.27 | CNY 614.50 | 48 | CNY 61.07 |
| V11 LambdaRank Top2 | 76.29% | 79.63% | -18.03% | 19.25% | CNY 2,123.41 | CNY 1,212.20 | 73 | CNY 36.23 |
| equal-weight hold | 70.44% | 70.50% | -15.42% | 18.36% | CNY 25.91 | CNY 29.88 | 4 | CNY 328.21 |

## Comparison to V25

- Versus 60d momentum Top2: net return -16.19 pp; maximum drawdown -1.86 pp (candidate minus baseline).
- Versus V11 LambdaRank Top2: net return -10.79 pp; maximum drawdown -1.24 pp (candidate minus baseline).
- Versus equal-weight hold: net return -4.94 pp; maximum drawdown -3.85 pp (candidate minus baseline).

## Audit

- Training-period feature selection: 13/16 quarterly V11 fits included this factor in their saved top 20 gain features.
- Signal audit records four factor values and ranks per monthly date, the T+1 execution date, and a false post-signal-data flag.
- All strategies share the same daily lifecycle; no end-of-period liquidation is imposed. Quarterly and annual return tables cover the lifecycle only.
- Cash remained nonnegative, blocked trade events were zero, and maximum accounting residual was 2.91e-11.
- The V11 Ranker baseline uses already-saved OOS scores; no fitting was run. The candidate uses only the single fixed factor and no parameter sweep.

A positive or superior backtest outcome would not establish stable alpha. See `protocol.json`, `audit.json`, `signal_audit.csv`, `feature_selection_audit.csv`, daily/trade files, and period tables for the reproducible record.
