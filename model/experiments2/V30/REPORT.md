# V30 — V26 factor Top2 + 60d momentum Top2, fixed 50/50

Strict holdout: 2024-01-02 to 2026-06-30; 30 shared monthly signals and 601 trading days per strategy.

The frozen protocol assigns 50% of total equity to V26's `mf_tier_flow_agreement_20` Top2 and 50% to 60d adjusted-price momentum Top2. Each sleeve assigns 25% to each selected bank. Overlapping names receive the sum of their sleeve weights. The rule was fixed before this backtest; weights were not adjusted from the result.

| Strategy | Net return | Gross same fills | Max drawdown | Ann. volatility | Fees | Slippage | Trades | Min cash |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V30 V26+60d momentum 50/50 Top2 mix | 83.59% | 85.69% | -15.36% | 18.67% | CNY 1,334.72 | CNY 761.56 | 76 | CNY 56.55 |
| V26 mf_tier_flow_agreement_20 Top2 | 84.71% | 87.72% | -15.50% | 18.77% | CNY 1,910.99 | CNY 1,103.43 | 63 | CNY 51.35 |
| 60d momentum Top2 | 81.69% | 83.40% | -17.41% | 19.03% | CNY 1,101.27 | CNY 614.50 | 48 | CNY 61.07 |
| V11 LambdaRank Top2 | 76.29% | 79.63% | -18.03% | 19.25% | CNY 2,123.41 | CNY 1,212.20 | 73 | CNY 36.23 |
| equal-weight hold | 70.44% | 70.50% | -15.42% | 18.36% | CNY 25.91 | CNY 29.88 | 4 | CNY 328.21 |

## V30 minus each baseline

- V26 mf_tier_flow_agreement_20 Top2: net return -1.12 pp; max drawdown +0.14 pp.
- 60d momentum Top2: net return +1.90 pp; max drawdown +2.05 pp.
- V11 LambdaRank Top2: net return +7.30 pp; max drawdown +2.67 pp.
- equal-weight hold: net return +13.15 pp; max drawdown +0.06 pp.

## Overlap and accounting audit

- Number of months by shared names: {"0": 2, "1": 21, "2": 7} (keys are 0, 1, or 2 overlapping stocks).
- Monthly target weights summed to 1.0 in every month; maximum single-name target weight was 50%.
- Every strategy uses the V11 final ledger on the same dates. Cash stayed nonnegative; there were zero blocked trade events.
- Maximum accounting residual: 2.91e-11.
- V26, 60d, V11 and equal-weight metrics and daily holdings reproduce the V26 cache; trades match exactly by identity and within 1e-5 CNY/share units on saved numeric fields.
- The end is marked at 2026-06-30 close without forced liquidation; quarter and year tables use this same lifecycle.
- SHA-256: run.py `04b3b5958875f7dc1f47b7ac6c413b1477ca8baa2964e4fe336b9558855ca67e`; frozen protocol.json `5173622c10b4275db784b0e027e23959e1633a909b4e0a5818867474fc5b0ab7`; cash_engine.py `64a89fdae173b057d7ef5277bde247f2c10bed96a7bdd2d2c0ce6f760df5bd38`.

See monthly_target_weights.csv for each component pair, overlap and total weight, plus daily_equity.csv, trades.csv, quarterly_returns.csv, annual_returns.csv, and audit.json. This historical four-bank backtest does not establish stable future alpha.
