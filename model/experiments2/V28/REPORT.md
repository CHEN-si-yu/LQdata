# V28 — 60d + 250d price momentum rank consensus

Strict holdout: 2024-01-02 to 2026-06-30; 30 monthly signals. All strategies use the same V11 final cash ledger and 601 trading-day lifecycle.

At each month’s first available close, the candidate ranks four banks separately by their 60-session and 250-session adjusted-close returns. It averages those two ordinal ranks with fixed 50/50 weights, selects the two strongest mean ranks, and executes at the next session open. Both return windows end on the signal-date close.

| Strategy | Net return | Same-fill gross | Max drawdown | Ann. volatility | Fees | Slippage | Trades | Min cash |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V28 60d+250d momentum rank Top2 | 76.45% | 77.70% | -19.28% | 19.08% | CNY 817.75 | CNY 429.07 | 45 | CNY 2.57 |
| 60d momentum Top2 | 81.69% | 83.40% | -17.41% | 19.03% | CNY 1,101.27 | CNY 614.50 | 48 | CNY 61.07 |
| V11 LambdaRank Top2 | 76.29% | 79.63% | -18.03% | 19.25% | CNY 2,123.41 | CNY 1,212.20 | 73 | CNY 36.23 |
| V24 stable-factor consensus Top2 | 71.36% | 75.45% | -13.42% | 18.64% | CNY 2,580.99 | CNY 1,503.17 | 81 | CNY 3.44 |
| equal-weight hold | 70.44% | 70.50% | -15.42% | 18.36% | CNY 25.91 | CNY 29.88 | 4 | CNY 328.21 |

## Comparison with V28

- Versus 60d momentum Top2: net return -5.24 pp; max drawdown -1.87 pp (candidate minus baseline).
- Versus V11 LambdaRank Top2: net return +0.16 pp; max drawdown -1.25 pp (candidate minus baseline).
- Versus V24 stable-factor consensus Top2: net return +5.09 pp; max drawdown -5.86 pp (candidate minus baseline).
- Versus equal-weight hold: net return +6.01 pp; max drawdown -3.86 pp (candidate minus baseline).

## Audit

- Signal audit stores adjusted closes, exact 60/250-session endpoints, returns, individual ranks, mean ranks, selected members, and T+1 execution date for all 30 signals.
- Both price windows end at or before the signal close. The candidate rule was fixed at 50/50; no holdout result was used to change it.
- V24 membership was copied from its frozen audit and reproduced under this run’s V11 cash engine. All baseline metrics matched the V24 cached run within 1e-8.
- All ledgers kept nonnegative cash, had zero blocked trade events, and maximum accounting residual 2.91e-11.
- Quarterly and annual tables include the common holdout lifecycle only; no exit-only day was appended.
- SHA-256: run.py `309ea107334ecede8908ce7b94691dc51ceca5fc57c1f7c76d9f42a5fe808f4f`; protocol.json `3beb9cc55298a30794e6f6ae06648d98103a19087ccae52f3aff0f6b5b6038d5`; cash_engine.py `64a89fdae173b057d7ef5277bde247f2c10bed96a7bdd2d2c0ce6f760df5bd38`.

The backtest is a four-bank historical holdout and does not establish stable future alpha. Protocol, detailed daily/trade records, and causal/ledger audits are provided alongside this report.
