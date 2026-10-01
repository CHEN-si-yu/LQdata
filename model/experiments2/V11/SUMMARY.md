# V11 walk-forward checkpoint

- Quarters: 26 (2020Q1-2026Q2)
- OOS dates: 2020-01-02 to 2026-06-30; monthly signals: 78

| Strategy | Net return | Same fills, before costs | Max drawdown | Ann. vol. | Fees | Slippage | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| V11 LambdaRank Top2 | 114.89% | 122.81% | -18.06% | 16.62% | CNY5,045 | CNY2,877 | 190 |
| 60d momentum Top2 | 117.56% | 121.83% | -17.41% | 16.97% | CNY2,773 | CNY1,489 | 140 |
| equal-weight hold | 102.17% | 102.23% | -15.48% | 15.99% | CNY26 | CNY30 | 4 |

V11 minus equal-weight hold: +12.71 pp; V11 minus 60d Top2: -2.68 pp.

Zero-weight exits include the odd-lot remainder only when the 1% participation cap can clear it; otherwise permitted whole lots are sold and the remainder stays held. This is a backtest comparison, not evidence of stable alpha. Full accounting paths and machine-readable detail are in `daily_equity.csv`, `trades.csv`, `annual_returns.csv`, `quarterly_returns.csv`, and `metrics.json`.
