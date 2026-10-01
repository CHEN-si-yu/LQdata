# V14 momentum-rank 60/40 allocation candidate

Snapshot: 2026-09-24; prices and traded amount only. No retraining or parameter sweep.

## Fixed rule

At each first trading session of the month, rank four banks by 60-session adjusted-close return. Hold the same Top2 as V4; allocate 60% to the stronger momentum name and 40% to the second name, then rebalance at the next open. The ledger applies the shared cash, 100-share lot, liquidity participation, price-limit and transaction-cost rules.

## Lifecycle returns

| Window | 60/40 net | 50/50 Top2 net | Four-bank hold net | vs Top2 | MDD | Fees |
|---|---:|---:|---:|---:|---:|---:|
| Official lifecycle incl. 2026-07-01 close exit | 12.95% | 11.39% | 1.52% | +1.57 pp | -18.12% | ¥369.88 |
| Extended incl. 2026-09-24 close exit | 170.41% | 169.70% | 151.28% | +0.71 pp | -18.26% | ¥2502.40 |

Lifecycle returns above retain each final exit-day mark and are unchanged. Quarterly and annual tables below use signal-period dates only and exclude the final exit-only date. Exit-day returns are listed separately.

## Signal-period quarterly returns

| Window | Quarter | Sessions | 60/40 net | Same-hold matched return | Friction |
|---|---|---:|---:|---:|---:|
| Official | 2025Q3 | 66 | 4.53% | 4.68% | -0.14% |
| Official | 2025Q4 | 60 | 11.90% | 11.99% | -0.10% |
| Official | 2026Q1 | 56 | -2.22% | -2.10% | -0.12% |
| Official | 2026Q2 | 60 | 0.87% | 0.93% | -0.06% |
| Extended | 2023Q1 | 59 | 6.62% | 6.82% | -0.20% |
| Extended | 2023Q2 | 59 | 12.40% | 12.47% | -0.07% |
| Extended | 2023Q3 | 64 | 5.82% | 5.86% | -0.04% |
| Extended | 2023Q4 | 60 | 1.91% | 2.17% | -0.27% |
| Extended | 2024Q1 | 58 | 8.46% | 8.58% | -0.12% |
| Extended | 2024Q2 | 59 | 9.09% | 9.17% | -0.07% |
| Extended | 2024Q3 | 64 | 8.44% | 8.65% | -0.21% |
| Extended | 2024Q4 | 61 | 11.14% | 11.50% | -0.36% |
| Extended | 2025Q1 | 57 | 0.78% | 0.89% | -0.10% |
| Extended | 2025Q2 | 60 | 8.54% | 8.75% | -0.21% |
| Extended | 2025Q3 | 66 | 8.25% | 8.37% | -0.12% |
| Extended | 2025Q4 | 60 | 11.97% | 12.06% | -0.09% |
| Extended | 2026Q1 | 56 | -2.32% | -2.19% | -0.13% |
| Extended | 2026Q2 | 60 | 0.86% | 0.92% | -0.06% |
| Extended | 2026Q3 | 61 | 12.08% | 12.26% | -0.18% |

## Signal-period annual returns

| Window | Year | Sessions | 60/40 net | Same-hold matched return | Friction |
|---|---:|---:|---:|---:|---:|
| Official | 2025 | 126 | 16.97% | 17.23% | -0.26% |
| Official | 2026 | 116 | -1.38% | -1.19% | -0.18% |
| Extended | 2023 | 242 | 29.24% | 29.95% | -0.71% |
| Extended | 2024 | 242 | 42.60% | 43.60% | -1.00% |
| Extended | 2025 | 243 | 32.59% | 33.23% | -0.64% |
| Extended | 2026 | 177 | 10.42% | 10.81% | -0.39% |

## Exit-only day returns

| Window | Portfolio | Exit date | Net return | Same-hold matched return | Friction |
|---|---|---|---:|---:|---:|
| official | 60/40 momentum rank | 2026-07-01 | -2.09% | -1.98% | -0.10% |
| official | 50/50 equal Top2 | 2026-07-01 | -2.08% | -1.98% | -0.10% |
| official | four-bank equal hold | 2026-07-01 | -2.09% | -1.99% | -0.10% |
| extended | 60/40 momentum rank | 2026-09-24 | 0.22% | 0.33% | -0.11% |
| extended | 50/50 equal Top2 | 2026-09-24 | 0.30% | 0.41% | -0.11% |
| extended | four-bank equal hold | 2026-09-24 | 0.20% | 0.30% | -0.11% |

## Audits and interpretation

Official vs four-bank hold: +11.43 pp; extended vs four-bank hold: +19.14 pp.
Both lifecycle audits retain their original cash, weight, execution and final-liquidation checks. Strict matched friction is the net result minus a no-fee/no-slippage return on the same realized holdings path; it describes execution friction, not selection alpha.
The 60/40 tilt improved returns slightly in these windows but had a larger maximum drawdown than both comparison portfolios. This is not evidence of stable improvement.

Runner SHA-256: 635410fbd3f2cb9017249de1d176e0de4a27cd679eb10422020cabffc101325f
