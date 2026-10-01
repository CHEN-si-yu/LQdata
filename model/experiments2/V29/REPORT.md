# V29: quarterly causal feature updates on a frozen holdout

At each holdout quarter boundary, the strategy uses only completed V11 training-fold importance artifacts through the previous quarter. It selects four factors by a fixed frequency/gain rule, then ranks the four banks at each monthly close and buys the two highest composites at the next open.

Holdout: 2024-01-02 to 2026-06-30 (30 signals). The factor cutoffs and selected lists are shown in protocol.json; all cutoffs precede their signal quarter.

## Results

| Strategy | Net return | Annualized | Max drawdown | Ann. vol | Fees | Slippage | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| V29 quarterly-updated factor consensus Top2 | 71.36% | 25.38% | -13.42% | 18.64% | 2580.99 | 1503.17 | 81 |
| V24 frozen four-factor consensus Top2 | 71.36% | 25.38% | -13.42% | 18.64% | 2580.99 | 1503.17 | 81 |
| V11 LambdaRank Top2 | 76.29% | 26.89% | -18.03% | 19.25% | 2123.41 | 1212.20 | 73 |
| 60d momentum Top2 | 81.69% | 28.50% | -17.41% | 19.03% | 1101.27 | 614.50 | 48 |
| equal-weight hold | 70.44% | 25.10% | -15.42% | 18.36% | 25.91 | 29.88 | 4 |

Quarterly/annual returns, daily equity, trades, membership audit, and cash-ledger audit are included in this directory.

This is a four-bank historical holdout. It does not establish that the strategy will retain its performance.
