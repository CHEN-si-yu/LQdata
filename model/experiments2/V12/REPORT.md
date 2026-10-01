# V12 inverse-vol allocation candidate

Snapshot: 2026-09-24; this run loaded only four-bank prices and traded amount. No model retraining or parameter sweep.

## Fixed rule

At each first trading session of the month, rank the four banks by 60-session adjusted-close return. Hold the same Top2 as V4; allocate total target capital to them in inverse weights of each bank's sample standard deviation over the 20 daily adjusted-close returns ending on the signal date. Execute at the next open, rebalance monthly, and liquidate on the designated final close. The ledger enforces cash, lots, liquidity participation, price-limit locks and transaction costs.

## Official lifecycle

Includes 242 signal-window sessions from 2025-07-01 through 2026-06-30 and the 2026-07-01 closing liquidation. See result.json for the inverse-vol candidate, equal-weight monthly Top2 and equal-four-bank hold metrics, quarterly returns, fees, drawdown, matched friction and audits.

## Extended lifecycle

Runs from 2023-01-03 through the latest available session, liquidating at that session's close. See result.json for annual/quarterly returns and baselines.

Strict matched friction is the net result minus a no-fee/no-slippage return on the same realized holdings path; it describes execution costs/friction, not stock-selection alpha. This candidate is exploratory and needs comparison across further regimes before claiming stable improvement.

Runner SHA-256: 3c0a1b11c6c28df421a381f056d6b843f0340a7549233016dd2080517949d873
