# V11 frozen protocol

## Hypothesis and scope
Test whether cross-sectional LightGBM LambdaRank, trained on daily four-bank queries, ranks future five-session returns more usefully for a monthly Top2 portfolio than V1's separate Huber regressors. This is a new model trained from the current factor/market panel; V1 cached scores are not consumed. No GPU, no hyperparameter sweep, and no OOS-driven early stopping.

## Point-in-time data and walk-forward
- Universe: 601288.SH, 601398.SH, 601939.SH, 601988.SH; V1-compatible factors, market factors, and four asset one-hot features.
- Factor snapshot must declare `zscore_win1_99_v1`: daily cross-section 1%/99% winsorization, z-score, factor direction harmonization; missing values become zero as in V1.
- Label: existing `label_ret_5d`; for each training date, keep finite rows, require at least two names, form the query group from the remaining rows, and map ascending unique within-date returns to integer relevance starting at 0 (ties share rank; levels never exceed 3).
- Tests: every calendar quarter 2020Q1 through 2026Q2, each fit independently using only labels dated at least five trading observations before that quarter's first test session. Exact rule: `train_day_index + 5 < first_test_day_index`.
- Each quarter's model scores only dates in that quarter. Monthly signal is the first available trading date of each calendar month. Rank the four scores, select two; execute at the next trading session's open.
- Keep the signal, model, prediction, training counts, purge boundary, process RSS, and scheduler resource snapshots per quarter.

## One predeclared CPU recipe
LightGBM 4.7.0 LGBMRanker, `objective=lambdarank`, NDCG@2 metric, `n_estimators=180`, `learning_rate=0.03`, `num_leaves=7`, `max_depth=3`, `min_child_samples=80`, `feature_fraction=0.80`, `bagging_fraction=0.80`, `bagging_freq=1`, `reg_alpha=2`, `reg_lambda=20`, `label_gain=[0,1,2,3]`, seed 17, one numerical/LightGBM thread per worker. No sample weighting, validation-driven choice, tuning, or early stopping.

## Execution and comparisons
The monthly Top2 account is 50/50 target weight, one shared cash balance, T+1 open execution. Charge V1-compatible commission (0.025%, CNY5 minimum), transfer fee (0.001%), sell stamp duty (0.05%), and adverse slippage (0.03%); use 100-share lots for ordinary fills, a 1% participation cap, and V1-style one-price limit-up/down entry/exit blocks. On a zero-weight full exit, the remaining odd-lot/fractional balance can be sold only when the participation cap covers the entire remainder; if it does not, sell permitted whole lots and keep the residual for the next rebalance. At each rebalance, sell first, then buy within available cash; mark daily using raw prices and adjustment-factor changes.
Compare on the identical walk-forward dates/capital/ledger: (1) equal-weight four-bank buy-and-hold from the first executable open; (2) monthly 60-session adjusted-close momentum Top2, signal at the same monthly close and execute T+1 open. Report net/gross cumulative and annual returns, volatility, drawdown, fees, turnover/trades, accounting residuals, and whether V11 beats each baseline. A ranking model or positive return alone is not evidence of stable alpha.

## Resource controls
CPU-only process pool capped at 8 workers. Workers load only the four-bank factor/market/label rows and set numerical thread counts to one. Scheduler records RAM/load every 10 seconds, pauses new submissions at estimated used RAM >=150 GiB or 1-minute load >=40 on the 48-core host. Hard user constraint remains under 180 GiB memory. No shared experiment or docs directories are written.
