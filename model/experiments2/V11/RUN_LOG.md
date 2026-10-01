# V11 run log

## Preflight
- Isolated unit: experiments2/V11. V1 and all other experiments were left unchanged.
- Verified the V1 loader: factor semantics zscore_win1_99_v1; factor/market inputs plus four asset indicators; target label_ret_5d is forward-looking and the existing V1 price-label audit is below 1e-8.
- LightGBM 4.7.0 LGBMRanker is installed. Fixed recipe and walk-forward protocol are in PROTOCOL.md.
- Host has 48 cores. Before V11, RAM used by MemTotal-MemAvailable was about 91 GiB and load1 about 14. No GPU was requested; existing GPU jobs were not modified.

## First batch: 2020Q1-2020Q4
- Four CPU workers completed in about 16 seconds each; LightGBM fit time was 3.53-4.39 seconds per quarter.
- Each model used 610 features and seed 17. Training groups were 482, 540, 599, and 665 daily queries, with 1,928-2,660 finite-label rows. All four stocks had labels in these first queries.
- 2020Q1 train labels ended 2019-12-24; the first test date was 2020-01-02, respecting train index + 5 < test index.
- RSS observed after panel load was 1.36-2.00 GiB per worker. During this batch estimated host RAM used peaked at 94 GiB and load1 at 11.8.

## Full walk-forward: 26 quarters
- Launched the remaining 22 quarters with an 8-worker, one-thread-per-worker CPU pool. All 26/26 quarter status files are complete; no failed/running quarters remain.
- Fit time ranged from 3.53 to 7.15 seconds per quarter; training rows ranged from 1,928 to 7,972. Each quarter has its own saved model, prediction table, and result/status JSON under quarters/<quarter>/. Launcher output is recorded in logs/first_batch_launcher.log and logs/full_batch_launcher.log.
- resource_log.csv contains 12 scheduler snapshots. Peak estimated host RAM used was 95.75 GiB (below 180 GiB); peak load1 was 23.51 on 48 cores. No resource gate was triggered.
- Full OOS range is 2020-01-02 to 2026-06-30 with 78 monthly signals. Shared-cash T+1 ledger: V11 net +114.89%, same fills before costs +122.81%, MDD -18.06%, annualized volatility 16.62%, 190 orders, fees CNY 5,045, slippage CNY 2,877. Fixed 60d Top2 net +117.56%, MDD -17.41%; equal-weight hold net +102.17%, MDD -15.48%.
- V11 beat 60d Top2 in 15/26 quarters and 4/7 calendar-year blocks (2026 is partial through June), but trailed it by 2.68 percentage points cumulatively. It beat equal-weight hold by 12.71 points but had a deeper drawdown than both baselines.
- Accounting audit: no blocked trades, minimum V11 cash CNY 13.85, maximum two V11 positions after full odd-lot liquidation on exits, and maximum equity/cash/holdings residual 2.91e-11.
- Result: retain as a research candidate only. It does not improve the fixed 60d Top2 benchmark over the full walk-forward and does not show stable alpha.

## Ledger exit-rule audit and replay
- Parent review found the zero-weight exit was still behind the 100-share minimum and the participation cap was rounded to board lots. Fixed the branch ordering: zero-weight liquidation bypasses the ordinary 100-share threshold; it uses exact cap shares to close the full odd-lot remainder only when the cap can carry it. Otherwise it sells only whole lots allowed by the cap and retains the remainder. One-price limit-down and missing volume/amount continue to block the sale.
- Replayed the full ledger and reports from all 26 saved prediction files. No quarter model was retrained and no prediction file was changed.
- Evidence of the corrected fills: 17 V11 sell orders have non-100-share quantities because each is a complete zero-weight exit. For example, 2021-02-02 sells 14,400.7559 shares of 601988.SH, including the 0.7559 adjusted-share remainder. The final daily ledger has no more than two V11 or 60d Top2 positions.
- Final audit after replay: V11 blocked events 0, minimum cash CNY 13.8452, max positions 2, max accounting residual 2.91e-11. 60d Top2 blocked 0, minimum cash CNY 9.1265, max positions 2. Equal hold blocked 0, minimum cash CNY 938.5489, max positions 4.
- Final net return differences: V11 is +12.7114 percentage points versus equal hold and -2.6791 points versus 60d Top2. Updated SUMMARY.md, metrics.json, annual_returns.csv, quarterly_returns.csv, daily_equity.csv, and trades.csv all come from this corrected replay.
