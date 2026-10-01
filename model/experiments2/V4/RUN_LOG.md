V4 final run record
==================

- Run started: 2026-09-27 17:22:55 (server local time); completed successfully in about 30 seconds.
- Command: python3 /autodl-fs/data/tmp/vlines/bank_momentum_v4/run.py
- Pre-run shared-host snapshot: load average 8.60 / 9.72 / 9.66; RAM used 64 GiB of 440 GiB, available 368 GiB.
- Data snapshot: built 2026-09-27 14:19:24, latest upstream session 2026-09-24.
- Loaded panel: 2,120 sessions × 4 banks × 610 features; the strategy uses adjusted prices and traded amount only.
- Official test: 2025-07-01 through 2026-06-30, exactly 242 sessions, 12 monthly signals. Net +13.63%; initial-Top2 hold +5.85%; four-bank equal-weight hold +3.58%; MDD -17.36%; 19 trades; fees ¥324.14.
- Net return difference: +7.78 percentage points vs initial-Top2 hold and +10.05 percentage points vs four-bank equal-weight hold.
- Exact realized-holdings, no-fee comparison: +14.14%, making the net minus matched difference -0.51 percentage points. This is transaction/implementation friction, not stock-selection alpha.
- Extended annual net / same-holdings no-fee returns: 2023 +30.17% / +30.74%; 2024 +42.60% / +43.58%; 2025 +29.67% / +30.24%; 2026 through 2026-09-24 +11.03% / +11.43%.
- Ledger audit passed: minimum cash ¥124.92 in the official test and ¥5.97 extended; at most 2 concurrent holdings; no blocked entries or exits; final liquidation complete; max equity-to-cash-plus-engine-mark residual 1.46e-11.
- Frozen research-runner SHA-256: f6a8db9c1b09d6900069545a6e390fc3d6a559476c121ae8238d10dd9e4f9b7c; it matches result.json and is preserved as archival provenance in run_frozen_reference.py (not the operational runner). The packaged run.py is path-rebased to the local account_engine.py and writes to this V4 directory; it was not executed during packaging.

