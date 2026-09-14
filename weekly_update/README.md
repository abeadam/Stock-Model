# Weekly Stock-Model update

Runs the full model refresh every **Monday at 08:00** and applies the new
buy/sell probability thresholds to `futures_trader.py`, if they pass sanity
checks and a held-out backtest. It never restarts the running trader — that
stays a manual step — and it never places orders.

## Install

```bash
cp /Users/abeadam/dev/model/Stock-Model/weekly_update/com.abeadam.stockmodel.weekly.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.abeadam.stockmodel.weekly.plist
launchctl print gui/$(id -u)/com.abeadam.stockmodel.weekly | head -20   # verify
```

Test it now without waiting for Monday (needs TWS running):

```bash
launchctl kickstart -p gui/$(id -u)/com.abeadam.stockmodel.weekly
tail -f /Users/abeadam/dev/model/Stock-Model/weekly_update/logs/run_*.log
```

Or run it straight from the shell:

```bash
bash /Users/abeadam/dev/model/Stock-Model/weekly_update/run_weekly_update.sh
```

Remove or change the schedule:

```bash
launchctl bootout gui/$(id -u)/com.abeadam.stockmodel.weekly
# edit the plist, then bootstrap again
```

## What runs

| Step | Command | Interpreter |
|---|---|---|
| 0 | preflight: TWS reachable on 127.0.0.1:7497 | — |
| 1 | `download_daily.py` — pulls each symbol-day into `daily_data/`, skipping days already on disk | ibkr venv |
| 2 | `future_prices/futures_price.py` | Stock-Model venv |
| 2b | back up the current models to `backups/models_<timestamp>/` before anything retrains | — |
| 3 | `basic_model/lightgbm_model_highest.py` + `lightgbm_model_lowest.py`, concurrently | ibkr venv |
| 3b | `verify_live_features.py` — every feature the new models selected must be computable live | ibkr venv |
| 4 | `gradient_value/prepare_data.py` | ibkr venv |
| 5 | `gradient_value/train_predictor.py` | ibkr venv |
| 6 | `gradient_value/optimize_thresholds.py` | ibkr venv |
| 6b | `compare_backtest.py` — scores the current and proposed thresholds on the same held-out bars | ibkr venv |
| 7 | `propose_thresholds.py` — applies the new thresholds if they pass every check | Stock-Model venv |

**What step 6 produces.** `optimize_thresholds.py` grids buy/sell probability
thresholds on a validation split and writes
`Chosen: policy=... buy=... sell=...` to
`gradient_value/threshold_optimization.txt`. Step 7 parses that line, sanity-checks
it, checks step 6b's held-out backtest of it, backs up `futures_trader.py`, and —
if every check passes — rewrites the two threshold constants in place.

(There used to be a step 6 running `future_prices/model_tester.py` for an RL
agent P&L report, with `optimize_thresholds.py` as "6b". That RL agent's
checkpoint doesn't currently exist, so model_tester.py was dropped from the
pipeline and optimize_thresholds.py was promoted to step 6.)

## Why step 3b exists

`futures_trader.py` predicts through `basic_model/es_features.py`, which returns one
column per row of `lightgbm_model_<kind>_feature_importances_splits_and_gain.csv` —
and step 3 rewrites that CSV on every retrain. Nothing in the model training
knows or cares what the live generator can compute, so a retrain is free to
select a feature `es_features.py` has no computation for. When that happens
`compute_es_features()` fills the column with NaN, prints a warning, and LightGBM
routes it through its missing-value branch: the trader keeps running, on degraded
input, and no step fails. Backtest metrics still look fine, because training used
the real values from the offline dataset.

That is not hypothetical. It was live: the `highest` model had selected 16 features
(`BB_50_*`, `EMA_10/20/50`, `MFI_7`, `PctChange_Lag_1/2/3/5/20`) that
`es_features.py` never implemented. Step 3b fails the run instead, listing the
offending names.

Two distinct failure modes are checked, because they need different fixes:

- **Not implemented** — port the calculation into `es_features.py`, matching the
  formula in `futures_price.py` / `lightgbm_utils.py` exactly.
- **No live data source** — the name is implemented but can only ever be NaN on a
  live bar window. The training set carries ~440 per-stock columns (`TSLA_*`,
  `NVDA_*`, ...) that the live ES+VXM feed has no source for. If a retrain selects
  one, the fix is to exclude it from training, not to write more feature code.

## Output

```
weekly_update/
├── logs/       run_<timestamp>.log  (driver)  +  one log per step
├── reports/    thresholds_<timestamp>.md, latest_recommendation.md, .json
│               backtest_comparison_<timestamp>.md, latest_backtest_comparison.md,
│               backtest_history.json  (step 6b's scores; step 7 gates on them)
└── backups/    futures_trader_<timestamp>.py  (copy taken every run)
                models_<timestamp>/basic_model/, models_<timestamp>/gradient_value/
                                                (models as they were before step 3)
```

Read `reports/latest_recommendation.md` each Monday to see whether
`futures_trader.py` was updated (and why, if it wasn't). If it was updated,
**restart the trader** — it only reads the thresholds at startup, and this
pipeline never restarts it for you.

## Rolling back a retrain

Steps 3 and 5 overwrite the models in place, and `*.pkl` is gitignored, so step 2b
is the only copy of the previous models. To put a run's predecessors back
(substitute the run's timestamp):

```bash
cp -p /Users/abeadam/dev/model/Stock-Model/weekly_update/backups/models_<timestamp>/basic_model/* /Users/abeadam/dev/model/Stock-Model/future_prices/basic_model/
cp -p /Users/abeadam/dev/model/Stock-Model/weekly_update/backups/models_<timestamp>/gradient_value/* /Users/abeadam/dev/model/Stock-Model/future_prices/gradient_value/
```

The trader loads its models the first time it predicts and keeps them, so a
restart picks up whichever models are on disk — restored or retrained.

## Failure behaviour

- Stops at the first failing step; nothing downstream runs on bad data.
- A clean exit code isn't trusted on its own: step logs are also scanned for
  `Traceback` and `Error:`. The IBKR download scripts print every TWS
  notification as `Error: reqId, code, message`, including routine ones
  (farm connection status, a single symbol-day with no data) — those specific,
  known-benign patterns are excluded from the check (see
  `BENIGN_IBKR_ERROR_PATTERN` in `run_weekly_update.sh`) so they don't fail an
  otherwise-successful step. Any other `Error:`/`Traceback` line still fails
  the run — check the step log before assuming a real failure.
- Per-step timeout of 4h (`STEP_TIMEOUT` env var to change).
- `download_daily.py` has no interactive prompts — it runs unattended and gets
  `/dev/null` on stdin like every other step, so an unexpected prompt anywhere
  in the pipeline fails immediately instead of hanging for the full timeout.

## Data download

Step 1 runs `download_daily.py`, which fetches one symbol-day at a time (SPY,
SPX, VIX, ES, VXM, and nine individual stocks) into
`daily_data/` and skips any file that's already on disk — so a weekly run only
pulls the handful of trading days since the last one. `futures_price.py`
(step 2) reads straight out of `daily_data/` via the symlink between
`Stock-Model/daily_data` (the real directory) and
`interactive-broker-python/Updated Stats/daily_data` (a symlink into it, which
is what `download_daily.py`'s hardcoded output path actually writes to) — so
there's no separate consolidation step.

IBKR only retains 5-second-bar historical data for a limited lookback window,
which is why `download_daily.py` defaults to the last 180 days: that's roughly
the practical ceiling of what's recoverable at this granularity, not an
arbitrary choice.
- A lock directory prevents two runs overlapping.
- Step 7 refuses to apply from a `threshold_optimization.txt` older than the
  current run, and refuses to touch `futures_trader.py` (exit 3, which fails
  the run) if the proposed pair fails a check:
  - sanity: buy outside 0.50–0.95, sell outside 0.05–0.50, or buy ≤ sell;
  - held-out backtest (step 6b, this run): the proposal's net P&L is negative,
    is lower than the current thresholds' net P&L, or is missing. A negative
    net P&L fails the run even when the optimizer re-picked the current
    thresholds, since that means the live values lose money on held-out data.

  `futures_trader.py` is backed up every run regardless of whether it ends up
  changed.

## Requirements on Monday morning

- TWS or IB Gateway running, API enabled, socket port 7497.
- Mac awake at 08:00. If it's asleep, launchd runs the job on the next wake —
  to have it wake on its own: `sudo pmset repeat wakeorpoweron M 07:55:00`.
