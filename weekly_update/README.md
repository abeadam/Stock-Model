# Weekly Stock-Model update

Runs the full model refresh every **Monday at 08:00** and produces a threshold
recommendation for `futures_trader.py`. It does not edit the trader and does not
place orders.

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
| 3 | `basic_model/lightgbm_model_highest.py` + `lightgbm_model_lowest.py`, concurrently | ibkr venv |
| 4 | `gradient_value/prepare_data.py` | ibkr venv |
| 5 | `gradient_value/train_predictor.py` | ibkr venv |
| 6 | `future_prices/model_tester.py` | ibkr venv |
| 6b | `gradient_value/optimize_thresholds.py` | ibkr venv |
| 7 | `propose_thresholds.py` — writes the recommendation | Stock-Model venv |

**Why 6b exists.** `model_tester.py` backtests the RL agent and prints portfolio
P&L — it never emits gradient probability thresholds. The values in
`futures_trader.py` come from `optimize_thresholds.py`, which grids buy/sell on a
validation split and writes `Chosen: policy=... buy=... sell=...` to
`gradient_value/threshold_optimization.txt`. Step 7 parses that line. Step 6 is
kept for its RL P&L report, which is included in the weekly recommendation.

## Output

```
weekly_update/
├── logs/       run_<timestamp>.log  (driver)  +  one log per step
├── reports/    thresholds_<timestamp>.md, latest_recommendation.md, .json
└── backups/    futures_trader_<timestamp>.py  (copy taken every run)
```

Read `reports/latest_recommendation.md` each Monday. If it proposes a change,
edit the two constants in `futures_trader.py` yourself and **restart the trader** —
it reads them at startup.

## Failure behaviour

- Stops at the first failing step; nothing downstream runs on bad data.
- A clean exit code isn't trusted on its own: step logs are also scanned for
  `Traceback` and `Error:`, because `model_tester.py` catches exceptions and still
  exits 0. A script that legitimately prints those strings will trip this — check
  the step log before assuming a real failure.
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
- Step 7 refuses to recommend from a `threshold_optimization.txt` older than the
  current run, and exits 3 if the proposed pair fails sanity checks
  (buy outside 0.50–0.95, sell outside 0.05–0.50, or buy ≤ sell).

## Requirements on Monday morning

- TWS or IB Gateway running, API enabled, socket port 7497.
- Mac awake at 08:00. If it's asleep, launchd runs the job on the next wake —
  to have it wake on its own: `sudo pmset repeat wakeorpoweron M 07:55:00`.
