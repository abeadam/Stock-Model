# Cross-instrument % change correlation study

Tests whether percent changes of ES, SPX, and SPY — each instrument on its own and
the divergences between them — predict ES's forward move, with **no input ever using
the current bar**.

## What it computes

For every lookback lag `n` (5-second bars) and each instrument, the n-bar percent
change anchored at the **previous** close:
`(Close[t-1] - Close[t-1-n]) / Close[t-1-n] * 100`. From those:

- **Per-instrument** changes: ES, SPX, SPY.
- **Pairwise differences**: ES−SPX, ES−SPY, SPY−SPX (lead/lag divergence).
- **ES-volume-normalized** variant of every feature, divided by the **previous** bar's
  ES volume (the divisor is lagged too, so it also excludes bar `t`).

Default lags span 5 s to 30 min, finer at the short end:
`{1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240, 300, 360}` bars.

Each feature is correlated against ES forward moves at horizons
`{12, 60, 120, 180, 240, 360}` bars (1, 5, 10, 15, 20, 30 min), in two forms:

- **signed** forward return `(Close[t+H] - Close[t]) / Close[t] * 100`, and
- **binary** up/down direction (flat bars excluded).

All changes and targets are computed **within a trading day** so no window crosses an
overnight gap. Bars are restricted to regular trading hours (09:30–16:00 ET) via an
inner join across the three instruments, which share an identical 5-second grid.

### No current bar in any input

Every feature is anchored at `Close[t-1]` and the volume divisor uses bar `t-1`, so the
current bar `t` — the bar the forward target is measured from — never enters an input.
This one-bar embargo removes the mechanical correlation that arises when a feature and a
target both depend on the noisy print `Close[t]` (bid-ask bounce, non-synchronous quotes).
A prior version that shared `Close[t]` showed a spurious ES−SPY correlation of −0.145 that
collapsed to ~−0.005 once the bar was excluded; that contaminated path has been removed.

## Run

```bash
python run.py        # all common trading days (2025-05-27 .. 2026-06-11, 264 days)
python run.py 20     # first 20 days (quick check)
```

Outputs (written next to the scripts):

- `feature_target_correlations.csv` — every (feature, target) pair, sorted by |corr|.
- `correlation_vs_lag_{signed,binary}.png` — correlation vs lag (log scale), split by
  raw/volume-normalized (rows) and forward horizon (columns).

## Headline findings (full 264-day run)

With the current bar excluded everywhere, **cross-instrument divergences are noise** and
the only structure left is **own-instrument**, with a horizon-dependent sign:

- **Short lookback → reversion.** A 45–60 bar (≈4–5 min) move predicts the opposite over
  the next ~5 min: `ES_pct_45` vs `ES_fwd_ret_60` ≈ **−0.016**. This holds across the
  short half of the grid.
- **Long lookback → momentum.** A 30-min move tends to continue: `ES_pct_360` vs
  `ES_fwd_ret_360` ≈ **+0.035** (the strongest pair overall). The sign flips from negative
  (reversion) to positive (momentum) along the lookback/horizon diagonal.
- **ES, SPX, SPY are interchangeable here** (corr ≈ 0.0351 / 0.0349 / 0.0349 for the
  30-min/30-min pair): SPX and SPY add nothing over ES's own past.
- **Differences are dead:** best `ES_minus_SPY` ≈ 0.002, `SPY_minus_SPX` ≈ −0.011.
- **Volume normalization halves the signal** (mean |corr| 0.0043 raw vs 0.0021 normalized).

**Stability.** The structure holds across the dataset: splitting the 264 days in half, the
30-min/30-min momentum pair is +0.0346 (first half) vs +0.0357 (second half), and the
short-reversion pair is −0.012 vs −0.018 — same sign, similar size in both. (A 10-day
subsample gave a misleading −0.097 for the momentum pair; that was small-sample noise, not
a regime, and is why per-pair samples here number in the millions.)

**Caveats.** Everything is small (|corr| ≤ 0.035): on its own this explains well under 0.2%
of forward-return variance. These are univariate, contemporaneous correlations, not a
backtest — no costs, fills, or latency — so treat the long-horizon momentum as a real but
faint regularity to combine with other signals, not a standalone edge.

## Modules

- `data_alignment.py` — load and inner-join ES/SPX/SPY onto the common RTH grid.
- `features.py` — build the percent-change features and their differences/normalizations.
- `targets.py` — build the forward ES signed/binary targets.
- `correlation.py` — streaming pairwise-complete Pearson correlation via sufficient statistics.
- `reporting.py` — CSV, console summary, and plots.
- `run.py` — orchestration.
