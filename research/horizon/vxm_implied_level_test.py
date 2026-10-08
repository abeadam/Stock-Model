"""Does an implied-volatility LEVEL help, when the model has never been given one?

The premium built from the VIX index failed: de-seasonalised, it cost the volatility head
0.0049 R2 at t -5.8. But that construction divided the clock out of the level, and the
columns that did help carried levels. So the prior question was never asked -- the base
163 columns contain **no implied volatility level at all**. `vxm_close`, `vxm_pct`,
`vxm_pct_lag1/2` and `vxm_es_ratio` all sit in `model_input.csv` and none of them survives
into the feature set, because `load()` restricts `usecols` to the 44 core features.

Four arms on top of the 163, paired, both heads:

  plus level            `vxm_close` alone. The arm that matters: if the implied level does
                        nothing by itself, anything engineered from it starts much weaker
  plus level and change `vxm_close` + `vxm_pct`
  plus range over vxm   `flow_range_pct_60 / vxm_close`, realized range against implied
                        level, which is the surprise ratio a tree only needs monotone
  plus detrended level  `vxm_close` minus its own trailing 20-session median

**Why the last arm exists.** `vxm_close` would be one of very few LEVEL columns in an
otherwise all-differences feature set, and levels are what make June and September
separable at AUC 1.0000. A level can therefore "help" by telling the model which regime it
is standing in, which is knowledge it will not have live. De-trending removes the slow
component and keeps the relative position: if the gain survives that, it is not the month
talking. Per-fold numbers are printed for the same reason -- a month-encoding column would
struggle to help in both test months.

usage: vxm_implied_level_test.py [HORIZON_BARS]   (default 18)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from deep_ladder_both_heads import head_scores, paired_t
from forward_extremes import forward_target
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from purged_folds import purged_fold_splits
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS, MODEL_INPUT

HORIZON_BARS = 18
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)
VXM_COLUMNS = ("vxm_close", "vxm_pct")
RANGE_COLUMN = "flow_range_pct_60"
DETREND_SESSIONS = 20
MIN_DETREND_SESSIONS = 5
EPSILON = 1e-9
RESULTS_PATH_TEMPLATE = "vxm_implied_level_test_h{horizon}.json"


def load_vxm_columns() -> pd.DataFrame:
    """The VXM columns `load()` leaves behind, keyed by bar-start epoch."""
    frame = pd.read_csv(MODEL_INPUT, usecols=["DateTime", *VXM_COLUMNS])
    frame = frame.drop_duplicates(subset="DateTime", keep="first")
    epochs = pd.to_datetime(frame.pop("DateTime")).astype("int64") // 10**9
    return frame.set_index(pd.Index(epochs, dtype="int64"))


def trailing_session_median(values: np.ndarray, session_days: np.ndarray) -> np.ndarray:
    """Each session's value minus the median of the previous sessions' medians."""
    frame = pd.DataFrame({"day": session_days, "value": values})
    per_day = frame.groupby("day", sort=True)["value"].median()
    profile = per_day.rolling(DETREND_SESSIONS, min_periods=MIN_DETREND_SESSIONS).median().shift(1)
    return frame["day"].map(profile).to_numpy()


def main(horizon: int = HORIZON_BARS) -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    vxm = load_vxm_columns().reindex(epochs)
    level = vxm["vxm_close"].to_numpy(float)
    change = vxm["vxm_pct"].to_numpy(float)
    session_days = data["session_day"].to_numpy()
    detrended = level - trailing_session_median(level, session_days)
    surprise = data[RANGE_COLUMN].to_numpy(float) / np.where(np.abs(level) > EPSILON, level, np.nan)
    data = data.assign(vxm_close=level, vxm_pct=change, vxm_detrended_level=detrended,
                       vxm_range_surprise=surprise)

    direction_target = average_target(data["mid"].to_numpy(float), epochs, horizon)
    decided = np.isfinite(direction_target) & (direction_target != 0)
    range_target = forward_target({"seconds": epochs, "close": data["Close"].to_numpy(float),
                                   "high": data["High"].to_numpy(float),
                                   "low": data["Low"].to_numpy(float)}, horizon, "range")
    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    arms = {"base": base,
            "plus level": base + ["vxm_close"],
            "plus level and change": base + ["vxm_close", "vxm_pct"],
            "plus range over vxm": base + ["vxm_range_surprise"],
            "plus detrended level": base + ["vxm_detrended_level"]}
    splits = purged_fold_splits(data["time"], horizon)
    print(f"horizon {horizon}, purge {horizon}; vxm_close covers "
          f"{np.isfinite(level).mean():.1%} of regular-hours bars, detrended "
          f"{np.isfinite(detrended).mean():.1%}; {len(SEEDS)} seeds x {len(splits)} folds "
          f"({time.time() - started:.0f}s to build)\n", flush=True)

    auc, r2 = {}, {}
    for name, columns in arms.items():
        X = data[columns].to_numpy(np.float32)
        for seed in SEEDS:
            aucs, r2s, _ = head_scores(X, direction_target, decided, range_target, splits,
                                       {**LGBM_PARAMS, "random_state": seed}, added=0)
            auc.setdefault(name, []).extend(aucs)
            r2.setdefault(name, []).extend(r2s)
        per_fold = "  ".join(f"fold {fold + 1} R2 {np.mean(r2[name][fold::len(splits)]):.4f}"
                             for fold in range(len(splits)))
        print(f"   {name:<22}direction {np.mean(auc[name]):.4f}  volatility R2 "
              f"{np.mean(r2[name]):.4f}   {per_fold}   ({time.time() - started:.0f}s)", flush=True)
        del X
        with open(RESULTS_PATH_TEMPLATE.format(horizon=horizon), "w") as handle:
            json.dump({"auc": auc, "r2": r2}, handle, indent=1)

    print(f"\n{'arm':<24}{'direction':>12}{'t':>7}{'volatility':>13}{'t':>7}")
    print(f"{'base':<24}{np.mean(auc['base']):>12.4f}{'':>7}{np.mean(r2['base']):>13.4f}")
    for name in arms:
        if name == "base":
            continue
        auc_gain, auc_t = paired_t(np.array(auc[name]) - np.array(auc["base"]))
        r2_gain, r2_t = paired_t(np.array(r2[name]) - np.array(r2["base"]))
        print(f"{name:<24}{auc_gain:>+12.5f}{auc_t:>7.1f}{r2_gain:>+13.5f}{r2_t:>7.1f}")
    print("\nIf the plain level helps but the detrended one does not, the level was "
          "telling the model which month it was standing in.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else HORIZON_BARS)
