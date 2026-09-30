"""Is the average-of-next-five target signal, or is it the bid-ask bounce again?

The attribution ranked `quote_trade_vs_mid_ticks` first by a distance -- the position of
the last trade relative to the mid, in ticks. The target it was explaining is built from
trade closes. If the current close printed at the bid, the next few will tend to print
nearer the ask, and the average of them rises: a feature that measures the bounce
predicting the bounce coming back, which no order can collect.

The mid does not bounce. Rebuilding the same target from mid prices and re-running the
same attribution separates the two readings, and the same is done for every feature that
scored highly, since they may all be drinking from the same well.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from feature_rank import prepare, standardise
from forward_extremes import forward_window, window_is_contiguous
from indicator_features import (EQUITY_INDICATOR_FEATURES, EQUITY_SYMBOLS, MACRO_INDICATOR_FEATURES,
                                MACRO_SYMBOLS, add_indicator_features)
from indicator_test import RETURNS_OF_NEW_DATA
from multi_timeframe_features import ALL_TIMEFRAME_FEATURES, add_timeframe_features
from projection_importance import variable_importance
from rotation_test import PLS_ROW_STRIDE, fold_splits
from segment_walk_forward import load_cut
from short_horizon_walk_forward import FEATURES
from volume_scaled_target import regular_hours

HORIZON_BARS = 5
COMPONENTS = 5
PERCENT = 100.0
WATCH = ["quote_trade_vs_mid_ticks", "xa_spy_ret_1", "macro_gld_ret_1", "quote_mid_change_1",
         "high_close_ratio", "xa_goog_ret_1", "low_close_ratio", "flow_bar_close_location",
         "close_pct_lag1", "xa_basket_ret_1"]


def average_target(price: np.ndarray, seconds: np.ndarray, bars: int) -> np.ndarray:
    reached = forward_window(price, bars, "mean")
    target = (reached - price) / price * PERCENT
    target[~window_is_contiguous(seconds, bars)] = np.nan
    return target


def main() -> None:
    data = load_cut()
    data = data[regular_hours(data["time"])].reset_index(drop=True)
    data = add_timeframe_features(add_indicator_features(data, (*MACRO_SYMBOLS, *EQUITY_SYMBOLS)))
    columns = (list(FEATURES) + RETURNS_OF_NEW_DATA + MACRO_INDICATOR_FEATURES
               + EQUITY_INDICATOR_FEATURES + ALL_TIMEFRAME_FEATURES)
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    targets = {"trade close": average_target(data["Close"].to_numpy(float), seconds, HORIZON_BARS),
               "mid": average_target(data["mid"].to_numpy(float), seconds, HORIZON_BARS)}
    X = data[columns].to_numpy(np.float32)

    table = {}
    for name, target in targets.items():
        vips, correlations = [], []
        for split in fold_splits(data, "dev"):
            rows = split.train[::PLS_ROW_STRIDE]
            rows = rows[np.isfinite(target[rows])]
            fitted = prepare(X[rows])
            standardised = standardise(X[rows], fitted)
            pls = PLSRegression(n_components=COMPONENTS, scale=False).fit(standardised, target[rows])
            vips.append(variable_importance(pls.x_weights_, pls.x_scores_, pls.y_loadings_))
            centred = target[rows] - target[rows].mean()
            correlations.append((standardised * centred[:, None]).mean(axis=0)
                                / np.maximum(standardised.std(axis=0) * centred.std(), 1e-12))
        table[name] = {"vip": np.mean(vips, axis=0), "corr": np.mean(correlations, axis=0)}

    print(f"{len(data):,} regular-hours bars; average over the next {HORIZON_BARS} bars\n")
    print(f"{'feature':<30}{'VIP trade':>11}{'VIP mid':>10}{'corr trade':>12}{'corr mid':>10}{'kept':>8}")
    survived = {}
    for name in WATCH:
        index = columns.index(name)
        trade_vip, mid_vip = table["trade close"]["vip"][index], table["mid"]["vip"][index]
        trade_corr, mid_corr = table["trade close"]["corr"][index], table["mid"]["corr"][index]
        share = abs(mid_corr) / max(abs(trade_corr), 1e-12)
        survived[name] = {"vip_trade": float(trade_vip), "vip_mid": float(mid_vip),
                          "corr_trade": float(trade_corr), "corr_mid": float(mid_corr)}
        print(f"{name:<30}{trade_vip:>11.2f}{mid_vip:>10.2f}{trade_corr:>+12.4f}{mid_corr:>+10.4f}"
              f"{share:>7.0%}")
    order = np.argsort(-table["mid"]["vip"])[:15]
    print("\nTop fifteen by VIP against the mid target (the honest one):")
    for index in order:
        print(f"   {columns[index]:<32}{table['mid']['vip'][index]:>7.2f}"
              f"   corr {table['mid']['corr'][index]:+.4f}")
    with open("mid_versus_trade_target.json", "w") as handle:
        json.dump({"watched": survived,
                   "top_mid": [(columns[i], float(table["mid"]["vip"][i])) for i in order]}, handle, indent=1)


if __name__ == "__main__":
    main()
