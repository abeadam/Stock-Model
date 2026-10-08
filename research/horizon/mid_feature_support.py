"""Loading the regular-hours frame with every feature family attached, once."""
from __future__ import annotations

import numpy as np

from cross_asset_features import load_symbol_closes, lookback_return_pct
from mid_microstructure import DIVERGENCE_WINDOWS, add_mid_features
from segment_walk_forward import load_cut
from volume_scaled_target import regular_hours


def build_data():
    data = load_cut()
    data = data[regular_hours(data["time"])].reset_index(drop=True)
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    symbol_returns = {f"{symbol}_{bars}": lookback_return_pct(load_symbol_closes(symbol), epochs, bars)
                      for symbol, windows in DIVERGENCE_WINDOWS.items() for bars in windows}
    return add_mid_features(data, symbol_returns)
