import numpy as np
import pandas as pd

from rate_futures_features import RATE_FEATURES, RATE_SYMBOLS, add_rate_features


def test_every_declared_column_is_named_consistently():
    assert all(name.startswith("rf_") for name in RATE_FEATURES)
    for symbol in RATE_SYMBOLS:
        assert f"rf_{symbol.lower()}_ret_1" in RATE_FEATURES
        assert f"rf_{symbol.lower()}_minus_mid_1" in RATE_FEATURES


def test_the_curve_is_the_short_end_less_the_long_end():
    assert "rf_curve_change_1" in RATE_FEATURES
    assert sum("curve" in name for name in RATE_FEATURES) == 3


def test_there_is_a_column_for_each_symbol_and_window():
    expected = len(RATE_SYMBOLS) * 3 + len(RATE_SYMBOLS) * 3 + 3
    assert len(RATE_FEATURES) == expected


def test_columns_appear_on_a_frame_even_with_no_data_downloaded(monkeypatch):
    """A symbol whose files are absent must leave blank columns, not raise."""
    import rate_futures_features as module
    monkeypatch.setattr(module, "load_symbol_closes",
                        lambda symbol: pd.Series(dtype=float, index=pd.Index([], dtype="int64")))
    frame = pd.DataFrame({"time": pd.to_datetime(np.arange(20) * 5 + 1_700_000_000, unit="s"),
                          "mid": np.full(20, 5000.0)})
    built = module.add_rate_features(frame)
    assert all(name in built.columns for name in RATE_FEATURES)
    assert built["rf_zt_ret_1"].isna().all()
