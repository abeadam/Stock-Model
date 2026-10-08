import numpy as np
import pandas as pd

from macro_features import MACRO_SYMBOLS, add_macro_features
from quote_features import add_quote_columns

START = 1_789_977_600


def es_rows(closes):
    return pd.DataFrame({"time": pd.to_datetime(START + np.arange(len(closes)) * 5, unit="s"),
                         "Close": np.asarray(closes, float)})


def quotes(bids, asks):
    index = START + np.arange(len(bids)) * 5
    return pd.DataFrame({"bid": np.asarray(bids, float), "ask": np.asarray(asks, float)}, index=index)


def test_spread_mid_and_trade_position():
    frame = add_quote_columns(es_rows([100.25, 100.0]), quotes([100.0, 100.0], [100.25, 100.5]))
    assert frame["mid"].tolist() == [100.125, 100.25]
    assert frame["quote_spread_ticks"].tolist() == [1.0, 2.0]
    assert frame["quote_trade_vs_mid_ticks"].tolist() == [0.5, -1.0]     # printed at the ask, then at the bid


def test_mid_change_looks_back_only():
    frame = add_quote_columns(es_rows([100, 100, 100]), quotes([100, 100.25, 100.5], [100.25, 100.5, 100.75]))
    assert np.isnan(frame["quote_mid_change_1"].iloc[0])
    assert frame["quote_mid_change_1"].iloc[2] == 1.0
    assert frame["quote_mid_change_3"].isna().all()                      # needs a quote 3 bars back


def test_missing_quote_gives_nan():
    frame = add_quote_columns(es_rows([100, 100]), quotes([100.0], [100.25]))
    assert np.isnan(frame["mid"].iloc[1])


def test_macro_returns():
    series = {symbol: pd.Series([50.0, 51.0], index=[START, START + 5]) for symbol in MACRO_SYMBOLS}
    frame = add_macro_features(es_rows([100, 100]), series)
    assert np.isclose(frame["macro_tlt_ret_1"].iloc[1], 2.0)
    assert np.isnan(frame["macro_uso_ret_12"].iloc[1])
