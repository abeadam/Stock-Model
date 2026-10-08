import pytest

from head_columns import (DROPPED_FEATURES, VOLATILITY_ONLY_FEATURES, HeadColumns,
                          split_by_head)

BASE = ["rolling_vol_5", "rolling_vol_20", "ms_uso_minus_mid_3", "ms_uup_minus_mid_1", "ema_9"]


def test_the_dropped_columns_leave_the_direction_head():
    assert not set(split_by_head(BASE).direction) & set(DROPPED_FEATURES)


def test_the_dropped_columns_leave_the_volatility_head_too():
    assert not set(split_by_head(BASE).volatility) & set(DROPPED_FEATURES)


def test_all_six_oil_and_dollar_columns_are_named():
    assert len(DROPPED_FEATURES) == 6
    assert sum("uso" in name for name in DROPPED_FEATURES) == 3
    assert sum("uup" in name for name in DROPPED_FEATURES) == 3


def test_the_volatility_head_gains_the_realized_columns():
    assert set(VOLATILITY_ONLY_FEATURES) <= set(split_by_head(BASE).volatility)


def test_the_direction_head_does_not():
    assert not set(VOLATILITY_ONLY_FEATURES) & set(split_by_head(BASE).direction)


def test_the_surviving_base_columns_are_kept_in_order():
    assert split_by_head(BASE).direction == ["rolling_vol_5", "rolling_vol_20", "ema_9"]


def test_the_added_columns_come_last_so_gain_can_be_read():
    columns = split_by_head(BASE)
    tail = columns.volatility[-columns.volatility_added:]
    assert tail == list(VOLATILITY_ONLY_FEATURES)


def test_the_added_count_matches_what_was_appended():
    columns = split_by_head(BASE)
    assert columns.volatility_added == len(columns.volatility) - len(columns.direction)


def test_nothing_is_counted_as_added_to_direction():
    assert split_by_head(BASE).direction_added == 0


def test_a_column_already_present_is_not_duplicated():
    base = BASE + list(VOLATILITY_ONLY_FEATURES)
    columns = split_by_head(base)
    assert len(columns.volatility) == len(set(columns.volatility))
    assert columns.volatility_added == 0


def test_a_base_with_nothing_to_drop_is_untouched():
    base = ["ema_9", "rolling_vol_5"]
    assert split_by_head(base).direction == base


def test_the_heads_differ_only_by_the_volatility_columns():
    columns = split_by_head(BASE)
    assert set(columns.volatility) - set(columns.direction) == set(VOLATILITY_ONLY_FEATURES)


def test_an_empty_base_still_gives_the_volatility_head_its_columns():
    columns = split_by_head([])
    assert columns.direction == [] and columns.volatility == list(VOLATILITY_ONLY_FEATURES)


def test_the_dataclass_is_frozen_so_a_fold_cannot_edit_it():
    with pytest.raises(Exception):
        split_by_head(BASE).direction = []


def test_custom_drops_are_honoured():
    assert "ema_9" not in split_by_head(BASE, dropped=("ema_9",)).direction
