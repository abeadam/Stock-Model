import numpy as np
import pandas as pd

from release_clock_features import FAR_AWAY, RELEASE_FEATURES, add_release_features


def bars(times_eastern, day="2026-06-01"):
    stamps = pd.to_datetime([f"{day} {t}" for t in times_eastern]).tz_localize(
        "America/New_York").tz_convert("UTC").tz_localize(None)
    return pd.DataFrame({"time": stamps, "session_day": [pd.Timestamp(day).date()] * len(stamps)})


def test_the_reaction_flag_is_on_only_just_after_the_slot():
    frame = add_release_features(bars(["09:59", "10:00", "10:02", "10:03", "10:30"]))
    assert frame["clock_in_reaction_ten"].tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]


def test_before_and_after_a_slot_are_distinguishable():
    frame = add_release_features(bars(["09:55", "10:05"]))
    offsets = frame["clock_minutes_from_ten"].to_numpy()
    assert offsets[0] == -5.0 and offsets[1] == 5.0


def test_distance_is_capped_so_the_far_morning_is_not_an_outlier():
    frame = add_release_features(bars(["09:30"], day="2026-06-01"))
    assert abs(frame["clock_minutes_from_two"].to_numpy()[0]) <= FAR_AWAY


def test_the_fed_slot_is_read_at_two_not_ten():
    frame = add_release_features(bars(["14:01"]))
    assert frame["clock_in_reaction_two"].to_numpy()[0] == 1.0
    assert frame["clock_in_reaction_ten"].to_numpy()[0] == 0.0


def test_minute_of_session_counts_from_the_opening_bell():
    frame = add_release_features(bars(["09:30", "10:30"]))
    assert frame["clock_minute_of_session"].tolist() == [0.0, 60.0]


def test_the_business_day_counts_trading_days_not_dates():
    frame = pd.concat([bars(["10:00"], day="2026-06-01"), bars(["10:00"], day="2026-06-04")])
    built = add_release_features(frame.reset_index(drop=True))
    assert built["clock_business_day_of_month"].tolist() == [1.0, 2.0]


def test_every_declared_column_is_produced():
    frame = add_release_features(bars(["10:00", "14:00"]))
    assert all(name in frame.columns for name in RELEASE_FEATURES)
