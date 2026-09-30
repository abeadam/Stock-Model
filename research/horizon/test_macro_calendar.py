from datetime import date

from macro_calendar import (IN_SESSION, RELEASES, RELEASE_HOUR_MINUTE, kinds_on, release_days)


def test_every_release_kind_has_a_published_time():
    assert set(RELEASES) == set(RELEASE_HOUR_MINUTE)


def test_only_the_fed_prints_after_the_opening_bell():
    for kind, (hour, minute) in RELEASE_HOUR_MINUTE.items():
        after_open = (hour, minute) >= (9, 30)
        assert after_open == (kind in IN_SESSION), kind


def test_the_monthly_series_have_twelve_dates_each():
    for kind in ("cpi", "nfp"):
        assert len(RELEASES[kind]) == 12, kind


def test_the_fed_meets_eight_times():
    assert len(RELEASES["fomc"]) == 8


def test_no_release_is_listed_on_a_weekend():
    for kind, days in RELEASES.items():
        assert all(day.weekday() < 5 for day in days), kind


def test_a_known_day_is_recognised():
    assert "cpi" in kinds_on(date(2026, 6, 10))
    assert "fomc" in kinds_on(date(2026, 6, 17))
    assert kinds_on(date(2026, 6, 12)) == ()


def test_a_day_can_carry_two_releases():
    """March 18 2026 is both a PPI morning and an FOMC afternoon."""
    assert set(kinds_on(date(2026, 3, 18))) == {"ppi", "fomc"}


def test_selecting_kinds_narrows_the_days():
    assert release_days(["fomc"]) < release_days()
    assert len(release_days(["cpi"])) == 12
