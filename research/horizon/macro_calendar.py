"""The dated calendar of American macro releases, taken from the agencies that publish it.

IBKR serves this through Wall Street Horizon, which this account is not subscribed to
(error 10276), so the dates are taken from the primary sources instead. They are more
authoritative than any vendor feed and they do not expire:

  FOMC   federalreserve.gov/monetarypolicy/fomccalendars.htm - statement at 14:00 ET on
         the second day of each two-day meeting
  CPI    bls.gov/schedule/news_release/cpi.htm      08:30 ET
  PPI    bls.gov/schedule/news_release/ppi.htm      08:30 ET
  NFP    bls.gov/schedule/news_release/empsit.htm   08:30 ET
  GDP    bea.gov/news/archive                       08:30 ET

**Only the Federal Reserve lands inside the stock session.** Everything else prints at
08:30, an hour before the opening bell, so for a regular-hours model those are yesterday's
news by the time the first bar forms. They fall squarely inside the *overnight* window a
position held from the close to the next open must sit through, which is where this
calendar is worth having.

Dates are 2026 only. Earlier years are on the same pages under their own schedules; add
them when a test needs the older window.
"""
from __future__ import annotations

from datetime import date

RELEASE_HOUR_MINUTE = {"fomc": (14, 0), "cpi": (8, 30), "ppi": (8, 30),
                       "nfp": (8, 30), "gdp": (8, 30)}

FOMC_STATEMENTS = tuple(date(2026, month, day) for month, day in
                        ((1, 28), (3, 18), (4, 29), (6, 17), (7, 29), (9, 16), (10, 28), (12, 9)))
CPI_RELEASES = tuple(date(2026, month, day) for month, day in
                     ((1, 13), (2, 13), (3, 11), (4, 10), (5, 12), (6, 10),
                      (7, 14), (8, 12), (9, 11), (10, 14), (11, 10), (12, 10)))
PPI_RELEASES = tuple(date(2026, month, day) for month, day in
                     ((1, 14), (1, 30), (2, 27), (3, 18), (4, 14), (5, 13), (6, 11),
                      (7, 15), (8, 13), (9, 10), (10, 15), (11, 13), (12, 15)))
NFP_RELEASES = tuple(date(2026, month, day) for month, day in
                     ((1, 9), (2, 11), (3, 6), (4, 3), (5, 8), (6, 5),
                      (7, 2), (8, 7), (9, 4), (10, 2), (11, 6), (12, 4)))
GDP_RELEASES = tuple(date(2026, month, day) for month, day in
                     ((3, 13), (4, 9), (4, 30), (5, 28), (6, 25), (7, 30),
                      (9, 30), (10, 29), (11, 25), (12, 23)))

RELEASES = {"fomc": FOMC_STATEMENTS, "cpi": CPI_RELEASES, "ppi": PPI_RELEASES,
            "nfp": NFP_RELEASES, "gdp": GDP_RELEASES}
IN_SESSION = ("fomc",)                      # the only one that prints after the opening bell


def release_days(kinds=None) -> set:
    """Every dated release day, or only those of the kinds asked for."""
    wanted = RELEASES if kinds is None else {kind: RELEASES[kind] for kind in kinds}
    return {day for days in wanted.values() for day in days}


def kinds_on(day: date) -> tuple:
    return tuple(sorted(kind for kind, days in RELEASES.items() if day in days))
