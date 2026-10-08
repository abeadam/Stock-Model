"""Backtest of the ES overnight-drift rule: long at the 16:00 ET close, out at the
next trading day's 09:30 ET open, flat during the day.

Index returns have historically accrued mostly overnight while the regular session
adds little. The rule has no fitted parameters. Trades that span a contract switch
have the switch's carry jump removed (see contract_rolls.py).

Run from this directory:  python overnight_backtest.py [--data-dir PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")  # futures_price imports pyplot
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contract_rolls import roll_gaps  # noqa: E402
from session_prices import closes_by_bar_end, prices_at_clock, trading_weekdays  # noqa: E402

ENTRY_CLOCK = "16:00"
EXIT_CLOCK = "09:30"
POINT_VALUE = 50.0             # dollars per ES point
ROUND_TRIP_COST = 5.00         # $2.50 commission per fill
TRADING_DAYS_PER_YEAR = 252
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "daily_data"


@dataclass(frozen=True)
class Summary:
    trades: int
    net: float
    hit_rate: float
    mean: float
    annualized_sharpe: float
    t_stat: float
    max_drawdown: float
    worst: float
    best: float


def daily_prices(closes: pd.Series) -> pd.DataFrame:
    """Per trading weekday: the 16:00 and 09:30 prices and any contract-switch jump that night."""
    days = trading_weekdays(closes)
    return pd.DataFrame({"close_price": prices_at_clock(closes, days, ENTRY_CLOCK),
                         "open_price": prices_at_clock(closes, days, EXIT_CLOCK),
                         "roll_gap": roll_gaps(closes, days).to_numpy()}, index=days)


def overnight_trades(closes: pd.Series) -> pd.DataFrame:
    """One row per entry day: long at 16:00, out at the next day that has a 09:30 price."""
    days = daily_prices(closes)
    days_with_open = days.index[days["open_price"].notna()]
    exit_slot = days_with_open.searchsorted(days.index, side="right")
    tradable = (exit_slot < len(days_with_open)) & days["close_price"].notna().to_numpy()

    trades = pd.DataFrame({"entry_price": days.loc[tradable, "close_price"],
                           "exit_day": days_with_open[exit_slot[tradable]]})
    trades["exit_price"] = days.loc[trades["exit_day"], "open_price"].to_numpy()
    # Every contract switch on a halt from the entry day up to (not including) the exit day.
    rolls_before = days["roll_gap"].cumsum() - days["roll_gap"]
    trades["roll_gap"] = (rolls_before.loc[trades["exit_day"]].to_numpy()
                          - rolls_before.loc[trades.index].to_numpy())
    trades["points"] = trades["exit_price"] - trades["entry_price"] - trades["roll_gap"]
    trades["net"] = trades["points"] * POINT_VALUE - ROUND_TRIP_COST
    return trades


def summarize(net: pd.Series) -> Summary:
    equity = net.cumsum()
    volatility = net.std(ddof=1)
    return Summary(trades=len(net), net=float(net.sum()), hit_rate=float((net > 0).mean()),
                   mean=float(net.mean()),
                   annualized_sharpe=float(net.mean() / volatility * np.sqrt(TRADING_DAYS_PER_YEAR)),
                   t_stat=float(net.mean() / volatility * np.sqrt(len(net))),
                   max_drawdown=float((equity - equity.cummax()).min()),
                   worst=float(net.min()), best=float(net.max()))


def format_summary(label: str, summary: Summary) -> str:
    return (f"{label:<24} trades {summary.trades:>4}  hit {summary.hit_rate:6.1%}  net ${summary.net:>9,.0f}  "
            f"per trade ${summary.mean:>7.2f}  Sharpe {summary.annualized_sharpe:5.2f}  t {summary.t_stat:+.2f}  "
            f"max drawdown ${summary.max_drawdown:>9,.0f}  worst ${summary.worst:>7,.0f}")


def report(trades: pd.DataFrame) -> None:
    print(f"ES overnight drift: long {ENTRY_CLOCK} -> next {EXIT_CLOCK} ET, "
          f"{trades.index[0].date()} .. {trades['exit_day'].iloc[-1].date()}")
    removed = trades.loc[trades["roll_gap"] != 0, "roll_gap"]
    print(f"contract-switch jumps removed: "
          f"{ {day.date().isoformat(): round(gap, 2) for day, gap in removed.items()} }")
    print(format_summary("all nights", summarize(trades["net"])))
    for year, group in trades.groupby(trades.index.year):
        print(format_summary(f"  {year}", summarize(group["net"])))
    monthly = trades["net"].groupby(trades.index.strftime("%Y-%m")).sum()
    print(f"positive months {(monthly > 0).sum()}/{len(monthly)}: "
          + " ".join(f"{month}:{net / 1000:+.1f}k" for month, net in monthly.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="directory of *_ES.txt 5-second files")
    arguments = parser.parse_args()
    # Imported here so the tests, which never load files, skip futures_price's heavy imports.
    from futures_price import load_es_data

    report(overnight_trades(closes_by_bar_end(load_es_data(arguments.data_dir))))


if __name__ == "__main__":
    main()
