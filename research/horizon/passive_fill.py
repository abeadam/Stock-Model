"""Would a limit order actually get filled, and at what cost?

A resting limit order pays no spread, but it fills only when the queue ahead of it
clears, and mostly when flow is pushing through its price. This simulates that from
the tick data:

  joining the bid at price P puts the size resting at P ahead of you;
  every trade printed at P consumes that queue, oldest first, and once it is gone you fill;
  a trade printed below P means the market traded through your level, so you fill there too;
  if neither happens inside the time limit, the order is cancelled unfilled.

Cancellations by orders ahead of you are invisible in this data and would only help,
so the fill rate here is a floor, not an estimate. Selling mirrors buying.
"""
from __future__ import annotations

import numpy as np

TICK = 0.25


class Book:
    """Trades and quotes of one session, indexed for fast forward walks."""

    def __init__(self, trades, quotes):
        self.trade_times = trades["Date"].to_numpy()
        self.trade_prices = trades["Price"].to_numpy()
        self.trade_sizes = trades["Size"].to_numpy()
        self.quote_times = quotes["Date"].to_numpy()
        self.bid = quotes["Bid"].to_numpy()
        self.ask = quotes["Ask"].to_numpy()
        self.bid_size = quotes["BidSize"].to_numpy()
        self.ask_size = quotes["AskSize"].to_numpy()

    def quote_row(self, moment: float) -> int:
        """The last quote at or before `moment`; -1 before the first quote."""
        return int(np.searchsorted(self.quote_times, moment, side="right")) - 1

    def first_trade_row(self, moment: float) -> int:
        return int(np.searchsorted(self.trade_times, moment, side="right"))


def simulate_limit_order(book: Book, placed_at: float, price: float, side: int, queue_ahead: float,
                         time_limit: float) -> float | None:
    """Time of the fill, or None if the order is still unfilled when the time limit passes.

    side +1 rests on the bid (buying), -1 rests on the ask (selling)."""
    deadline = placed_at + time_limit
    for row in range(book.first_trade_row(placed_at), len(book.trade_times)):
        moment, traded_price, size = book.trade_times[row], book.trade_prices[row], book.trade_sizes[row]
        if moment > deadline:
            return None
        through = traded_price < price if side > 0 else traded_price > price
        if through:                                     # the market traded past the level: we are done
            return float(moment)
        if traded_price == price:
            queue_ahead -= size
            if queue_ahead <= 0:
                return float(moment)
    return None


def stop_exit(book: Book, entry_time: float, entry_price: float, side: int, stop_ticks: float,
              deadline: float) -> tuple[float, float] | None:
    """(time, price) where a stop would trigger and leave at market, or None if it never does."""
    stop_price = entry_price - side * stop_ticks * TICK
    for row in range(book.first_trade_row(entry_time), len(book.trade_times)):
        moment, traded_price = book.trade_times[row], book.trade_prices[row]
        if moment > deadline:
            return None
        if (traded_price <= stop_price) if side > 0 else (traded_price >= stop_price):
            quote = book.quote_row(moment)
            return float(moment), float(book.bid[quote] if side > 0 else book.ask[quote])
    return None


def passive_round_trip(book: Book, decision_at: float, side: int, target_ticks: float,
                       entry_time_limit: float, hold_time_limit: float,
                       stop_ticks: float | None = None) -> dict:
    """Join the touch, then try to exit a tick better; if that does not fill, leave at market.
    With stop_ticks set, leave at market as soon as the market trades that far against the fill.

    Returns the attempt's outcome in points (entry-to-exit, signed by side)."""
    quote = book.quote_row(decision_at)
    if quote < 0:
        return {"filled": False}
    entry_price = book.bid[quote] if side > 0 else book.ask[quote]
    queue = book.bid_size[quote] if side > 0 else book.ask_size[quote]
    entry_time = simulate_limit_order(book, decision_at, entry_price, side, queue, entry_time_limit)
    if entry_time is None:
        return {"filled": False}

    exit_price = entry_price + side * target_ticks * TICK
    exit_quote = book.quote_row(entry_time)
    # Resting inside the spread means nobody is ahead of us at that price.
    resting_ahead = 0.0 if (side > 0) == (exit_price < book.ask[exit_quote]) else (
        book.ask_size[exit_quote] if side > 0 else book.bid_size[exit_quote])
    exit_time = simulate_limit_order(book, entry_time, exit_price, -side, resting_ahead, hold_time_limit)
    stopped = (stop_exit(book, entry_time, entry_price, side, stop_ticks, entry_time + hold_time_limit)
               if stop_ticks else None)
    if exit_time is not None and not (stopped and stopped[0] < exit_time):
        return {"filled": True, "points": side * (exit_price - entry_price), "exit": "target"}
    if stopped:
        return {"filled": True, "points": side * (stopped[1] - entry_price), "exit": "stop"}

    # Time is up: cross the spread and take what the book offers.
    final = book.quote_row(entry_time + hold_time_limit)
    market_exit = book.bid[final] if side > 0 else book.ask[final]
    return {"filled": True, "points": side * (market_exit - entry_price), "exit": "market"}
