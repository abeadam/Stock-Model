#!/usr/bin/env python3
"""
Rebuild per-day daily_data files from an es_with_indicators.csv snapshot.

Maintenance tool, not part of the weekly run. It exists because
es_with_indicators.csv carries the raw OHLCV of every symbol it was built from
(ES in the unprefixed columns, then VXM_* and one block per stock), so an old
snapshot of it can regenerate the daily_data/<date>_<symbol>.txt files that
produced it — including days IBKR will no longer serve, since 5-second history
ages out of its retention window.

Only days strictly before --cutoff are written, and a file that already exists
is never touched, so real downloads always win over reconstructed data.

Bars are bucketed into day files by their ET calendar date, which can differ at
session edges from how download_daily.py bucketed them. That is harmless:
futures_price.py globs every day file, concatenates, sorts by Date and drops
duplicate timestamps, so which file a bar lands in does not affect the result.

Exit codes:
    0  reconstruction finished (or dry run completed)
    2  could not run (missing snapshot, unreadable columns, no cutoff)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

OHLCV = ("Open", "High", "Low", "Close", "Volume")
# ES lives in the unprefixed columns; every other symbol is prefixed.
ES_SYMBOL = "ES"
PREFIXED_SYMBOLS = ("VXM", "TSLA", "NVDA", "MSFT", "META", "JPM",
                    "GOOG", "AVGO", "AMZN", "AAPL")
DATE_COL = "Date"
ET_COL = "DateTime_ET"
CHUNK_ROWS = 250_000

# The snapshot is a left join onto ES timestamps, so every symbol has a value on
# every ES bar — including hours it does not trade. For the equities those
# out-of-session values are forward-filled: the price is frozen AND the volume is
# a repeat of the last real bar, so summing it invents volume that never traded.
# Writing that back as a day file would put fabricated bars into training data,
# so equities are clipped to the session their real downloads cover
# (04:00:00-15:59:55 ET, 8640 bars — verified against a downloaded file).
#
# ES and VXM need no clipping: they trade nearly around the clock, and where VXM
# sits frozen overnight it does so with zero volume in the real downloads too,
# so the snapshot is faithful there rather than filled.
EQUITY_SESSION_ET = ("04:00:00", "15:59:55")
UNCLIPPED_SYMBOLS = frozenset({ES_SYMBOL, "VXM"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True,
                   help="es_with_indicators.csv (or a backup copy of one)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="daily_data directory to write <date>_<symbol>.txt into")
    p.add_argument("--cutoff", default=None,
                   help="write only days strictly before this YYYY-MM-DD; "
                        "defaults to the earliest existing *_ES.txt in --out-dir")
    p.add_argument("--limit-chunks", type=int, default=None,
                   help="stop after this many chunks (for a dry run)")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be written without creating files")
    return p.parse_args()


def fail(message: str) -> None:
    print(f"Error: {message}")
    sys.exit(2)


def symbol_columns(symbol: str) -> dict[str, str]:
    """Source column -> canonical OHLCV name for one symbol."""
    if symbol == ES_SYMBOL:
        return {field: field for field in OHLCV}
    return {f"{symbol}_{field}": field for field in OHLCV}


def default_cutoff(out_dir: Path) -> str:
    """Earliest day already present as a real download, so we stay strictly before it."""
    existing = sorted(p.name[:10] for p in out_dir.glob(f"*_{ES_SYMBOL}.txt"))
    if not existing:
        fail(f"no *_{ES_SYMBOL}.txt in {out_dir} to infer --cutoff from; pass it explicitly")
    return existing[0]


def read_columns(snapshot: Path) -> list[str]:
    header = pd.read_csv(snapshot, nrows=0)
    return list(header.columns)


def main() -> None:
    args = parse_args()
    if not args.snapshot.exists():
        fail(f"snapshot not found: {args.snapshot}")
    if not args.out_dir.is_dir():
        fail(f"output directory not found: {args.out_dir}")

    available = set(read_columns(args.snapshot))
    for required in (DATE_COL, ET_COL):
        if required not in available:
            fail(f"snapshot has no {required!r} column; cannot rebuild day files")

    symbols = [ES_SYMBOL] + [s for s in PREFIXED_SYMBOLS
                             if set(symbol_columns(s)) <= available]
    skipped = [s for s in PREFIXED_SYMBOLS if set(symbol_columns(s)) > available]
    if skipped:
        print(f"Note: no complete OHLCV block for {skipped} — those are not recoverable here")

    cutoff = args.cutoff or default_cutoff(args.out_dir)
    print(f"Snapshot: {args.snapshot}")
    print(f"Out dir:  {args.out_dir}")
    print(f"Cutoff:   writing only days < {cutoff}")
    print(f"Symbols:  {symbols}")
    if args.dry_run:
        print("DRY RUN — no files will be written")
    print()

    usecols = [DATE_COL, ET_COL]
    for symbol in symbols:
        usecols.extend(symbol_columns(symbol))

    # Files this run has opened, so a day spanning several chunks appends rather
    # than truncating. Anything already on disk beforehand is left alone.
    opened: set[Path] = set()
    preexisting_skipped: set[Path] = set()
    rows_written: Counter[str] = Counter()
    days_seen: set[str] = set()
    chunks = 0
    rows_read = 0

    reader = pd.read_csv(args.snapshot, usecols=usecols, chunksize=CHUNK_ROWS)
    for chunk in reader:
        chunks += 1
        rows_read += len(chunk)
        stamps = chunk[ET_COL].astype(str)
        chunk = chunk.assign(_day=stamps.str.slice(0, 10), _time=stamps.str.slice(11, 19))
        chunk = chunk[chunk["_day"] < cutoff]
        if not chunk.empty:
            days_seen.update(chunk["_day"].unique().tolist())
            in_session = ((chunk["_time"] >= EQUITY_SESSION_ET[0])
                          & (chunk["_time"] <= EQUITY_SESSION_ET[1]))
            for symbol in symbols:
                mapping = symbol_columns(symbol)
                rows = chunk if symbol in UNCLIPPED_SYMBOLS else chunk[in_session]
                frame = rows[[DATE_COL, "_day", *mapping]].rename(columns=mapping)
                frame = frame.dropna(subset=list(OHLCV))
                if frame.empty:
                    continue
                frame = frame.astype({DATE_COL: "int64", "Volume": "int64"})
                for day_value, group in frame.groupby("_day", sort=True):
                    path = args.out_dir / f"{day_value}_{symbol}.txt"
                    if path in preexisting_skipped:
                        continue
                    if path not in opened and path.exists():
                        preexisting_skipped.add(path)
                        continue
                    rows_written[symbol] += len(group)
                    if args.dry_run:
                        opened.add(path)
                        continue
                    is_new = path not in opened
                    group[[DATE_COL, *OHLCV]].to_csv(
                        path, index=False, header=is_new,
                        mode="w" if is_new else "a",
                    )
                    opened.add(path)

        if chunks % 20 == 0:
            print(f"  {chunks} chunks / {rows_read:,} rows read — "
                  f"{len(days_seen)} days, {len(opened)} files, "
                  f"{sum(rows_written.values()):,} rows written")
        if args.limit_chunks is not None and chunks >= args.limit_chunks:
            print(f"  stopping after {chunks} chunks (--limit-chunks)")
            break

    print()
    print(f"Chunks read:  {chunks} ({rows_read:,} rows)")
    print(f"Days below cutoff: {len(days_seen)}")
    print(f"Files {'that would be' if args.dry_run else ''} written: {len(opened)}")
    if preexisting_skipped:
        print(f"Left untouched (already existed): {len(preexisting_skipped)} file(s)")
    for symbol in symbols:
        if rows_written[symbol]:
            print(f"  {symbol:6} {rows_written[symbol]:>12,} rows")


if __name__ == "__main__":
    main()
