#!/usr/bin/env python3
"""Compare two gradient models' held-out results on trend days vs chop days.

Built for the session-feature experiment: does knowing where a bar sits within
its session help the strategy on the days that trend, without hurting the days
that chop? Two steps, because each model is scored against its own
model_input.csv and the candidate's retrain overwrites it:

  snapshot  score the current predictor_model.pkl on its held-out test split and
            save each bar's time, prices and up-probability to a small CSV
  compare   line up two snapshots on their common bars in the untouched second
            half of the test window, label each session trend / mixed / chop,
            and report the strategy's results per label for both models

Day labels use each session's whole path, known only after it closes. They
exist to split the evaluation and are never a model input.

Places no orders and modifies no model or source file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "basic_model"))

import backtest  # noqa: E402
from market_clock import EASTERN_TIMEZONE  # noqa: E402
from optimize_thresholds import VALIDATION_FRACTION  # noqa: E402
from session_features import session_start_times  # noqa: E402

# A session's efficiency: |close - open| / (high - low) over the whole session.
# Near 1 it went one way all day; near 0 it ended where it started.
TREND_DAY_MIN_EFFICIENCY = 0.6
CHOP_DAY_MAX_EFFICIENCY = 0.3
DAY_TYPES = ("trend", "mixed", "chop", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    snapshot = commands.add_parser("snapshot", help="score the current model; save its held-out bars")
    snapshot.add_argument("--out", type=Path, required=True)

    compare = commands.add_parser("compare", help="compare two snapshots by day type")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--buy", type=float, required=True)
    compare.add_argument("--sell", type=float, required=True)
    compare.add_argument("--policy", choices=("hold", "flat"), default="hold")
    return parser.parse_args()


def snapshot(out: Path) -> None:
    test = backtest.load_test_frame(extra_columns=("High", "Low"))
    untouched = test.iloc[int(len(test) * VALIDATION_FRACTION):]
    untouched.to_csv(out, index=False)
    print(f"Saved {len(untouched):,} untouched held-out bars to {out}")


def label_sessions(bars: pd.DataFrame) -> pd.DataFrame:
    """Each bar's session start and day type (trend / mixed / chop), judged
    from that session's whole path."""
    eastern = (pd.to_datetime(bars["DateTime"], utc=True)   # model_input.csv holds UTC
               .dt.tz_convert(EASTERN_TIMEZONE).dt.tz_localize(None))
    session = session_start_times(eastern)
    by_session = bars.groupby(session, sort=False)
    net_move = (by_session["Close"].transform("last") - by_session["Close"].transform("first")).abs()
    day_range = by_session["High"].transform("max") - by_session["Low"].transform("min")
    efficiency = net_move / day_range.where(day_range > 0)
    labels = np.select([efficiency >= TREND_DAY_MIN_EFFICIENCY,
                        efficiency < CHOP_DAY_MAX_EFFICIENCY], ["trend", "chop"], "mixed")
    return pd.Series(labels, index=bars.index).rename("day_type").to_frame().assign(session=session)


def score_by_day_type(bars: pd.DataFrame, probability: np.ndarray,
                      buy: float, sell: float, policy: str) -> dict[str, dict]:
    """Strategy results over the common bars, attributed to each bar's day type."""
    close = bars["Close"].to_numpy(float)
    positions = backtest.build_positions(probability, buy, sell, policy)
    result = backtest.simulate(close, positions)
    next_move = np.sign(np.diff(close, append=close[-1]))
    traded = np.abs(np.diff(positions, prepend=0)) > 0
    scored = (positions != 0) & (next_move != 0)
    correct = np.sign(positions) == next_move

    scores = {}
    for day_type in DAY_TYPES:
        on_day = np.ones(len(bars), bool) if day_type == "all" else (bars["day_type"] == day_type).to_numpy()
        scores[day_type] = {
            "sessions": bars.loc[on_day, "session"].nunique(),
            "net_pnl": float(result["net_bar_pnl"][on_day].sum()),
            "position_changes": int(traded[on_day].sum()),
            "directional_accuracy_pct": float(correct[scored & on_day].mean() * 100)
            if (scored & on_day).any() else float("nan"),
        }
    return scores


def compare(args: argparse.Namespace) -> None:
    baseline = pd.read_csv(args.baseline)
    candidate = pd.read_csv(args.candidate)
    common = baseline.merge(candidate[["DateTime", "probability"]], on="DateTime",
                            suffixes=("_baseline", "_candidate"))
    common = pd.concat([common, label_sessions(common)], axis=1)
    print(f"Common untouched held-out bars: {len(common):,} "
          f"(baseline had {len(baseline):,}, candidate {len(candidate):,})")
    print(f"Thresholds: buy>={args.buy:.2f} sell<={args.sell:.2f} policy={args.policy}\n")

    results = {name: score_by_day_type(common, common[f"probability_{name}"].to_numpy(),
                                       args.buy, args.sell, args.policy)
               for name in ("baseline", "candidate")}
    header = (f"{'day type':<8}{'sessions':>9}{'baseline net':>15}{'candidate net':>15}"
              f"{'delta':>12}{'changes b/c':>16}{'dir acc b/c':>16}")
    print(header)
    print("-" * len(header))
    for day_type in DAY_TYPES:
        b, c = results["baseline"][day_type], results["candidate"][day_type]
        print(f"{day_type:<8}{b['sessions']:>9}{b['net_pnl']:>15,.0f}{c['net_pnl']:>15,.0f}"
              f"{c['net_pnl'] - b['net_pnl']:>+12,.0f}"
              f"{b['position_changes']:>8,}/{c['position_changes']:<7,}"
              f"{b['directional_accuracy_pct']:>8.2f}/{c['directional_accuracy_pct']:<6.2f}")


def main() -> None:
    args = parse_args()
    if args.command == "snapshot":
        snapshot(args.out)
    else:
        compare(args)


if __name__ == "__main__":
    main()
