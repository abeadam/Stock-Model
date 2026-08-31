#!/usr/bin/env python3
"""
Step 6b of the weekly Stock-Model refresh: backtest the thresholds currently in
futures_trader.py against the ones optimize_thresholds.py just chose.

Both are scored on the same freshly rebuilt data, so the report answers the only
question that matters at this point in the pipeline: what does switching cost or
buy? Without it, threshold_optimization.txt reports the proposal alone and there
is nothing to compare it against.

This must run BEFORE step 7 (propose_thresholds.py), which overwrites the
incumbent values in futures_trader.py.

Scoring uses the untouched second half of the test window — the same half
optimize_thresholds.py reports on, and the only span neither configuration was
tuned on. The incumbent is scored under the proposed policy so the two differ
only in their thresholds.

Every run's scores are appended to backtest_history.json, and the report carries
a second table comparing the incumbent's numbers now against the last run that
scored those same thresholds. A configuration that looked strong when it was
selected and is weak a week later was fitting noise, which no single run's
report can reveal.

Reads nothing from TWS, places no orders, and modifies no source file. It only
writes its report.

Exit codes:
    0  comparison written
    2  could not parse the results file or futures_trader.py (nothing written)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

# propose_thresholds.py sits alongside this file and already knows how to read
# both threshold sources; importing it keeps a single parser for each.
sys.path.insert(0, str(Path(__file__).parent))

from propose_thresholds import (  # noqa: E402
    fail,
    parse_current_thresholds,
    parse_results,
)

# Rows of the comparison table: (display name, metrics key, number format).
# Bar count is deliberately absent — it is identical across configurations.
METRIC_ROWS = [
    ("Trades", "n_trades", "{:,.0f}"),
    ("Directional accuracy %", "directional_accuracy_pct", "{:.2f}"),
    ("Gross P&L", "gross_pnl", "${:,.0f}"),
    ("Commission", "commission", "${:,.0f}"),
    ("Slippage", "slippage", "${:,.0f}"),
    ("Net P&L", "net_pnl", "${:,.0f}"),
    ("Win rate %", "win_rate_pct", "{:.2f}"),
    ("Profit factor", "profit_factor", "{:.3f}"),
    ("Sharpe (annualized)", "sharpe_annualized", "{:.2f}"),
    ("Max drawdown", "max_drawdown", "${:,.0f}"),
]


# Every run appends one record here so later runs can detect decay.
HISTORY_FILENAME = "backtest_history.json"


class ReportColumn(NamedTuple):
    """One configuration's column in the comparison table."""
    label: str
    buy: float
    sell: float
    metrics: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="gradient_value/threshold_optimization.txt")
    parser.add_argument("--trader", type=Path, required=True,
                        help="futures_trader.py, read for the incumbent thresholds")
    parser.add_argument("--gradient-dir", type=Path, required=True,
                        help="gradient_value/, providing backtest.py and the model")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--min-mtime", type=float, default=0.0,
                        help="reject a results file written before this epoch time")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    return parser.parse_args()


def load_untouched_half(backtest, validation_fraction: float):
    """Return (close, probabilities) for the half neither configuration was tuned on."""
    close, probabilities = backtest.load_test_split()
    split = int(len(close) * validation_fraction)
    return close[split:], probabilities[split:]


def score(backtest, close, probabilities, buy: float, sell: float, policy: str) -> dict:
    positions = backtest.build_positions(probabilities, buy, sell, policy)
    return backtest.summarize(close, positions, backtest.simulate(close, positions))


def format_delta(incumbent_value: float, proposed_value: float, number_format: str) -> str:
    difference = proposed_value - incumbent_value
    if difference == 0:
        return "—"
    return ("+" if difference > 0 else "-") + number_format.format(abs(difference))


def load_history(report_dir: Path) -> list[dict]:
    """Past runs, oldest first. A missing or unreadable file just means no history."""
    history_path = report_dir / HISTORY_FILENAME
    if not history_path.exists():
        return []
    try:
        records = json.loads(history_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        print(f"Warning: ignoring unreadable {HISTORY_FILENAME} ({error})")
        return []
    return records if isinstance(records, list) else []


def append_history(report_dir: Path, record: dict, history: list[dict]) -> None:
    (report_dir / HISTORY_FILENAME).write_text(json.dumps(history + [record], indent=2))


def matches(configuration: dict, buy: float, sell: float) -> bool:
    """Thresholds come from text on both sides, so compare at the precision shown."""
    return (round(configuration["buy"], 4) == round(buy, 4)
            and round(configuration["sell"], 4) == round(sell, 4))


def find_previous_scoring(history: list[dict], buy: float, sell: float) -> dict | None:
    """The most recent earlier run that scored these exact thresholds, if any."""
    for record in reversed(history):
        for configuration in record.get("configurations", []):
            if matches(configuration, buy, sell):
                return {
                    "run_id": record["run_id"],
                    "n_bars": record["n_bars"],
                    "role": configuration["role"],
                    "metrics": configuration["metrics"],
                }
    return None


def render_week_over_week(previous: dict, current: dict, buy: float, sell: float,
                          current_bars: int, current_run_id: str) -> list[str]:
    """Same thresholds, previous run's data vs this run's."""
    lines = [
        "",
        f"## Same thresholds ({buy:.2f} / {sell:.2f}), run over run",
        "",
        f"Scored as the **{previous['role']}** in run `{previous['run_id']}` "
        f"({previous['n_bars']:,} bars), and as the incumbent now "
        f"({current_bars:,} bars).",
        "",
        f"| Metric | {previous['run_id']} | {current_run_id} | Delta |",
        "|---|---|---|---|",
    ]
    for display_name, key, number_format in METRIC_ROWS:
        before, after = previous["metrics"].get(key), current[key]
        if before is None:
            continue
        lines.append(
            f"| {display_name} | {number_format.format(before)} | "
            f"{number_format.format(after)} | {format_delta(before, after, number_format)} |"
        )
    lines += [
        "",
        "_The model was retrained and the test window moved between these runs, so "
        "this is not a controlled comparison. It is still the clearest signal "
        "available that a selection is decaying rather than holding up._",
    ]
    return lines


def choose_columns(
    incumbent: dict | None,
    proposed: dict,
    incumbent_thresholds: tuple[float | None, float | None],
    proposed_thresholds: tuple[float, float],
) -> tuple[list[ReportColumn], str]:
    """Pick the table's columns, plus a note explaining any missing comparison."""
    incumbent_buy, incumbent_sell = incumbent_thresholds
    proposed_buy, proposed_sell = proposed_thresholds

    if incumbent is None:
        return (
            [ReportColumn("Proposed", proposed_buy, proposed_sell, proposed)],
            "> **Could not read the incumbent thresholds from `futures_trader.py`.**\n"
            "> Only the proposed configuration is reported below.",
        )

    if (incumbent_buy, incumbent_sell) == (proposed_buy, proposed_sell):
        return (
            [ReportColumn("Current", proposed_buy, proposed_sell, proposed)],
            "> The optimizer chose the thresholds already in `futures_trader.py`. "
            "**No change proposed.**",
        )

    return (
        [
            ReportColumn("Incumbent", incumbent_buy, incumbent_sell, incumbent),
            ReportColumn("Proposed", proposed_buy, proposed_sell, proposed),
        ],
        "",
    )


def render_table(columns: list[ReportColumn]) -> list[str]:
    """Render the comparison as markdown rows, with a delta column when comparing."""
    comparing = len(columns) == 2
    trailing = " | Delta |" if comparing else " |"

    rows = [
        "| Metric | " + " | ".join(column.label for column in columns) + trailing,
        "|---|" + "---|" * (len(columns) + comparing),
        "| **buy / sell** | "
        + " | ".join(f"{column.buy:.2f} / {column.sell:.2f}" for column in columns)
        + (" | |" if comparing else " |"),
    ]

    for display_name, key, number_format in METRIC_ROWS:
        cells = " | ".join(number_format.format(column.metrics[key]) for column in columns)
        row = f"| {display_name} | {cells}"
        if comparing:
            row += " | " + format_delta(
                columns[0].metrics[key], columns[1].metrics[key], number_format
            )
        rows.append(row + " |")
    return rows


def build_report(
    columns: list[ReportColumn],
    note: str,
    policy: str,
    n_bars: int,
    run_id: str,
    week_over_week: list[str] | None = None,
) -> str:
    lines = [
        f"# Backtest — incumbent vs proposed thresholds ({run_id})",
        "",
        f"Scored on the untouched second half of the test window: **{n_bars:,} bars**, "
        f"policy `{policy}` for both.",
        "",
    ]
    if note:
        lines += [note, ""]
    lines += render_table(columns)
    lines += week_over_week or [
        "",
        "_No earlier run has scored these thresholds yet, so there is no run-over-run "
        "comparison. The next run will have one._",
    ]
    lines += [
        "",
        "_Backtested thresholds describe past data. This is not a forecast, and it "
        "assumes the slippage in `backtest.py` is correct._",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    sys.path.insert(0, str(args.gradient_dir))
    try:
        import backtest
        from optimize_thresholds import VALIDATION_FRACTION
    except ImportError as error:
        fail(f"cannot import backtest.py from {args.gradient_dir}: {error}")

    results = parse_results(args.results, args.min_mtime)
    proposed_buy, proposed_sell, policy = results["buy"], results["sell"], results["policy"]
    incumbent_buy, incumbent_sell = parse_current_thresholds(args.trader)

    print(f"Incumbent (futures_trader.py): buy={incumbent_buy} sell={incumbent_sell}")
    print(f"Proposed  (optimizer):         buy={proposed_buy} sell={proposed_sell} policy={policy}")
    print("Loading model and test split ...", flush=True)

    close, probabilities = load_untouched_half(backtest, VALIDATION_FRACTION)
    print(f"Untouched half: {len(close):,} bars\n", flush=True)

    proposed = score(backtest, close, probabilities, proposed_buy, proposed_sell, policy)
    incumbent = (
        score(backtest, close, probabilities, incumbent_buy, incumbent_sell, policy)
        if incumbent_buy is not None and incumbent_sell is not None
        else None
    )

    columns, note = choose_columns(
        incumbent, proposed,
        (incumbent_buy, incumbent_sell), (proposed_buy, proposed_sell),
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    history = load_history(args.report_dir)

    # Track the incumbent: it is normally the previous run's proposal, so this is
    # where a selection that decayed since it was chosen becomes visible.
    week_over_week = None
    if incumbent is not None:
        previous = find_previous_scoring(history, incumbent_buy, incumbent_sell)
        if previous is not None:
            week_over_week = render_week_over_week(
                previous, incumbent, incumbent_buy, incumbent_sell,
                len(close), args.run_id,
            )

    report = build_report(columns, note, policy, len(close), args.run_id, week_over_week)

    (args.report_dir / f"backtest_comparison_{args.run_id}.md").write_text(report)
    (args.report_dir / "latest_backtest_comparison.md").write_text(report)

    # Proposed first, so a later run matching both reports the more meaningful role.
    configurations = [
        {"role": "proposed", "buy": proposed_buy, "sell": proposed_sell,
         "policy": policy, "metrics": proposed},
    ]
    if incumbent is not None:
        configurations.append(
            {"role": "incumbent", "buy": incumbent_buy, "sell": incumbent_sell,
             "policy": policy, "metrics": incumbent}
        )
    append_history(args.report_dir, {
        "run_id": args.run_id,
        "n_bars": len(close),
        "configurations": configurations,
    }, history)

    print(report)
    print(f"Saved to {args.report_dir / 'latest_backtest_comparison.md'}")


if __name__ == "__main__":
    main()
