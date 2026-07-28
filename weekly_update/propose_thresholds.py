#!/usr/bin/env python3
"""
Step 7 of the weekly Stock-Model refresh: recommend GRADIENT_BUY_THRESHOLD and
GRADIENT_SELL_THRESHOLD for futures_trader.py.

This script is deliberately read-only with respect to the trading code. It:
  - parses gradient_value/threshold_optimization.txt (written by
    optimize_thresholds.py) for the validation-selected buy/sell pair,
  - reads the values currently hardcoded in futures_trader.py,
  - sanity-checks the proposal,
  - backs up futures_trader.py,
  - writes a markdown recommendation for a human to apply.

It never edits futures_trader.py, never connects to TWS, and never places an order.

Exit codes:
    0  recommendation written, values look sane
    2  could not parse / results file is stale (nothing written)
    3  recommendation written but FAILED sanity checks — do not apply blindly
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Guard rails for a probability-threshold pair. A gradient buy threshold below
# 0.50 would mean "go long when the model leans short", which is never intended.
BUY_RANGE = (0.50, 0.95)
SELL_RANGE = (0.05, 0.50)
LARGE_MOVE = 0.15  # absolute change worth calling out explicitly

CHOSEN_RE = re.compile(
    r"Chosen:\s*policy=(?P<policy>\S+)\s+buy=(?P<buy>[0-9.]+)\s+sell=(?P<sell>[0-9.]+)"
)
TABLE_ROW_RE = re.compile(
    r"^\s*(?P<policy>\S+)\s+(?P<buy>[0-9.]+)\s+(?P<sell>[0-9.]+)\s+"
    r"\$\s*(?P<pnl>-?[\d,]+)\s+(?P<trades>[\d,]+)\s*$"
)
CURRENT_BUY_RE = re.compile(
    r"^\s*GRADIENT_BUY_THRESHOLD\s*=\s*(?P<value>-?[0-9.]+)", re.MULTILINE
)
CURRENT_SELL_RE = re.compile(
    r"^\s*GRADIENT_SELL_THRESHOLD\s*=\s*(?P<value>-?[0-9.]+)", re.MULTILINE
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True,
                   help="threshold_optimization.txt produced by optimize_thresholds.py")
    p.add_argument("--trader", type=Path, required=True,
                   help="futures_trader.py (read + backed up, never written)")
    p.add_argument("--tester-log", type=Path, default=None,
                   help="model_tester.py step log, for the RL P&L summary")
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--backup-dir", type=Path, required=True)
    p.add_argument("--min-mtime", type=float, default=0.0,
                   help="results file must be newer than this epoch (staleness guard)")
    p.add_argument("--run-id", default=datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    return p.parse_args()


def fail(message: str) -> None:
    """Print in the format the driver script scans for, then exit."""
    print(f"Error: {message}", file=sys.stderr)
    print(f"Error: {message}")
    sys.exit(2)


def parse_results(path: Path, min_mtime: float) -> dict:
    """Extract the chosen thresholds, the ranked grid, and the held-out metrics."""
    if not path.exists():
        fail(f"results file not found: {path} — did optimize_thresholds.py run?")

    if path.stat().st_mtime < min_mtime:
        written = datetime.fromtimestamp(path.stat().st_mtime)
        fail(
            f"results file {path.name} is stale (written {written:%Y-%m-%d %H:%M}, "
            "before this run started). Refusing to recommend thresholds from an "
            "older optimization."
        )

    text = path.read_text()

    chosen = CHOSEN_RE.search(text)
    if not chosen:
        fail(f"no 'Chosen: policy=... buy=... sell=...' line in {path}")

    ranked = []
    in_table = False
    for line in text.splitlines():
        if line.strip().startswith("policy") and "val_net_pnl" in line:
            in_table = True
            continue
        if in_table:
            row = TABLE_ROW_RE.match(line)
            if row:
                ranked.append({
                    "policy": row["policy"],
                    "buy": float(row["buy"]),
                    "sell": float(row["sell"]),
                    "val_net_pnl": float(row["pnl"].replace(",", "")),
                    "val_trades": int(row["trades"].replace(",", "")),
                })
            elif ranked:
                in_table = False

    metrics = {}
    for line in text.splitlines():
        if ":" in line and line.startswith("  ") and "=" not in line:
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key and value and not key.startswith("policy"):
                metrics[key] = value

    return {
        "policy": chosen["policy"],
        "buy": float(chosen["buy"]),
        "sell": float(chosen["sell"]),
        "ranked": ranked,
        "metrics": metrics,
    }


def parse_current_thresholds(trader: Path) -> tuple[float | None, float | None, int | None, int | None]:
    """Read the thresholds hardcoded in futures_trader.py, with line numbers."""
    if not trader.exists():
        fail(f"futures_trader.py not found: {trader}")
    text = trader.read_text()

    buy_m, sell_m = CURRENT_BUY_RE.search(text), CURRENT_SELL_RE.search(text)
    buy = float(buy_m["value"]) if buy_m else None
    sell = float(sell_m["value"]) if sell_m else None
    buy_line = text[: buy_m.start()].count("\n") + 1 if buy_m else None
    sell_line = text[: sell_m.start()].count("\n") + 1 if sell_m else None
    return buy, sell, buy_line, sell_line


def sanity_check(buy: float, sell: float,
                 cur_buy: float | None, cur_sell: float | None) -> list[str]:
    """Return a list of problems. Empty list means the proposal looks reasonable."""
    problems: list[str] = []
    if not BUY_RANGE[0] <= buy <= BUY_RANGE[1]:
        problems.append(f"proposed buy {buy:.2f} outside expected range {BUY_RANGE}")
    if not SELL_RANGE[0] <= sell <= SELL_RANGE[1]:
        problems.append(f"proposed sell {sell:.2f} outside expected range {SELL_RANGE}")
    if buy <= sell:
        problems.append(f"proposed buy {buy:.2f} <= sell {sell:.2f} — bands overlap")
    if cur_buy is None:
        problems.append("could not read current GRADIENT_BUY_THRESHOLD from futures_trader.py")
    if cur_sell is None:
        problems.append("could not read current GRADIENT_SELL_THRESHOLD from futures_trader.py")
    return problems


def extract_tester_summary(log_path: Path | None) -> list[str]:
    """Pull the closing summary lines out of the model_tester.py step log."""
    if not log_path or not log_path.exists():
        return []
    lines = log_path.read_text().splitlines()
    keep = [ln for ln in lines
            if ln.startswith(("Total days tested:", "Starting capital",
                              "Final portfolio value:", "Total return:"))]
    return keep[-6:]


def build_report(res: dict, cur_buy, cur_sell, buy_line, sell_line,
                 problems: list[str], tester_lines: list[str],
                 backup_path: Path, results_path: Path, trader: Path) -> str:
    buy, sell = res["buy"], res["sell"]
    changed = (cur_buy != buy) or (cur_sell != sell)
    big_move = (cur_buy is not None and abs(buy - cur_buy) >= LARGE_MOVE) or \
               (cur_sell is not None and abs(sell - cur_sell) >= LARGE_MOVE)

    out = [
        f"# Threshold recommendation — {datetime.now():%A %d %B %Y, %H:%M}",
        "",
        "**Nothing has been changed.** `futures_trader.py` is untouched; this is a",
        "proposal for you to review and apply by hand.",
        "",
        "## Verdict",
        "",
    ]

    if problems:
        out += ["> **DO NOT APPLY WITHOUT CHECKING.** Sanity checks failed:", ""]
        out += [f"> - {p}" for p in problems] + [""]
    elif not changed:
        out += ["No change needed — the optimizer picked the values already in the trader.", ""]
    elif big_move:
        out += [f"Change proposed, and it is a large move (>= {LARGE_MOVE:.2f}). Worth a second look.", ""]
    else:
        out += ["Change proposed and within normal range.", ""]

    out += [
        "## Values",
        "",
        "| Constant | Current | Proposed | Delta |",
        "|---|---|---|---|",
        f"| `GRADIENT_BUY_THRESHOLD` | {fmt(cur_buy)} | **{buy:.2f}** | {delta(cur_buy, buy)} |",
        f"| `GRADIENT_SELL_THRESHOLD` | {fmt(cur_sell)} | **{sell:.2f}** | {delta(cur_sell, sell)} |",
        "",
        f"Policy selected by the optimizer: `{res['policy']}`",
        "",
    ]

    if changed and not problems:
        out += [
            "## To apply",
            "",
            f"Edit `{trader}`"
            + (f" (lines {buy_line} and {sell_line})" if buy_line and sell_line else "")
            + ":",
            "",
            "```python",
            f"GRADIENT_BUY_THRESHOLD  =  {buy:.2f}  # BUY  when prob(up) > this",
            f"GRADIENT_SELL_THRESHOLD =  {sell:.2f}  # SELL when prob(up) < this",
            "```",
            "",
            "The running trader loads these at startup — restart it for the change to take effect.",
            f"Backup of the current file: `{backup_path}`",
            "",
        ]

    if res["ranked"]:
        out += [
            "## Top configurations on validation",
            "",
            "| policy | buy | sell | val net P&L | val trades |",
            "|---|---|---|---|---|",
        ]
        out += [f"| {r['policy']} | {r['buy']:.2f} | {r['sell']:.2f} | "
                f"${r['val_net_pnl']:,.0f} | {r['val_trades']:,} |" for r in res["ranked"]]
        out += [""]

    if res["metrics"]:
        out += ["## Held-out test half (chosen thresholds)", ""]
        out += [f"- **{k}**: {v}" for k, v in res["metrics"].items()] + [""]

    if tester_lines:
        out += ["## RL agent check (model_tester.py)", "", "```"] + tester_lines + ["```", ""]

    out += [
        "## Provenance",
        "",
        f"- Optimizer output: `{results_path}` "
        f"(written {datetime.fromtimestamp(results_path.stat().st_mtime):%Y-%m-%d %H:%M})",
        f"- Trader read from: `{trader}`",
        "",
        "_Backtested thresholds describe past data. Validation P&L is not a forecast._",
        "",
    ]
    return "\n".join(out)


def fmt(value: float | None) -> str:
    return "unreadable" if value is None else f"{value:.2f}"


def delta(current: float | None, proposed: float) -> str:
    if current is None:
        return "—"
    diff = proposed - current
    return "no change" if abs(diff) < 1e-9 else f"{diff:+.2f}"


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.backup_dir.mkdir(parents=True, exist_ok=True)

    res = parse_results(args.results, args.min_mtime)
    cur_buy, cur_sell, buy_line, sell_line = parse_current_thresholds(args.trader)
    problems = sanity_check(res["buy"], res["sell"], cur_buy, cur_sell)

    backup_path = args.backup_dir / f"futures_trader_{args.run_id}.py"
    shutil.copy2(args.trader, backup_path)

    report = build_report(res, cur_buy, cur_sell, buy_line, sell_line, problems,
                          extract_tester_summary(args.tester_log),
                          backup_path, args.results, args.trader)

    dated = args.report_dir / f"thresholds_{args.run_id}.md"
    latest = args.report_dir / "latest_recommendation.md"
    dated.write_text(report)
    latest.write_text(report)

    (args.report_dir / "latest_recommendation.json").write_text(json.dumps({
        "run_id": args.run_id,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "policy": res["policy"],
        "proposed": {"buy": res["buy"], "sell": res["sell"]},
        "current": {"buy": cur_buy, "sell": cur_sell},
        "changed": (cur_buy != res["buy"]) or (cur_sell != res["sell"]),
        "problems": problems,
        "applied": False,
        "backup": str(backup_path),
    }, indent=2))

    print(f"Proposed: buy={res['buy']:.2f} sell={res['sell']:.2f} "
          f"(current: buy={fmt(cur_buy)} sell={fmt(cur_sell)}, policy={res['policy']})")
    print(f"Report:   {latest}")
    print(f"Backup:   {backup_path}")
    print("futures_trader.py was NOT modified.")

    if problems:
        for problem in problems:
            print(f"SANITY CHECK FAILED: {problem}")
        sys.exit(3)


if __name__ == "__main__":
    main()
