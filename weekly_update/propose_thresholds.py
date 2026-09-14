#!/usr/bin/env python3
"""
Step 7 of the weekly Stock-Model refresh: apply GRADIENT_BUY_THRESHOLD and
GRADIENT_SELL_THRESHOLD to futures_trader.py.

This script:
  - parses gradient_value/threshold_optimization.txt (written by
    optimize_thresholds.py) for the validation-selected buy/sell pair,
  - reads the values currently hardcoded in futures_trader.py,
  - sanity-checks the proposal,
  - reads this run's held-out backtest from backtest_history.json (written by
    compare_backtest.py, step 6b) and refuses a proposal that loses money
    there, or earns less there than the thresholds already in futures_trader.py,
  - backs up futures_trader.py,
  - if the proposal passes every check, writes the new values directly into
    futures_trader.py (if it fails, the file is left untouched),
  - writes a markdown report describing what was (or was not) changed.

It never restarts the running trader process — restart it yourself for a
threshold change to take effect. It never connects to TWS and never places an
order.

Exit codes:
    0  applied (or no change was needed) — see the report for which
    2  could not parse / results file is stale (nothing written)
    3  proposal FAILED a check — futures_trader.py was NOT modified
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

# Held-out gate. Step 6b scores the proposed and the current thresholds on the
# same untouched half of the test window. A proposal is refused if its net P&L
# there is below this floor, or below the net P&L of the thresholds it replaces.
MIN_HELD_OUT_NET_PNL = 0.0  # dollars

# Thresholds reach the backtest history as floats parsed from text, so compare
# them at the precision they are written with rather than exactly.
THRESHOLD_MATCH_DECIMALS = 4

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

# Same two constants, but capturing the prefix (indent/name/spacing) and
# trailing comment so a rewrite can swap only the number and leave everything
# else — comments included — exactly as it was.
BUY_LINE_RE = re.compile(
    r"^(?P<prefix>\s*GRADIENT_BUY_THRESHOLD\s*=\s*)(?P<value>-?[0-9.]+)(?P<suffix>.*)$",
    re.MULTILINE,
)
SELL_LINE_RE = re.compile(
    r"^(?P<prefix>\s*GRADIENT_SELL_THRESHOLD\s*=\s*)(?P<value>-?[0-9.]+)(?P<suffix>.*)$",
    re.MULTILINE,
)


def apply_thresholds(trader: Path, buy: float, sell: float) -> None:
    """Rewrite the two threshold constants in futures_trader.py in place."""
    text = trader.read_text()
    text, n_buy = BUY_LINE_RE.subn(lambda m: f"{m['prefix']}{buy:.2f}{m['suffix']}", text, count=1)
    text, n_sell = SELL_LINE_RE.subn(lambda m: f"{m['prefix']}{sell:.2f}{m['suffix']}", text, count=1)
    if n_buy != 1 or n_sell != 1:
        fail(
            f"expected exactly one GRADIENT_BUY_THRESHOLD and one "
            f"GRADIENT_SELL_THRESHOLD line to update, found buy={n_buy} sell={n_sell} "
            f"— refusing to write a partial or ambiguous edit"
        )
    trader.write_text(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True,
                   help="threshold_optimization.txt produced by optimize_thresholds.py")
    p.add_argument("--trader", type=Path, required=True,
                   help="futures_trader.py (read, backed up, and updated if sane)")
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--backup-dir", type=Path, required=True)
    p.add_argument("--backtest-history", type=Path, required=True,
                   help="backtest_history.json from compare_backtest.py (step 6b), holding "
                        "the held-out net P&L of the proposed and current thresholds")
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


def parse_current_thresholds(trader: Path) -> tuple[float | None, float | None]:
    """Read the thresholds currently hardcoded in futures_trader.py."""
    if not trader.exists():
        fail(f"futures_trader.py not found: {trader}")
    text = trader.read_text()

    buy_m, sell_m = CURRENT_BUY_RE.search(text), CURRENT_SELL_RE.search(text)
    buy = float(buy_m["value"]) if buy_m else None
    sell = float(sell_m["value"]) if sell_m else None
    return buy, sell


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


def thresholds_match(configuration: dict, buy: float, sell: float) -> bool:
    """True when a scored configuration holds exactly this buy/sell pair."""
    return (round(configuration["buy"], THRESHOLD_MATCH_DECIMALS) == round(buy, THRESHOLD_MATCH_DECIMALS)
            and round(configuration["sell"], THRESHOLD_MATCH_DECIMALS) == round(sell, THRESHOLD_MATCH_DECIMALS))


def load_held_out_scores(history_path: Path, run_id: str) -> dict[str, dict]:
    """Configurations step 6b scored for this run, keyed by role ("proposed", "incumbent").

    Empty when the history is missing, unreadable, or has no record for run_id,
    so held_out_check refuses: nothing has vouched for the proposal.
    """
    try:
        records = json.loads(history_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(records, list):
        return {}
    for record in reversed(records):
        if isinstance(record, dict) and record.get("run_id") == run_id:
            return {configuration["role"]: configuration
                    for configuration in record.get("configurations", [])}
    return {}


def held_out_check(buy: float, sell: float,
                   cur_buy: float | None, cur_sell: float | None,
                   scored: dict[str, dict]) -> list[str]:
    """Return held-out P&L problems. Empty list means the proposal earned its place."""
    proposed = scored.get("proposed")
    if proposed is None or not thresholds_match(proposed, buy, sell):
        return [f"no held-out backtest of the proposed {buy:.2f} / {sell:.2f} from step 6b for "
                f"this run — refusing to apply thresholds that were not scored"]

    problems: list[str] = []
    proposed_pnl = proposed["metrics"]["net_pnl"]
    if proposed_pnl < MIN_HELD_OUT_NET_PNL:
        problems.append(f"proposed {buy:.2f} / {sell:.2f} loses money on the held-out half "
                        f"(net P&L ${proposed_pnl:,.0f})")

    if cur_buy is None or cur_sell is None:
        return problems  # sanity_check already reports the unreadable current values
    incumbent = scored.get("incumbent")
    if incumbent is None or not thresholds_match(incumbent, cur_buy, cur_sell):
        problems.append(f"no held-out backtest of the current {cur_buy:.2f} / {cur_sell:.2f} from "
                        f"step 6b for this run — cannot confirm the proposal beats them")
    elif proposed_pnl < incumbent["metrics"]["net_pnl"]:
        problems.append(f"proposed {buy:.2f} / {sell:.2f} earns less on the held-out half than the "
                        f"current {cur_buy:.2f} / {cur_sell:.2f} (net P&L ${proposed_pnl:,.0f} vs "
                        f"${incumbent['metrics']['net_pnl']:,.0f})")
    return problems


def build_report(res: dict, cur_buy, cur_sell,
                 problems: list[str], applied: bool, scored: dict[str, dict],
                 backup_path: Path, results_path: Path, trader: Path) -> str:
    buy, sell = res["buy"], res["sell"]
    changed = (cur_buy != buy) or (cur_sell != sell)
    big_move = (cur_buy is not None and abs(buy - cur_buy) >= LARGE_MOVE) or \
               (cur_sell is not None and abs(sell - cur_sell) >= LARGE_MOVE)

    out = [
        f"# Threshold update — {datetime.now():%A %d %B %Y, %H:%M}",
        "",
    ]
    if applied:
        out += [f"**`futures_trader.py` was updated.** Backup: `{backup_path}`.",
                "Restart the trader for the change to take effect — this script never does.", ""]
    else:
        out += ["**`futures_trader.py` was NOT modified.**", ""]

    out += ["## Verdict", ""]

    if problems:
        out += ["> **NOT APPLIED — checks failed:**", ""]
        out += [f"> - {p}" for p in problems] + [""]
    elif not changed:
        out += ["No change needed — the optimizer picked the values already in the trader.", ""]
    elif big_move:
        out += [f"Applied. This was a large move (>= {LARGE_MOVE:.2f}) — worth a second look.", ""]
    else:
        out += ["Applied, and within normal range.", ""]

    out += [
        "## Values",
        "",
        "| Constant | Previous | New | Delta |",
        "|---|---|---|---|",
        f"| `GRADIENT_BUY_THRESHOLD` | {fmt(cur_buy)} | **{buy:.2f}** | {delta(cur_buy, buy)} |",
        f"| `GRADIENT_SELL_THRESHOLD` | {fmt(cur_sell)} | **{sell:.2f}** | {delta(cur_sell, sell)} |",
        "",
        f"Policy selected by the optimizer: `{res['policy']}`",
        "",
    ]
    out += render_held_out_gate(scored)

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


def render_held_out_gate(scored: dict[str, dict]) -> list[str]:
    """The held-out net P&L the gate decided on, so a pass is as explainable as a refusal."""
    rows = [(label, scored[role])
            for role, label in (("incumbent", "Current"), ("proposed", "Proposed"))
            if role in scored]
    if not rows:
        return []
    lines = [
        "## Held-out gate (step 6b backtest, same bars)",
        "",
        "| Thresholds | buy / sell | net P&L |",
        "|---|---|---|",
    ]
    lines += [f"| {label} | {configuration['buy']:.2f} / {configuration['sell']:.2f} | "
              f"${configuration['metrics']['net_pnl']:,.0f} |" for label, configuration in rows]
    lines += [
        "",
        f"_Refused when the proposal's net P&L is below ${MIN_HELD_OUT_NET_PNL:,.0f} "
        "or below the current thresholds'._",
        "",
    ]
    return lines


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
    cur_buy, cur_sell = parse_current_thresholds(args.trader)
    scored = load_held_out_scores(args.backtest_history, args.run_id)
    problems = (sanity_check(res["buy"], res["sell"], cur_buy, cur_sell)
                + held_out_check(res["buy"], res["sell"], cur_buy, cur_sell, scored))
    changed = (cur_buy != res["buy"]) or (cur_sell != res["sell"])

    backup_path = args.backup_dir / f"futures_trader_{args.run_id}.py"
    shutil.copy2(args.trader, backup_path)

    applied = False
    if not problems and changed:
        apply_thresholds(args.trader, res["buy"], res["sell"])
        applied = True

    report = build_report(res, cur_buy, cur_sell, problems, applied, scored,
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
        "previous": {"buy": cur_buy, "sell": cur_sell},
        "changed": changed,
        "held_out_net_pnl": {role: configuration["metrics"]["net_pnl"]
                             for role, configuration in scored.items()},
        "problems": problems,
        "applied": applied,
        "backup": str(backup_path),
    }, indent=2))

    print(f"{'Applied' if applied else 'Proposed'}: buy={res['buy']:.2f} sell={res['sell']:.2f} "
          f"(previous: buy={fmt(cur_buy)} sell={fmt(cur_sell)}, policy={res['policy']})")
    print(f"Report:   {latest}")
    print(f"Backup:   {backup_path}")

    if problems:
        for problem in problems:
            print(f"CHECK FAILED: {problem}")
        print("futures_trader.py was NOT modified — fix the underlying issue and rerun.")
        sys.exit(3)
    elif applied:
        print("futures_trader.py UPDATED. Restart the trader for the change to take effect.")
    else:
        print("No change needed — futures_trader.py already matches the optimizer's choice.")


if __name__ == "__main__":
    main()
