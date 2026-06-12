"""Optimize the gradient strategy's entry/exit thresholds on a validation split.

The backtest showed the 0.65/0.35 rule churns far too much to survive realistic
slippage. This grids the buy/sell probability thresholds (and two position
policies) on the first half of the held-out test window, picks the configuration
with the best net-of-cost P&L, and reports it on the untouched second half.

Two policies are compared because they trade very differently under slippage:
  * "flat" — long above buy, short below sell, flat in between (the original rule).
  * "hold" — long above buy, short below sell, otherwise HOLD the current
    position. Only flips on the opposite signal, so it trades far less.

Outputs: threshold_optimization.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from backtest import (
    COST_PER_CONTRACT,
    POINT_VALUE,
    build_positions,
    load_test_split,
    simulate,
    summarize,
)

_HERE = Path(__file__).parent
RESULTS_TXT = _HERE / "threshold_optimization.txt"

VALIDATION_FRACTION = 0.5  # first half of the test window tunes; second half reports
BUY_THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
SELL_THRESHOLDS = (0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10)
POLICIES = ("flat", "hold")


def net_pnl(close: np.ndarray, positions: np.ndarray) -> tuple[float, int]:
    """Fast net-of-cost P&L and trade count, skipping the full trade ledger."""
    price_change = np.diff(close, append=close[-1])
    hold_pnl = positions * price_change * POINT_VALUE
    position_change = np.diff(positions, prepend=0)
    contracts_traded = np.abs(position_change).astype(float)
    contracts_traded[-1] += abs(positions[-1])
    pnl = float((hold_pnl - contracts_traded * COST_PER_CONTRACT).sum())
    n_trades = int((position_change != 0).sum())
    return pnl, n_trades


def search(close_val: np.ndarray, prob_val: np.ndarray) -> list[dict]:
    """Evaluate every (policy, buy, sell) on the validation window."""
    rows = []
    for policy in POLICIES:
        for buy in BUY_THRESHOLDS:
            for sell in SELL_THRESHOLDS:
                positions = build_positions(prob_val, buy, sell, policy)
                pnl, n_trades = net_pnl(close_val, positions)
                rows.append({"policy": policy, "buy": buy, "sell": sell,
                             "val_net_pnl": pnl, "val_trades": n_trades})
    return sorted(rows, key=lambda r: r["val_net_pnl"], reverse=True)


def main() -> None:
    print("Loading model and test split ...")
    close, probabilities = load_test_split()

    split = int(len(close) * VALIDATION_FRACTION)
    close_val, prob_val = close[:split], probabilities[:split]
    close_test, prob_test = close[split:], probabilities[split:]
    print(f"Validation bars: {len(close_val):,}  |  Test bars: {len(close_test):,}")

    ranked = search(close_val, prob_val)
    best = ranked[0]
    print(f"\nBest on validation: policy={best['policy']} buy={best['buy']} sell={best['sell']} "
          f"net=${best['val_net_pnl']:,.0f} trades={best['val_trades']:,}")

    # Report the chosen configuration on the untouched test half.
    test_positions = build_positions(prob_test, best["buy"], best["sell"], best["policy"])
    metrics = summarize(close_test, test_positions, simulate(close_test, test_positions))

    lines = [
        "Threshold optimization — gradient ES strategy",
        "=" * 60,
        f"Validation = first {VALIDATION_FRACTION*100:.0f}% of test window; reported on the rest.",
        f"Cost per fill: ${COST_PER_CONTRACT:.2f} (commission + slippage)",
        "",
        "Top 8 configurations by validation net P&L:",
        f"  {'policy':>5}  {'buy':>5}  {'sell':>5}  {'val_net_pnl':>14}  {'val_trades':>11}",
    ]
    for row in ranked[:8]:
        lines.append(f"  {row['policy']:>5}  {row['buy']:>5.2f}  {row['sell']:>5.2f}  "
                     f"${row['val_net_pnl']:>12,.0f}  {row['val_trades']:>11,}")
    lines += [
        "",
        f"Chosen: policy={best['policy']}  buy={best['buy']:.2f}  sell={best['sell']:.2f}",
        "",
        "Held-out TEST half with chosen thresholds:",
        f"  Bars:                 {metrics['n_bars']:,}",
        f"  Exposure:             {metrics['exposure_pct']:.1f}%",
        f"  Trades:               {metrics['n_trades']:,}",
        f"  Directional accuracy: {metrics['directional_accuracy_pct']:.2f}%",
        f"  Net P&L:              ${metrics['net_pnl']:,.2f}",
        f"    Gross P&L:          ${metrics['gross_pnl']:,.2f}",
        f"    Commission:         ${metrics['commission']:,.2f}",
        f"    Slippage:           ${metrics['slippage']:,.2f}",
        f"  Buy & hold:           ${metrics['buy_hold_pnl']:,.2f}",
        f"  Win rate:             {metrics['win_rate_pct']:.2f}%",
        f"  Profit factor:        {metrics['profit_factor']:.3f}",
        f"  Sharpe (annualized):  {metrics['sharpe_annualized']:.2f}",
        f"  Max drawdown:         ${metrics['max_drawdown']:,.2f}",
    ]
    RESULTS_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved to {RESULTS_TXT}")


if __name__ == "__main__":
    main()
