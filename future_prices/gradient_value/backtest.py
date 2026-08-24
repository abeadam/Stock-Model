"""Backtest the gradient_value direction classifier as an ES futures strategy.

Trading rule (long & short, decided each bar from p = P(up over next 5 bars)):
    p >= BUY_THRESHOLD   -> hold long  1 ES contract
    p <= SELL_THRESHOLD  -> hold short 1 ES contract
    otherwise            -> flat

The position chosen at bar t is held into bar t+1, so the signal never uses
information from the bar whose return it earns. Evaluated on the same held-out
last-20% test split used by train_predictor.py.

Thresholds default to the constants below; override them with --buy/--sell/--policy.

Outputs: backtest_results.txt and backtest_equity_curve.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).parent
INPUT_CSV = _HERE / "model_input.csv"
MODEL_PKL = _HERE / "predictor_model.pkl"
RESULTS_TXT = _HERE / "backtest_results.txt"
EQUITY_PNG = _HERE / "backtest_equity_curve.png"

# Strategy / market parameters. The thresholds here are only defaults for a
# standalone run -- pass --buy/--sell/--policy to override. They do NOT track
# optimize_thresholds.py: its current selection lives in threshold_optimization.txt,
# and the weekly pipeline reads that file rather than these constants, so treat a
# bare `python backtest.py` as "whatever is written below", not "what we trade".
# A "hold until opposite" policy trades far less than "flat" and is the only kind
# that survives realistic slippage.
BUY_THRESHOLD = 0.75
SELL_THRESHOLD = 0.20
POLICY = "hold"                # "hold" = hold until opposite signal; "flat" = exit in the neutral zone
POINT_VALUE = 50.0              # ES = $50 per index point
TICK_VALUE = 12.50              # ES: 1 tick = 0.25 pt = $12.50
COMMISSION_PER_CONTRACT = 2.50  # broker commission per contract per fill
SLIPPAGE_TICKS = 1.0            # adverse ticks per fill (entry and exit each pay this)
TEST_FRACTION = 0.20

# Total cost charged every time one contract is filled (a position change of N
# contracts incurs this N times). Entry and exit are separate fills, so a round
# trip pays it twice. One ES tick of slippage per fill is a realistic-to-
# conservative assumption for a market order on this very liquid contract.
COST_PER_CONTRACT = COMMISSION_PER_CONTRACT + SLIPPAGE_TICKS * TICK_VALUE

# ES regular-hours 5-second bars per year, for annualizing the Sharpe ratio.
_BARS_PER_YEAR = 4680 * 252


def load_test_split() -> tuple[np.ndarray, np.ndarray]:
    """Return (close_prices, up_probabilities) for the held-out test split.

    Replicates train_predictor.py's preprocessing exactly: dedup by timestamp,
    drop rows with missing features/target, then take the final TEST_FRACTION.
    """
    bundle = joblib.load(MODEL_PKL)
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    columns = ["DateTime", "Close", *feature_cols, "target_avg5_close_pct"]
    df = pd.read_csv(INPUT_CSV, usecols=columns)
    df = df.drop_duplicates(subset="DateTime", keep="first").reset_index(drop=True)

    target_present = df["target_avg5_close_pct"].notna()
    features_present = ~df[feature_cols].isna().any(axis=1)
    df = df[target_present & features_present].reset_index(drop=True)

    split = int(len(df) * (1 - TEST_FRACTION))
    test = df.iloc[split:]

    probabilities = model.predict_proba(test[feature_cols])[:, 1]
    return test["Close"].to_numpy(float), probabilities


def positions_from_probabilities(
    probabilities: np.ndarray,
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
) -> np.ndarray:
    """Flat policy: long above buy, short below sell, flat in the neutral zone."""
    positions = np.zeros(len(probabilities), dtype=int)
    positions[probabilities >= buy_threshold] = 1
    positions[probabilities <= sell_threshold] = -1
    return positions


def positions_hold_until_opposite(
    probabilities: np.ndarray,
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
) -> np.ndarray:
    """Hold policy: long above buy, short below sell, otherwise keep the current
    position. Only flips on the opposite signal, so it trades far less."""
    signal = np.where(probabilities >= buy_threshold, 1,
                      np.where(probabilities <= sell_threshold, -1, 0))
    held = pd.Series(signal, dtype=float).replace(0, np.nan).ffill().fillna(0)
    return held.to_numpy().astype(int)


def build_positions(
    probabilities: np.ndarray,
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
    policy: str = POLICY,
) -> np.ndarray:
    """Target positions under the chosen policy ('flat' or 'hold')."""
    if policy == "flat":
        return positions_from_probabilities(probabilities, buy_threshold, sell_threshold)
    if policy == "hold":
        return positions_hold_until_opposite(probabilities, buy_threshold, sell_threshold)
    raise ValueError(f"unknown policy {policy!r}")


def simulate(close: np.ndarray, positions: np.ndarray) -> dict:
    """Simulate the strategy bar by bar and return P&L arrays and a trade ledger."""
    # Position held over bar t earns the t -> t+1 price change (no lookahead).
    price_change = np.diff(close, append=close[-1])
    hold_pnl = positions * price_change * POINT_VALUE

    # A contract is traded whenever the target position changes (entry at bar 0
    # through the last transition); the final bar closes any open position to flat.
    position_change = np.diff(positions, prepend=0)
    contracts_traded = np.abs(position_change).astype(float)
    contracts_traded[-1] += abs(positions[-1])  # close out remaining position

    commission = contracts_traded * COMMISSION_PER_CONTRACT
    slippage = contracts_traded * SLIPPAGE_TICKS * TICK_VALUE
    cost = commission + slippage

    net_bar_pnl = hold_pnl - cost
    equity = np.cumsum(net_bar_pnl)
    return {
        "hold_pnl": hold_pnl,
        "cost": cost,
        "commission": commission,
        "slippage": slippage,
        "net_bar_pnl": net_bar_pnl,
        "equity": equity,
        "trades": _extract_trades(close, positions),
    }


def _extract_trades(close: np.ndarray, positions: np.ndarray) -> list[dict]:
    """Segment the position series into round-trip trades with gross P&L."""
    trades: list[dict] = []
    entry_index: int | None = None

    for index in range(len(positions)):
        previous = positions[index - 1] if index > 0 else 0
        if positions[index] != previous:
            if previous != 0 and entry_index is not None:
                trades.append(_close_trade(close, entry_index, index, previous))
            entry_index = index if positions[index] != 0 else None

    if entry_index is not None and positions[-1] != 0:
        trades.append(_close_trade(close, entry_index, len(positions) - 1, positions[-1]))
    return trades


def _close_trade(close: np.ndarray, entry: int, exit_: int, direction: int) -> dict:
    gross = direction * (close[exit_] - close[entry]) * POINT_VALUE
    return {"direction": direction, "entry": entry, "exit": exit_, "gross_pnl": gross}


def summarize(close: np.ndarray, positions: np.ndarray, result: dict) -> dict:
    """Compute headline performance metrics from a simulation result."""
    trades = result["trades"]
    trade_pnl = np.array([t["gross_pnl"] for t in trades]) if trades else np.array([])
    wins = trade_pnl[trade_pnl > 0]
    losses = trade_pnl[trade_pnl < 0]

    net_bar = result["net_bar_pnl"]
    sharpe = (
        net_bar.mean() / net_bar.std() * np.sqrt(_BARS_PER_YEAR)
        if net_bar.std() > 0 else 0.0
    )
    running_max = np.maximum.accumulate(result["equity"])
    max_drawdown = float((result["equity"] - running_max).min())

    # Directional accuracy over bars that were both in-market and actually moved
    # (a flat next bar carries no up/down information on 5-second ES data).
    next_move = np.sign(np.diff(close, append=close[-1]))
    scored = (positions != 0) & (next_move != 0)
    directional_accuracy = (
        float(np.mean(np.sign(positions[scored]) == next_move[scored]) * 100)
        if scored.any() else float("nan")
    )

    return {
        "n_bars": len(positions),
        "exposure_pct": float((positions != 0).mean() * 100),
        "n_trades": len(trades),
        "net_pnl": float(result["net_bar_pnl"].sum()),
        "gross_pnl": float(result["hold_pnl"].sum()),
        "total_cost": float(result["cost"].sum()),
        "commission": float(result["commission"].sum()),
        "slippage": float(result["slippage"].sum()),
        "buy_hold_pnl": float((close[-1] - close[0]) * POINT_VALUE),
        "win_rate_pct": float(len(wins) / len(trades) * 100) if trades else float("nan"),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) else float("inf"),
        "directional_accuracy_pct": directional_accuracy,
        "sharpe_annualized": float(sharpe),
        "max_drawdown": max_drawdown,
    }


def write_results(
    metrics: dict,
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
    policy: str = POLICY,
) -> None:
    lines = [
        "Gradient-value strategy backtest — ES futures",
        "=" * 60,
        f"Rule: p>={buy_threshold:.2f} long, p<={sell_threshold:.2f} short, "
        f"else {'hold' if policy == 'hold' else 'flat'} (long & short, '{policy}' policy)",
        f"Sizing: 1 contract | ${POINT_VALUE:.0f}/pt",
        f"Costs:  ${COMMISSION_PER_CONTRACT:.2f} commission + {SLIPPAGE_TICKS:.2f} tick "
        f"(${SLIPPAGE_TICKS * TICK_VALUE:.2f}) slippage per fill",
        f"Evaluated on held-out last {TEST_FRACTION*100:.0f}% test split",
        "",
        f"Bars:                  {metrics['n_bars']:,}",
        f"Exposure:              {metrics['exposure_pct']:.1f}% of bars in market",
        f"Trades:                {metrics['n_trades']:,}",
        f"Directional accuracy:  {metrics['directional_accuracy_pct']:.2f}% (active bars, next-bar)",
        "",
        f"Net P&L:               ${metrics['net_pnl']:,.2f}",
        f"  Gross P&L:           ${metrics['gross_pnl']:,.2f}",
        f"  Commission:          ${metrics['commission']:,.2f}",
        f"  Slippage:            ${metrics['slippage']:,.2f}",
        f"Buy & hold (1 long):   ${metrics['buy_hold_pnl']:,.2f}",
        "",
        f"Win rate:              {metrics['win_rate_pct']:.2f}%",
        f"Avg win / loss:        ${metrics['avg_win']:,.2f} / ${metrics['avg_loss']:,.2f}",
        f"Profit factor:         {metrics['profit_factor']:.3f}",
        f"Sharpe (annualized):   {metrics['sharpe_annualized']:.2f}",
        f"Max drawdown:          ${metrics['max_drawdown']:,.2f}",
    ]
    RESULTS_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nResults saved to {RESULTS_TXT}")


def plot_equity(equity: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity, color="#2E86AB", lw=1.2)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_title("Gradient-value strategy — cumulative net P&L (test set)", fontweight="bold")
    ax.set_xlabel("Bar (5s) in test set")
    ax.set_ylabel("Cumulative net P&L ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(EQUITY_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Equity curve saved to {EQUITY_PNG}")


def parse_args() -> argparse.Namespace:
    """Defaults are the module constants, so a bare run behaves exactly as before."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buy", type=float, default=BUY_THRESHOLD,
                        help=f"go long when p >= this (default {BUY_THRESHOLD})")
    parser.add_argument("--sell", type=float, default=SELL_THRESHOLD,
                        help=f"go short when p <= this (default {SELL_THRESHOLD})")
    parser.add_argument("--policy", choices=("hold", "flat"), default=POLICY,
                        help=f"what to do in the neutral zone (default {POLICY})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Thresholds: buy>={args.buy:.2f}  sell<={args.sell:.2f}  policy={args.policy}")
    print("Loading model and test split ...")
    close, probabilities = load_test_split()
    positions = build_positions(probabilities, args.buy, args.sell, args.policy)
    result = simulate(close, positions)
    metrics = summarize(close, positions, result)
    print()
    write_results(metrics, args.buy, args.sell, args.policy)
    plot_equity(result["equity"])


if __name__ == "__main__":
    main()
