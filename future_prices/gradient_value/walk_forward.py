"""Anchored walk-forward evaluation of the gradient ES strategy.

Confirms the edge is not specific to the single last-20% test window. The
timeline is split into an initial training block followed by N sequential test
folds. For each fold:

  1. train the classifier on every bar before the fold (expanding window),
     early-stopping on a validation tail of that training block;
  2. select the position policy and thresholds on that same validation tail
     (never on the test fold);
  3. trade the fold out-of-sample and record P&L net of realistic slippage.

Per-fold results plus the pooled out-of-sample equity curve show whether the
strategy is consistently profitable across different market periods.

Caveat: the pred_CloseToHigh/Low features come from basic models trained on the
full history, so they carry mild lookahead for the earliest folds; the other 36
features are causal. Retraining the 22GB basic-model pipeline per fold is
infeasible, so this evaluates temporal robustness at the gradient layer.

Outputs: walk_forward_results.txt, walk_forward_equity.png
"""

from __future__ import annotations

from pathlib import Path

import joblib  # noqa: F401  (kept for parity with sibling scripts)
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backtest import COST_PER_CONTRACT, POINT_VALUE, build_positions, simulate, summarize
from optimize_thresholds import BUY_THRESHOLDS, POLICIES, SELL_THRESHOLDS, net_pnl
from train_predictor import TARGET_COL, _BEST_PLUS_ASI as FEATURES

_HERE = Path(__file__).parent
INPUT_CSV = _HERE / "model_input.csv"
RESULTS_TXT = _HERE / "walk_forward_results.txt"
EQUITY_PNG = _HERE / "walk_forward_equity.png"

INITIAL_TRAIN_FRACTION = 0.40
N_FOLDS = 6
VALIDATION_TAIL_FRACTION = 0.15   # of each fold's training block; tunes thresholds + early stopping
MAX_TREES = 2000
EARLY_STOPPING_ROUNDS = 50

# Best configuration from tune_predictor.py, reused for every fold (we test
# temporal robustness of the chosen model, not per-fold re-tuning).
TUNED_PARAMS = dict(
    objective="binary", num_leaves=511, max_depth=-1, learning_rate=0.05,
    min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.6,
    lambda_l1=0.0, lambda_l2=0.05, n_jobs=-1, verbose=-1, random_state=42,
)


def load_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Chronological feature matrix, labels, close prices, and timestamps."""
    columns = ["DateTime", "Close", *FEATURES, TARGET_COL]
    df = pd.read_csv(INPUT_CSV, usecols=columns)
    df = df.drop_duplicates(subset="DateTime", keep="first").reset_index(drop=True)
    df = df.dropna(subset=[*FEATURES, TARGET_COL]).reset_index(drop=True)

    X = df[FEATURES].to_numpy(np.float64)
    y = (df[TARGET_COL].to_numpy() > 0).astype(np.int32)
    close = df["Close"].to_numpy(np.float64)
    return X, y, close, df["DateTime"]


def select_thresholds(close_val: np.ndarray, prob_val: np.ndarray) -> tuple[str, float, float]:
    """Pick the (policy, buy, sell) with the best validation net P&L."""
    best = (-np.inf, "hold", 0.75, 0.20)
    for policy in POLICIES:
        for buy in BUY_THRESHOLDS:
            for sell in SELL_THRESHOLDS:
                positions = build_positions(prob_val, buy, sell, policy)
                pnl, _ = net_pnl(close_val, positions)
                if pnl > best[0]:
                    best = (pnl, policy, buy, sell)
    return best[1], best[2], best[3]


def run_fold(X, y, close, train_end, test_end) -> dict:
    """Train before `train_end`, tune thresholds on the train tail, test to `test_end`."""
    val_size = int(train_end * VALIDATION_TAIL_FRACTION)
    fit_end = train_end - val_size

    model = lgb.LGBMClassifier(n_estimators=MAX_TREES, **TUNED_PARAMS)
    model.fit(
        X[:fit_end], y[:fit_end],
        eval_set=[(X[fit_end:train_end], y[fit_end:train_end])], eval_metric="auc",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                   lgb.log_evaluation(0)],
    )

    prob_val = model.predict_proba(X[fit_end:train_end])[:, 1]
    policy, buy, sell = select_thresholds(close[fit_end:train_end], prob_val)

    prob_test = model.predict_proba(X[train_end:test_end])[:, 1]
    close_test = close[train_end:test_end]
    positions = build_positions(prob_test, buy, sell, policy)
    result = simulate(close_test, positions)
    metrics = summarize(close_test, positions, result)

    metrics.update({
        "policy": policy, "buy": buy, "sell": sell,
        "test_auc": float(roc_auc_score(y[train_end:test_end], prob_test)),
        "net_bar_pnl": result["net_bar_pnl"],
    })
    return metrics


def main() -> None:
    print("Loading data ...")
    X, y, close, timestamps = load_arrays()
    n = len(X)
    print(f"  {n:,} bars, {len(FEATURES)} features")

    initial = int(n * INITIAL_TRAIN_FRACTION)
    fold_size = (n - initial) // N_FOLDS
    pooled_pnl: list[np.ndarray] = []
    rows = []

    for fold in range(N_FOLDS):
        train_end = initial + fold * fold_size
        test_end = train_end + fold_size if fold < N_FOLDS - 1 else n
        print(f"\nFold {fold + 1}/{N_FOLDS}: train<{train_end:,}  test[{train_end:,}:{test_end:,}]")
        m = run_fold(X, y, close, train_end, test_end)
        pooled_pnl.append(m["net_bar_pnl"])
        rows.append({
            "fold": fold + 1,
            "test_start": timestamps.iloc[train_end][:10],
            "test_end": timestamps.iloc[test_end - 1][:10],
            "bars": m["n_bars"], "policy": m["policy"], "buy": m["buy"], "sell": m["sell"],
            "auc": m["test_auc"], "net_pnl": m["net_pnl"], "buy_hold": m["buy_hold_pnl"],
            "pf": m["profit_factor"], "win": m["win_rate_pct"], "trades": m["n_trades"],
            "max_dd": m["max_drawdown"],
        })
        print(f"  {rows[-1]['test_start']}..{rows[-1]['test_end']}  policy={m['policy']} "
              f"buy={m['buy']} sell={m['sell']}  AUC={m['test_auc']:.4f}  "
              f"net=${m['net_pnl']:,.0f}  PF={m['profit_factor']:.2f}  B&H=${m['buy_hold_pnl']:,.0f}")

    _report(rows, np.concatenate(pooled_pnl))


def _report(rows: list[dict], pooled_pnl: np.ndarray) -> None:
    total_net = sum(r["net_pnl"] for r in rows)
    total_bh = sum(r["buy_hold"] for r in rows)
    profitable = sum(r["net_pnl"] > 0 for r in rows)
    equity = np.cumsum(pooled_pnl)
    pooled_dd = float((equity - np.maximum.accumulate(equity)).min())

    header = (f"  {'fold':>4} {'test_start':>10} {'test_end':>10} {'bars':>9} {'policy':>5} "
              f"{'buy':>4} {'sell':>4} {'AUC':>6} {'net_pnl':>12} {'B&H':>10} {'PF':>6} {'win%':>6} {'trades':>7}")
    lines = ["Walk-forward evaluation — gradient ES strategy", "=" * len(header),
             f"{N_FOLDS} anchored folds, expanding train; thresholds chosen per fold on a "
             f"validation tail.", f"Cost: ${COST_PER_CONTRACT:.2f}/fill | ${POINT_VALUE:.0f}/pt", "",
             header, "  " + "-" * (len(header) - 2)]
    for r in rows:
        lines.append(f"  {r['fold']:>4} {r['test_start']:>10} {r['test_end']:>10} {r['bars']:>9,} "
                     f"{r['policy']:>5} {r['buy']:>4.2f} {r['sell']:>4.2f} {r['auc']:>6.4f} "
                     f"${r['net_pnl']:>10,.0f} ${r['buy_hold']:>8,.0f} {r['pf']:>6.2f} "
                     f"{r['win']:>6.1f} {r['trades']:>7,}")
    lines += ["",
              f"Folds profitable:        {profitable}/{len(rows)}",
              f"Total OOS net P&L:       ${total_net:,.2f}",
              f"Total buy & hold:        ${total_bh:,.2f}",
              f"Pooled OOS max drawdown: ${pooled_dd:,.2f}",
              f"Mean per-fold AUC:       {np.mean([r['auc'] for r in rows]):.4f}"]
    text = "\n".join(lines)
    RESULTS_TXT.write_text(text + "\n")
    print("\n" + text)
    print(f"\nSaved to {RESULTS_TXT}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity, color="#2E86AB", lw=1.2)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_title("Walk-forward pooled out-of-sample cumulative net P&L", fontweight="bold")
    ax.set_xlabel("Out-of-sample bar (5s, folds concatenated)")
    ax.set_ylabel("Cumulative net P&L ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(EQUITY_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Equity curve saved to {EQUITY_PNG}")


if __name__ == "__main__":
    main()
