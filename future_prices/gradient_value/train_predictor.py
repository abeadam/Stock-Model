"""
LightGBM binary classifier — predicts direction of average close change over next 5 bars.

Input:   gradient_value/model_input.csv  (from prepare_data.py)
Output:  predictor_results.txt           (metrics summary)
         predictor_predictions.png       (confidence vs accuracy plot)
         predictor_model.pkl             (trained model for reuse)

Target
------
Binary: 1 if mean(next-5-bar close % changes) > 0, else 0.
Derived inline from `target_avg5_close_pct` in model_input.csv.

Feature-set sweep
-----------------
Two feature sets are compared to measure the uplift from lagged features.

  base      (10 features): current-bar OHLC, time, MFI, BB bands,
                           LightGBM high/low predictions
  base+lags (19 features): base + 5 lagged close % changes (momentum),
                           rolling vol 5 & 20 (volatility regime),
                           prev-bar LightGBM signals (pred_high_lag1/low_lag1)

Loss: binary log-loss, 500 trees, LR=0.01 — no early stopping.

Split:
    Train = first 80% (chronological)
    Test  = last 20%  (never seen during training)
"""

import joblib
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, log_loss

# ── Paths ───────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).parent
INPUT_CSV   = _HERE / "model_input.csv"
RESULTS_TXT = _HERE / "predictor_results.txt"
PLOT_PNG    = _HERE / "predictor_predictions.png"
MODEL_PKL   = _HERE / "predictor_model.pkl"

# ── Feature / target config ─────────────────────────────────────────────────
TARGET_COL = "target_avg5_close_pct"

BASE_FEATURES = [
    # Stationary bar-shape features replace raw High/Low/Close — trees
    # trained on absolute price levels collapse to one leaf region once
    # live prices drift above the training range.
    "high_close_ratio", "low_close_ratio",
    "hour_sin", "hour_cos",
    "MFI_14",
    # Raw BB_20_2_Upper/Lower removed: bb_upper_dist/bb_lower_dist
    # (see BB_DIST_FEATURES below) encode the same info stationarily.
    "pred_CloseToHigh", "pred_CloseToLow",
]

LAG_FEATURES = [
    "close_pct_lag1", "close_pct_lag2", "close_pct_lag3",
    "close_pct_lag4", "close_pct_lag5",
    "rolling_vol_5", "rolling_vol_20",
    "pred_high_lag1", "pred_low_lag1",
]

BB_DIST_FEATURES = [
    "bb_upper_dist",   # (Close - BB_Upper) / Close
    "bb_lower_dist",   # (Close - BB_Lower) / Close
]

EMA_FEATURES = [
    "ema10_dist",   # (Close - EMA_10) / Close  — short-term deviation
    "ema20_dist",   # (Close - EMA_20) / Close  — medium-term deviation
    "ema50_dist",   # (Close - EMA_50) / Close  — long-term deviation
    "ema_cross",    # EMA_10 / EMA_50           — trend alignment ratio
]

INTERACTION_FEATURES = [
    "mfi_x_hsin", "mfi_x_hcos",         # MFI conditioned on session phase
    "ema10_x_hsin", "ema10_x_hcos",      # EMA deviation at this time of day
    "bbud_x_hsin", "bbud_x_hcos",        # BB upper distance at this time of day
    "rvol5_x_hsin", "rvol5_x_hcos",      # rolling vol at this time of day
]

VXM_FEATURES = [
    "vxm_close",     # raw VXM level (regime: high = fearful market)
    "vxm_pct",       # bar-to-bar VXM % change (vol spike/drop)
    "vxm_pct_lag1",  # prior bar's VXM change
    "vxm_pct_lag2",  # two bars ago
    "vxm_es_ratio",  # VXM / ES Close — cross-asset stress normalised by price
]

ASI_FEATURES = [
    # Absolute Strength Index — bull/bear power (EMA of up/down % moves).
    "asi_bull_7", "asi_bear_7",
    "asi_bull_14", "asi_bear_14",
    "asi_bull_28", "asi_bear_28",
    # 14-period pair lagged 1..5 bars (recent buying/selling-pressure trajectory).
    "asi_bull_14_lag1", "asi_bull_14_lag2", "asi_bull_14_lag3",
    "asi_bull_14_lag4", "asi_bull_14_lag5",
    "asi_bear_14_lag1", "asi_bear_14_lag2", "asi_bear_14_lag3",
    "asi_bear_14_lag4", "asi_bear_14_lag5",
]

_BEST = BASE_FEATURES + LAG_FEATURES + BB_DIST_FEATURES + EMA_FEATURES
_BEST_PLUS_ASI = _BEST + ASI_FEATURES

# Sweep both so the test-set metrics directly show the uplift from adding ASI.
FEATURE_SETS = {
    "best_so_far": _BEST,
    "best_plus_asi": _BEST_PLUS_ASI,
}

# ── LightGBM hyperparameters ────────────────────────────────────────────────
LGBM_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    n_estimators=500,
    learning_rate=0.01,
    max_depth=6,
    num_leaves=63,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l1=0.05,
    lambda_l2=0.05,
    min_data_in_leaf=50,
    n_jobs=-1,
    verbose=-1,
    random_state=42,
)


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str,
) -> dict:
    y_pred  = (y_prob >= 0.5).astype(int)
    dir_acc = np.mean(y_true == y_pred) * 100
    auc     = roc_auc_score(y_true, y_prob)
    logloss = log_loss(y_true, y_prob)

    frac_pos, mean_prob = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    ece = float(np.mean(np.abs(frac_pos - mean_prob)))

    print(f"\n{label}")
    print(f"  Directional acc:   {dir_acc:.2f}%")
    print(f"  AUC-ROC:           {auc:.4f}")
    print(f"  Log loss:          {logloss:.4f}")
    print(f"  ECE:               {ece:.4f}")
    return dict(dir_acc=dir_acc, auc=auc, logloss=logloss, ece=ece)


# ── Plot ────────────────────────────────────────────────────────────────────

def plot_confidence_vs_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path,
    title_tag: str,
    n_bins_cal: int = 10,
    n_thresh_steps: int = 100,
) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle(
        f"LightGBM binary classifier [{title_tag}] — confidence vs accuracy",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── Subplot 1: Calibration curve ────────────────────────────────────────
    frac_pos, mean_prob = calibration_curve(
        y_true, y_prob, n_bins=n_bins_cal, strategy="uniform"
    )
    ax1.plot(mean_prob, frac_pos, marker="o", color="#2E86AB", lw=2,
             label="Model calibration")
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.5,
             label="Perfect calibration")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Mean predicted probability", fontsize=11)
    ax1.set_ylabel("Fraction of positives (actual up)", fontsize=11)
    ax1.set_title("Calibration curve — does 70% confidence mean 70% accuracy?",
                  fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── Subplot 2: Accuracy vs confidence threshold ──────────────────────────
    confidence = np.abs(y_prob - 0.5)
    thresholds = np.linspace(0, 0.48, n_thresh_steps)
    acc_list   = []
    frac_list  = []

    for t in thresholds:
        mask = confidence >= t
        n    = mask.sum()
        if n == 0:
            acc_list.append(np.nan)
            frac_list.append(0.0)
        else:
            acc = np.mean((y_prob[mask] >= 0.5).astype(int) == y_true[mask]) * 100
            acc_list.append(acc)
            frac_list.append(n / len(y_true))

    acc_arr  = np.array(acc_list)
    frac_arr = np.array(frac_list)

    ax2.plot(thresholds, acc_arr, color="#2E86AB", lw=2, label="Directional accuracy")
    ax2.axhline(50, linestyle="--", color="gray", alpha=0.5, lw=1.5,
                label="50% baseline (random)")
    ax2.set_xlabel("Confidence threshold  |p − 0.5|", fontsize=11)
    ax2.set_ylabel("Directional accuracy (%)", fontsize=11, color="#2E86AB")
    ax2.tick_params(axis="y", labelcolor="#2E86AB")
    ax2.set_title("Accuracy vs confidence threshold — high-confidence predictions only",
                  fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(thresholds, frac_arr * 100, color="#A23B72", lw=1.5, linestyle=":",
              label="% of samples above threshold")
    ax2b.set_ylabel("Samples above threshold (%)", fontsize=11, color="#A23B72")
    ax2b.tick_params(axis="y", labelcolor="#A23B72")
    ax2b.set_ylim(0, 100)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10)

    # ── Subplot 3: Probability distribution by actual direction ──────────────
    prob_up   = y_prob[y_true == 1]
    prob_down = y_prob[y_true == 0]

    ax3.hist(prob_up,   bins=50, alpha=0.5, color="#2E86AB", density=True,
             label=f"Actual up   (n={len(prob_up):,})")
    ax3.hist(prob_down, bins=50, alpha=0.5, color="#A23B72", density=True,
             label=f"Actual down (n={len(prob_down):,})")
    ax3.axvline(0.5, color="black", linestyle="--", lw=1.5, alpha=0.7,
                label="Decision boundary (p=0.5)")
    ax3.set_xlabel("Predicted probability of up", fontsize=11)
    ax3.set_ylabel("Density", fontsize=11)
    ax3.set_title("Predicted probability distribution — do up/down predictions separate?",
                  fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {save_path}")


# ── Persist ─────────────────────────────────────────────────────────────────

def write_results(
    best_fset: str,
    best_feats: list,
    train_metrics: dict,
    test_metrics: dict,
    all_results: dict,
    n_train: int,
    n_test: int,
) -> None:
    lines = [
        "LightGBM Binary Classifier — Direction of Avg Next-5-Bar Close % Change",
        "=" * 70,
        f"Target: 1 if mean(next-5-bar close % changes) > 0, else 0",
        f"Loss:   binary log-loss, 500 trees, LR=0.01",
        f"Best feature set: {best_fset}  ({len(best_feats)} features)",
        f"Features: {', '.join(best_feats)}",
        f"\nSplit:",
        f"  Train: {n_train:,} rows (first 80%)",
        f"  Test:  {n_test:,} rows (last 20%)",
        "\nFEATURE SET SWEEP — Test metrics:",
        "-" * 70,
    ]
    for fset, m in all_results.items():
        marker = "  ← BEST" if fset == best_fset else ""
        lines += [
            f"\n  [{fset}]{marker}",
            f"    Dir%:    {m['dir_acc']:.2f}%",
            f"    AUC:     {m['auc']:.4f}",
            f"    LogLoss: {m['logloss']:.4f}",
            f"    ECE:     {m['ece']:.4f}",
        ]
    lines += [
        "\nBEST MODEL TRAINING SET:",
        "-" * 40,
        f"  Directional acc: {train_metrics['dir_acc']:.2f}%",
        f"  AUC-ROC:         {train_metrics['auc']:.4f}",
        f"  Log loss:        {train_metrics['logloss']:.4f}",
        f"  ECE:             {train_metrics['ece']:.4f}",
        "\nBEST MODEL TEST SET (last 20% — unseen):",
        "-" * 40,
        f"  Directional acc: {test_metrics['dir_acc']:.2f}%",
        f"  AUC-ROC:         {test_metrics['auc']:.4f}",
        f"  Log loss:        {test_metrics['logloss']:.4f}",
        f"  ECE:             {test_metrics['ece']:.4f}",
    ]
    with open(RESULTS_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Results saved to {RESULTS_TXT}")


# ── Per-feature-set run ──────────────────────────────────────────────────────

def run_feature_set(
    name: str,
    feat_cols: list,
    df: pd.DataFrame,
    y_binary: np.ndarray,
) -> tuple:
    """Train and evaluate for one feature set. Returns (test metrics, model, splits)."""
    print(f"\n{'─'*60}")
    print(f"  feature set = {name}  ({len(feat_cols)} features)")
    print(f"{'─'*60}")

    X_all = df[feat_cols].astype(np.float64)
    valid = ~np.isnan(y_binary.astype(np.float64)) & ~(X_all.isna().any(axis=1).values)
    X     = X_all[valid]
    y     = y_binary[valid]

    split      = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y[:split],      y[split:]

    model     = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_tr, y_tr)

    y_prob_te = model.predict_proba(X_te)[:, 1]
    metrics   = evaluate(y_te, y_prob_te, f"  Test set [{name}]")

    return metrics, model, X_tr, y_tr, X_te, y_te, y_prob_te


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LightGBM Binary Classifier — direction prediction sweep")
    print("=" * 60)

    print(f"\nLoading {INPUT_CSV.name} ...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df):,} rows loaded")

    # Deduplicate to one row per timestamp so the model is evaluated on
    # genuinely unseen bars, not label-consistent twins of training rows.
    n_before = len(df)
    df = df.drop_duplicates(subset="DateTime", keep="first").reset_index(drop=True)
    print(f"  {n_before - len(df):,} duplicate-timestamp rows dropped → {len(df):,} unique bars")

    y_binary = (df[TARGET_COL].values > 0).astype(np.int32)
    print(f"  Class balance: {y_binary.mean()*100:.1f}% up / {(1-y_binary.mean())*100:.1f}% down")

    results = {}
    stored  = {}   # name → (model, feat_cols, X_tr, y_tr, X_te, y_te, y_prob_te)

    for fset_name, feat_cols in FEATURE_SETS.items():
        metrics, model, X_tr, y_tr, X_te, y_te, y_prob_te = run_feature_set(
            fset_name, feat_cols, df, y_binary
        )
        results[fset_name] = metrics
        stored[fset_name]  = (model, feat_cols, X_tr, y_tr, X_te, y_te, y_prob_te)

    # ── Comparison table ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY — Test set metrics by feature set")
    print(f"{'='*60}")
    hdr = f"{'Feature set':>12}  {'Dir%':>7}  {'AUC':>8}  {'LogLoss':>10}  {'ECE':>8}"
    print(hdr)
    print("─" * len(hdr))
    best_dir     = max(results, key=lambda t: results[t]["dir_acc"])
    best_auc     = max(results, key=lambda t: results[t]["auc"])
    best_logloss = min(results, key=lambda t: results[t]["logloss"])
    for name, m in results.items():
        flags = "".join([
            " ← best dir%"    if name == best_dir     else "",
            " ← best AUC"     if name == best_auc     else "",
            " ← best logloss" if name == best_logloss else "",
        ])
        print(f"{name:>12}  {m['dir_acc']:>7.2f}  {m['auc']:>8.4f}  "
              f"{m['logloss']:>10.4f}  {m['ece']:>8.4f}{flags}")

    best_fset = best_dir   # winner by directional accuracy
    print(f"\nBest feature set: {best_fset}")

    # ── Save best model ─────────────────────────────────────────────────────
    print("\nSaving best model ...")
    model, feat_cols, X_tr, y_tr, X_te, y_te, y_prob_te = stored[best_fset]

    y_prob_tr     = model.predict_proba(X_tr)[:, 1]
    train_metrics = evaluate(y_tr, y_prob_tr, "  Train set [best model]")

    plot_confidence_vs_accuracy(y_te, y_prob_te, PLOT_PNG,
                                title_tag=f"features={best_fset}")
    write_results(best_fset, feat_cols, train_metrics, results[best_fset],
                  results, len(X_tr), len(X_te))
    joblib.dump({"model": model, "feature_cols": feat_cols}, MODEL_PKL)
    print(f"  Model saved to {MODEL_PKL}  (feature_set={best_fset})")

    print("\n" + "=" * 60)
    print("Sweep complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
