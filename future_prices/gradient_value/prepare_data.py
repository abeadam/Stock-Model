"""
Data preparation for gradient_value predictor.

Thin wrapper around the two basic_model LightGBM models:
  lightgbm_model_highest.pkl  →  PctChange_ToMaxHigh_5  (max-high next 5 bars)
  lightgbm_model_lowest.pkl   →  PctChange_ToMinLow_5   (min-low  next 5 bars)

Both target columns are pre-computed in es_with_indicators.csv by futures_price.py.

Pipeline
--------
1. Load es_with_indicators.csv
2. Run inference with both basic_model pkls via prepare_features_and_target
   (guarantees identical feature order to training)
3. Assemble model_input.csv for train_predictor.py

Columns in model_input.csv
---------------------------
DateTime, Close, High, Low, Volume,      <- bookkeeping only (not features)
hour_sin, hour_cos, MFI_14,
high_close_ratio,        <- (High - Close) / Close  (stationary upper wick)
low_close_ratio,         <- (Close - Low)  / Close  (stationary lower wick)
pred_CloseToHigh,        <- predicted % from Close to next-5-bar max High
pred_CloseToLow,         <- predicted % from Close to next-5-bar min Low
target_avg5_close_pct    <- mean of next 5 bar-to-bar close % changes

Note: raw High/Low/Close and BB_20_2_Upper/Lower are intentionally NOT
exposed as model features — tree splits on non-stationary price levels
collapse to a single leaf region once live prices drift above the
training range. Use the ratio/distance features instead.

Volume is not used as a model feature; it is used as a sample weight in
train_predictor.py so high-volume bars dominate the training loss.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).parent
BASIC_DIR    = _HERE.parent / "basic_model"
GRADIENT_DIR = _HERE
CSV_PATH     = _HERE.parent / "es_with_indicators.csv"
OUT_PATH     = _HERE / "model_input.csv"

# lightgbm_utils lives in basic_model
sys.path.insert(0, str(BASIC_DIR))
from lightgbm_utils import (  # noqa: E402
    prepare_features_and_target,
    apply_top_n_features_from_csv_or_correlation,
)

# ── Constants ──────────────────────────────────────────────────────────────
_ASINH_SCALE = 50.0   # matches ASINH_SCALE used during training


# ── Inference ──────────────────────────────────────────────────────────────

def run_inference(
    df: pd.DataFrame,
    model_path: Path,
    target_col: str,
) -> np.ndarray:
    """
    Run inference using the same feature pipeline as training.

    Builds the full 122-feature matrix via prepare_features_and_target, then
    selects the subset the model was actually trained on by reading the feature
    count from model.n_features_in_ and the ranked feature list from the
    model's importance CSV (same selection apply_top_n_features_from_csv_or_correlation
    applies during training).

    df must already contain `target_col` (used only to determine the valid mask;
    target values are ignored at inference time).

    Returns float array of length len(df) with NaN for rows where any feature
    or the target is missing.
    """
    # Build full feature matrix — same pipeline as training
    X, _y, feature_cols, _scale, _method, _, kept_rows = prepare_features_and_target(
        df, target_column=target_col, return_valid_mask=True
    )

    model = joblib.load(model_path)

    # Select the exact feature subset the model was trained on.
    # n_features_in_ is the authoritative count; the importance CSV provides the
    # ranked order so we pick the same top-N columns as the training script did.
    n_features     = model.n_features_in_
    importance_csv = model_path.with_name(
        model_path.stem + "_feature_importances_splits_and_gain.csv"
    )
    X, feature_cols, _ = apply_top_n_features_from_csv_or_correlation(
        X, feature_cols, None, importance_csv, n_features
    )

    y_scaled = model.predict(X)
    y_pred   = np.sign(y_scaled) * np.sinh(np.abs(y_scaled)) / _ASINH_SCALE

    # Scatter back onto exactly the rows kept. Counting in from the ends assumed
    # five NaN-target rows at the tail, but futures_price.py already trims them,
    # so every prediction landed five bars early: bar i carried the prediction
    # made from bar i+5, the window the gradient target measures. A NaN
    # mid-dataset would have shifted every later row further still.
    result = np.full(len(df), np.nan)
    result[kept_rows] = y_pred
    return result


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Data Preparation: gradient_value predictor")
    print("=" * 60)

    # 1. Load data (PctChange_ToMaxHigh_5 / PctChange_ToMinLow_5 pre-computed by futures_price.py)
    print(f"\n[1/4] Loading {CSV_PATH.name} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"      {len(df):,} rows  ×  {df.shape[1]} columns")
    print(f"      PctChange_ToMaxHigh_5 — mean: {df['PctChange_ToMaxHigh_5'].mean():.4f}%  "
          f"std: {df['PctChange_ToMaxHigh_5'].std():.4f}%")
    print(f"      PctChange_ToMinLow_5  — mean: {df['PctChange_ToMinLow_5'].mean():.4f}%  "
          f"std: {df['PctChange_ToMinLow_5'].std():.4f}%")

    # 2. Run inference with both basic_model LightGBM models
    print("\n[2/4] Running LightGBM inference ...")

    print("      lightgbm_model_highest  (pred_CloseToHigh) ...")
    pred_high = run_inference(
        df,
        model_path=BASIC_DIR / "lightgbm_model_highest.pkl",
        target_col="PctChange_ToMaxHigh_5",
    )
    n_valid_high = int((~np.isnan(pred_high)).sum())
    print(f"        valid: {n_valid_high:,} / {len(df):,}  |  "
          f"mean={np.nanmean(pred_high):.4f}%  std={np.nanstd(pred_high):.4f}%")

    print("      lightgbm_model_lowest   (pred_CloseToLow) ...")
    pred_low = run_inference(
        df,
        model_path=BASIC_DIR / "lightgbm_model_lowest.pkl",
        target_col="PctChange_ToMinLow_5",
    )
    n_valid_low = int((~np.isnan(pred_low)).sum())
    print(f"        valid: {n_valid_low:,} / {len(df):,}  |  "
          f"mean={np.nanmean(pred_low):.4f}%  std={np.nanstd(pred_low):.4f}%")

    # 3. Assemble output
    print("\n[3/4] Assembling predictor dataset ...")


    # ── Lagged features (all shift forward so no lookahead) ──────────────────
    close_pct = df["Close"].pct_change() * 100   # bar-to-bar % change

    # Recent momentum: last 5 bars of close % change
    lag_close = {f"close_pct_lag{k}": close_pct.shift(k) for k in range(1, 6)}

    # Volatility regime: rolling std of close % change (non-overlapping windows)
    rolling_vol_5  = close_pct.shift(1).rolling(5,  min_periods=3).std()
    rolling_vol_20 = close_pct.shift(1).rolling(20, min_periods=10).std()

    # Previous bar's LightGBM signals
    pred_high_s = pd.Series(pred_high)
    pred_low_s  = pd.Series(pred_low)

    out = pd.DataFrame({
        "DateTime":          df["DateTime"],
        "Close":             df["Close"],
        "High":              df["High"],
        "Low":               df["Low"],
        "Volume":            df["Volume"],
        # ── base features ──────────────────────────────────────────────────
        "hour_sin":          df["hour_sin"],
        "hour_cos":          df["hour_cos"],
        "MFI_14":            df["MFI_14"],
        "pred_CloseToHigh":  pred_high,
        "pred_CloseToLow":   pred_low,
        # Stationary replacements for raw High/Low/Close: bar wick sizes
        # relative to close. These stay in a stable range even as the
        # absolute price level drifts well beyond the training window.
        "high_close_ratio":  ((df["High"] - df["Close"]) / df["Close"]).values,
        "low_close_ratio":   ((df["Close"] - df["Low"])  / df["Close"]).values,
        # ── lagged features ────────────────────────────────────────────────
        **lag_close,
        "rolling_vol_5":     rolling_vol_5.values,
        "rolling_vol_20":    rolling_vol_20.values,
        "pred_high_lag1":    pred_high_s.shift(1).values,
        "pred_low_lag1":     pred_low_s.shift(1).values,
        # ── normalised BB distance features ────────────────────────────────
        # (Close - Band) / Close: scale-invariant distance from each band.
        # Positive → price above band (upper: overbought; lower: comfortable).
        # Negative → price below band (upper: room to rise; lower: oversold).
        "bb_upper_dist": ((df["Close"] - df["BB_20_2_Upper"]) / df["Close"]).values,
        "bb_lower_dist": ((df["Close"] - df["BB_20_2_Lower"]) / df["Close"]).values,
        # ── normalised EMA distance features ───────────────────────────────
        # (Close - EMA_N) / Close: deviation from each EMA.
        # Positive → price above EMA (bullish momentum / overbought).
        # Negative → price below EMA (bearish momentum / oversold).
        # EMA_10/EMA_50 ratio captures short-vs-long trend alignment.
        "ema10_dist":  ((df["Close"] - df["EMA_10"]) / df["Close"]).values,
        "ema20_dist":  ((df["Close"] - df["EMA_20"]) / df["Close"]).values,
        "ema50_dist":  ((df["Close"] - df["EMA_50"]) / df["Close"]).values,
        "ema_cross":   (df["EMA_10"] / df["EMA_50"]).values,
        # ── time-of-day interaction terms ──────────────────────────────────
        # indicator × hour_sin/cos: lets the model condition on "how
        # overbought/volatile is this signal *at this phase of the session*".
        # E.g. high MFI at open → momentum; high MFI near close → fade.
        "mfi_x_hsin":    (df["MFI_14"]      * df["hour_sin"]).values,
        "mfi_x_hcos":    (df["MFI_14"]      * df["hour_cos"]).values,
        "ema10_x_hsin":  (((df["Close"] - df["EMA_10"]) / df["Close"]) * df["hour_sin"]).values,
        "ema10_x_hcos":  (((df["Close"] - df["EMA_10"]) / df["Close"]) * df["hour_cos"]).values,
        "bbud_x_hsin":   (((df["Close"] - df["BB_20_2_Upper"]) / df["Close"]) * df["hour_sin"]).values,
        "bbud_x_hcos":   (((df["Close"] - df["BB_20_2_Upper"]) / df["Close"]) * df["hour_cos"]).values,
        "rvol5_x_hsin":  (rolling_vol_5  * df["hour_sin"]).values,
        "rvol5_x_hcos":  (rolling_vol_5  * df["hour_cos"]).values,
        # ── VXM (VIX futures) features ─────────────────────────────────────
        # VXM measures implied volatility expectations; useful regime signal.
        # vxm_pct / lags: bar-level vol spike/drop (computed from VXM_Close).
        # vxm_es_ratio:   VXM_Close / ES_Close — cross-asset stress indicator.
        "vxm_close":    df["VXM_Close"].values,
        "vxm_pct":      (df["VXM_Close"].pct_change() * 100).values,
        "vxm_pct_lag1": (df["VXM_Close"].pct_change() * 100).shift(1).values,
        "vxm_pct_lag2": (df["VXM_Close"].pct_change() * 100).shift(2).values,
        "vxm_es_ratio": (df["VXM_Close"] / df["Close"]).values,
        # ── Absolute Strength Index (bull / bear power) ─────────────────────
        # EMA of up/down % moves at 7/14/28 (stationary, like MFI). The
        # 14-period pair is also lagged 1..5 bars, matching the close_pct lag
        # structure, so the model sees the recent trajectory of buying/selling
        # pressure rather than only its current level.
        "asi_bull_7":   df["ASI_Bull_7"].values,
        "asi_bear_7":   df["ASI_Bear_7"].values,
        "asi_bull_14":  df["ASI_Bull_14"].values,
        "asi_bear_14":  df["ASI_Bear_14"].values,
        "asi_bull_28":  df["ASI_Bull_28"].values,
        "asi_bear_28":  df["ASI_Bear_28"].values,
        **{f"asi_bull_14_lag{k}": df["ASI_Bull_14"].shift(k).values for k in range(1, 6)},
        **{f"asi_bear_14_lag{k}": df["ASI_Bear_14"].shift(k).values for k in range(1, 6)},
        # ── target ─────────────────────────────────────────────────────────
        "target_avg5_close_pct": pd.concat(
            [df["Close"].pct_change().shift(-i) for i in range(1, 6)], axis=1
        ).mean(axis=1) * 100,
    })

    total = len(out)
    out.dropna(inplace=True)
    print(f"      Total rows:  {total:,}")
    print(f"      Valid rows:  {len(out):,}")
    print(f"      Dropped:     {total - len(out):,}")

    # 4. Save
    print(f"\n[4/4] Saving to {OUT_PATH.name} ...")
    out.to_csv(OUT_PATH, index=False)
    print(f"      Columns: {list(out.columns)}")

    split = int(len(out) * 0.8)
    print(f"\nTime-series split preview:")
    print(f"  Train: first {split:,} rows  ({split / len(out) * 100:.1f}%)")
    print(f"  Test:  last  {len(out) - split:,} rows   ({(len(out) - split) / len(out) * 100:.1f}%)")

    print("\n" + "=" * 60)
    print("Data preparation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
