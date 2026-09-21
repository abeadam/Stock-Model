"""
ES (E-mini S&P 500) feature computation and LightGBM prediction.

Builds features from OHLCV DataFrames and runs trained highest/lowest models.
Feature set and column order are driven by each model's feature_importances CSV.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

from lightgbm_utils import unscale_target

ModelKind = Literal["highest", "lowest"]
MODEL_KINDS: tuple[ModelKind, ...] = ("highest", "lowest")

_BASE_DIR = Path(__file__).resolve().parent
TARGET_SCALE = 50.0
SCALING_METHOD = "asinh"
OHLCV_COLS = ("Open", "High", "Low", "Close", "Volume")


def _path_importance_csv(model_kind: ModelKind) -> Path:
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"model_kind must be in {MODEL_KINDS}, got {model_kind!r}")
    return _BASE_DIR / f"lightgbm_model_{model_kind}_feature_importances_splits_and_gain.csv"


def _path_model(model_kind: ModelKind) -> Path:
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"model_kind must be in {MODEL_KINDS}, got {model_kind!r}")
    return _BASE_DIR / f"lightgbm_model_{model_kind}.pkl"


def get_feature_names(
    model_kind: ModelKind = "highest",
    *,
    csv_path: Path | str | None = None,
) -> list[str]:
    """Feature names in CSV order for the given model kind."""
    path = Path(csv_path) if csv_path is not None else _path_importance_csv(model_kind)
    if not path.exists():
        raise FileNotFoundError(f"Importance CSV not found: {path}")
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        raise ValueError(f"CSV must have column 'feature': {path}")
    return df["feature"].astype(str).tolist()


def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    missing = [c for c in OHLCV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}. Required: {OHLCV_COLS}")
    return df[list(OHLCV_COLS)]


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range (rolling mean of true range)."""
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr = np.nanmax(
        np.column_stack([
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]),
        axis=1,
    )
    out = np.full_like(tr, np.nan, dtype=np.float64)
    for i in range(period - 1, len(tr)):
        out[i] = np.nanmean(tr[i - period + 1 : i + 1])
    return out


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """RSI (EWM of gains/losses on close deltas), matching futures_price.calculate_rsi.

    Division is left to propagate the way pandas does it there: no losses in the
    window gives rs=inf and RSI 100, while no movement at all gives 0/0 -> NaN,
    which the caller's NaN filter drops instead of reading as a real RSI of 100.
    """
    delta = np.diff(close.astype(np.float64), prepend=np.nan)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))


def _bollinger(close: np.ndarray, period: int, num_std: float) -> tuple[np.ndarray, np.ndarray]:
    """Bollinger upper and lower bands."""
    ma = pd.Series(close).rolling(period).mean().values
    std = pd.Series(close).rolling(period).std().values
    return ma + num_std * std, ma - num_std * std


def _roll_mean(values: np.ndarray, period: int) -> np.ndarray:
    """Rolling mean (the BB_{period}_MA of futures_price.calculate_bollinger_bands)."""
    return pd.Series(values).rolling(period).mean().to_numpy(dtype=np.float64, na_value=np.nan)


def _roll_std(values: np.ndarray, period: int) -> np.ndarray:
    """Rolling standard deviation."""
    return pd.Series(values).rolling(period).std().to_numpy(dtype=np.float64, na_value=np.nan)


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    """EMA with adjust=False, matching futures_price.calculate_ema."""
    return pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy(dtype=np.float64, na_value=np.nan)


def _mfi(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int,
) -> np.ndarray:
    """Money Flow Index, matching futures_price.calculate_mfi.

    A zero negative-flow sum means every flow in the window was positive, which
    is MFI 100. Rows in the rolling warmup keep NaN rather than collapsing to
    100, so they match the training data and get dropped by the caller's
    NaN filter instead of entering the model as a real-looking value.
    """
    typical_price = (high + low + close) / 3.0
    money_flow    = typical_price * volume
    tp_delta      = np.diff(typical_price, prepend=np.nan)
    pos_flow      = np.where(tp_delta > 0, money_flow, 0.0)
    neg_flow      = np.where(tp_delta < 0, money_flow, 0.0)
    pos_sum       = pd.Series(pos_flow).rolling(period).sum().values
    neg_sum       = pd.Series(neg_flow).rolling(period).sum().values
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pos_sum / neg_sum
    return np.where(neg_sum == 0, 100.0, 100.0 - (100.0 / (1.0 + ratio)))


def _parse_datetime_column(date_col: pd.Series) -> pd.Series:
    """Parse a date column to datetimes without the dateutil per-element fallback.

    Handles the formats this code sees: Unix seconds as a number (offline ES
    files) or as a string (the live IB feed under formatDate=2), and string
    timestamps from either the IB feed ("YYYYMMDD HH:MM:SS", sometimes
    double-spaced) or pandas/ISO ("YYYY-MM-DD HH:MM:SS"). Passing an explicit
    format / "mixed" avoids pandas' "Could not infer format" warning, which is
    both noisy and slow in the live per-bar loop.
    """
    if pd.api.types.is_numeric_dtype(date_col):
        return pd.to_datetime(date_col, unit="s", errors="coerce")
    cleaned = date_col.astype(str).str.strip().str.replace("  ", " ", regex=False)
    # The TWS API hands epoch seconds back as a *string* under formatDate=2, so a
    # column of all-digit strings is still Unix seconds and must not go down the
    # date-string path — neither format below matches it, which would silently
    # yield NaT and turn every time-of-day feature into NaN.
    as_epoch = pd.to_numeric(cleaned, errors="coerce")
    if as_epoch.notna().all():
        return pd.to_datetime(as_epoch, unit="s", errors="coerce")
    parsed = pd.to_datetime(cleaned, format="%Y%m%d %H:%M:%S", errors="coerce")
    if parsed.isna().all():  # not the IB compact format — fall back to general parsing
        parsed = pd.to_datetime(cleaned, errors="coerce", format="mixed")
    return parsed


def _time_features(date_col: pd.Series, n: int) -> dict[str, np.ndarray]:
    """
    Compute time-of-day and calendar features from a date column.
    Timestamps are assumed to be in ET (or naive ET) — no tz conversion is applied.
    Falls back to NaN arrays on any parse error.
    """
    try:
        dt        = _parse_datetime_column(date_col)
        hour      = dt.dt.hour.values.astype(np.float64)
        minute    = dt.dt.minute.values.astype(np.float64)
        hour_frac = hour + minute / 60.0
        # Formal trading starts at 09:00 ET
        hours_formal    = hour_frac - 9.0
        # Overnight session starts at 18:00 ET; before 18:00 use previous day's 18:00
        hours_overnight = np.where(hour_frac >= 18.0, hour_frac - 18.0, hour_frac + 6.0)
        return {
            "Hours_From_Formal_Trading":    hours_formal,
            "Hours_From_Overnight_Trading": hours_overnight,
            "hour_sin":  np.sin(2 * np.pi * hour_frac / 24),
            "hour_cos":  np.cos(2 * np.pi * hour_frac / 24),
            "day_sin":   np.sin(2 * np.pi * dt.dt.dayofweek.values / 7),
            "day_cos":   np.cos(2 * np.pi * dt.dt.dayofweek.values / 7),
            "month_sin": np.sin(2 * np.pi * dt.dt.month.values / 12),
            "month_cos": np.cos(2 * np.pi * dt.dt.month.values / 12),
        }
    except Exception:
        nan_arr = np.full(n, np.nan, dtype=np.float64)
        return {k: nan_arr.copy() for k in (
            "Hours_From_Formal_Trading", "Hours_From_Overnight_Trading",
            "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
        )}


def _build_all_feature_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Build every supported feature from df.

    Required columns: High, Low, Close, Volume.
    Optional columns used when present:
      - date / DateTime / Date  → time-of-day and calendar features
      - VXM_Close               → VXM volatility-regime features:
                                  VXM_BB_50_3_Upper, VXM_BB_50_STD, VXM_ES_Ratio
                                  vxm_close, vxm_pct, vxm_pct_lag1, vxm_pct_lag2, vxm_es_ratio

    Returns NaN arrays for any feature whose source data is absent.
    """
    n      = len(df)
    high   = df["High"].values.astype(np.float64)
    low    = df["Low"].values.astype(np.float64)
    close  = df["Close"].values.astype(np.float64)
    volume = df["Volume"].values.astype(np.float64)

    safe_close = np.where(close > 0, close, np.nan)
    safe_high  = np.where(high  > 0, high,  np.nan)
    safe_low   = np.where(low   > 0, low,   np.nan)

    def rel(band: np.ndarray, price: np.ndarray, safe: np.ndarray) -> np.ndarray:
        out = np.full(n, np.nan, dtype=np.float64)
        ok  = np.isfinite(safe) & (safe > 0)
        np.divide(band - price, safe, out=out, where=ok)
        return out

    # ── % change of close (basis for rolling stats) ────────────────────────
    pct    = np.full(n, np.nan, dtype=np.float64)
    pct[1:] = (close[1:] - close[:-1]) / safe_close[:-1] * 100.0
    pct_s  = pd.Series(pct)

    # ── Indicators ─────────────────────────────────────────────────────────
    atr_14 = _atr(high, low, close, 14)
    rsi_7  = _rsi(close,  7)
    rsi_14 = _rsi(close, 14)
    rsi_28 = _rsi(close, 28)
    mfi_7  = _mfi(high, low, close, volume,  7)
    mfi_14 = _mfi(high, low, close, volume, 14)
    mfi_28 = _mfi(high, low, close, volume, 28)

    # ── Bollinger Bands ─────────────────────────────────────────────────────
    bb20_1_u, bb20_1_l = _bollinger(close, 20, 1.0)
    bb20_2_u, bb20_2_l = _bollinger(close, 20, 2.0)
    bb20_3_u, bb20_3_l = _bollinger(close, 20, 3.0)
    bb50_1_u, bb50_1_l = _bollinger(close, 50, 1.0)
    bb50_2_u, bb50_2_l = _bollinger(close, 50, 2.0)
    bb50_3_u, bb50_3_l = _bollinger(close, 50, 3.0)
    bb10_std = _roll_std(close, 10)
    bb20_std = _roll_std(close, 20)
    bb50_std = _roll_std(close, 50)
    bb50_ma  = _roll_mean(close, 50)

    # ── Momentum ────────────────────────────────────────────────────────────
    momentum_5  = np.full(n, np.nan, dtype=np.float64)
    momentum_10 = np.full(n, np.nan, dtype=np.float64)
    if n >= 6:  momentum_5[5:]   = pct[5:]   - pct[: n - 5]
    if n >= 11: momentum_10[10:] = pct[10:]  - pct[: n - 10]

    # ── Pct-change lags ─────────────────────────────────────────────────────
    # Matches lightgbm_utils.prepare_features_and_target, which lags
    # Close.pct_change()*100 by each of these offsets.
    pct_lags: dict[str, np.ndarray] = {
        f"PctChange_Lag_{lag}": pct_s.shift(lag).to_numpy(dtype=np.float64, na_value=np.nan)
        for lag in (1, 2, 3, 5, 10, 20)
    }

    # ── Rolling stats on pct (windows 5 / 10 / 20 / 50) ───────────────────
    rolling: dict[str, np.ndarray] = {}
    for w in (5, 10, 20, 50):
        rolling[f"RollingMean_{w}"] = pct_s.rolling(w).mean().values
        rolling[f"RollingStd_{w}"]  = pct_s.rolling(w).std().values
        rolling[f"RollingMin_{w}"]  = pct_s.rolling(w).min().values
        rolling[f"RollingMax_{w}"]  = pct_s.rolling(w).max().values

    # ── Volatility ──────────────────────────────────────────────────────────
    vol_10      = pct_s.rolling(10).std().values
    vol_20      = pct_s.rolling(20).std().values
    vol_50      = pct_s.rolling(50).std().values
    safe_vol_50 = np.where(vol_50 > 0, vol_50, 1.0)
    vol_ratio   = np.where(vol_50 > 0, vol_10 / safe_vol_50, 1.0)

    # ── Price position (Close within rolling price range) ──────────────────
    close_s = pd.Series(close)
    price_pos: dict[str, np.ndarray] = {}
    for w in (10, 20, 50):
        rmin    = close_s.rolling(w).min().values
        rmax    = close_s.rolling(w).max().values
        rng     = rmax - rmin
        safe_rng = np.where(rng > 0, rng, 1.0)
        price_pos[f"PricePosition_{w}"] = np.where(rng > 0, (close - rmin) / safe_rng, 0.5)

    # ── Trend strength (fraction of prior N bars with same sign as current) ─
    pct_vals  = np.where(np.isfinite(pct), pct, 0.0)
    pct_signs = np.sign(pct_vals).astype(np.int8)
    is_up_s   = pd.Series((pct_signs == 1).astype(np.float64))
    is_dn_s   = pd.Series((pct_signs == -1).astype(np.float64))
    trend_str: dict[str, np.ndarray] = {}
    for w in (10, 20):
        prev_up = is_up_s.shift(1).rolling(w).mean().values
        prev_dn = is_dn_s.shift(1).rolling(w).mean().values
        ts = np.where(pct_signs > 0, prev_up,
             np.where(pct_signs < 0, prev_dn, 0.5))
        trend_str[f"TrendStrength_{w}"] = np.where(np.isnan(ts), 0.5, ts)

    # ── Absolute Strength Index (bull/bear power; EMA of up/down % moves) ────
    # Matches futures_price.calculate_absolute_strength: clip the % return into
    # up/down components, then EMA each. Selected into both basic models.
    up_move   = pct_s.clip(lower=0)
    down_move = (-pct_s).clip(lower=0)
    asi: dict[str, np.ndarray] = {}
    for p in (7, 14, 28):
        asi[f"ASI_Bull_{p}"] = up_move.ewm(span=p, adjust=False).mean().to_numpy(dtype=np.float64, na_value=np.nan)
        asi[f"ASI_Bear_{p}"] = down_move.ewm(span=p, adjust=False).mean().to_numpy(dtype=np.float64, na_value=np.nan)

    # ── Core dict ───────────────────────────────────────────────────────────
    arrays: dict[str, np.ndarray] = {
        "Volume":           volume,
        "ATR_14":           atr_14,
        "ATR_14_Pct":       np.where(safe_close > 0, (atr_14 / safe_close) * 100.0, 0.0),
        "RSI_7":            rsi_7,
        "RSI_14":           rsi_14,
        "RSI_28":           rsi_28,
        "MFI_7":            mfi_7,
        "MFI_14":           mfi_14,
        "MFI_28":           mfi_28,
        "RSI_7_x_Volume":   rsi_7 * volume,
        "Momentum_5":       momentum_5,
        "Momentum_10":      momentum_10,
        **pct_lags,
        "EMA_10":           _ema(close, 10),
        "EMA_20":           _ema(close, 20),
        "EMA_50":           _ema(close, 50),
        "BB_10_STD":        bb10_std,
        "BB_20_STD":        bb20_std,
        "BB_50_STD":        bb50_std,
        "BB_50_MA":         bb50_ma,
        "BB_50_1_Upper":    bb50_1_u,
        "BB_50_1_Lower":    bb50_1_l,
        "BB_50_2_Upper":    bb50_2_u,
        "BB_50_2_Lower":    bb50_2_l,
        "BB_50_3_Upper":    bb50_3_u,
        "BB_50_3_Lower":    bb50_3_l,
        "BB_20_1_Upper":    bb20_1_u,
        "BB_20_1_Lower":    bb20_1_l,
        "BB_20_2_Upper":    bb20_2_u,
        "BB_20_2_Lower":    bb20_2_l,
        "BB_20_3_Upper":    bb20_3_u,
        "BB_20_3_Lower":    bb20_3_l,
        "BB_20_1_Upper_vs_High":  rel(bb20_1_u, high,  safe_high),
        "BB_20_1_Upper_vs_Low":   rel(bb20_1_u, low,   safe_low),
        "BB_20_1_Upper_vs_Close": rel(bb20_1_u, close, safe_close),
        "BB_20_1_Lower_vs_High":  rel(bb20_1_l, high,  safe_high),
        "BB_20_1_Lower_vs_Low":   rel(bb20_1_l, low,   safe_low),
        "BB_20_1_Lower_vs_Close": rel(bb20_1_l, close, safe_close),
        "BB_20_2_Upper_vs_High":  rel(bb20_2_u, high,  safe_high),
        "BB_20_2_Upper_vs_Low":   rel(bb20_2_u, low,   safe_low),
        "BB_20_2_Upper_vs_Close": rel(bb20_2_u, close, safe_close),
        "BB_20_2_Lower_vs_High":  rel(bb20_2_l, high,  safe_high),
        "BB_20_2_Lower_vs_Low":   rel(bb20_2_l, low,   safe_low),
        "BB_20_2_Lower_vs_Close": rel(bb20_2_l, close, safe_close),
        "BB_20_3_Upper_vs_High":  rel(bb20_3_u, high,  safe_high),
        "BB_20_3_Upper_vs_Low":   rel(bb20_3_u, low,   safe_low),
        "BB_20_3_Upper_vs_Close": rel(bb20_3_u, close, safe_close),
        "BB_20_3_Lower_vs_Low":   rel(bb20_3_l, low,   safe_low),
        "BB_20_3_Lower_vs_High":  rel(bb20_3_l, high,  safe_high),
        "BB_20_3_Lower_vs_Close": rel(bb20_3_l, close, safe_close),
        "Volatility_10":    vol_10,
        "Volatility_20":    vol_20,
        "Volatility_50":    vol_50,
        "Volatility_Ratio": vol_ratio,
        **rolling,
        **price_pos,
        **trend_str,
        **asi,
    }

    # ── VXM features (volatility regime) ───────────────────────────────────
    # Full set matching the offline pipeline: BB(10/20/50, 1/2/3 std) and EMA on
    # VXM_Close (futures_price.calculate_bollinger_bands / calculate_ema), the
    # BB_20_1-vs-OHLC distances and pct-change lags from prepare_features_and_target,
    # plus the lowercase names the gradient model uses. VXM bars are the same
    # 5-second IBKR stream as ES, so live values match training. Open/High/Low/
    # Volume are used when supplied by the feed; absent fields stay NaN.
    vxm_col = next((c for c in ("VXM_Close", "VXM_CLOSE") if c in df.columns), None)
    _VXM_KEYS = (
        "VXM_Open", "VXM_High", "VXM_Low", "VXM_Close", "VXM_Volume",
        "VXM_BB_10_STD", "VXM_BB_20_STD", "VXM_BB_50_STD",
        "VXM_BB_20_1_Upper", "VXM_BB_20_1_Lower", "VXM_BB_20_2_Upper", "VXM_BB_20_2_Lower",
        "VXM_BB_20_3_Upper", "VXM_BB_20_3_Lower", "VXM_BB_50_MA",
        "VXM_BB_50_1_Upper", "VXM_BB_50_1_Lower", "VXM_BB_50_2_Upper", "VXM_BB_50_2_Lower",
        "VXM_BB_50_3_Upper", "VXM_BB_50_3_Lower", "VXM_EMA_10", "VXM_EMA_20", "VXM_EMA_50",
        "VXM_BB_20_1_Upper_vs_VXM_High", "VXM_BB_20_1_Upper_vs_VXM_Close", "VXM_BB_20_1_Upper_vs_VXM_Low",
        "VXM_BB_20_1_Lower_vs_VXM_Low", "VXM_BB_20_1_Lower_vs_VXM_Close", "VXM_BB_20_1_Lower_vs_VXM_High",
        "VXM_PctChange", "VXM_PctChange_Lag_1", "VXM_PctChange_Lag_2", "VXM_ES_Ratio",
        "vxm_close", "vxm_pct", "vxm_pct_lag1", "vxm_pct_lag2", "vxm_es_ratio",
    )
    if vxm_col is not None:
        vxm_close = df[vxm_col].values.astype(np.float64)
        safe_vxm  = np.where(vxm_close > 0, vxm_close, np.nan)

        def _vxm_field(field: str) -> np.ndarray:
            col = next((c for c in (f"VXM_{field}", f"VXM_{field.upper()}") if c in df.columns), None)
            return df[col].values.astype(np.float64) if col is not None else np.full(n, np.nan, dtype=np.float64)
        vxm_high = _vxm_field("High")
        vxm_low  = _vxm_field("Low")
        safe_vxm_high = np.where(vxm_high > 0, vxm_high, np.nan)
        safe_vxm_low  = np.where(vxm_low  > 0, vxm_low,  np.nan)

        vxm_pct      = np.concatenate([[np.nan], np.diff(vxm_close) / safe_vxm[:-1] * 100.0])
        vxm_pct_lag1 = np.concatenate([[np.nan], vxm_pct[:-1]])
        vxm_pct_lag2 = np.concatenate([[np.nan, np.nan], vxm_pct[:-2]])
        vxm_es_ratio = np.where(safe_close > 0, vxm_close / close, np.nan)

        vbb20_1_u, vbb20_1_l = _bollinger(vxm_close, 20, 1.0)
        vbb20_2_u, vbb20_2_l = _bollinger(vxm_close, 20, 2.0)
        vbb20_3_u, vbb20_3_l = _bollinger(vxm_close, 20, 3.0)
        vbb50_1_u, vbb50_1_l = _bollinger(vxm_close, 50, 1.0)
        vbb50_2_u, vbb50_2_l = _bollinger(vxm_close, 50, 2.0)
        vbb50_3_u, vbb50_3_l = _bollinger(vxm_close, 50, 3.0)

        arrays.update({
            "VXM_Open":   _vxm_field("Open"),
            "VXM_High":   vxm_high,
            "VXM_Low":    vxm_low,
            "VXM_Close":  vxm_close,
            "VXM_Volume": _vxm_field("Volume"),
            "VXM_BB_10_STD": _roll_std(vxm_close, 10),
            "VXM_BB_20_STD": _roll_std(vxm_close, 20),
            "VXM_BB_50_STD": _roll_std(vxm_close, 50),
            "VXM_BB_20_1_Upper": vbb20_1_u, "VXM_BB_20_1_Lower": vbb20_1_l,
            "VXM_BB_20_2_Upper": vbb20_2_u, "VXM_BB_20_2_Lower": vbb20_2_l,
            "VXM_BB_20_3_Upper": vbb20_3_u, "VXM_BB_20_3_Lower": vbb20_3_l,
            "VXM_BB_50_MA":    _roll_mean(vxm_close, 50),
            "VXM_BB_50_1_Upper": vbb50_1_u, "VXM_BB_50_1_Lower": vbb50_1_l,
            "VXM_BB_50_2_Upper": vbb50_2_u, "VXM_BB_50_2_Lower": vbb50_2_l,
            "VXM_BB_50_3_Upper": vbb50_3_u, "VXM_BB_50_3_Lower": vbb50_3_l,
            "VXM_EMA_10": _ema(vxm_close, 10),
            "VXM_EMA_20": _ema(vxm_close, 20),
            "VXM_EMA_50": _ema(vxm_close, 50),
            "VXM_BB_20_1_Upper_vs_VXM_High":  rel(vbb20_1_u, vxm_high,  safe_vxm_high),
            "VXM_BB_20_1_Upper_vs_VXM_Close": rel(vbb20_1_u, vxm_close, safe_vxm),
            "VXM_BB_20_1_Upper_vs_VXM_Low":   rel(vbb20_1_u, vxm_low,   safe_vxm_low),
            "VXM_BB_20_1_Lower_vs_VXM_Low":   rel(vbb20_1_l, vxm_low,   safe_vxm_low),
            "VXM_BB_20_1_Lower_vs_VXM_Close": rel(vbb20_1_l, vxm_close, safe_vxm),
            "VXM_BB_20_1_Lower_vs_VXM_High":  rel(vbb20_1_l, vxm_high,  safe_vxm_high),
            "VXM_PctChange": vxm_pct,
            "VXM_PctChange_Lag_1": vxm_pct_lag1,
            "VXM_PctChange_Lag_2": vxm_pct_lag2,
            "VXM_ES_Ratio": vxm_es_ratio,
            "vxm_close": vxm_close, "vxm_pct": vxm_pct,
            "vxm_pct_lag1": vxm_pct_lag1, "vxm_pct_lag2": vxm_pct_lag2,
            "vxm_es_ratio": vxm_es_ratio,
        })
    else:
        nan_arr = np.full(n, np.nan, dtype=np.float64)
        for key in _VXM_KEYS:
            arrays[key] = nan_arr.copy()

    # ── Time features (optional — NaN when no date column present) ──────────
    date_col = next((c for c in ("date", "DateTime", "Date") if c in df.columns), None)
    if date_col is not None:
        arrays.update(_time_features(df[date_col], n))
    else:
        nan_arr = np.full(n, np.nan, dtype=np.float64)
        for key in ("Hours_From_Formal_Trading", "Hours_From_Overnight_Trading",
                    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"):
            arrays[key] = nan_arr.copy()

    return arrays


def compute_es_features(
    df: pd.DataFrame,
    model_kind: ModelKind = "highest",
    *,
    feature_names: list[str] | None = None,
    csv_path: Path | str | None = None,
    print_time: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute feature matrix for the given model kind. Column order matches that model's CSV.

    Returns:
        X: (n_rows, n_features) float64; early rows may be NaN (indicator warmup).
        feature_names: names for each column, in order.
    """
    t0 = time.perf_counter()
    _validate_ohlcv(df)   # raises ValueError if required OHLCV columns are missing
    arrays = _build_all_feature_arrays(df)
    if feature_names is None:
        path = Path(csv_path) if csv_path is not None else _path_importance_csv(model_kind)
        feature_names = get_feature_names(csv_path=path)
    n = len(df)
    cols: list[np.ndarray] = []
    missing: list[str] = []
    for name in feature_names:
        if name in arrays:
            cols.append(arrays[name])
        else:
            cols.append(np.full(n, np.nan, dtype=np.float64))
            missing.append(name)
    if not cols:
        raise ValueError(
            "Feature name list is empty. "
            "Check model_kind and that the importance CSV is valid."
        )
    if missing:
        print(f"  Warning: {len(missing)} feature(s) not computable, filled with NaN: {missing}")
    X = np.column_stack(cols).astype(np.float64)
    if print_time:
        print(f"Feature generation took {time.perf_counter() - t0:.4f} s ({n} rows, {len(feature_names)} features)")
    return X, list(feature_names)


class ESPredictor:
    """
    Loads one model's feature list and pkl once; reuses them for all predictions.
    Use model_kind="highest" for Close→High, model_kind="lowest" for Close→Low.
    """

    def __init__(
        self,
        model_kind: ModelKind = "highest",
        *,
        importance_csv_path: Path | str | None = None,
        model_path: Path | str | None = None,
    ) -> None:
        if model_kind not in MODEL_KINDS:
            raise ValueError(f"model_kind must be in {MODEL_KINDS}, got {model_kind!r}")
        self._model_kind = model_kind
        csv_path = Path(importance_csv_path) if importance_csv_path is not None else _path_importance_csv(model_kind)
        if not csv_path.exists():
            raise FileNotFoundError(f"Importance CSV not found: {csv_path}")
        imp = pd.read_csv(csv_path)
        if "feature" not in imp.columns:
            raise ValueError(f"CSV must have column 'feature': {csv_path}")
        self._feature_names = imp["feature"].astype(str).tolist()
        pkl_path = Path(model_path) if model_path is not None else _path_model(model_kind)
        if not pkl_path.exists():
            raise FileNotFoundError(f"Model not found: {pkl_path}")
        self._model = joblib.load(pkl_path)

    def compute_features(self, df: pd.DataFrame, *, print_time: bool = True) -> tuple[np.ndarray, list[str]]:
        """Feature matrix and names; uses cached feature list (no CSV read)."""
        return compute_es_features(
            df,
            model_kind=self._model_kind,
            feature_names=self._feature_names,
            print_time=print_time,
        )

    def predict(self, df: pd.DataFrame, *, print_time: bool = True) -> np.ndarray:
        """Per-row predicted pct change (%). NaN where features are invalid (e.g. warmup)."""
        X, _ = self.compute_features(df, print_time=print_time)
        n = X.shape[0]
        valid = ~np.isnan(X).any(axis=1)
        if not np.any(valid):
            return np.full(n, np.nan, dtype=np.float64)
        y = unscale_target(self._model.predict(X[valid]), TARGET_SCALE, SCALING_METHOD)
        out = np.full(n, np.nan, dtype=np.float64)
        out[valid] = y
        return out

    def predict_single(self, df: pd.DataFrame, *, print_time: bool = True) -> tuple[float, float]:
        """
        Single prediction from the last row: (predicted_pct, expected_price).
        For highest, expected_price is expected high; for lowest, expected low.
        """
        X, _ = self.compute_features(df, print_time=print_time)
        last = X[-1:]
        nan_count = int(np.isnan(last).sum())
        if nan_count == last.size:
            raise ValueError(
                "All features are NaN (insufficient history). "
                "Provide at least 50 bars."
            )
        if nan_count > 0:
            # Partial NaN (e.g. VXM absent, date column missing) — LightGBM
            # routes NaN through its default missing-value branch. Predictions
            # will be slightly degraded but will not crash.
            print(f"  Warning: {nan_count}/{last.size} feature(s) are NaN — "
                  "predicting with missing-value passthrough (provide VXM_Close "
                  "and a date column for full accuracy).")
        pct = float(unscale_target(self._model.predict(last), TARGET_SCALE, SCALING_METHOD)[0])
        data = _validate_ohlcv(df)
        close = float(data["Close"].iloc[-1])
        expected = close * (1.0 + pct / 100.0)
        return pct, expected
