"""Is the model blind to the indicator setups, or does it see them and rate them small?

The production model is scored across every bar, and a band touch with an MFI extreme
happens on a few percent of them. This scores the same model twice on the same fold:
on all test bars, and on only the bars where the classic rule fires.
"""
from __future__ import annotations

import numpy as np

from indicator_rules import MFI_OVERBOUGHT, MFI_OVERSOLD
from segment_walk_forward import DATA_END_UTC, FOLD_STARTS_UTC, FOLDS_BY_STAGE, load_cut, mid_move
from short_horizon_walk_forward import FEATURES, direction_auc, fold_split, train_direction_model

TARGET_BARS = 12


def rule_fires(data) -> np.ndarray:
    touch_low = (data["bb_lower_dist"].to_numpy() < 0) & (data["MFI_14"].to_numpy() < MFI_OVERSOLD)
    touch_high = (data["bb_upper_dist"].to_numpy() > 0) & (data["MFI_14"].to_numpy() > MFI_OVERBOUGHT)
    return touch_low | touch_high


def main() -> None:
    data = load_cut()
    X = data[FEATURES].to_numpy(np.float32)
    move = mid_move(data, TARGET_BARS)
    fires = rule_fires(data)
    print(f"the rule fires on {fires.mean():.1%} of bars")
    for fold in FOLDS_BY_STAGE["dev"]:
        split = fold_split(data["time"], fold, purge_bars=TARGET_BARS,
                           fold_starts=FOLD_STARTS_UTC, data_end=DATA_END_UTC)
        model = train_direction_model(X, move, split)
        score = model.predict_proba(X[split.test])[:, 1]
        on_signal = fires[split.test]
        print(f"  fold {fold}: AUC all bars {direction_auc(score, move[split.test]):.4f}   "
              f"AUC on signal bars {direction_auc(score[on_signal], move[split.test][on_signal]):.4f}   "
              f"({on_signal.sum():,} of {len(on_signal):,} test bars)")


if __name__ == "__main__":
    main()
