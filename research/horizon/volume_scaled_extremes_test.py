"""The highest, lowest, average and range targets, each divided by the volume that made it.

The move-per-volume idea applied to the other four things a sub-model could predict.
Each is measured twice: how much of its own variance the model explains, and -- for the
two targets that carry a direction -- how well the prediction ranks which way the market
went. Regular trading hours only, five bars ahead, dev folds, fitted inside each fold.

The extremes come from traded High and Low, as production defines them, so unlike the
mid-based targets they carry some bid-ask bounce; the comparison between scaled and
unscaled is unaffected, since both sides carry it equally.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score

from forward_extremes import forward_target
from rotation_test import fold_splits
from segment_walk_forward import load_cut
from short_horizon_walk_forward import FEATURES
from volume_scaled_target import forward_volume, regular_hours
from volume_target_test import HORIZON_BARS, STAGE, fit_regressor

TARGET_KINDS = ("highest", "lowest", "average", "range")
DIRECTIONAL = ("average",)


def main() -> None:
    data = load_cut()
    data = data[regular_hours(data["time"])].reset_index(drop=True)
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    prices = {"seconds": seconds, "close": data["Close"].to_numpy(float),
              "high": data["High"].to_numpy(float), "low": data["Low"].to_numpy(float)}
    volume = forward_volume(data["Volume"].to_numpy(float), HORIZON_BARS)
    X = data[FEATURES].to_numpy(np.float32)
    print(f"regular hours only: {len(data):,} bars over {data['session_day'].nunique()} days\n")
    print(f"{'target':<28}{'R2':>9}{'R2 / volume':>14}{'AUC':>9}{'AUC / volume':>14}")
    results = {}
    for kind in TARGET_KINDS:
        raw = forward_target(prices, HORIZON_BARS, kind)
        versions = {"plain": raw, "scaled": raw / volume}
        scores = {name: {"r2": [], "auc": []} for name in versions}
        decided = np.isfinite(raw) & (raw != 0)
        for split in fold_splits(data, STAGE):
            for name, target in versions.items():
                model = fit_regressor(X, target, split)
                test = split.test[np.isfinite(target[split.test])]
                scores[name]["r2"].append(r2_score(target[test], model.predict(X[test])))
                if kind in DIRECTIONAL:
                    ranked = split.test[decided[split.test]]
                    scores[name]["auc"].append(
                        roc_auc_score((raw[ranked] > 0).astype(int), model.predict(X[ranked])))
        results[kind] = {name: {key: float(np.mean(values)) if values else None
                                for key, values in score.items()} for name, score in scores.items()}
        auc = results[kind]["plain"]["auc"]
        scaled_auc = results[kind]["scaled"]["auc"]
        print(f"{kind:<28}{results[kind]['plain']['r2']:>9.4f}{results[kind]['scaled']['r2']:>14.4f}"
              f"{(f'{auc:.4f}' if auc else '-'):>9}{(f'{scaled_auc:.4f}' if scaled_auc else '-'):>14}",
              flush=True)
    with open("volume_scaled_extremes_test.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
