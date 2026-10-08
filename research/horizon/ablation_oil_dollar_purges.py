"""Does the oil and dollar verdict survive the purge being set correctly?

`ablation_new_columns.py` is why the USO columns are in the model: it removed them and the
score fell, t +2.46. But it ran a five-bar horizon against `rotation_test`'s hardcoded
twelve-bar purge. Over-purging does not leak, so the number is not wrong -- but the purge
turned out to move base AUC by 0.0017, which is the same size as the effect, so the verdict
might be an artefact of a gap nobody chose deliberately.

Both purges are run here in one process, on the same seeds and folds, so the two verdicts
can be read side by side rather than across runs. If the sign and significance hold at
purge 5, the columns stay. If they only appear at purge 12, they were never earned.

usage: ablation_oil_dollar_purges.py
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from purged_folds import purged_fold_splits
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS, train_direction_model

HORIZON_BARS = 5
ORIGINAL_PURGE_BARS = 12
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)
RESULTS_PATH = "ablation_oil_dollar_purges.json"


def fold_scores(X, target, decided, splits, params) -> list[float]:
    scores = []
    for split in splits:
        model = train_direction_model(X, target, split, params)
        test = split.test[decided[split.test]]
        scores.append(roc_auc_score((target[test] > 0).astype(int), model.predict_proba(X[test])[:, 1]))
    return scores


def paired_t(differences: np.ndarray) -> tuple[float, float]:
    error = differences.std(ddof=1) / np.sqrt(len(differences))
    return differences.mean(), differences.mean() / error if error > 0 else 0.0


def main() -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    target = average_target(data["mid"].to_numpy(float), epochs, HORIZON_BARS)
    decided = np.isfinite(target) & (target != 0)

    full = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    oil = [name for name in full if "uso_minus_mid" in name]
    dollar = [name for name in full if "uup_minus_mid" in name]
    sets = {"full": full,
            "no_new": [name for name in full if name not in oil + dollar],
            "no_uso": [name for name in full if name not in oil],
            "no_uup": [name for name in full if name not in dollar]}
    matrices = {name: data[columns].to_numpy(np.float32) for name, columns in sets.items()}
    print(f"{len(oil)} oil and {len(dollar)} dollar columns inside {len(full)}; horizon "
          f"{HORIZON_BARS}; purges {HORIZON_BARS} and {ORIGINAL_PURGE_BARS}; {len(SEEDS)} seeds "
          f"x 2 folds ({time.time() - started:.0f}s to build)\n", flush=True)

    scores: dict[str, list[float]] = {}
    for purge in (HORIZON_BARS, ORIGINAL_PURGE_BARS):
        splits = purged_fold_splits(data["time"], HORIZON_BARS, purge_bars=purge)
        for seed in SEEDS:
            params = {**LGBM_PARAMS, "random_state": seed}
            for name, X in matrices.items():
                scores.setdefault(f"{name}_p{purge}", []).extend(
                    fold_scores(X, target, decided, splits, params))
        print(f"   purge {purge}: " + "  ".join(f"{name} {np.mean(scores[f'{name}_p{purge}']):.4f}"
                                                for name in sets)
              + f"   ({time.time() - started:.0f}s)", flush=True)
        with open(RESULTS_PATH, "w") as handle:
            json.dump({name: list(map(float, values)) for name, values in scores.items()}, handle, indent=1)

    print(f"\n{'comparison':<22}" + "".join(f"{'purge ' + str(purge):>26}"
                                            for purge in (HORIZON_BARS, ORIGINAL_PURGE_BARS)))
    print(f"{'full mean AUC':<22}" + "".join(f"{np.mean(scores[f'full_p{purge}']):>26.4f}"
                                             for purge in (HORIZON_BARS, ORIGINAL_PURGE_BARS)))
    for removed in ("no_new", "no_uso", "no_uup"):
        cells = []
        for purge in (HORIZON_BARS, ORIGINAL_PURGE_BARS):
            mean, t = paired_t(np.array(scores[f"full_p{purge}"]) - np.array(scores[f"{removed}_p{purge}"]))
            cells.append(f"{mean:>+12.5f}  t{t:>6.2f}")
        print(f"{'full vs ' + removed:<22}" + "".join(f"{cell:>26}" for cell in cells))
    print("\nPositive means the columns earn their place. The purge-5 column is the one that counts.")


if __name__ == "__main__":
    main()
