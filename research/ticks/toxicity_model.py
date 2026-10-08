"""Can a toxic fill be spotted before the order is placed?

The decomposition says the loss is 84% pre-fill -- $9.45 given up while waiting against
$1.82 after the fill -- and splits cleanly by mechanism:

    queue cleared    54% of fills   +$3.68 at 30s
    traded through   46% of fills  -$12.35 at 30s

So the question is not which fills to keep but whether to rest at all, and the target is
whether the fill that follows will come from the market trading through the level. A $16
gap between two nearly equal groups is a lot to aim at.

Every feature is read at the **placement** second. Nothing from the fill itself is used,
including the size of the print that filled -- that arrives too late to inform a decision
not to place, and flattening after a bad fill pays the spread again.

Walk-forward by session through `session_folds`, whose tests assert no fold trains on a
later session; that exact leak cost +0.0051 AUC in `tick_walk_forward.py`.

Step 5's bar: the intended edge measured at placement is **$7.30**, not $6.25, because the
spread is wider than a tick often enough to matter. A filter only counts if the retained
fills turn net positive while keeping a usable share.

usage: toxicity_model.py [DECISIONS_PER_SIDE]
"""
from __future__ import annotations

import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fill_features import FILL_FEATURES, at_placement, session_frame
from markout import BUY, POINT_VALUE, SELL, fresh_seconds, mid_at, queue_ahead, touch_price
from markout_study import CLEAN_SESSIONS, load_book
from session_folds import session_folds
from tick_probe import load_day

DECISIONS_PER_SIDE = 3000
ENTRY_LIMIT_SECONDS = 30.0
HOLD_SECONDS = 30
SEED = 11
KEEP_QUANTILES = (0.3, 0.5, 0.7)
LGBM_PARAMS = dict(objective="binary", num_leaves=31, max_depth=5, learning_rate=0.05,
                   n_estimators=600, min_child_samples=200, colsample_bytree=0.8,
                   subsample=0.8, subsample_freq=1, reg_lambda=10.0, n_jobs=-1,
                   verbose=-1, random_state=SEED)
RESULTS_PATH = "/Users/abeadam/dev/model/Stock-Model/scratchpad_backup/toxicity_model.json"


def fill_with_reason(book, placed_at, price, side, ahead, limit):
    """Fill time and mechanism: 'through' when the market trades past us, else 'queue'."""
    deadline = placed_at + limit
    for row in range(book.first_trade_row(placed_at), len(book.trade_times)):
        moment, traded, size = book.trade_times[row], book.trade_prices[row], book.trade_sizes[row]
        if moment > deadline:
            return None
        if (traded < price) if side > 0 else (traded > price):
            return float(moment), "through"
        if traded == price:
            ahead -= size
            if ahead <= 0:
                return float(moment), "queue"
    return None


def session_fills(day: str, decisions_per_side: int, seed: int) -> pd.DataFrame:
    """Every simulated fill of one session, with placement-time features and its outcome."""
    book, grid = load_book(day), load_day(day)
    fresh, frame = fresh_seconds(grid), session_frame(grid)
    pool = np.array(sorted(fresh))
    generator = np.random.default_rng(seed)
    rows = []
    for side in (BUY, SELL):
        for place in generator.choice(pool, size=min(decisions_per_side, len(pool)), replace=False):
            price, ahead = touch_price(book, float(place), side), queue_ahead(book, float(place), side)
            if not np.isfinite(price) or not np.isfinite(ahead):
                continue
            features = at_placement(frame, int(place), side)
            if features is None or not all(np.isfinite(list(features.values()))):
                continue
            got = fill_with_reason(book, float(place), price, side, ahead, ENTRY_LIMIT_SECONDS)
            if got is None:
                continue
            fill_time, reason = got
            if int(fill_time) not in fresh or int(place) not in fresh:
                continue
            mid_place, mid_fill = mid_at(book, float(place)), mid_at(book, fill_time)
            later = int(fill_time) + HOLD_SECONDS
            if later not in fresh:
                continue
            mid_later = mid_at(book, float(later))
            if not all(np.isfinite([mid_place, mid_fill, mid_later])):
                continue
            rows.append({"day": day, "side": side, "toxic": int(reason == "through"),
                         "intended": side * (mid_place - price) * POINT_VALUE,
                         "total": side * (mid_later - price) * POINT_VALUE, **features})
    return pd.DataFrame(rows)


def main(decisions_per_side: int = DECISIONS_PER_SIDE) -> None:
    started = time.time()
    frames = []
    for index, day in enumerate(CLEAN_SESSIONS):
        frames.append(session_fills(day, decisions_per_side, SEED + index))
        print(f"{day}: {len(frames[-1]):,} fills ({time.time() - started:.0f}s)", flush=True)
    fills = pd.concat(frames, ignore_index=True)
    print(f"\n{len(fills):,} fills, {fills['toxic'].mean():.1%} toxic, "
          f"intended edge ${fills['intended'].mean():.2f}, total ${fills['total'].mean():.2f}\n")

    folds = session_folds(sorted(fills["day"].unique()))
    per_session, kept_rows = [], []
    for fold in folds:
        train = fills[fills["day"].isin(fold.train)]
        validation = fills[fills["day"] == fold.validation]
        test = fills[fills["day"] == fold.test]
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(train[FILL_FEATURES], train["toxic"],
                  eval_set=[(validation[FILL_FEATURES], validation["toxic"])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        score = model.predict_proba(test[FILL_FEATURES])[:, 1]
        auc = roc_auc_score(test["toxic"], score)
        row = {"session": fold.test, "fills": len(test), "toxic": test["toxic"].mean(),
               "auc": auc, "all_total": test["total"].mean()}
        for quantile in KEEP_QUANTILES:
            cut = np.quantile(score, quantile)
            keep = test[score <= cut]
            row[f"keep_{quantile}"] = keep["total"].mean() if len(keep) else float("nan")
            row[f"toxic_{quantile}"] = keep["toxic"].mean() if len(keep) else float("nan")
        per_session.append(row)
        kept_rows.append(test.assign(score=score))
        print(f"   {fold.test}: {len(test):,} fills, AUC {auc:.4f}, all ${row['all_total']:+.2f}, "
              f"keep-30% ${row['keep_0.3']:+.2f}   ({time.time() - started:.0f}s)", flush=True)

    table = pd.DataFrame(per_session)
    print(f"\nSTEP 4 - per session, trained only on earlier sessions")
    print(f"  {'session':<13}{'fills':>8}{'toxic':>8}{'AUC':>8}{'all $':>9}"
          + "".join(f"{'keep ' + str(int(q * 100)) + '%':>11}" for q in KEEP_QUANTILES))
    for row in per_session:
        print(f"  {row['session']:<13}{row['fills']:>8,}{row['toxic']:>8.1%}{row['auc']:>8.4f}"
              f"{row['all_total']:>9.2f}"
              + "".join(f"{row[f'keep_{q}']:>11.2f}" for q in KEEP_QUANTILES))
    print(f"  {'mean':<13}{table['fills'].mean():>8,.0f}{table['toxic'].mean():>8.1%}"
          f"{table['auc'].mean():>8.4f}{table['all_total'].mean():>9.2f}"
          + "".join(f"{table[f'keep_{q}'].mean():>11.2f}" for q in KEEP_QUANTILES))

    everything = pd.concat(kept_rows, ignore_index=True)
    print(f"\nSTEP 5 - the bar: intended edge ${everything['intended'].mean():.2f} a fill")
    print(f"  {'filter':<14}{'fills kept':>12}{'share':>8}{'toxic':>8}{'net $/fill':>12}")
    print(f"  {'none':<14}{len(everything):>12,}{1.0:>8.1%}"
          f"{everything['toxic'].mean():>8.1%}{everything['total'].mean():>12.2f}")
    for quantile in KEEP_QUANTILES:
        cut = everything.groupby("day")["score"].transform(lambda s: s.quantile(quantile))
        keep = everything[everything["score"] <= cut]
        print(f"  {f'keep {int(quantile * 100)}%':<14}{len(keep):>12,}"
              f"{len(keep) / len(everything):>8.1%}{keep['toxic'].mean():>8.1%}"
              f"{keep['total'].mean():>12.2f}")
    with open(RESULTS_PATH, "w") as handle:
        json.dump(per_session, handle, indent=1, default=float)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else DECISIONS_PER_SIDE)
