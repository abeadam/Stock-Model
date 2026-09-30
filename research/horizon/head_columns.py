"""Which columns each head gets, and the evidence for the difference.

The harness fed one matrix to both heads, so every finding had to be applied to both or
neither. That is not what the measurements say. Two of them point in opposite directions:

**Long realized volatility helps the volatility head and does nothing for direction.**
Over 13 monthly folds and 315 sessions, `realized_vol_60/180` lift the volatility head
+0.0191 R2 (t 6.6), gaining in 12 of 13 folds, while direction moves -0.00016 (t -0.5)
with sign agreement 7/13. Giving them to direction would add two columns it has no use
for. See [realized_volatility_features].

**The oil and dollar divergence columns are not supported at all.** Their place in the
set rests on `ablation_new_columns`, which reported t +2.46 for the oil group -- but it
imported `fold_splits` from `rotation_test`, which hardcodes a 12-bar purge, while
running a 5-bar horizon. Re-run with the purge tied to the horizon, in one process,
8 seeds x 2 folds, the verdict flips: no_uso t -1.40, no_uup t -1.38, both negative and
neither significant. Nothing survives to justify carrying them.

That ablation measured **direction only**, so dropping them from the volatility head is
a judgment rather than a measurement: there is no evidence they help either head, and
their sole justification was a direction result that did not survive its own purge. If
that judgement is ever worth revisiting, `DROPPED_FEATURES` is the one place to change.
"""
from __future__ import annotations

from dataclasses import dataclass

from realized_volatility_features import REALIZED_VOLATILITY_FEATURES

# Three oil and three dollar divergence columns, named for their lookback in bars.
DROPPED_FEATURES = ("ms_uso_minus_mid_3", "ms_uso_minus_mid_6", "ms_uso_minus_mid_12",
                    "ms_uup_minus_mid_1", "ms_uup_minus_mid_3", "ms_uup_minus_mid_12")
VOLATILITY_ONLY_FEATURES = tuple(REALIZED_VOLATILITY_FEATURES)


@dataclass(frozen=True)
class HeadColumns:
    """The column list for each head, plus how many were appended to each.

    `added` counts trailing columns so a caller can read their share of tree gain, which
    is what separates a column the model ignores from one it uses and gains nothing from.
    """
    direction: list[str]
    volatility: list[str]
    direction_added: int = 0
    volatility_added: int = 0


def split_by_head(base: list[str], volatility_only: tuple[str, ...] = VOLATILITY_ONLY_FEATURES,
                  dropped: tuple[str, ...] = DROPPED_FEATURES) -> HeadColumns:
    """Both heads lose `dropped`; only the volatility head gains `volatility_only`."""
    kept = [name for name in base if name not in dropped]
    extra = [name for name in volatility_only if name not in kept]
    return HeadColumns(direction=kept, volatility=kept + extra, volatility_added=len(extra))
