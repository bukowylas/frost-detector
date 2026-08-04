"""The frost-risk season -- the one definition the training and service paths share.

The model is trained only on nights inside these windows (outside them, day-of-year
alone would "solve" the task and the score would mean nothing). These define what
the MODEL covers. The live service must never forecast OUTSIDE them (that is
extrapolation), but it MAY serve a subset -- see ``service.config``, where this
build deliberately serves spring only while the model covers spring and autumn.
Keeping the model's windows in one place is why ``prepare`` (which builds the
training rows) and the service cannot drift about what the model was trained on.
"""

from __future__ import annotations

# Inclusive (month, day) ranges. Spring and autumn are BOTH frost-risk seasons for
# growers -- spring threatens blossom, autumn threatens unharvested fruit on the
# tree -- and training on both measurably improves accuracy over spring alone.
RISK_WINDOWS: list[tuple[tuple[int, int], tuple[int, int]]] = [
    ((3, 1), (5, 31)),     # spring: 1 March - 31 May
    ((9, 15), (11, 15)),   # autumn: 15 September - 15 November
]


def in_risk_window(month: int, day: int) -> bool:
    """Whether a (month, day) falls in any frost-risk window."""
    return any(lo <= (month, day) <= hi for lo, hi in RISK_WINDOWS)
