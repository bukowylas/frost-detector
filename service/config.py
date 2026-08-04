"""Single source of truth for the service's operational configuration.

The project's thesis is "one feature builder, one source of truth". These are the
values that would otherwise drift between modules -- the serviceable stations,
their labels, the frost-risk season, and the alarm threshold -- so they live here
and everything imports them.

Serviceability is asserted against the live provider at import: a serviceable
station that the provider does not know about is a configuration error that must
fail loudly, not a station quietly dropped or (worse) a fallback that serves
every station the provider has, including those the parity gate rejected.
"""

from __future__ import annotations

from frostlib import live, season

# The stations the service forecasts for. An explicit literal -- NOT derived by
# filtering, so it can never silently widen to "every provider station" on a
# config mismatch. These are exactly the stations Stage 2's parity gate proved
# reproduce the training features.
SERVICEABLE_STATIONS: tuple[str, ...] = ("uk_waddington", "uk_cranwell")

# Human-readable labels for the UI dropdown and the SMS messages.
STATION_LABELS: dict[str, str] = {
    "uk_waddington": "Waddington",
    "uk_cranwell": "Cranwell",
}

# What the MODEL covers: the frost-risk windows it was trained on (spring AND
# autumn), shared from frostlib.season.
MODEL_RISK_WINDOWS = season.RISK_WINDOWS

# What the SERVICE currently serves. Deliberately narrower than the model can
# handle: this build runs spring only, to keep the operational surface small. The
# model is trained on autumn too and reproduces it fine, so enabling autumn is a
# one-line change -- set this to ``season.RISK_WINDOWS``. It must always be a
# SUBSET of what the model covers (asserted below), so the service can never be
# configured to forecast a season the model was not trained on.
SERVICE_RISK_WINDOWS: list[tuple[tuple[int, int], tuple[int, int]]] = [
    ((3, 1), (5, 31)),     # spring: 1 March - 31 May
]

# The recommended alarm threshold (predicted tmin at/below this fires an alarm).
# An operational parameter, kept out of the training module so the service does
# not reach into training code for it.
RECOMMENDED_ALARM_C: float = 1.5


def station_label(station: str) -> str:
    return STATION_LABELS.get(station, station)


def _in_windows(date, windows) -> bool:
    import pandas as pd
    d = pd.Timestamp(date)
    md = (d.month, d.day)
    return any(lo <= md <= hi for lo, hi in windows)


def service_in_season(date) -> bool:
    """Whether the SERVICE forecasts on this evening date (its current windows)."""
    return _in_windows(date, SERVICE_RISK_WINDOWS)


def model_covers(date) -> bool:
    """Whether the MODEL was trained for this evening date (spring or autumn)."""
    return _in_windows(date, MODEL_RISK_WINDOWS)


def _assert_config_coherent() -> None:
    missing = [s for s in SERVICEABLE_STATIONS if s not in live.PROVIDER]
    if missing:
        raise RuntimeError(
            f"serviceable stations absent from the live provider: {missing}. "
            "A serviceable station must have a verified provider entry -- refusing "
            "to start rather than serve a station the parity gate never validated.")
    # The service must never forecast a season the model was not trained on.
    extra = [w for w in SERVICE_RISK_WINDOWS if w not in MODEL_RISK_WINDOWS]
    if extra:
        raise RuntimeError(
            f"service risk windows not covered by the model: {extra}. The service "
            "may serve fewer seasons than the model, never more.")


_assert_config_coherent()
