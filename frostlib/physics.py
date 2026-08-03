"""Frost feature formulas shared by the prepare and predict steps.

``prepare.py`` computes these features for every training night and
``predict.py`` recomputes them for a single evening at forecast time. If the two
implementations drifted the model would be served inputs it was never trained
on -- a silent accuracy loss -- so both call these functions.
"""

from __future__ import annotations

import math

# Fallbacks when an evening observation lacks cloud or wind: mid cloud (4 of 8
# oktas) and a light breeze. Chosen so a missing field cannot masquerade as the
# extreme (perfectly clear / dead calm) that maximises frost risk.
DEFAULT_CLOUD_OKTAS = 4.0
DEFAULT_WIND_MS = 3.0

MAX_OKTAS = 8.0


def _or_default(value, default: float) -> float:
    if value is None:
        return default
    value = float(value)
    return default if math.isnan(value) else value


def dewpoint_depression_c(temp_c: float, dewpoint_c: float) -> float:
    """Temperature minus dew point: how far the air must cool to saturate."""
    return float(temp_c) - float(dewpoint_c)


def clear_sky_fraction(cloud_oktas) -> float:
    """1 = cloudless, 0 = overcast (missing cloud -> mid cover)."""
    return 1.0 - _or_default(cloud_oktas, DEFAULT_CLOUD_OKTAS) / MAX_OKTAS


def radiative_potential(cloud_oktas, wind_ms) -> float:
    """Radiative-cooling potential = (clear fraction) / (1 + wind).

    Clear and calm favours strong nocturnal radiative cooling -- the dominant
    driver of orchard frost -- expressed as one number. Missing inputs fall back
    to DEFAULT_CLOUD_OKTAS / DEFAULT_WIND_MS.
    """
    wind = _or_default(wind_ms, DEFAULT_WIND_MS)
    return clear_sky_fraction(cloud_oktas) / (1.0 + wind)


# --- column-wise variants, for scoring a whole nights frame at once ----------


def clear_sky_fraction_col(cloud_oktas):
    """clear_sky_fraction over a pandas Series."""
    return 1.0 - cloud_oktas.fillna(DEFAULT_CLOUD_OKTAS) / MAX_OKTAS


def radiative_potential_col(cloud_oktas, wind_ms):
    """radiative_potential over pandas Series (same formula, vectorised)."""
    return clear_sky_fraction_col(cloud_oktas) / (
        1.0 + wind_ms.fillna(DEFAULT_WIND_MS))
