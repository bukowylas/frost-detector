"""NOAA ISD packed-field decoding, quality flags and station time zones.

Every ISD numeric field arrives as a comma-packed string ("+0031,1" = 3.1 C,
quality flag 1), with a per-field missing sentinel. Both the pipeline and the
station-probing scripts under ``tests/`` decode these, so the sentinels, the
scale factors and the rejected quality flags are defined once here.

Reject suspect (2, 6) and erroneous (3, 7) quality flags. Flag '9' (likely
"QC not applied") is not listed, but this admits no unchecked data: in this
dataset every flag-'9' temperature/dew-point row also carries the missing
sentinel (+9999), so the sentinel check drops it regardless.
(tests/check_flag9.py checks this holds for the current stations.)
"""

from __future__ import annotations

import math

BAD_QUALITY_FLAGS = {"2", "6", "3", "7"}

# Missing-value sentinels, per ISD field.
MISSING_TEMP = "+9999"      # TMP, DEW
MISSING_SLP = "99999"       # SLP
MISSING_WIND = "9999"       # WND speed

# Fixed Local Standard Time offset (hours from UTC) per station-name prefix.
# Standard, not daylight, so there is no discontinuity mid-spring inside the
# frost window.
LST_OFFSET_HOURS = {"pl_": 1, "uk_": 0}


def packed(value) -> list[str]:
    """Split a packed ISD field into its sub-fields ([] when absent)."""
    return value.split(",") if isinstance(value, str) and value else []


def scaled_num(raw, sentinel: str, scale: float, value_idx: int = 0,
               flag_idx: int = 1) -> float:
    """Decode a packed ISD numeric sub-field to a real value, or NaN."""
    parts = packed(raw)
    if len(parts) <= value_idx:
        return math.nan
    token = parts[value_idx]
    if token == sentinel:
        return math.nan
    if len(parts) > flag_idx and parts[flag_idx] in BAD_QUALITY_FLAGS:
        return math.nan
    try:
        return int(token) / scale
    except ValueError:
        return math.nan


def temp_c(raw) -> float:
    """TMP / DEW in degrees C, or NaN."""
    return scaled_num(raw, MISSING_TEMP, 10.0)


def slp_hpa(raw) -> float:
    """SLP sea-level pressure in hPa, or NaN."""
    return scaled_num(raw, MISSING_SLP, 10.0)


def wind_speed_ms(raw) -> float:
    """WND wind speed in m/s, or NaN."""
    return scaled_num(raw, MISSING_WIND, 10.0, value_idx=3, flag_idx=4)


def cloud_oktas(raw) -> float:
    """GA1 lowest-layer coverage in oktas (0-8), or NaN.

    Clear sky (low oktas) favours radiative cooling, so cloud cover is a frost
    signal.
    """
    parts = packed(raw)
    if not parts:
        return math.nan
    if len(parts) > 1 and parts[1] in BAD_QUALITY_FLAGS:
        return math.nan
    try:
        code = int(parts[0])
    except ValueError:
        return math.nan
    return float(code) if 0 <= code <= 8 else math.nan


def has_temp(raw) -> bool:
    """Whether a TMP/DEW field carries a usable value (coverage counting)."""
    return not math.isnan(temp_c(raw))


def lst_offset(station: str) -> int:
    """Local Standard Time offset for a station name (pl_/uk_ prefix)."""
    for prefix, off in LST_OFFSET_HOURS.items():
        if station.startswith(prefix):
            return off
    raise ValueError(f"no LST offset known for station {station!r}")
