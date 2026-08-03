"""NOAA's ISD station-history file: the authoritative station list.

Several exploratory scripts resolve station ids from this file rather than
guessing USAF+WBAN pairs. Downloading it, coercing the numeric columns and
filtering to still-active stations is the same work every time.
"""

from __future__ import annotations

import io

import pandas as pd

from . import net

# Stations whose record ends before this date are no longer reporting.
DEFAULT_ACTIVE_THROUGH = 20231231


def load_isd_history(timeout: float = net.TIMEOUT_SECONDS) -> pd.DataFrame:
    """Download isd-history.csv with END/LAT/LON coerced to numbers."""
    resp = net.get(net.ISD_HISTORY_URL, timeout=timeout)
    resp.raise_for_status()
    hist = pd.read_csv(io.StringIO(resp.text), dtype=str)
    for col in ("END", "LAT", "LON"):
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
    return hist


def active_stations(hist: pd.DataFrame, country: str | None = None,
                    active_through: int = DEFAULT_ACTIVE_THROUGH) -> pd.DataFrame:
    """Stations still reporting at ``active_through``, optionally one country.

    ``country`` is an ISD FIPS code (UK, FR, PL for Poland, GM for Germany).
    """
    sub = hist[hist["END"] >= active_through]
    if country is not None:
        sub = sub[sub["CTRY"] == country]
    return sub


def station_id(row) -> str:
    """The USAF+WBAN id in the exact form the access URL expects."""
    return f"{row['USAF']}{row['WBAN']}"
