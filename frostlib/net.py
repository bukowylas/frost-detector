"""NCEI access URLs and HTTP helpers (requests only, no pandas).

Both ``fetch_data.py`` and the station-probing scripts under ``tests/`` hit the
same public NCEI "global-hourly" service, with the same User-Agent and the same
URL shape; those live here so a service move or a UA change is a one-line edit.
"""

from __future__ import annotations

import csv as csvmod
import io

import requests

GLOBAL_HOURLY_BASE = "https://www.ncei.noaa.gov/data/global-hourly/access"
ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"

USER_AGENT = "frost-detector/0.1 (research/portfolio; contact via README)"
HEADERS = {"User-Agent": USER_AGENT}

TIMEOUT_SECONDS = 120.0


def station_year_url(station_id: str, year: int) -> str:
    """The global-hourly access URL for one station-year CSV."""
    return f"{GLOBAL_HOURLY_BASE}/{year}/{station_id}.csv"


def get(url: str, timeout: float = TIMEOUT_SECONDS,
        session: requests.Session | None = None) -> requests.Response:
    """GET with the project User-Agent (on a caller-supplied session if given)."""
    client = session if session is not None else requests
    return client.get(url, headers=HEADERS, timeout=timeout)


def parse_csv_rows(text: str) -> list[dict]:
    """Parse an ISD CSV body into dict rows."""
    return list(csvmod.DictReader(io.StringIO(text)))


def fetch_station_year_rows(station_id: str, year: int,
                            timeout: float = TIMEOUT_SECONDS) -> list[dict]:
    """Download one station-year and return its rows.

    Raises ``requests.HTTPError`` on a non-200, so probing callers can report the
    status without every one of them re-implementing the check.
    """
    resp = get(station_year_url(station_id, year), timeout=timeout)
    resp.raise_for_status()
    return parse_csv_rows(resp.text)
