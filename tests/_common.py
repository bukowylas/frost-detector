"""Shared helpers for the exploratory scripts in this directory.

These scripts are one-off checks (does a station exist? does it freeze? is a
quality flag safe?), and they all did the same three things: put the repository
root on sys.path, download a station-year from NCEI, and print a frost base-rate
line. Importing this module does the first; the helpers below do the rest.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Imported after the sys.path bootstrap above, so frostlib resolves.
import requests

from frostlib import isd, net

# Temperature thresholds these checks report on: frost, and the frost-warning
# level a grower acts at.
FROST_C = 0.0
WARN_C = 2.0


def temperatures(rows: list[dict], field: str = "TMP") -> list[float]:
    """Every decodable temperature in the rows (missing/suspect values dropped)."""
    return [isd.temp_c(row.get(field, "")) for row in rows
            if isd.has_temp(row.get(field, ""))]


def coverage(rows: list[dict], field: str) -> int:
    """How many rows carry a usable value for a TMP/DEW field."""
    return sum(1 for row in rows if isd.has_temp(row.get(field, "")))


def month_rows(rows: list[dict], months: tuple[str, ...]) -> list[dict]:
    """Rows whose DATE falls in the given zero-padded month strings."""
    return [row for row in rows if row.get("DATE", "")[5:7] in months]


def frost_line(temps: list[float]) -> str:
    """The shared `rows=... <=0C ...% <=2C ...% min=...` summary."""
    frost = sum(1 for t in temps if t <= FROST_C)
    warn = sum(1 for t in temps if t <= WARN_C)
    return (f"rows={len(temps):6d}  <=0C {100 * frost / len(temps):5.1f}%  "
            f"<=2C {100 * warn / len(temps):5.1f}%  min={min(temps):6.1f}C")


def cloud_line(rows: list[dict]) -> str:
    """GA1 (cloud) availability -- clear sky matters for radiative frost."""
    ga1 = sum(1 for row in rows if row.get("GA1"))
    return f"GA1={100 * ga1 / max(len(rows), 1):3.0f}%"


def report_station_frost(name: str, station_id: str, year: int, width: int = 26,
                         timeout: float = 60.0, with_cloud: bool = False,
                         suffix: str = "") -> None:
    """Download one station-year and print its frost base rate (or why not).

    The candidate-checking scripts differ only in their station lists, so the
    fetch / decode / report / error-handling loop lives here.
    """
    label = f"{name:{width}s}"
    try:
        rows = net.fetch_station_year_rows(station_id, year, timeout=timeout)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        print(f"{label} HTTP {status}{suffix}")
        return
    except requests.RequestException as exc:
        print(f"{label} ERROR {exc}{suffix}")
        return
    temps = temperatures(rows)
    if not temps:
        print(f"{label} resolved but no temp data{suffix}")
        return
    extra = f"  {cloud_line(rows)}" if with_cloud else ""
    print(f"{label} {frost_line(temps)}{extra}{suffix}")
