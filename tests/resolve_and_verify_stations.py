"""Resolve REAL UK+PL station ids from isd-history, then verify coverage.

No hand-guessing ids. Load the authoritative history file, pick inland stations
in the right latitude bands, use the EXACT USAF+WBAN string, probe each for a
real 2023 file, and report TMP/DEW coverage + spring frost rate.

Clear, flushed progress on every step so silence never looks like a hang.

Usage:
    python3 tests/resolve_and_verify_stations.py --limit 2   # small test chunk
    python3 tests/resolve_and_verify_stations.py             # full (8 per country)
"""

import argparse

import requests
from _common import FROST_C, coverage, month_rows, temperatures

from frostlib import isd_history, net
from frostlib.progress import Timer

YEAR = 2023
SPRING_MONTHS = ("04", "05")

_timer = Timer()
log = _timer.log


def probe(usaf, wban, name):
    sid = f"{usaf}{wban}"
    try:
        resp = net.get(net.station_year_url(sid, YEAR), timeout=30)
    except requests.RequestException as exc:
        log(f"    {sid} {name}: request error {exc}")
        return None
    if resp.status_code != 200:
        log(f"    {sid} {name}: HTTP {resp.status_code}")
        return None
    rows = net.parse_csv_rows(resp.text)
    if not rows:
        log(f"    {sid} {name}: empty")
        return None
    n = len(rows)
    spring = month_rows(rows, SPRING_MONTHS)
    frost = sum(1 for t in temperatures(spring) if t <= FROST_C)
    size_mb = len(resp.content) / 1e6
    log(f"    {sid} {name:22s} OK rows={n:5d} ({size_mb:.1f}MB) "
        f"TMP={100*coverage(rows, 'TMP')/n:3.0f}% "
        f"DEW={100*coverage(rows, 'DEW')/n:3.0f}% "
        f"springFrost={100*frost/max(len(spring),1):4.1f}%")
    return sid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8,
                    help="max stations to probe per country")
    args = ap.parse_args()

    log("downloading isd-history.csv (a few MB) ...")
    hist = isd_history.load_isd_history()
    active = isd_history.active_stations(hist, active_through=20231201)
    log(f"parsed: {len(hist)} stations, {len(active)} active through 2023")

    for country, lat_lo, lat_hi, lon_lo, lon_hi in [
        ("PL", 50.0, 53.0, 16.0, 23.0),
        ("UK", 51.0, 53.5, -2.5, 1.8),
    ]:
        sub = active[(active["CTRY"] == country)
                     & active["LAT"].between(lat_lo, lat_hi)
                     & active["LON"].between(lon_lo, lon_hi)]
        log(f"{country}: {len(sub)} candidate stations; probing up to {args.limit}")
        found = 0
        for _, row in sub.iterrows():
            if found >= args.limit:
                break
            name = str(row["STATION NAME"])[:22]
            log(f"  probing {country} {found+1}/{args.limit}: {name} ...")
            if probe(row["USAF"], row["WBAN"], name):
                found += 1
    log("done.")


if __name__ == "__main__":
    main()
