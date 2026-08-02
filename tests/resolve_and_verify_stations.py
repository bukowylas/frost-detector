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
import io
import csv as csvmod
import sys
import time

import pandas as pd
import requests

UA = {"User-Agent": "frost-detector/0.1 (research; contact via README)"}
t0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - t0:5.1f}s] {msg}", flush=True)


def temp_ok(raw: str) -> bool:
    p = raw.split(",") if raw else []
    return len(p) > 1 and p[0] not in ("+9999", "9999") and p[1] not in ("2", "3", "6", "7")


def probe(usaf, wban, name):
    sid = f"{usaf}{wban}"
    url = f"https://www.ncei.noaa.gov/data/global-hourly/access/2023/{sid}.csv"
    try:
        r = requests.get(url, headers=UA, timeout=30)
    except requests.RequestException as exc:
        log(f"    {sid} {name}: request error {exc}")
        return None
    if r.status_code != 200:
        log(f"    {sid} {name}: HTTP {r.status_code}")
        return None
    rows = list(csvmod.DictReader(io.StringIO(r.text)))
    if not rows:
        log(f"    {sid} {name}: empty")
        return None
    n = len(rows)
    tmp = sum(1 for x in rows if temp_ok(x.get("TMP", "")))
    dew = sum(1 for x in rows if temp_ok(x.get("DEW", "")))
    spring = [x for x in rows if x.get("DATE", "")[5:7] in ("04", "05")]
    frost = sum(1 for x in spring
                if temp_ok(x.get("TMP", "")) and int(x["TMP"].split(",")[0]) <= 0)
    size_mb = len(r.content) / 1e6
    log(f"    {sid} {name:22s} OK rows={n:5d} ({size_mb:.1f}MB) "
        f"TMP={100*tmp/n:3.0f}% DEW={100*dew/n:3.0f}% "
        f"springFrost={100*frost/max(len(spring),1):4.1f}%")
    return sid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8,
                    help="max stations to probe per country")
    args = ap.parse_args()

    log("downloading isd-history.csv (a few MB) ...")
    r = requests.get("https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv",
                     headers=UA, timeout=120)
    r.raise_for_status()
    log(f"history downloaded ({len(r.content)/1e6:.1f}MB), parsing ...")
    hist = pd.read_csv(io.StringIO(r.text), dtype=str)
    hist["END"] = pd.to_numeric(hist["END"], errors="coerce")
    hist["LAT"] = pd.to_numeric(hist["LAT"], errors="coerce")
    hist["LON"] = pd.to_numeric(hist["LON"], errors="coerce")
    active = hist[hist["END"] >= 20231201]
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
        for i, (_, row) in enumerate(sub.iterrows(), start=1):
            if found >= args.limit:
                break
            name = str(row["STATION NAME"])[:22]
            log(f"  probing {country} {found+1}/{args.limit}: {name} ...")
            if probe(row["USAF"], row["WBAN"], name):
                found += 1
    log("done.")


if __name__ == "__main__":
    main()
