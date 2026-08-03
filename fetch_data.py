"""Fetch raw hourly surface observations from NOAA/NCEI ISD for Frost Detector.

Downloads one CSV per station-year from the NCEI "global-hourly" access service
and saves it under ``data_raw/`` (a symlink to the Linux filesystem). This is
the one and only network step; the rest of the pipeline reads the local files,
so cleaning, feature-engineering and training are fully offline and reproducible.

Stations are real frost-sensitive agricultural regions in **Poland** (primary --
the model's core, and the target user) and **England** (secondary -- a milder
maritime contrast). Every station below was verified against NOAA's station
history to have 100% temperature AND dewpoint coverage; dewpoint is the single
most important frost predictor, so poor-coverage stations were rejected.

Source: NOAA / NCEI Integrated Surface Database (ISD), global-hourly access.
Public service, no API key.

Usage:
    python3 fetch_data.py                        # all default stations, all years
    python3 fetch_data.py --years 2021 2022      # specific years
    python3 fetch_data.py --stations 12105499999 # specific station ids
"""

from __future__ import annotations

import argparse
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from frostlib import net, paths

RAW_DIR = paths.RAW_DIR

# Downloads are I/O-bound (waiting on the NCEI server), and these European
# station-years are large (~9 MB) and slow (~70 s each sequentially), so we
# fetch several at once. Kept modest to be a polite client to a public service.
MAX_WORKERS = 6

# Confirmed stations (name -> ISD id). Ids resolved from NOAA's isd-history file
# and verified for TMP+DEW coverage. Poland is primary (frost signal + the
# target user); England is a milder maritime contrast. See tests/ for the
# resolution/verification scripts.
POLAND_STATIONS = {
    "pl_tomaszow": "12105499999",
    "pl_lublinek_lodz": "12105599999",
    "pl_inowroclaw": "12105299999",
    "pl_leczyca": "12105399999",
    "pl_krzesiny_poznan": "12326099999",
}
UK_STATIONS = {
    "uk_waddington": "03377099999",
    "uk_cranwell": "03379099999",
    "uk_manchester": "03334099999",
}
DEFAULT_STATIONS = {**POLAND_STATIONS, **UK_STATIONS}

# A station name and id both flow, unmodified, into a filesystem path
# (isd_<name>_<id>_<year>.csv under data_raw/) AND into the fetch URL
# (<BASE_URL>/<year>/<id>.csv). Constrain them to safe characters so a crafted
# --stations value (e.g. containing "/" or "..") cannot traverse out of the
# data directory or repoint the request at another path on the host.
NAME_RE = re.compile(r"\A[A-Za-z0-9_]+\Z")
STATION_ID_RE = re.compile(r"\A[0-9]+\Z")

# Multiple years so year-to-year variability (the dominant uncertainty for a
# weather target) is represented, and leave-one-year-out evaluation has enough
# held-out years to be meaningful.
DEFAULT_YEARS = [2019, 2020, 2021, 2022, 2023]


# Each worker thread gets its own Session (requests Sessions are not safe to
# share across threads).
_thread_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def fetch_station_year(
    station_id: str, year: int, out_path: Path, session: requests.Session
) -> dict:
    """Download one station-year CSV to out_path. Returns a small summary.

    Raises on a non-200, or when the response does not look like an ISD CSV (a
    substring test would pass on an HTML error page mentioning "STATION", so we
    require the header to actually START with the STATION column). The write
    happens only AFTER the full response is buffered and validated, so a failed
    or truncated download can never leave a partial CSV on disk for the prepare
    step to silently consume.
    """
    resp = net.get(net.station_year_url(station_id, year), session=session)
    resp.raise_for_status()

    text = resp.text
    first_line = text.splitlines()[0].lstrip("﻿") if text else ""
    if not first_line.startswith('"STATION"') and not first_line.startswith("STATION"):
        raise ValueError(
            f"unexpected response for {station_id} {year} "
            f"(does not start with the ISD STATION header)"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write raw bytes to avoid a decode/re-encode round-trip.
    out_path.write_bytes(resp.content)
    return {
        "rows": len(text.splitlines()) - 1,
        "size_mb": round(len(resp.content) / 1_000_000, 2),
        "output": str(out_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    ap.add_argument(
        "--stations",
        nargs="+",
        default=None,
        help="stations as name=id, e.g. pl_lublin=12345099999 "
             "(default: the built-in verified set)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the output file already exists",
    )
    args = ap.parse_args()

    # An override must be given as name=id (e.g. pl_lublin=12345099999): the
    # name prefix (pl_/uk_) carries the LST offset the prepare step needs, so a
    # bare id would produce files prepare.py cannot place in a time zone.
    if args.stations:
        stations = {}
        for spec in args.stations:
            if "=" not in spec:
                raise SystemExit(
                    f"--stations entries must be name=id (got {spec!r}); the "
                    "name prefix pl_/uk_ sets the local-standard-time offset."
                )
            name, sid = spec.split("=", 1)
            if not NAME_RE.match(name):
                raise SystemExit(
                    f"invalid station name {name!r}: only letters, digits and "
                    "underscore are allowed (it becomes part of a file path)."
                )
            if not STATION_ID_RE.match(sid):
                raise SystemExit(
                    f"invalid station id {sid!r}: ISD ids are digits only "
                    "(it becomes part of a file path and the fetch URL)."
                )
            stations[name] = sid
    else:
        stations = DEFAULT_STATIONS
    jobs = [(name, sid, year) for name, sid in stations.items() for year in args.years]

    # Skip already-downloaded files up front, so the pool only does real work.
    to_fetch = []
    skipped = 0
    for name, sid, year in jobs:
        out = paths.raw_csv_path(name, sid, year, RAW_DIR)
        if out.exists() and not args.force:
            skipped += 1
        else:
            to_fetch.append((name, sid, year, out))

    t0 = time.monotonic()
    print(
        f"fetching {len(stations)} stations x {len(args.years)} years "
        f"= {len(jobs)} station-years ({skipped} already present, "
        f"{len(to_fetch)} to fetch, {MAX_WORKERS} at a time)",
        flush=True,
    )

    total_rows = fetched = 0
    failures: list[tuple[str, int, str]] = []

    def worker(job):
        _name, sid, year, out = job
        return job, fetch_station_year(sid, year, out, _session())

    n = len(to_fetch)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(worker, job): job for job in to_fetch}
        for done, future in enumerate(as_completed(futures), start=1):
            name, sid, year, _ = futures[future]
            elapsed = time.monotonic() - t0
            prefix = f"[{done:>2}/{n}  {elapsed:5.0f}s]"
            try:
                _, summary = future.result()
            except Exception as exc:  # noqa: BLE001 -- batch-continue by design
                # One bad station-year must not abort the batch nor leave a
                # partial dataset; collect, report, and exit non-zero.
                failures.append((name, year, repr(exc)))
                print(f"{prefix} FAILED {name} {year}: {exc}", flush=True)
                continue
            total_rows += summary["rows"]
            fetched += 1
            print(
                f"{prefix} {name} {year}: {summary['rows']} rows, "
                f"{summary['size_mb']} MB",
                flush=True,
            )

    print(
        f"\ndone in {time.monotonic() - t0:.0f}s: {fetched} fetched, "
        f"{skipped} skipped, {len(failures)} failed; {total_rows} rows total",
        flush=True,
    )
    if failures:
        print(f"\n{len(failures)} station-year(s) failed:")
        for name, year, err in failures:
            print(f"  {name} {year}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
