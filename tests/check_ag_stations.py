"""Verify candidate agricultural-region ISD stations exist and have frost.

Fetches one year for each candidate station in a real frost-sensitive farming
region, decodes temperature, and reports the frost base rate. We want stations
that (a) resolve, (b) sit in genuine agricultural country, and (c) actually
freeze enough to learn from -- but not so extreme that frost is trivial.
"""

import io
import csv as csvmod

import requests

UA = {"User-Agent": "cloud-cover-portfolio/0.1 (research; contact via README)"}
YEAR = 2023

# name -> ISD id. Real frost-sensitive US ag regions.
CANDIDATES = {
    "fresno_ca_citrus_nuts": "72389093193",       # CA Central Valley
    "bakersfield_ca": "72384023155",              # S. Central Valley
    "yakima_wa_apples": "72781024243",            # WA fruit country
    "medford_or_pears": "72597024225",            # OR pears/wine
    "grand_rapids_mi_fruit": "72635094860",       # MI fruit belt
    "macon_ga_peaches": "72217013860",            # GA peaches/pecans
    "fresno_alt": "72389023940",                  # alt Fresno WBAN
}


def temp_c(raw: str) -> float | None:
    parts = raw.split(",") if raw else []
    if len(parts) < 2 or parts[0] == "+9999" or parts[1] in ("2", "3", "6", "7"):
        return None
    try:
        return int(parts[0]) / 10.0
    except ValueError:
        return None


for name, sid in CANDIDATES.items():
    url = f"https://www.ncei.noaa.gov/data/global-hourly/access/{YEAR}/{sid}.csv"
    try:
        r = requests.get(url, headers=UA, timeout=60)
    except Exception as exc:
        print(f"{name:26s} ERROR {exc}")
        continue
    if r.status_code != 200:
        print(f"{name:26s} HTTP {r.status_code}")
        continue
    rows = list(csvmod.DictReader(io.StringIO(r.text)))
    temps = [t for t in (temp_c(x.get("TMP", "")) for x in rows) if t is not None]
    if not temps:
        print(f"{name:26s} no temp data")
        continue
    frost = sum(1 for t in temps if t <= 0)
    warn = sum(1 for t in temps if t <= 2)
    print(f"{name:26s} rows={len(temps):6d}  <=0C {100*frost/len(temps):5.1f}%  "
          f"<=2C {100*warn/len(temps):5.1f}%  min={min(temps):6.1f}C")
