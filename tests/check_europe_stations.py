"""Verify European (esp. UK) agricultural-region ISD stations exist + have frost.

NOAA ISD is global (aggregates national met services). Check whether UK and
European stations in frost-sensitive growing regions resolve and carry usable
temperature, so a frost forecaster could be built on European data too.
"""

import io
import csv as csvmod

import requests

UA = {"User-Agent": "cloud-cover-portfolio/0.1 (research; contact via README)"}
YEAR = 2023

# name -> ISD id (USAF+WBAN; European stations use WBAN 99999).
CANDIDATES = {
    # UK -- fruit/horticulture regions
    "london_heathrow_uk": "037720099999",
    "manston_kent_uk": "037970099999",       # Kent = "garden of England", fruit
    "waddington_uk": "033770099999",          # Lincolnshire arable/veg
    "shawbury_uk": "033790099999",            # Shropshire
    # Europe -- viticulture / orchard regions
    "bordeaux_fr_wine": "075100099999",       # Bordeaux vineyards
    "geneva_ch": "067000099999",              # Swiss vineyards / orchards
    "poznan_pl": "123300099999",              # Poland (cloudatus home region)
    "krakow_pl": "125660099999",              # Poland
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
        print(f"{name:22s} ERROR {exc}")
        continue
    if r.status_code != 200:
        print(f"{name:22s} HTTP {r.status_code}")
        continue
    rows = list(csvmod.DictReader(io.StringIO(r.text)))
    temps = [t for t in (temp_c(x.get("TMP", "")) for x in rows) if t is not None]
    if not temps:
        print(f"{name:22s} resolved but no temp data")
        continue
    frost = sum(1 for t in temps if t <= 0)
    warn = sum(1 for t in temps if t <= 2)
    # also check GA1 (cloud) presence, since clear-sky matters for radiative frost
    ga1 = sum(1 for x in rows if x.get("GA1"))
    print(f"{name:22s} rows={len(temps):6d}  <=0C {100*frost/len(temps):5.1f}%  "
          f"<=2C {100*warn/len(temps):5.1f}%  min={min(temps):6.1f}C  "
          f"GA1={100*ga1/len(rows):3.0f}%")
