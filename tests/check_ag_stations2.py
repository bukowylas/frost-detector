"""Verify the second batch of frost-sensitive fruit-region ISD stations."""

import io
import csv as csvmod

import requests

UA = {"User-Agent": "cloud-cover-portfolio/0.1 (research; contact via README)"}
YEAR = 2023

CANDIDATES = {
    "geneva_finger_lakes_ny": "72522014748",   # Finger Lakes vineyards (Rochester)
    "wenatchee_wa_apples": "72788024243",       # WA apple country
    "traverse_city_mi_cherry": "72636014850",   # MI cherries
    "salem_or_willamette": "72694024232",       # Willamette valley
    "penn_yan_ny_alt": "72522594705",           # alt Finger Lakes
    "hood_river_or_alt": "72698024219",          # OR fruit (Portland area proxy)
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
