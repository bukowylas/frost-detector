"""Find Poland (PL) stations and spot-check frost on real European ids.

Corrects the earlier PO=Portugal mistake (Poland is PL in ISD FIPS). Lists
inland Polish stations, then downloads a representative UK/FR/PL/DE set for 2023
and reports frost rate + data cadence (hourly vs sparser) + cloud availability.
"""

import io
import csv as csvmod

import pandas as pd
import requests

UA = {"User-Agent": "cloud-cover-portfolio/0.1 (research; contact via README)"}
YEAR = 2023

# First, list active Polish stations from the history file.
hist = pd.read_csv(
    io.StringIO(requests.get(
        "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv",
        headers=UA, timeout=120).text),
    dtype=str,
)
hist["END"] = pd.to_numeric(hist["END"], errors="coerce")
pl = hist[(hist["CTRY"] == "PL") & (hist["END"] >= 20231231)]
print(f"=== Poland (PL): {len(pl)} active stations (inland sample) ===")
pl_inland = pl[pd.to_numeric(pl["LON"], errors="coerce") > 16.0]  # skip coast
for _, row in pl_inland[["USAF", "WBAN", "STATION NAME", "LAT", "LON"]].head(10).iterrows():
    print(f"  {row['USAF']}{row['WBAN']}  {str(row['STATION NAME'])[:30]:30s} "
          f"lat={row['LAT']} lon={row['LON']}")


def temp_c(raw: str) -> float | None:
    parts = raw.split(",") if raw else []
    if len(parts) < 2 or parts[0] == "+9999" or parts[1] in ("2", "3", "6", "7"):
        return None
    try:
        return int(parts[0]) / 10.0
    except ValueError:
        return None


# Representative set with CORRECT ids (WBAN 099999).
CHECK = {
    "manston_kent_uk": "03797099999",       # Kent fruit
    "lille_fr": "07015099999",              # N. France arable
    "hamburg_de": "10147099999",            # N. Germany
    "krakow_pl": None,                       # filled from lookup below
}
# Pick the first inland PL station as the Poland representative.
if len(pl_inland):
    first = pl_inland.iloc[0]
    CHECK["poland_" + str(first["STATION NAME"]).split()[0].lower()] = (
        f"{first['USAF']}{first['WBAN']}")
CHECK.pop("krakow_pl", None)

print("\n=== frost spot-check (2023) ===")
for name, sid in CHECK.items():
    url = f"https://www.ncei.noaa.gov/data/global-hourly/access/{YEAR}/{sid}.csv"
    r = requests.get(url, headers=UA, timeout=90)
    if r.status_code != 200:
        print(f"{name:22s} HTTP {r.status_code} (id {sid})")
        continue
    rows = list(csvmod.DictReader(io.StringIO(r.text)))
    temps = [t for t in (temp_c(x.get("TMP", "")) for x in rows) if t is not None]
    if not temps:
        print(f"{name:22s} no temp (id {sid})")
        continue
    frost = sum(1 for t in temps if t <= 0)
    warn = sum(1 for t in temps if t <= 2)
    ga1 = sum(1 for x in rows if x.get("GA1"))
    print(f"{name:22s} rows={len(temps):6d}  <=0C {100*frost/len(temps):5.1f}%  "
          f"<=2C {100*warn/len(temps):5.1f}%  min={min(temps):6.1f}  "
          f"GA1={100*ga1/len(rows):3.0f}%  (id {sid})")
