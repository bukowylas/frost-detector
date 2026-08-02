"""Look up real European/UK ISD station ids from NOAA's station-history file.

Rather than guess USAF+WBAN ids, fetch the authoritative station list and
search it for UK / France / Poland / Germany stations that were active in 2023.
Prints candidate ids in the exact form the access URL expects (USAF+WBAN).
"""

import io

import pandas as pd
import requests

UA = {"User-Agent": "cloud-cover-portfolio/0.1 (research; contact via README)"}

# NOAA ISD station history (fixed dataset of all stations).
URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
print("fetching isd-history.csv ...")
r = requests.get(URL, headers=UA, timeout=120)
r.raise_for_status()
hist = pd.read_csv(io.StringIO(r.text), dtype=str)
print("columns:", list(hist.columns))
print("total stations:", len(hist))

# Keep stations active through 2023.
hist["END"] = pd.to_numeric(hist["END"], errors="coerce")
active = hist[hist["END"] >= 20231231].copy()

for country in ["UK", "FR", "PO", "GM"]:  # ISD uses FIPS: UK, FR, PO(Poland), GM(Germany)
    sub = active[active["CTRY"] == country]
    print(f"\n=== {country}: {len(sub)} active stations (showing a sample) ===")
    show = sub[["USAF", "WBAN", "STATION NAME", "CTRY", "LAT", "LON"]].head(12)
    for _, row in show.iterrows():
        sid = f"{row['USAF']}{row['WBAN']}"
        print(f"  {sid}  {str(row['STATION NAME'])[:34]:34s} "
              f"lat={row['LAT']} lon={row['LON']}")
