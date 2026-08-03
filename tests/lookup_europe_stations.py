"""Look up real European/UK ISD station ids from NOAA's station-history file.

Rather than guess USAF+WBAN ids, fetch the authoritative station list and
search it for UK / France / Poland / Germany stations that were active in 2023.
Prints candidate ids in the exact form the access URL expects (USAF+WBAN).
"""

import _common  # noqa: F401  -- puts the repository root on sys.path

from frostlib import isd_history

print("fetching isd-history.csv ...")
hist = isd_history.load_isd_history()
print("columns:", list(hist.columns))
print("total stations:", len(hist))

for country in ["UK", "FR", "PL", "GM"]:  # ISD FIPS codes (PL=Poland, GM=Germany)
    sub = isd_history.active_stations(hist, country=country)
    print(f"\n=== {country}: {len(sub)} active stations (showing a sample) ===")
    show = sub[["USAF", "WBAN", "STATION NAME", "CTRY", "LAT", "LON"]].head(12)
    for _, row in show.iterrows():
        print(f"  {isd_history.station_id(row)}  "
              f"{str(row['STATION NAME'])[:34]:34s} "
              f"lat={row['LAT']} lon={row['LON']}")
