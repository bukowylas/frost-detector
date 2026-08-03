"""Find Poland (PL) stations and spot-check frost on real European ids.

Corrects the earlier PO=Portugal mistake (Poland is PL in ISD FIPS). Lists
inland Polish stations, then downloads a representative UK/FR/PL/DE set for 2023
and reports frost rate + data cadence (hourly vs sparser) + cloud availability.
"""

from _common import report_station_frost

from frostlib import isd_history

YEAR = 2023

# First, list active Polish stations from the history file.
hist = isd_history.load_isd_history()
pl = isd_history.active_stations(hist, country="PL")
print(f"=== Poland (PL): {len(pl)} active stations (inland sample) ===")
pl_inland = pl[pl["LON"] > 16.0]  # skip coast
for _, row in pl_inland[["USAF", "WBAN", "STATION NAME", "LAT", "LON"]].head(10).iterrows():
    print(f"  {isd_history.station_id(row)}  {str(row['STATION NAME'])[:30]:30s} "
          f"lat={row['LAT']} lon={row['LON']}")

# Representative set with CORRECT ids (WBAN 099999).
CHECK = {
    "manston_kent_uk": "03797099999",       # Kent fruit
    "lille_fr": "07015099999",              # N. France arable
    "hamburg_de": "10147099999",            # N. Germany
}
# Pick the first inland PL station as the Poland representative.
if len(pl_inland):
    first = pl_inland.iloc[0]
    CHECK["poland_" + str(first["STATION NAME"]).split()[0].lower()] = (
        isd_history.station_id(first))

print("\n=== frost spot-check (2023) ===")
for name, sid in CHECK.items():
    report_station_frost(name, sid, YEAR, width=22, timeout=90,
                         with_cloud=True, suffix=f"  (id {sid})")
