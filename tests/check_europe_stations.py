"""Verify European (esp. UK) agricultural-region ISD stations exist + have frost.

NOAA ISD is global (aggregates national met services). Check whether UK and
European stations in frost-sensitive growing regions resolve and carry usable
temperature, so a frost forecaster could be built on European data too.
"""

from _common import report_station_frost

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

for name, sid in CANDIDATES.items():
    # Cloud availability matters too: clear sky drives radiative frost.
    report_station_frost(name, sid, YEAR, width=22, with_cloud=True)
