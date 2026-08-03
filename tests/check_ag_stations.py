"""Verify candidate agricultural-region ISD stations exist and have frost.

Fetches one year for each candidate station in a real frost-sensitive farming
region, decodes temperature, and reports the frost base rate. We want stations
that (a) resolve, (b) sit in genuine agricultural country, and (c) actually
freeze enough to learn from -- but not so extreme that frost is trivial.
"""

from _common import report_station_frost

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

for name, sid in CANDIDATES.items():
    report_station_frost(name, sid, YEAR)
