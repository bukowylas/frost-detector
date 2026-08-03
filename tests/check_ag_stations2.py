"""Verify the second batch of frost-sensitive fruit-region ISD stations."""

from _common import report_station_frost

YEAR = 2023

CANDIDATES = {
    "geneva_finger_lakes_ny": "72522014748",   # Finger Lakes vineyards (Rochester)
    "wenatchee_wa_apples": "72788024243",       # WA apple country
    "traverse_city_mi_cherry": "72636014850",   # MI cherries
    "salem_or_willamette": "72694024232",       # Willamette valley
    "penn_yan_ny_alt": "72522594705",           # alt Finger Lakes
    "hood_river_or_alt": "72698024219",          # OR fruit (Portland area proxy)
}

for name, sid in CANDIDATES.items():
    report_station_frost(name, sid, YEAR)
