"""Verify UK + Polish agricultural stations for Frost Detector.

For each candidate, check across 2021-2023: does it resolve, is TMP and
(critically) DEW well-covered, and what is the frost-night rate in the spring
risk window (Apr-May)? Dewpoint is the single most important frost feature, so a
station with poor DEW coverage is disqualified regardless of anything else.

Station ids resolved from NOAA isd-history (USAF+WBAN, WBAN=099999 for Europe).
"""

import requests
from _common import FROST_C, coverage, month_rows, temperatures

from frostlib import net

YEARS = [2021, 2022, 2023]
SPRING_MONTHS = ("04", "05")

# Candidates in real fruit / horticulture regions. Verified to exist via the
# isd-history lookup; names are the ISD station names.
# Poland: Grojec area = apple capital; Sandomierz = orchards; Lublin = fruit/veg.
# UK: Kent/East Anglia = fruit; use well-known long-record stations.
CANDIDATES = {
    "PL_warszawa_okecie": "123750099999",
    "PL_lublin": "122950099999",
    "PL_kielce": "122500099999",
    "PL_wroclaw": "123600099999",
    "UK_wattisham_suffolk": "035900099999",
    "UK_marham_norfolk": "034820099999",
    "UK_manston_kent": "037960099999",
}

for name, sid in CANDIDATES.items():
    total_rows = tmp_ok = dew_ok = 0
    spring_lows: list[float] = []
    resolved_years = 0
    for year in YEARS:
        try:
            rows = net.fetch_station_year_rows(sid, year, timeout=90)
        except requests.RequestException:
            continue
        resolved_years += 1
        total_rows += len(rows)
        tmp_ok += coverage(rows, "TMP")
        dew_ok += coverage(rows, "DEW")
        # spring-window (Apr-May) daily-ish min temp as a rough frost signal
        spring_lows += temperatures(month_rows(rows, SPRING_MONTHS))
    if resolved_years == 0:
        print(f"{name:24s} HTTP 404 (id {sid})")
        continue
    frost = sum(1 for v in spring_lows if v <= FROST_C)
    print(f"{name:24s} yrs={resolved_years} rows={total_rows:6d}  "
          f"TMP={100*tmp_ok/total_rows:3.0f}%  DEW={100*dew_ok/total_rows:3.0f}%  "
          f"spring<=0C={100*frost/max(len(spring_lows),1):4.1f}%  (id {sid})")
