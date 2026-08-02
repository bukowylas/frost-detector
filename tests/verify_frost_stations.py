"""Verify UK + Polish agricultural stations for Frost Detector.

For each candidate, check across 2021-2023: does it resolve, is TMP and
(critically) DEW well-covered, and what is the frost-night rate in the spring
risk window (Apr-May)? Dewpoint is the single most important frost feature, so a
station with poor DEW coverage is disqualified regardless of anything else.

Station ids resolved from NOAA isd-history (USAF+WBAN, WBAN=099999 for Europe).
"""

import io
import csv as csvmod

import pandas as pd
import requests

UA = {"User-Agent": "frost-detector/0.1 (research; contact via README)"}
YEARS = [2021, 2022, 2023]

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


def sub(raw: str, idx: int) -> str:
    parts = raw.split(",") if raw else []
    return parts[idx] if len(parts) > idx else ""


def temp_ok(raw: str) -> bool:
    return bool(raw) and sub(raw, 0) not in ("+9999", "9999") and sub(raw, 1) not in ("2", "3", "6", "7")


def temp_val(raw: str):
    return int(sub(raw, 0)) / 10.0 if temp_ok(raw) else None


for name, sid in CANDIDATES.items():
    total_rows = tmp_ok = dew_ok = 0
    spring_lows = []
    resolved_years = 0
    for year in YEARS:
        url = f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{sid}.csv"
        r = requests.get(url, headers=UA, timeout=90)
        if r.status_code != 200:
            continue
        resolved_years += 1
        rows = list(csvmod.DictReader(io.StringIO(r.text)))
        total_rows += len(rows)
        tmp_ok += sum(1 for x in rows if temp_ok(x.get("TMP", "")))
        dew_ok += sum(1 for x in rows if temp_ok(x.get("DEW", "")))
        # spring-window (Apr-May) daily-ish min temp as a rough frost signal
        for x in rows:
            date = x.get("DATE", "")
            if len(date) >= 7 and date[5:7] in ("04", "05"):
                v = temp_val(x.get("TMP", ""))
                if v is not None:
                    spring_lows.append(v)
    if resolved_years == 0:
        print(f"{name:24s} HTTP 404 (id {sid})")
        continue
    frost = sum(1 for v in spring_lows if v <= 0) if spring_lows else 0
    print(f"{name:24s} yrs={resolved_years} rows={total_rows:6d}  "
          f"TMP={100*tmp_ok/total_rows:3.0f}%  DEW={100*dew_ok/total_rows:3.0f}%  "
          f"spring<=0C={100*frost/max(len(spring_lows),1):4.1f}%  (id {sid})")
