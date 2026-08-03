"""Live observation providers for the forecast service.

The model is trained on NCEI-decoded ISD observations. To forecast a real
evening, the live path must fetch the *same* observations, from a source that
carries the *same* stream each station reported on, and present them in the
exact schema ``prepare.build_feature_row`` expects. Which stream matters:

- **METAR** (from the Iowa Environmental Mesonet, ASOS archive) matches the
  stations whose ISD training data came from METAR. Temperature/dew point are
  already in degC; wind is in knots (convert); sky is METAR letter codes
  (map to oktas); some stations do not report MSLP in METAR at all.
- **SYNOP** (from OGIMET) matches the stations trained on the hourly SYNOP
  stream -- tenths-of-a-degree temperature/dew point and a sea-level pressure
  their METARs never carry. The report is raw FM-12 code and must be decoded.

``PROVIDER`` records, per station, which stream reproduces its training data --
established by the parity test (``tests/test_parity.py``), not assumed.

Every value that cannot be trusted is left missing (NaN); nothing is imputed.
The feature builder and its rejection handling deal with missing data.
"""

from __future__ import annotations

import csv as csvmod
import io
import math

import pandas as pd
import requests

HEADERS = {"User-Agent": "frost-detector/0.1 (research/portfolio; contact via README)"}
KNOTS_TO_MS = 0.514444

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
OGIMET_SYNOP_URL = "https://www.ogimet.com/cgi-bin/getsynop"

# The stations the live service can serve: those whose real-time observations
# are published to OGIMET's civilian SYNOP network under a WMO index that matches
# the station the model was trained on. `wmo` is that verified index (NOT simply
# the ISD USAF prefix -- those don't decode 1:1).
#
# Four of the eight training stations are deliberately absent: Tomaszow,
# Inowroclaw, Leczyca and Krzesiny are Polish military airfields whose ISD
# training data came from a military feed, not the civilian SYNOP network. They
# have no live SYNOP report under a matching index (the nearest civilian index,
# e.g. 12330 Lawica for Krzesiny, is a different station ~10 km away, which would
# feed the model another site's observations). Rather than substitute a wrong
# station, the live service does not cover them -- a real data-availability
# boundary, stated plainly.
PROVIDER = {
    "uk_waddington": {"stream": "synop", "wmo": "03377"},
    "uk_cranwell": {"stream": "synop", "wmo": "03379"},
    "uk_manchester": {"stream": "synop", "wmo": "03334"},
    "pl_lublinek_lodz": {"stream": "synop", "wmo": "12105"},
}

# Training stations not live-serviceable (see PROVIDER note), kept for reference.
UNSERVICED_STATIONS = (
    "pl_tomaszow", "pl_inowroclaw", "pl_leczyca", "pl_krzesiny_poznan",
)

# Static geography (lat/lon/elev), which build_feature_row carries onto the row.
# These must equal the values already in the training data; sourced from it.
STATION_META = {
    "uk_waddington": {"lat": 53.166167, "lon": -0.523811, "elev": 70.4},
    "uk_cranwell": {"lat": 53.03035, "lon": -0.483242, "elev": 66.44},
    "uk_manchester": {"lat": 53.353744, "lon": -2.27495, "elev": 78.33},
    "pl_lublinek_lodz": {"lat": 51.717, "lon": 19.4, "elev": 184.0},
}


# --- SYNOP (OGIMET) ---------------------------------------------------------


def _signed_tenths(group: str) -> float | None:
    """Decode a 1sTTT / 2sTdTdTd temperature group to degC, or None."""
    if len(group) != 5 or "/" in group[1:]:
        return None
    sign = -1 if group[1] == "1" else 1
    try:
        return sign * int(group[2:]) / 10.0
    except ValueError:
        return None


def decode_synop(report: str) -> dict:
    """Decode section 1 of an FM-12 SYNOP report to the model's raw fields.

    Only section 1 (before the ``333`` regional-section marker) carries the
    instantaneous temperature, dew point and sea-level pressure; the 1.../2...
    groups after ``333`` are max/min temps and tendencies and must be ignored.
    """
    tokens = report.replace("=", "").split()
    out: dict = {"temp_c": math.nan, "dewpoint_c": math.nan,
                 "slp_hpa": math.nan, "wind_ms": math.nan, "cloud_oktas": math.nan}
    if len(tokens) < 5:
        return out

    # tokens: AAXX  ddhhiw  WMO  iRiXhVV  Nddff  <groups...>
    # We deliberately take only WIND from the Nddff group, not cloud: SYNOP's
    # total-cloud digit N and NCEI's ISD oktas do not decode 1:1 (N=8 "obscured"
    # vs an ISD 7, etc.), so reproducing it faithfully is a rabbit hole for a
    # feature that only feeds the (marginal) radiative-cooling term. Cloud is
    # left missing here; the model handles missing cloud natively, exactly as it
    # does for training nights whose cloud was missing.
    nddff = tokens[4]
    if len(nddff) >= 5 and nddff[3:5].isdigit():
        out["wind_ms"] = int(nddff[3:5]) * KNOTS_TO_MS

    section1 = tokens[5:]
    if "333" in section1:
        section1 = section1[: section1.index("333")]
    for tok in section1:
        if len(tok) != 5:
            continue
        if tok[0] == "1":
            out["temp_c"] = _signed_tenths(tok)
        elif tok[0] == "2":
            out["dewpoint_c"] = _signed_tenths(tok)
        elif tok[0] == "4" and "/" not in tok[1:]:
            p = int(tok[1:])
            out["slp_hpa"] = (1000 + p / 10.0) if p < 5000 else (900 + p / 10.0)
    return out


def fetch_synop_window(wmo: str, cutoff, hours: int = 30) -> pd.DataFrame:
    """Fetch SYNOP reports for [cutoff - hours, cutoff] and decode them.

    ``cutoff`` is a naive Local Standard Time timestamp; UK/PL SYNOP is reported
    on UTC hours and (for these stations) LST aligns with the reporting clock as
    the training path assumes, so we query the corresponding UTC window.
    """
    begin = (cutoff - pd.Timedelta(hours=hours)).strftime("%Y%m%d%H%M")
    end = cutoff.strftime("%Y%m%d%H%M")
    r = requests.get(OGIMET_SYNOP_URL,
                     params={"block": wmo, "begin": begin, "end": end},
                     headers=HEADERS, timeout=60)
    r.raise_for_status()
    rows = []
    for rec in csvmod.reader(io.StringIO(r.text)):
        if len(rec) < 7 or rec[0] != wmo or not rec[1].isdigit():
            continue
        ts = pd.Timestamp(int(rec[1]), int(rec[2]), int(rec[3]), int(rec[4]), int(rec[5]))
        fields = decode_synop(rec[6])
        rows.append({"lst": ts, **fields})
    return pd.DataFrame(rows)


# --- METAR (IEM) ------------------------------------------------------------

# METAR sky cover -> oktas, matching the NCEI ISD convention the model trained
# on (FEW/SCT map to their lower bound, BKN to its lower bound, OVC to 8, clear
# to 0). NSC/NCD/no-report are NOT zero cloud -> missing, as ISD leaves them.
_SKY_TO_OKTAS = {"CLR": 0.0, "SKC": 0.0, "FEW": 2.0, "SCT": 4.0,
                 "BKN": 5.0, "OVC": 8.0}


def _sky_to_oktas(code: str) -> float:
    return _SKY_TO_OKTAS.get((code or "").strip().upper(), math.nan)


def fetch_metar_window(icao: str, cutoff, hours: int = 30) -> pd.DataFrame:
    """Fetch the IEM ASOS (METAR) window and convert to the model's schema."""
    start = cutoff - pd.Timedelta(hours=hours)
    params = {
        "station": icao, "data": ["tmpc", "dwpc", "sknt", "skyc1", "mslp"],
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": cutoff.year, "month2": cutoff.month, "day2": cutoff.day,
        "tz": "UTC", "format": "comma", "latlon": "no", "missing": "M",
    }
    r = requests.get(IEM_URL, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    lines = [ln for ln in r.text.splitlines() if ln and not ln.startswith("#")]
    reader = csvmod.DictReader(lines)

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return math.nan

    rows = []
    for rec in reader:
        ts = pd.to_datetime(rec.get("valid"), errors="coerce")
        if pd.isna(ts):
            continue
        knots = num(rec.get("sknt"))
        rows.append({
            "lst": ts.tz_localize(None) if ts.tzinfo else ts,
            "temp_c": num(rec.get("tmpc")),
            "dewpoint_c": num(rec.get("dwpc")),
            "slp_hpa": num(rec.get("mslp")),
            "wind_ms": knots * KNOTS_TO_MS if not math.isnan(knots) else math.nan,
            "cloud_oktas": _sky_to_oktas(rec.get("skyc1")),
        })
    return pd.DataFrame(rows)


# --- unified entry point ----------------------------------------------------


def fetch_window(station: str, cutoff, hours: int = 30) -> pd.DataFrame:
    """Observations for one station up to ``cutoff``, in build_feature_row's
    schema, from whichever provider matches that station's training stream.

    Adds the station's static geography (lat/lon/elev) as columns, since
    build_feature_row carries them onto the feature row.
    """
    if station not in PROVIDER:
        raise ValueError(f"no live provider configured for {station!r}")
    cfg = PROVIDER[station]
    if cfg["stream"] == "synop":
        obs = fetch_synop_window(cfg["wmo"], cutoff, hours)
    else:
        obs = fetch_metar_window(cfg["icao"], cutoff, hours)

    meta = STATION_META.get(station, {})
    for col in ("lat", "lon", "elev"):
        obs[col] = meta.get(col, math.nan)
    return obs.sort_values("lst").reset_index(drop=True)
