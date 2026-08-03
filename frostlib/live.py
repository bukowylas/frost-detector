"""Live observation provider for the forecast service.

The model is trained on NCEI-decoded ISD observations. To forecast a real
evening, the live path must fetch the *same* observations and present them in the
exact schema ``prepare.build_feature_row`` expects.

Each serviceable station is served from the hourly **SYNOP** stream (via OGIMET)
that its ISD training data was derived from -- tenths-of-a-degree temperature and
dew point, and a sea-level pressure their METARs do not carry. The report is raw
FM-12 code and is decoded here (``decode_synop``).

SERVICEABLE STATIONS -- decided by the parity gate, not by ambition:
    A station is served only where its live OGIMET feed provably reproduces the
    NCEI-decoded training features (to a tenth of a degree). Of the eight training
    stations, two UK stations (Waddington, Cranwell) meet that bar; the other six
    do not, each for a verified reason recorded on ``UNSERVICED_STATIONS`` --
    military-feed airfields with no civilian SYNOP, a station OGIMET does not
    archive, and one where OGIMET and NCEI simply disagree on the observations.
    Feeding the model another site's data, or a feed that does not match training,
    is the precise failure this layer exists to prevent, so coverage stops where
    parity stops. (NCEI's own archive lags ~1 year, so it cannot be the live
    source; a real-time feed is required, and it is trusted only where it matches.)

CLOUD -- a known, deliberate limitation, not a silent gap:
    Cloud cover is a genuine frost driver (clear, calm nights radiate heat away
    fastest -- that is when frost forms), so it is NOT a throwaway feature. But
    the serviceable SYNOP stations do not supply cloud in a form that matches how
    NCEI derived the training ``cloud_oktas``, so the live path runs with cloud
    MISSING. The model still produces a forecast (it handles missing cloud), but
    with reduced skill on exactly the clear nights that matter most.

    Two honest ways to close this properly, when it is worth doing:
      1. Retrain the deployed model WITHOUT cloud (and its derived
         ``radiative_potential``) and publish that model's accuracy -- then the
         live system and its quoted number are the same thing.
      2. SUPPLY cloud from a separate source: a cloud-cover model (the sibling
         "Cloud Cover" project predicts sky condition from surface variables) or
         a METAR sky-cover feed mapped to oktas.
    Until one of those is done, live forecasts run cloud-blank; the parity test
    asserts cloud is missing (so the gap is visible, never silent) and checks
    that every other feature reproduces the training value exactly.

Every value that cannot be trusted is left missing (NaN); nothing is imputed, and
an implausible decode is dropped to NaN rather than fed to the model.
"""

from __future__ import annotations

import csv as csvmod
import io
import math

import pandas as pd
import requests

from frostlib import isd

HEADERS = {"User-Agent": "frost-detector/0.1 (research/portfolio; contact via README)"}
KNOTS_TO_MS = 0.514444

OGIMET_SYNOP_URL = "https://www.ogimet.com/cgi-bin/getsynop"

# The stations the live service can serve: those whose real-time OGIMET SYNOP
# observations provably reproduce the NCEI-decoded ISD features the model was
# trained on. `wmo` is the verified civilian SYNOP index (NOT the ISD USAF prefix
# -- those don't decode 1:1).
#
# Live serviceability is decided by the parity gate, not by station count. Only
# stations whose live feed matches training to a tenth of a degree are served;
# a station is included here only after the parity test confirms it. The two UK
# stations below reproduce temperature, dew point, sea-level pressure and wind
# exactly (or within sub-tenth rounding) across cold, cyclonic and windy nights.
PROVIDER = {
    "uk_waddington": {"wmo": "03377"},
    "uk_cranwell": {"wmo": "03379"},
}

# Training stations that are NOT live-serviceable, each for a concrete, verified
# reason. Serving any of these would feed the model data that is not the data it
# was trained on -- the exact failure the parity gate exists to prevent.
#
#   Polish military airfields (Tomaszow, Inowroclaw, Leczyca, Krzesiny): their
#     ISD training data came from a military feed, not the civilian SYNOP network,
#     and they have no live SYNOP under a matching index (the nearest civilian
#     index, e.g. 12330 Lawica for Krzesiny, is a different station ~10 km away).
#
#   Manchester (03334): OGIMET's civilian SYNOP archive carries no reports for it
#     -- the parity fetch returns empty windows -- so there is no live feed to
#     serve, matching or otherwise.
#
#   Lodz-Lublinek (12105): OGIMET's archived SYNOP and NCEI's ISD disagree on the
#     instantaneous observations for this station (e.g. 2023-03-10 18:00: OGIMET
#     temp 0.3 C vs training 9.5 C, with dew point and wind also diverging) while
#     agreeing on pressure -- two providers rendering different values for the
#     same station-hour. NCEI (training) is the meteorologically consistent one;
#     OGIMET cannot reproduce it, so Lodz is not live-serviceable via this feed.
#
# NCEI's own archive is not an alternative live source: its global-hourly access
# files lag ~1 year (the current-year file does not yet exist), far too stale for
# a nightly forecast. A real-time feed is required, and where that feed does not
# match training, the station is excluded rather than served with wrong data.
UNSERVICED_STATIONS = (
    "pl_tomaszow", "pl_inowroclaw", "pl_leczyca", "pl_krzesiny_poznan",
    "uk_manchester", "pl_lublinek_lodz",
)

# Static geography (lat/lon/elev), which build_feature_row carries onto the row.
# These must equal the values already in the training data; sourced from it.
STATION_META = {
    "uk_waddington": {"lat": 53.166167, "lon": -0.523811, "elev": 70.4},
    "uk_cranwell": {"lat": 53.03035, "lon": -0.483242, "elev": 66.44},
}


# --- SYNOP (OGIMET) ---------------------------------------------------------


# The columns every observation frame carries (so an empty frame still has the
# schema build_feature_row and downstream code expect -- an empty frame with no
# columns is a KeyError waiting to happen). cloud_oktas is present but always NaN
# live: build_feature_row reads it, and the missing-cloud contract is that the
# column exists and is empty (visible gap), never absent (silent KeyError).
OBS_COLUMNS = ["lst", "temp_c", "dewpoint_c", "slp_hpa", "wind_ms", "cloud_oktas"]

# Physical plausibility bounds; a decoded value outside these is a mis-decode,
# not an observation, so it is dropped to NaN rather than fed to the model.
_TEMP_RANGE = (-90.0, 60.0)
_SLP_RANGE = (850.0, 1085.0)
_WIND_MAX = 60.0


class OgimetError(RuntimeError):
    """OGIMET declined to answer (rate limit, error body). Distinct from a
    usable-but-empty response, so callers can skip on this but must not treat a
    genuine data problem as skippable."""


def _signed_tenths(group: str, valid_signs: str = "01") -> float:
    """Decode a s + TTT tenths-of-a-degree group (temperature/dew point).

    Returns NaN (never None) for anything that is not a valid such group, so the
    output field type stays a consistent float. ``valid_signs`` restricts the sign
    digit: temperature/dew point use 0/1 (positive/negative). This rejects e.g. a
    2-group's ``9`` sign, so a relative-humidity group (2 9UUU) is not mistaken for
    a dew point.
    """
    if len(group) != 5 or "/" in group[1:] or group[1] not in valid_signs:
        return math.nan
    sign = -1 if group[1] == "1" else 1
    try:
        return sign * int(group[2:]) / 10.0
    except ValueError:
        return math.nan


# The section-2/3 markers; section 1 (the instantaneous obs) ends at the first.
_SECTION_MARKERS = ("222", "333", "444", "555")


def decode_synop(report: str) -> dict:
    """Decode section 1 of an FM-12 land SYNOP report to the model's raw fields.

    Only section 1 -- before the first regional/national section marker -- holds
    the instantaneous temperature (1sTTT), dew point (2sTdTdTd), sea-level
    pressure (4PPPP) and wind (Nddff). Later sections repeat 1.../2.../4... groups
    with different meanings (max/min temps, tendencies, geopotential) and must be
    ignored. Wind is read only if the report's unit indicator says knots; every
    decoded value is range-checked (a mis-decode yields an implausible number,
    which we drop to NaN rather than feed the model).
    """
    out: dict = {k: math.nan for k in ("temp_c", "dewpoint_c", "slp_hpa", "wind_ms")}
    tokens = report.replace("=", "").split()
    # Anchor on the AAXX header + its day-hour-unit group, robust to bulletin
    # prefixes and to NIL reports.
    try:
        i = tokens.index("AAXX")
    except ValueError:
        return out
    if i + 4 >= len(tokens):
        return out
    # FM-12: AAXX(i) ddhhiw(i+1) IIiii(i+2) iRiXhVV(i+3) Nddff(i+4) <groups...>
    ddhhiw = tokens[i + 1]  # YYGGiw: last digit is the wind-speed unit indicator
    nddff = tokens[i + 4]   # Nddff: total cloud, wind dir, wind speed

    # iw: 0/1 = wind in m/s, 3/4 = knots. Convert only when we know the unit.
    iw = ddhhiw[-1] if len(ddhhiw) == 5 and ddhhiw[-1].isdigit() else None
    ff = nddff[3:5] if len(nddff) >= 5 else ""
    if ff.isdigit() and ff != "99" and iw in ("0", "1", "3", "4"):
        speed = int(ff)
        out["wind_ms"] = round(speed * (KNOTS_TO_MS if iw in ("3", "4") else 1.0), 1)

    groups = tokens[i + 5:]
    for m in _SECTION_MARKERS:
        if m in groups:
            groups = groups[: groups.index(m)]
    for tok in groups:
        if len(tok) != 5 or not tok[1:].replace("/", "").isdigit():
            continue
        if tok[0] == "1" and math.isnan(out["temp_c"]):
            out["temp_c"] = _signed_tenths(tok)
        elif tok[0] == "2" and math.isnan(out["dewpoint_c"]):
            out["dewpoint_c"] = _signed_tenths(tok)  # sign 0/1 only -> not 29UUU RH
        elif (tok[0] == "4" and tok[1] in "09" and "/" not in tok[1:]
              and math.isnan(out["slp_hpa"])):
            # 4PPPP sea-level pressure in tenths of hPa. Leading 0/9 distinguishes
            # it from 4a3hhh (geopotential, leading digit 1-8). The value already
            # includes its base -- 9765 -> 976.5, 0032 -> 1003.2 -- so >=5000 is
            # the 900s band, else the 1000s band.
            p = int(tok[1:])
            out["slp_hpa"] = p / 10.0 if p >= 5000 else 1000 + p / 10.0

    # Plausibility guard: a decoded value outside physical bounds is a mis-decode,
    # dropped to NaN. A value already NaN stays NaN; a genuine 0.0 (a frost-point
    # temperature, a calm wind -- both peak frost signals) must survive, so this is
    # an explicit range test, never a truthiness test (0.0 is falsy).
    def _keep_in_range(value, lo, hi):
        return value if (math.isnan(value) or lo <= value <= hi) else math.nan

    out["temp_c"] = _keep_in_range(out["temp_c"], *_TEMP_RANGE)
    out["dewpoint_c"] = _keep_in_range(out["dewpoint_c"], *_TEMP_RANGE)
    out["slp_hpa"] = _keep_in_range(out["slp_hpa"], *_SLP_RANGE)
    out["wind_ms"] = _keep_in_range(out["wind_ms"], 0.0, _WIND_MAX)
    # Dew point cannot exceed temperature (+ a hair for rounding).
    if (not math.isnan(out["temp_c"]) and not math.isnan(out["dewpoint_c"])
            and out["dewpoint_c"] > out["temp_c"] + 0.2):
        out["temp_c"] = out["dewpoint_c"] = math.nan
    return out


def _empty_obs() -> pd.DataFrame:
    return pd.DataFrame(columns=OBS_COLUMNS)


def fetch_synop_window(station: str, wmo: str, cutoff, hours: int = 30) -> pd.DataFrame:
    """Fetch and decode the SYNOP window ending at ``cutoff`` (naive LST).

    OGIMET reports on UTC; we convert the LST cutoff to UTC using the SAME offset
    the training path uses (``isd.lst_offset``), query that UTC window, then stamp
    the decoded reports back to LST -- so the live and training clocks cannot
    drift. Raises OgimetError when the provider declines (rate-limit / error
    body, which it returns as an HTTP-200 text); returns an empty (but correctly
    columned) frame when the query simply had no reports.
    """
    off = pd.Timedelta(hours=isd.lst_offset(station))
    begin = (cutoff - pd.Timedelta(hours=hours) - off).strftime("%Y%m%d%H%M")
    end = (cutoff - off).strftime("%Y%m%d%H%M")
    r = requests.get(OGIMET_SYNOP_URL,
                     params={"block": wmo, "begin": begin, "end": end},
                     headers=HEADERS, timeout=60)
    r.raise_for_status()
    body = r.text
    # OGIMET signals problems in the body with HTTP 200, so raise_for_status is
    # not enough -- detect its error/quota text explicitly.
    low = body.lower()
    if "quota" in low or "sorry" in low or "#error" in low or "no valid" in low:
        raise OgimetError(body.strip().splitlines()[0][:200] if body.strip() else "empty")

    rows = []
    for rec in csvmod.reader(io.StringIO(body)):
        if len(rec) < 7 or rec[0] != wmo or not rec[1].isdigit():
            continue
        ts_utc = pd.Timestamp(int(rec[1]), int(rec[2]), int(rec[3]),
                              int(rec[4]), int(rec[5]))
        rows.append({"lst": ts_utc + off, **decode_synop(rec[6])})
    if not rows:
        return _empty_obs()
    obs = pd.DataFrame(rows)
    # OGIMET can return a corrected report for the same hour; keep the most
    # complete (fewest NaNs) per timestamp so the cutoff snapshot isn't a partial.
    obs["_completeness"] = obs.notna().sum(axis=1)
    obs = (obs.sort_values(["lst", "_completeness"])
              .drop_duplicates("lst", keep="last").drop(columns="_completeness"))
    return obs.reset_index(drop=True)


# --- unified entry point ----------------------------------------------------


def fetch_window(station: str, cutoff, hours: int = 30) -> pd.DataFrame:
    """Observations for one station up to ``cutoff`` (naive LST), in the schema
    ``build_feature_row`` expects, from the station's SYNOP provider.

    Adds the station's static geography (lat/lon/elev) as columns, since
    build_feature_row carries them onto the feature row. All serviceable stations
    use SYNOP; the cloud feature is not sourced (the deployed model excludes it),
    so no cloud provider is needed.
    """
    if station not in PROVIDER:
        raise ValueError(f"no live provider configured for {station!r}")
    obs = fetch_synop_window(station, PROVIDER[station]["wmo"], cutoff, hours)

    meta = STATION_META.get(station, {})
    for col in ("lat", "lon", "elev"):
        obs[col] = meta.get(col, math.nan)
    # Cloud is not sourced live (see module docstring): present so build_feature_row
    # can read it, always NaN so the gap is visible rather than silently imputed.
    obs["cloud_oktas"] = math.nan
    return obs.sort_values("lst").reset_index(drop=True)
