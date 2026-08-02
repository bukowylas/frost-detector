"""Prepare step: raw ISD CSVs -> one modelling row per (station, frost-risk night).

The task is an overnight-minimum-temperature forecast: from what a weather
station observes by early evening, predict how cold it will get overnight. Each
output row is therefore ONE NIGHT:

  features  = the station's state at the 18:00 LST cutoff (+ recent trends)
  label     = the minimum temperature observed 20:00 LST -> 08:00 LST next day

with the 18:00-20:00 window excluded from both, so no feature is contemporaneous
with the label period (the core anti-leakage rule -- see README / consult notes).

Design decisions made here (documented so they can be revisited / reviewed):

- **Local Standard Time, not UTC and not DST.** The whole task pivots on the
  day/night boundary; a UTC cutoff would smear it by the station's offset. We use
  fixed standard offsets: Poland UTC+1, UK UTC+0. (Standard, not daylight, to
  avoid a discontinuity mid-spring inside the frost window.)
- **Frost-risk windows only.** Spring (1 Mar - 31 May) and autumn (15 Sep -
  15 Nov). Year-round nights would let day-of-year alone "solve" the task
  (January is ~always frost, July ~never) -- a good score that means nothing.
  Restricting to the windows where a frost warning has economic value also forces
  the model onto the physics. Spring is widened to March (vs a stricter April)
  because Polish budbreak risk starts then; tune per region later if needed.
- **Features = 18:00 snapshot + 3 h and 24 h trends.** Evening temperature,
  dewpoint, dewpoint depression, wind, cloud, pressure -- plus temperature change
  and pressure tendency, a cheap synoptic (advection) signal. Not a full evening
  curve; kept deliberately simple.
- **Radiative-frost context, not filtering.** We do NOT drop windy/cloudy nights;
  the model must handle them. But we record evening wind and cloud so error
  analysis can separate radiative from advective frost later.

Reads every ``isd_*.csv`` under ``data_raw/`` (offline) and writes
``data/nights.csv`` (one row per station-night) plus a coverage report. The ISD
field decoding (sentinels + quality flags) is carried over verified from the
prior project.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "data_raw"
OUT_DIR = HERE / "data"

# Fixed Local Standard Time offset (hours from UTC) per station-name prefix.
LST_OFFSET_HOURS = {"pl_": 1, "uk_": 0}

# Cutoff / label window in LST.
CUTOFF_HOUR = 18            # features use observations at/just before this
LABEL_START_HOUR = 20      # overnight-min window start (same evening)
LABEL_END_HOUR = 8         # overnight-min window end (next morning)
# The 18:00-20:00 gap between them is excluded from both by construction.

# Frost-risk windows as (month, day) ranges, inclusive.
RISK_WINDOWS = [((3, 1), (5, 31)), ((9, 15), (11, 15))]

# --- ISD packed-field decoding ----------------------------------------------
# Reject suspect (2, 6) and erroneous (3, 7) quality flags. Flag '9' is NOT
# rejected here, but that is moot in practice: verified across all 40 frost
# station-years, EVERY flag-'9' TMP/DEW row also carries the missing sentinel
# (+9999) and is dropped by the sentinel check anyway -- 0 flag-'9' rows carry a
# real value. So whatever '9' denotes (likely "QC not applied"), no unchecked
# value enters the model. (See tests/check_flag9.py.)
_BAD_QUALITY_FLAGS = {"2", "6", "3", "7"}


def _packed(value: str) -> list[str]:
    return value.split(",") if isinstance(value, str) and value else []


def _scaled_num(raw, sentinel, scale, value_idx=0, flag_idx=1) -> float:
    """Decode a packed ISD numeric sub-field to a real value, or NaN."""
    parts = _packed(raw)
    if len(parts) <= value_idx:
        return math.nan
    token = parts[value_idx]
    if token == sentinel:
        return math.nan
    if len(parts) > flag_idx and parts[flag_idx] in _BAD_QUALITY_FLAGS:
        return math.nan
    try:
        return int(token) / scale
    except ValueError:
        return math.nan


def _temp_c(raw) -> float:
    return _scaled_num(raw, "+9999", 10.0)


def _slp_hpa(raw) -> float:
    return _scaled_num(raw, "99999", 10.0)


def _wind_speed_ms(raw) -> float:
    return _scaled_num(raw, "9999", 10.0, value_idx=3, flag_idx=4)


def _cloud_oktas(raw) -> float:
    """GA1 lowest-layer coverage in oktas (0-8), or NaN. Clear sky (low oktas)
    favours radiative cooling, so this is a real frost signal even though it was
    a poor *target* in the prior project."""
    parts = _packed(raw)
    if not parts:
        return math.nan
    if len(parts) > 1 and parts[1] in _BAD_QUALITY_FLAGS:
        return math.nan
    try:
        code = int(parts[0])
    except ValueError:
        return math.nan
    return float(code) if 0 <= code <= 8 else math.nan


def _station_name(csv_path: Path) -> str:
    # isd_<name>_<id>_<year>.csv -> <name>. removeprefix (not replace) so an
    # "isd_" substring elsewhere in the name can't be mangled.
    return csv_path.stem.removeprefix("isd_").rsplit("_", 2)[0]


def _lst_offset(station: str) -> int:
    for prefix, off in LST_OFFSET_HOURS.items():
        if station.startswith(prefix):
            return off
    raise ValueError(f"no LST offset known for station {station!r}")


def _in_risk_window(month: int, day: int) -> bool:
    for (m0, d0), (m1, d1) in RISK_WINDOWS:
        if (m0, d0) <= (month, day) <= (m1, d1):
            return True
    return False


def decode_station(csv_path: Path) -> pd.DataFrame:
    """Decode one raw station-year CSV into LST-stamped hourly observations."""
    usecols = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION",
               "TMP", "DEW", "SLP", "WND", "GA1"]
    raw = pd.read_csv(csv_path, dtype=str, low_memory=False, usecols=usecols)
    station = _station_name(csv_path)
    # Start the frame from a length-carrying column; assigning a scalar to a
    # still-empty DataFrame would create a zero-length column and silently drop
    # every row (the station column would be empty, breaking the later groupby).
    utc = pd.to_datetime(raw["DATE"], utc=True, errors="coerce")
    out = pd.DataFrame({"lst": utc + pd.to_timedelta(_lst_offset(station), unit="h")})
    out["station"] = station
    out["lat"] = pd.to_numeric(raw["LATITUDE"], errors="coerce")
    out["lon"] = pd.to_numeric(raw["LONGITUDE"], errors="coerce")
    out["elev"] = pd.to_numeric(raw["ELEVATION"], errors="coerce")
    out["temp_c"] = raw["TMP"].map(_temp_c)
    out["dewpoint_c"] = raw["DEW"].map(_temp_c)
    out["slp_hpa"] = raw["SLP"].map(_slp_hpa)
    out["wind_ms"] = raw["WND"].map(_wind_speed_ms)
    out["cloud_oktas"] = raw["GA1"].map(_cloud_oktas)
    return out.dropna(subset=["lst"])


def _nearest_at_or_before(day_obs: pd.DataFrame, target, tol_minutes=90):
    """The last observation at or before `target`, within a tolerance.

    ISD reports irregularly (2-3x/hour with gaps), so 'the 18:00 value' is really
    'the most recent observation by 18:00'. A tolerance guards against using a
    stale value across a long data gap.
    """
    prior = day_obs[day_obs["lst"] <= target]
    if prior.empty:
        return None
    row = prior.iloc[-1]
    if (target - row["lst"]).total_seconds() > tol_minutes * 60:
        return None
    return row


def build_nights(obs: pd.DataFrame) -> pd.DataFrame:
    """Collapse hourly observations into one row per (station, frost-risk night).

    For each station and each evening date D in a risk window, features come from
    the 18:00 LST cutoff on D (plus 3 h / 24 h trends), and the label is the
    minimum temperature in [20:00 LST D, 08:00 LST D+1].
    """
    rows = []
    for station, g in obs.groupby("station", sort=False):
        g = g.sort_values("lst").reset_index(drop=True)
        lst = g["lst"]
        # Iterate candidate evening dates present in this station's record.
        dates = pd.Series(lst.dt.normalize().unique())
        for day in dates:
            month, dom = day.month, day.day
            if not _in_risk_window(month, dom):
                continue
            cutoff = day + pd.Timedelta(hours=CUTOFF_HOUR)
            label_start = day + pd.Timedelta(hours=LABEL_START_HOUR)
            label_end = day + pd.Timedelta(days=1, hours=LABEL_END_HOUR)

            # Features: observation window up to the cutoff (last 30 h, so 24 h
            # trends are available).
            window = g[(lst >= cutoff - pd.Timedelta(hours=30)) & (lst <= cutoff)]
            at = _nearest_at_or_before(window, cutoff)
            if at is None or pd.isna(at["temp_c"]) or pd.isna(at["dewpoint_c"]):
                continue  # no trustworthy evening snapshot -> no example

            at_3h = _nearest_at_or_before(window, cutoff - pd.Timedelta(hours=3))
            at_24h = _nearest_at_or_before(window, cutoff - pd.Timedelta(hours=24))

            # Radiative-cooling potential: clear + calm favours strong nocturnal
            # radiative cooling (the dominant driver of orchard frost). One
            # physically-motivated number = (clear fraction) / (1 + wind). This
            # was the winning feature in the improvement experiments (a small but
            # consistent gain over the raw cloud/wind features). Missing cloud ->
            # mid (4 oktas), missing wind -> a light 3 m/s, matching the tested
            # convention.
            cloud = at["cloud_oktas"] if pd.notna(at["cloud_oktas"]) else 4.0
            wind = at["wind_ms"] if pd.notna(at["wind_ms"]) else 3.0
            radiative_potential = (1.0 - cloud / 8.0) / (1.0 + wind)

            # Label: overnight minimum temperature.
            night = g[(lst >= label_start) & (lst <= label_end)]
            night_temps = night["temp_c"].dropna()
            if len(night_temps) < 3:
                continue  # too few overnight obs to trust the minimum
            tmin = float(night_temps.min())

            def trend(col, past_row, now_row=at):
                if past_row is None or pd.isna(past_row[col]) or pd.isna(now_row[col]):
                    return math.nan
                return float(now_row[col] - past_row[col])

            rows.append({
                "station": station,
                "date": day.date().isoformat(),
                "month": month,
                "doy": int(day.dayofyear),
                "lat": at["lat"], "lon": at["lon"], "elev": at["elev"],
                # 18:00 snapshot
                "temp_c": float(at["temp_c"]),
                "dewpoint_c": float(at["dewpoint_c"]),
                "dewpoint_depression_c": float(at["temp_c"] - at["dewpoint_c"]),
                "slp_hpa": float(at["slp_hpa"]) if pd.notna(at["slp_hpa"]) else math.nan,
                "wind_ms": float(at["wind_ms"]) if pd.notna(at["wind_ms"]) else math.nan,
                "cloud_oktas": float(at["cloud_oktas"]) if pd.notna(at["cloud_oktas"]) else math.nan,
                "radiative_potential": float(radiative_potential),
                # trends (cheap synoptic / advection signal)
                "temp_change_3h": trend("temp_c", at_3h),
                "temp_change_24h": trend("temp_c", at_24h),
                "slp_tendency_3h": trend("slp_hpa", at_3h),
                # label
                "tmin_overnight_c": tmin,
            })
    return pd.DataFrame(rows)


def main() -> None:
    csv_paths = sorted(RAW_DIR.glob("isd_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No isd_*.csv in {RAW_DIR}. Run `python3 fetch_data.py` first."
        )
    print(f"decoding {len(csv_paths)} station-years ...", flush=True)
    obs = pd.concat([decode_station(p) for p in csv_paths], ignore_index=True)
    print(f"  {len(obs)} hourly observations decoded", flush=True)

    nights = build_nights(obs)
    print(f"  built {len(nights)} station-night examples "
          f"(frost-risk windows only)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nights.to_csv(OUT_DIR / "nights.csv", index=False)
    print(f"wrote {OUT_DIR / 'nights.csv'}", flush=True)

    # Quick honest summary.
    frost = (nights["tmin_overnight_c"] <= 0).mean()
    print("\nper-station night counts and frost rate (tmin <= 0C):", flush=True)
    for st, g in nights.groupby("station"):
        print(f"  {st:22s} nights={len(g):5d}  "
              f"frost={100*(g['tmin_overnight_c'] <= 0).mean():4.1f}%  "
              f"tmin range {g['tmin_overnight_c'].min():.1f}..{g['tmin_overnight_c'].max():.1f}",
              flush=True)
    print(f"\noverall frost-night rate: {100*frost:.1f}%", flush=True)


if __name__ == "__main__":
    main()
