"""Prepare step: raw ISD CSVs -> one modelling row per (station, frost-risk night).

The task is an overnight-minimum-temperature forecast: from what a weather
station observes by early evening, predict how cold it will get overnight. Each
output row is therefore ONE NIGHT:

  features  = the station's state at the 18:00 LST cutoff (+ recent trends)
  label     = the minimum temperature observed 20:00 LST -> 08:00 LST next day

with the 18:00-20:00 window excluded from both, so no feature is contemporaneous
with the label period (the core anti-leakage rule -- see README / consult notes).

Design decisions made here (documented so they can be revisited):

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
``data/nights.csv`` (one row per station-night) plus a coverage report. ISD
field decoding honours the missing-value sentinels and quality flags.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import pandas as pd

from frostlib import isd, paths, physics

# Where this step reads and writes (frostlib.paths owns the defaults).
RAW_DIR = paths.RAW_DIR
OUT_DIR = paths.DATA_DIR

# Cutoff / label window in LST.
CUTOFF_HOUR = 18            # features use observations at/just before this
LABEL_START_HOUR = 20      # overnight-min window start (same evening)
LABEL_END_HOUR = 8         # overnight-min window end (next morning)
LATE_HOUR = 4              # a night must have an obs at/after this (near dawn)
# How far back from the cutoff the feature window reaches: enough for the 24 h
# trend lookback plus the tolerance _nearest_at_or_before allows.
FEATURE_WINDOW_HOURS = 30
# The 18:00-20:00 gap between them is excluded from both by construction.

# Frost-risk windows as (month, day) ranges, inclusive.
RISK_WINDOWS = [((3, 1), (5, 31)), ((9, 15), (11, 15))]


def _in_risk_window(month: int, day: int) -> bool:
    for (m0, d0), (m1, d1) in RISK_WINDOWS:
        if (m0, d0) <= (month, day) <= (m1, d1):
            return True
    return False


def decode_station(csv_path: Path) -> pd.DataFrame:
    """Decode one raw station-year CSV into LST-stamped hourly observations."""
    # GA1 (cloud) is an optional ISD group -- some station-years lack the column
    # entirely -- so read only the columns actually present and fill the rest
    # with NaN, rather than let usecols hard-fail and kill the whole run.
    wanted = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION",
              "TMP", "DEW", "SLP", "WND", "GA1"]
    header = pd.read_csv(csv_path, nrows=0).columns
    present = [c for c in wanted if c in header]
    raw = pd.read_csv(csv_path, dtype=str, low_memory=False, usecols=present)
    for col in wanted:
        if col not in raw.columns:
            raw[col] = pd.NA
    station = paths.station_name_from_path(csv_path)
    # Start the frame from a length-carrying column; assigning a scalar to a
    # still-empty DataFrame would create a zero-length column and silently drop
    # every row (the station column would be empty, breaking the later groupby).
    # Shift UTC to Local Standard Time, then drop the tz so the column is naive
    # LST -- keeping a UTC tz-label on LST values is a trap for anything that
    # later tz-converts or compares against a genuinely-UTC series.
    # ISD DATE is a fixed ISO format; specifying it is faster and deterministic
    # (no per-element guessing) and silences pandas' format-inference warning.
    # errors="coerce" still turns a malformed value into NaT rather than raising.
    utc = pd.to_datetime(raw["DATE"], format="%Y-%m-%dT%H:%M:%S",
                         utc=True, errors="coerce")
    lst = (utc + pd.to_timedelta(isd.lst_offset(station), unit="h")).dt.tz_localize(None)
    out = pd.DataFrame({"lst": lst})
    out["station"] = station
    out["lat"] = pd.to_numeric(raw["LATITUDE"], errors="coerce")
    out["lon"] = pd.to_numeric(raw["LONGITUDE"], errors="coerce")
    out["elev"] = pd.to_numeric(raw["ELEVATION"], errors="coerce")
    out["temp_c"] = raw["TMP"].map(isd.temp_c)
    out["dewpoint_c"] = raw["DEW"].map(isd.temp_c)
    out["slp_hpa"] = raw["SLP"].map(isd.slp_hpa)
    out["wind_ms"] = raw["WND"].map(isd.wind_speed_ms)
    out["cloud_oktas"] = raw["GA1"].map(isd.cloud_oktas)
    return out.dropna(subset=["lst"])


def _nearest_at_or_before(day_obs: pd.DataFrame, target, tol_minutes=90,
                          require_col=None):
    """The last observation at or before `target`, within a tolerance.

    ISD reports irregularly (2-3x/hour with gaps), so 'the 18:00 value' is really
    'the most recent observation by 18:00'. A tolerance guards against using a
    stale value across a long data gap. If ``require_col`` is given, only rows
    with a non-null value in that column are considered -- so a partial report
    landing at 17:55 with no temperature doesn't shadow a good 17:30 reading.
    """
    prior = day_obs[day_obs["lst"] <= target]
    if require_col is not None:
        prior = prior[prior[require_col].notna()]
    if prior.empty:
        return None
    row = prior.iloc[-1]
    if (target - row["lst"]).total_seconds() > tol_minutes * 60:
        return None
    return row


def build_feature_row(obs: pd.DataFrame, cutoff) -> dict | None:
    """The model's input features for one station-night, or None if unusable.

    THE single feature builder: the training path (``build_nights`` below) and
    the live path must both call this, so the live service cannot drift into
    feeding the model a subtly different vector than it was trained on.

    ``obs`` is one station's observations covering (at least) the run-up to
    ``cutoff``; only the last ``FEATURE_WINDOW_HOURS`` up to the cutoff are used,
    so passing a wider frame is safe. Nothing here looks past the cutoff -- the
    label window is the caller's business, which is what makes this reusable
    live, where the night has not happened yet.

    Returns None when the window has no usable cutoff snapshot (no observation
    within tolerance, or one lacking temperature or dew point). Missing data is
    never imputed: the caller records the rejection instead.
    """
    # Sorted here, not assumed: "the last observation by 18:00" is a positional
    # lookup, so an unsorted live payload would silently pick the wrong row. The
    # sort must be STABLE: ISD sometimes reports two rows at the same timestamp,
    # and an unstable sort would reorder them, changing which one is "last".
    window = obs[(obs["lst"] >= cutoff - pd.Timedelta(hours=FEATURE_WINDOW_HOURS))
                 & (obs["lst"] <= cutoff)].sort_values("lst", kind="stable")
    at = _nearest_at_or_before(window, cutoff, require_col="temp_c")
    if at is None or pd.isna(at["dewpoint_c"]):
        return None

    # Trend lookups require a real value in the column being differenced, so a
    # valueless partial report doesn't shadow an earlier good reading.
    lag3 = cutoff - pd.Timedelta(hours=3)
    lag24 = cutoff - pd.Timedelta(hours=24)
    at_3h_t = _nearest_at_or_before(window, lag3, require_col="temp_c")
    at_24h_t = _nearest_at_or_before(window, lag24, require_col="temp_c")
    at_3h_p = _nearest_at_or_before(window, lag3, require_col="slp_hpa")

    # Radiative-cooling potential, from frostlib.physics so a live caller and
    # the training rows cannot compute it differently. Its missing-data defaults
    # feed only this derived term; the raw cloud/wind features stay missing,
    # which the model handles natively.
    radiative_potential = physics.radiative_potential(
        at["cloud_oktas"], at["wind_ms"])

    def trend(col, past_row, now_row=at):
        if past_row is None or pd.isna(past_row[col]) or pd.isna(now_row[col]):
            return math.nan
        return float(now_row[col] - past_row[col])

    return {
        "month": cutoff.month,
        "doy": int(cutoff.dayofyear),
        "lat": at["lat"], "lon": at["lon"], "elev": at["elev"],
        # cutoff snapshot
        "temp_c": float(at["temp_c"]),
        "dewpoint_c": float(at["dewpoint_c"]),
        "dewpoint_depression_c": physics.dewpoint_depression_c(
            at["temp_c"], at["dewpoint_c"]),
        "slp_hpa": float(at["slp_hpa"]) if pd.notna(at["slp_hpa"]) else math.nan,
        "wind_ms": float(at["wind_ms"]) if pd.notna(at["wind_ms"]) else math.nan,
        "cloud_oktas": float(at["cloud_oktas"]) if pd.notna(at["cloud_oktas"]) else math.nan,
        "radiative_potential": float(radiative_potential),
        # trends (cheap synoptic / advection signal)
        "temp_change_3h": trend("temp_c", at_3h_t),
        "temp_change_24h": trend("temp_c", at_24h_t),
        "slp_tendency_3h": trend("slp_hpa", at_3h_p),
    }


def build_nights(obs: pd.DataFrame) -> tuple[pd.DataFrame, Counter]:
    """Collapse hourly observations into one row per (station, frost-risk night).

    For each station and each evening date D in a risk window, features come from
    the 18:00 LST cutoff on D (plus 3 h / 24 h trends), and the label is the
    minimum temperature in [20:00 LST D, 08:00 LST D+1].
    """
    rows = []
    rejected: Counter = Counter()
    for station, g in obs.groupby("station", sort=False):
        g = g.sort_values("lst").reset_index(drop=True)
        lst = g["lst"]
        # Iterate candidate evening dates present in this station's record.
        dates = pd.Series(lst.dt.normalize().unique())
        for day in dates:
            month, dom = day.month, day.day
            if not _in_risk_window(month, dom):
                rejected["out_of_window"] += 1
                continue
            cutoff = day + pd.Timedelta(hours=CUTOFF_HOUR)
            label_start = day + pd.Timedelta(hours=LABEL_START_HOUR)
            label_end = day + pd.Timedelta(days=1, hours=LABEL_END_HOUR)

            # Features: the shared builder, exactly as the live path will call it.
            features = build_feature_row(g, cutoff)
            if features is None:
                rejected["no_evening_snapshot"] += 1
                continue

            # Label: overnight minimum temperature. The true minimum usually
            # falls near dawn, so a night observed only in the evening would give
            # a warm-biased label -- an under-warning error, the worst kind here.
            # Require at least one observation late in the window (after
            # LATE_HOUR LST) as well as a minimum count. Accepted trade-off: this
            # bounds *coverage near dawn* but not gaps within the window, so a
            # night sampled 20:00 / 23:00 / 04:05 still passes and could miss a
            # 05:30 minimum -- diminishing returns to police further.
            night = g[(lst >= label_start) & (lst <= label_end)]
            night_obs = night.dropna(subset=["temp_c"])
            night_temps = night_obs["temp_c"]
            late = night_obs["lst"] >= (day + pd.Timedelta(days=1, hours=LATE_HOUR))
            if len(night_temps) < 3 or not late.any():
                rejected["no_late_obs"] += 1
                continue  # too few / no late (near-dawn) obs -> untrustworthy min
            tmin = float(night_temps.min())

            rows.append({
                "station": station,
                "date": day.date().isoformat(),
                **features,
                # label (training only -- the live path has no label yet)
                "tmin_overnight_c": tmin,
            })
    return pd.DataFrame(rows), rejected


def main() -> None:
    csv_paths = paths.raw_station_years(RAW_DIR)
    if not csv_paths:
        raise FileNotFoundError(
            f"No isd_*.csv in {RAW_DIR}. Run `python3 fetch_data.py` first."
        )
    print(f"decoding {len(csv_paths)} station-years ...", flush=True)
    obs = pd.concat([decode_station(p) for p in csv_paths], ignore_index=True)
    print(f"  {len(obs)} hourly observations decoded", flush=True)

    nights, rejected = build_nights(obs)
    print(f"  built {len(nights)} station-night examples "
          f"(frost-risk windows only)", flush=True)

    # Coverage report: why candidate nights were rejected (out-of-window nights
    # are expected; the others show what the quality filters cost).
    print("  candidate nights rejected:", flush=True)
    for reason, n in rejected.most_common():
        print(f"    {reason:22s} {n}", flush=True)

    if nights.empty:
        raise SystemExit("no usable nights -- see the rejection counts above")

    out_csv = OUT_DIR / paths.NIGHTS_NAME
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nights.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}", flush=True)

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
