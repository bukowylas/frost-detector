"""Parity gate: the live path must reproduce the trained model.

Go/no-go test for the live service. For each serviceable station and a set of
stratified nights (coldest, lowest-pressure, windiest -- chosen from the training
data so the frost case and the cyclonic sub-1000 hPa case are exercised, not just
convenient clear evenings), it fetches the live window, builds the feature vector
through the SAME ``prepare.build_feature_row`` the model was trained through, and
checks it against the training row.

What is asserted, and the one deliberate gap -- cloud:

- Every live-observed and derived feature EXCEPT cloud must match the training
  value (temperature, dew point, dew-point depression, sea-level pressure, wind,
  the three trends), as must the static geography and calendar fields. These are
  the fields the live path is meant to reproduce, so they are checked directly
  and tightly (FEATURE_TOL, ~float noise).
- ``cloud_oktas`` is NOT sourced live -- no serviceable SYNOP station supplies it
  in a form matching how NCEI derived the training value. It is a genuine frost
  driver (clear, calm nights radiate fastest), so this is a real limitation, not
  a throwaway. The gate does not pretend it away:
    * it asserts cloud is *missing* live (visible gap, never a silent zero);
    * it measures what missing cloud costs on this night -- the gap between the
      forecast the live (cloud-blank) features produce and the forecast the full
      training features produce -- and reports it, so the cost is quantified;
    * it checks the live forecast equals the same-night cloud-blanked training
      forecast (to PRED_TOL_C), confirming that missing cloud is the ONLY thing
      that differs between live and training -- no other feature has drifted.
  The honest reading is: the live system reproduces every feature it can, and
  runs cloud-blank until cloud is either dropped from the model on retrain or
  supplied from a cloud-cover source (see ``frostlib.live`` module docstring).

**Not part of the offline unit suite.** It makes real, rate-limited network calls
to OGIMET (one query per IP per 20 s), so it is excluded from the default
collection (``pytest.ini`` ``testpaths = tests/unit``) and run deliberately:

    python3 -m pytest tests/test_parity.py -v -o addopts=""

Fail-closed: a run in which no case actually executed its assertions (all
skipped, e.g. rate-limited) fails ``test_parity_coverage`` -- a gate that can
pass having checked nothing is not a gate.
"""

import math
import time

import pandas as pd
import pytest

import prepare
from frostlib import live, model_io, paths, physics
from service import config
from service.nightly_job import LIVE_CUTOFF_TOL_MIN

# Features that must reproduce exactly from live observations (cloud excepted).
EXACT_FEATURES = [
    "temp_c", "dewpoint_c", "dewpoint_depression_c", "slp_hpa", "wind_ms",
    "temp_change_3h", "temp_change_24h", "slp_tendency_3h",
    "lat", "lon", "elev", "month", "doy",
]
# Serviceable stations reproduce training to a tenth: temperature can differ by
# one sub-tenth rounding step (0.1 C) between OGIMET's SYNOP and NCEI's ISD, which
# round the same reading independently. 0.15 admits that step and nothing larger
# -- the divergences that made stations unserviceable were 1-9 C, far outside it.
FEATURE_TOL = 0.15
# The live vs cloud-blanked-training forecast check confirms cloud is the only
# MATERIAL difference. It cannot be bit-exact, because the permitted 0.1 C feature
# rounding (above) propagates through the tree ensemble into the forecast -- a
# 0.1 C input step moves a prediction by a few hundredths. 0.05 admits that
# propagated rounding while still catching any real feature drift (which would
# move the forecast by tenths to whole degrees), against a model whose MAE is
# ~1.8 C.
PRED_TOL_C = 0.05

_OGIMET_COOLDOWN_S = 21
_executed = {"count": 0, "stations": set()}  # cases that actually asserted (fail-closed)


def _load():
    if not paths.MODEL_PATH.exists() or not paths.NIGHTS_CSV.exists():
        pytest.skip("model artifact or nights.csv missing")
    try:
        artifact = model_io.load_model(path=paths.MODEL_PATH)
    except model_io.ModelContractError as exc:
        pytest.skip(f"model artifact needs re-freezing (predict.py --fit): {exc}")
    return artifact.model, artifact.features, pd.read_csv(paths.NIGHTS_CSV)


def _extreme_date(g, col, how):
    """The date of the row with the min/max value in ``col`` for group ``g``, or
    None when the column is entirely NaN for this station (e.g. Manchester never
    reported SLP -- ``idxmin`` on an all-NaN column raises, so this is guarded)."""
    valid = g[g[col].notna()]
    if valid.empty:
        return None
    idx = valid[col].idxmin() if how == "min" else valid[col].idxmax()
    return g.loc[idx, "date"]


def _stratified_cases():
    """One test case per (station, purposeful night): the coldest, the lowest-SLP
    and the windiest night per serviceable station -- chosen from the training
    data so the gate exercises frost and cyclonic regimes, not just clear ones.
    A stratifier whose column is all-NaN for a station is skipped for it, not an
    error (see ``_extreme_date``)."""
    if not paths.NIGHTS_CSV.exists():
        return []
    nights = pd.read_csv(paths.NIGHTS_CSV)
    cases = []
    # Iterate the SERVICEABLE set (what the service actually serves), not the whole
    # provider -- the gate must validate exactly the stations that will ship.
    for st in config.SERVICEABLE_STATIONS:
        g = nights[nights.station == st]
        if g.empty:
            continue
        picks = {
            _extreme_date(g, "tmin_overnight_c", "min"),   # coldest (frost)
            _extreme_date(g, "slp_hpa", "min"),            # cyclonic sub-1000
            _extreme_date(g, "wind_ms", "max"),            # windiest
        }
        picks.discard(None)
        cases.extend((st, d) for d in sorted(picks))
    return cases


CASES = _stratified_cases()


@pytest.fixture(autouse=True)
def _pace_ogimet():
    """Sleep AFTER every case (pass or fail) so a failing assertion never removes
    the pacing and cascades the rest into rate-limit skips."""
    yield
    time.sleep(_OGIMET_COOLDOWN_S)


@pytest.mark.parametrize("station,date", CASES)
def test_live_matches_training(station, date):
    model, feat_names, nights = _load()
    row = nights[(nights.station == station) & (nights.date == date)]
    if row.empty:
        pytest.skip(f"{station} {date} not in training set")
    row = row.iloc[0]
    cutoff = pd.Timestamp(date) + pd.Timedelta(hours=prepare.CUTOFF_HOUR)

    try:
        obs = live.fetch_window(station, cutoff)
    except live.OgimetError as exc:
        pytest.skip(f"provider declined (rate-limit/error body): {exc}")
    except Exception as exc:  # noqa: BLE001 -- network unreachable: skip, not fail
        pytest.skip(f"provider unreachable: {exc}")

    # Build with the SAME tolerance the nightly job uses, so the gate validates the
    # path production runs -- not the 90-min default the service never uses.
    feats = prepare.build_feature_row(
        obs, cutoff, cutoff_tol_minutes=LIVE_CUTOFF_TOL_MIN)
    assert feats is not None, f"{station} {date}: live window unusable"

    # The new snapshot_ts field's contract: present, and within the live tolerance.
    assert "snapshot_ts" in feats, f"{station} {date}: snapshot_ts missing"
    snap_gap = cutoff - pd.Timestamp(feats["snapshot_ts"])
    assert pd.Timedelta(0) <= snap_gap <= pd.Timedelta(minutes=LIVE_CUTOFF_TOL_MIN), (
        f"{station} {date}: snapshot {feats['snapshot_ts']} is {snap_gap} from cutoff")

    # Per-feature exact equality (cloud excepted).
    for f in EXACT_FEATURES:
        live_v, train_v = feats[f], float(row[f])
        assert not (pd.isna(live_v) ^ pd.isna(train_v)), (
            f"{station} {date}: {f} missingness differs "
            f"(live={live_v}, train={train_v})")
        if not pd.isna(live_v):
            assert abs(live_v - train_v) <= FEATURE_TOL, (
                f"{station} {date}: {f} live={live_v} vs train={train_v}")

    # Cloud is not sourced live -> must be missing, not silently zero.
    assert pd.isna(feats["cloud_oktas"]), (
        f"{station} {date}: cloud_oktas should be missing live, got "
        f"{feats['cloud_oktas']}")

    # The same training night AS IF cloud had been missing -- exactly what
    # build_feature_row computes when cloud is absent. This isolates the cloud gap
    # from every other feature, so the prediction check below confirms cloud is
    # the ONLY difference between the live and training feature vectors.
    blank = row.to_dict()
    blank["cloud_oktas"] = math.nan
    blank["radiative_potential"] = physics.radiative_potential(math.nan, row["wind_ms"])
    assert abs(feats["radiative_potential"] - blank["radiative_potential"]) <= FEATURE_TOL

    live_pred = float(model.predict(pd.DataFrame([feats])[feat_names])[0])
    blank_pred = float(model.predict(pd.DataFrame([blank])[feat_names])[0])
    full_pred = float(model.predict(pd.DataFrame([row.to_dict()])[feat_names])[0])

    assert abs(live_pred - blank_pred) <= PRED_TOL_C, (
        f"{station} {date}: live forecast {live_pred:.3f} != same-night "
        f"cloud-blanked training forecast {blank_pred:.3f} -- a feature other "
        f"than cloud has drifted")

    # The quantified cloud limitation: what running cloud-blank costs on this
    # night (informational, not an assertion) -- reported, never hidden.
    print(f"  {station} {date}: forecast {live_pred:.2f} C; "
          f"running cloud-blank costs {abs(blank_pred - full_pred):.3f} C here")
    _executed["count"] += 1
    _executed["stations"].add(station)


def test_parity_coverage():
    """Fail-closed: EVERY serviceable station must have validated at least one case.

    A run where only some stations executed (e.g. one rate-limited) is not a pass:
    the gate would go green while half the fleet went unvalidated. So this requires
    coverage of the full serviceable set, not merely that *some* case ran."""
    if not CASES:
        pytest.skip("no cases (nights.csv missing)")
    expected = set(config.SERVICEABLE_STATIONS)
    missing = expected - _executed["stations"]
    assert not missing, (
        f"parity did not validate every serviceable station -- missing {missing} "
        f"(provider rate-limited/unreachable?). A gate that skips a served station "
        "does not pass.")
