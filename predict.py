"""Fit the final model on all data and predict tomorrow's overnight minimum.

The evaluation (train.py) measures how well the approach works. This script is
the deployable artifact behind it: it fits one model on every available night
and either saves it or uses it to forecast a single evening.

Usage:
    # Fit on all nights and save the model:
    python3 predict.py --fit

    # Forecast from an evening's 18:00 LST observations (frost alarm derived at
    # the +1.5 C ground-frost threshold):
    python3 predict.py --temp 3.0 --dewpoint 0.5 --wind 1.5 --cloud 1 \
                       --pressure 1018 --lat 51.7 --lon 19.4 --elev 180 \
                       --doy 110
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from train import (
    DATA,
    FEATURES,
    RECOMMENDED_ALARM_C,
    TARGET,
)

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "data" / "model.joblib"

# Fixed hyperparameters (a representative point from train.py's search); this
# script is the deployment artifact, not the tuning experiment.
MODEL_PARAMS = {
    "learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 31,
    "min_samples_leaf": 20, "early_stopping": True, "random_state": 42,
}


def fit_and_save() -> None:
    if not DATA.exists():
        raise SystemExit(f"{DATA} not found -- run `python3 prepare.py` first")
    df = pd.read_csv(DATA)
    missing = [c for c in [*FEATURES, TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{DATA} is missing required column(s) {missing} -- re-run "
            "`python3 prepare.py`"
        )
    if df.empty or df[TARGET].isna().any():
        raise SystemExit(
            f"{DATA} has no usable rows ({len(df)} rows, "
            f"{int(df[TARGET].isna().sum())} missing labels)"
        )
    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(df[FEATURES], df[TARGET])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    print(f"fitted on {len(df)} nights; saved {MODEL_PATH}")


# Physically impossible inputs are far more likely a typo or a unit mix-up than a
# real observation. The model would still return a confident number for any of
# them, so a frost alarm computed from a bad input would look exactly like a good
# one -- reject them instead.
_INPUT_BOUNDS = {
    "temp": (-60.0, 60.0),
    "dewpoint": (-60.0, 60.0),
    "wind": (0.0, 120.0),
    "cloud": (0.0, 8.0),
    "pressure": (850.0, 1100.0),
    "lat": (-90.0, 90.0),
    "lon": (-180.0, 180.0),
    "elev": (-500.0, 9000.0),
    "doy": (1, 366),
}


def _validate(args, parser) -> None:
    """Reject forecast inputs that are absent or physically impossible.

    lat/lon/doy have no sensible default: left unset they reach the model as NaN,
    which HistGradientBoosting accepts silently and turns into a prediction from a
    site and season it was never told about.
    """
    for name in ("temp", "dewpoint", "lat", "lon", "doy"):
        if getattr(args, name) is None:
            parser.error(f"--{name} is required to forecast a night")

    for name, (low, high) in _INPUT_BOUNDS.items():
        value = getattr(args, name)
        if value is not None and not low <= value <= high:
            parser.error(f"--{name}={value} is outside the plausible range "
                         f"[{low}, {high}]")

    # Dew point above air temperature means supersaturation: a measurement or
    # unit error. A small margin absorbs rounding in reported observations.
    if args.dewpoint > args.temp + 0.5:
        parser.error(
            f"--dewpoint ({args.dewpoint} C) exceeds --temp ({args.temp} C); "
            "dew point cannot be warmer than the air"
        )


def _derived(args) -> dict:
    """Assemble the feature row from CLI inputs, computing the derived ones."""
    cloud = args.cloud if args.cloud is not None else 4.0
    wind = args.wind if args.wind is not None else 3.0
    return {
        "temp_c": args.temp,
        "dewpoint_c": args.dewpoint,
        "dewpoint_depression_c": args.temp - args.dewpoint,
        "slp_hpa": args.pressure,
        "wind_ms": args.wind,
        "cloud_oktas": args.cloud,
        "radiative_potential": (1.0 - cloud / 8.0) / (1.0 + wind),
        "temp_change_3h": args.temp_change_3h,
        "temp_change_24h": args.temp_change_24h,
        "slp_tendency_3h": args.slp_tendency_3h,
        "lat": args.lat, "lon": args.lon, "elev": args.elev,
        "doy": args.doy,
    }


def _load_bundle() -> dict:
    """Load the saved model, refusing anything stale or corrupt.

    A truncated/foreign joblib file, or one saved by an older FEATURES list,
    would otherwise either blow up deep inside sklearn or -- worse -- predict
    from a feature set that no longer means what this script thinks it does.
    """
    if not MODEL_PATH.exists():
        raise SystemExit("no saved model -- run `python3 predict.py --fit` first")
    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise SystemExit(
            f"could not load {MODEL_PATH} ({exc!r}) -- delete it and re-run "
            "`python3 predict.py --fit`"
        ) from exc
    if not isinstance(bundle, dict) or not {"model", "features"} <= bundle.keys():
        raise SystemExit(
            f"{MODEL_PATH} is not a Frost Detector model bundle -- re-run "
            "`python3 predict.py --fit`"
        )
    if list(bundle["features"]) != list(FEATURES):
        raise SystemExit(
            f"{MODEL_PATH} was fitted on a different feature set "
            f"({bundle['features']}) than this code expects ({FEATURES}) -- "
            "re-run `python3 predict.py --fit`"
        )
    return bundle


def forecast(args) -> None:
    bundle = _load_bundle()
    row = pd.DataFrame([_derived(args)])[bundle["features"]]
    tmin = float(bundle["model"].predict(row)[0])
    alarm = tmin <= RECOMMENDED_ALARM_C
    print(f"predicted overnight minimum: {tmin:+.1f} C")
    print(f"frost alarm (predicted <= {RECOMMENDED_ALARM_C:+.1f} C): "
          f"{'YES' if alarm else 'no'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", action="store_true", help="fit on all data and save")
    # Forecast inputs (18:00 LST evening state).
    ap.add_argument("--temp", type=float, help="air temperature (C)")
    ap.add_argument("--dewpoint", type=float, help="dew point (C)")
    ap.add_argument("--wind", type=float, default=None, help="wind speed (m/s)")
    ap.add_argument("--cloud", type=float, default=None, help="cloud cover (oktas 0-8)")
    ap.add_argument("--pressure", type=float, default=None, help="sea-level pressure (hPa)")
    ap.add_argument("--temp-change-3h", type=float, default=0.0, dest="temp_change_3h")
    ap.add_argument("--temp-change-24h", type=float, default=0.0, dest="temp_change_24h")
    ap.add_argument("--slp-tendency-3h", type=float, default=0.0, dest="slp_tendency_3h")
    ap.add_argument("--lat", type=float, help="station latitude")
    ap.add_argument("--lon", type=float, help="station longitude")
    ap.add_argument("--elev", type=float, default=100.0, help="station elevation (m)")
    ap.add_argument("--doy", type=int, help="day of year")
    args = ap.parse_args()

    if args.fit:
        fit_and_save()
    elif args.temp is not None or args.dewpoint is not None:
        _validate(args, ap)
        forecast(args)
    else:
        ap.error("either --fit, or forecast inputs "
                 "(--temp --dewpoint --lat --lon --doy)")


if __name__ == "__main__":
    main()
