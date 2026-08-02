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
    df = pd.read_csv(DATA)
    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(df[FEATURES], df[TARGET])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    print(f"fitted on {len(df)} nights; saved {MODEL_PATH}")


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


def forecast(args) -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("no saved model -- run `python3 predict.py --fit` first")
    bundle = joblib.load(MODEL_PATH)
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
    elif args.temp is not None and args.dewpoint is not None:
        forecast(args)
    else:
        ap.error("either --fit, or provide at least --temp and --dewpoint")


if __name__ == "__main__":
    main()
