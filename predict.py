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

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from frostlib import model_io, physics
from frostlib.paths import METRICS_JSON, MODEL_PATH
from train import (
    DATA,
    FEATURES,
    FIXED_MODEL_PARAMS,
    RECOMMENDED_ALARM_C,
    TARGET,
)


def _measured_mae() -> tuple[float | None, float | None]:
    """(full-feature MAE, live/cloud-blank MAE) from the evaluation metrics.

    The full number is measured with real cloud; the live number is measured with
    ``cloud_oktas`` forced missing -- the regime the live service actually runs in
    (no serviceable station supplies cloud). The live figure is the accuracy a
    grower receives; the artifact stores both so the distinction is never lost.
    Absent if the metrics file has not been written; the freeze still proceeds.
    """
    if not METRICS_JSON.exists():
        return None, None
    metrics = json.loads(METRICS_JSON.read_text())
    full = metrics.get("leave_one_year_out", {}).get("mae_mean")
    live = metrics.get("leave_one_year_out_cloud_blank_live", {}).get("mae_mean")
    return full, live


def fit_and_save() -> None:
    df = pd.read_csv(DATA)
    # Fixed hyperparameters (shared with train.py); this script is the
    # deployment artifact, not the tuning experiment.
    model = HistGradientBoostingRegressor(**FIXED_MODEL_PARAMS)
    model.fit(df[FEATURES], df[TARGET])
    years = pd.to_datetime(df["date"]).dt.year.unique()
    mae_full, mae_live = _measured_mae()
    version = model_io.save_model(
        model, FEATURES,
        training_years=years,
        training_stations=df["station"].unique(),
        n_training_nights=len(df),
        # mae_c is the LIVE number -- what the deployed service delivers -- not the
        # with-cloud headline. Both are kept in extra so the skew is explicit.
        mae_c=mae_live if mae_live is not None else mae_full,
        extra={
            "mae_c_full": mae_full,
            "mae_c_live": mae_live,
            "cloud_train_serve_skew": (
                "ACCEPTED train/serve skew (deliberate, not a bug). The model is "
                "trained with cloud_oktas, but the live service runs with it "
                "missing (no serviceable station supplies it in the training form), "
                "which shifts radiative_potential to its cloud-missing default. "
                "This is accepted rather than closed: cloud is a real frost driver "
                "and kept in the model, and the cost is small and measured -- "
                "mae_c (and mae_c_live) is the cloud-blank accuracy the service "
                "actually delivers, vs mae_c_full with cloud. To close it later, "
                "either retrain without cloud or supply cloud from a separate "
                "source. See README (Live service)."),
        },
        path=MODEL_PATH,
    )
    print(f"fitted on {len(df)} nights; froze model {version} "
          f"(mae_live={mae_live}, mae_full={mae_full})")


def _derived(args) -> dict:
    """Assemble the feature row from CLI inputs, computing the derived ones.

    The derived features come from frostlib.physics -- the same code prepare.py
    used to build the training rows, so a served row cannot drift from what the
    model was fit on.
    """
    return {
        "temp_c": args.temp,
        "dewpoint_c": args.dewpoint,
        "dewpoint_depression_c": physics.dewpoint_depression_c(
            args.temp, args.dewpoint),
        "slp_hpa": args.pressure,
        "wind_ms": args.wind,
        "cloud_oktas": args.cloud,
        "radiative_potential": physics.radiative_potential(args.cloud, args.wind),
        "temp_change_3h": args.temp_change_3h,
        "temp_change_24h": args.temp_change_24h,
        "slp_tendency_3h": args.slp_tendency_3h,
        "lat": args.lat, "lon": args.lon, "elev": args.elev,
        "doy": args.doy,
    }


def forecast(args) -> None:
    # Assert the artifact's features match the code's, so a served row cannot be
    # scored in the wrong column order.
    try:
        artifact = model_io.load_model(expected_features=FEATURES, path=MODEL_PATH)
    except model_io.ModelContractError as exc:
        raise SystemExit(str(exc))
    row = pd.DataFrame([_derived(args)])
    tmin = float(artifact.predict(row)[0])
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
    # Trends default to NaN, not 0.0: zero is a real, informative value ("no
    # change") -- a substantive claim about the synoptic situation -- so an absent
    # trend must be missing, which the model handles natively, not silently zero.
    nan = float("nan")
    ap.add_argument("--temp-change-3h", type=float, default=nan, dest="temp_change_3h")
    ap.add_argument("--temp-change-24h", type=float, default=nan, dest="temp_change_24h")
    ap.add_argument("--slp-tendency-3h", type=float, default=nan, dest="slp_tendency_3h")
    ap.add_argument("--lat", type=float, help="station latitude")
    ap.add_argument("--lon", type=float, help="station longitude")
    ap.add_argument("--elev", type=float, help="station elevation (m)")
    ap.add_argument("--doy", type=int, help="day of year")
    args = ap.parse_args()

    if args.fit:
        fit_and_save()
    elif args.temp is not None and args.dewpoint is not None:
        # Geography and day-of-year are real inputs, not defaultable -- an invented
        # elevation or coordinate is the never-impute rule broken at the CLI.
        missing = [n for n in ("lat", "lon", "doy") if getattr(args, n) is None]
        if missing:
            ap.error(f"a forecast needs {', '.join('--' + m for m in missing)}")
        forecast(args)
    else:
        ap.error("either --fit, or provide at least --temp and --dewpoint")


if __name__ == "__main__":
    main()
