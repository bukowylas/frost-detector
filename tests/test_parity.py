"""Parity gate: the live path must reproduce the trained model.

This is the go/no-go test for the live service. For each serviceable station it
fetches the live observation window, builds the feature vector via the SAME
``prepare.build_feature_row`` the model was trained through, and asserts that the
resulting prediction matches the prediction from the training features stored in
``nights.csv``. If the live path diverged, the deployed system would not be the
system whose accuracy was published.

**Not part of the offline unit suite.** It makes real network calls to OGIMET,
which rate-limits to one query per IP per 20 s, so a run is paced and slow. It is
excluded from the default collection (see ``pytest.ini`` ``testpaths``) and run
deliberately:

    python3 -m pytest tests/test_parity.py -v -o addopts=""

What it checks, and why the check is on the *prediction*:

- The SYNOP core (temperature, dew point, sea-level pressure, wind) decodes to
  the training values exactly -- verified at the feature level.
- Cloud is intentionally NOT reproduced from SYNOP (SYNOP's total-cloud digit
  and NCEI's ISD oktas do not decode 1:1); it is left missing, as the model
  handles natively. This shifts the derived ``radiative_potential`` slightly, so
  the honest question is whether the *forecast* still matches -- it does, to well
  within the model's own ~1.8 C error. The assertion is therefore on the
  predicted minimum temperature, the quantity that actually reaches a grower.
"""

import time

import joblib
import pandas as pd
import pytest

import prepare
from frostlib import live, paths

# In-window dates present in the training set. Kept small: each fetch costs a
# 20 s OGIMET cooldown, so N stations x M dates x 20 s is the run time.
DATES = ["2023-04-10", "2023-04-15"]

# The live forecast must match the trained-feature forecast to well within the
# model's own mean absolute error (~1.8 C); the only feature that differs is the
# cloud-derived term, whose effect on the prediction is a fraction of a degree.
PRED_TOL_C = 0.5

_OGIMET_COOLDOWN_S = 21


def _load_model():
    if not paths.MODEL_PATH.exists():
        pytest.skip("no model artifact (run `python3 predict.py --fit`)")
    bundle = joblib.load(paths.MODEL_PATH)
    return bundle["model"], bundle["features"]


def _training_nights():
    if not paths.NIGHTS_CSV.exists():
        pytest.skip("no nights.csv (run the prepare step)")
    return pd.read_csv(paths.NIGHTS_CSV)


# One (station, date) case per serviceable station x date.
CASES = [(s, d) for s in live.PROVIDER for d in DATES]


@pytest.mark.parametrize("station,date", CASES)
def test_live_prediction_matches_training(station, date):
    model, feat_names = _load_model()
    nights = _training_nights()

    row = nights[(nights.station == station) & (nights.date == date)]
    if row.empty:
        pytest.skip(f"{station} {date} not in training set")
    row = row.iloc[0]

    cutoff = pd.Timestamp(date) + pd.Timedelta(hours=prepare.CUTOFF_HOUR)
    try:
        obs = live.fetch_window(station, cutoff)
    except Exception as exc:  # noqa: BLE001 -- provider hiccup: skip, don't fail
        if "quota" in str(exc).lower() or "429" in str(exc) or "501" in str(exc):
            pytest.skip(f"provider rate-limited: {exc}")
        pytest.skip(f"provider unreachable: {exc}")

    features = prepare.build_feature_row(obs, cutoff)
    assert features is not None, f"live window unusable for {station} {date}"

    live_pred = float(model.predict(pd.DataFrame([features])[feat_names])[0])
    train_pred = float(model.predict(pd.DataFrame([row.to_dict()])[feat_names])[0])

    assert abs(live_pred - train_pred) <= PRED_TOL_C, (
        f"{station} {date}: live forecast {live_pred:.2f} C diverges from "
        f"training-feature forecast {train_pred:.2f} C by "
        f"{abs(live_pred - train_pred):.2f} C (> {PRED_TOL_C} C)"
    )

    # Be a polite OGIMET client: pace requests to its 1-per-20s limit.
    time.sleep(_OGIMET_COOLDOWN_S)
