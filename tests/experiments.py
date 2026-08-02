"""Feature-improvement experiment bench for Frost Detector.

Idea: don't argue about which feature helps -- try several cheap, physically
motivated variants and let the data rank them. Each experiment is scored on the
SAME held-out fold(s) so results are comparable, against the same FAO baseline.

Fast by design: one LOYO fold ~20s, so a handful of experiments run in a couple
of minutes. Winners get promoted into prepare.py / train.py for a full run.

Usage:
    python3 tests/experiments.py            # score on 1 fold (hold out 2023)
    python3 tests/experiments.py --folds 3  # average over 3 held-out years
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

import train  # reuse TARGET, FAO baseline, DATA

BASE = list(train.FEATURES)


def add_radiative_proxy(df):
    df = df.copy()
    clear = 1 - df["cloud_oktas"].fillna(4) / 8.0        # 1=clear, 0=overcast
    df["radiative_proxy"] = df["dewpoint_depression_c"] * clear
    df["calm_flag"] = (df["wind_ms"].fillna(3) < 2).astype(float)
    return df, ["radiative_proxy", "calm_flag"]


def add_radiative_potential(df):
    df = df.copy()
    clear = 1 - df["cloud_oktas"].fillna(4) / 8.0
    df["radiative_potential"] = clear / (1 + df["wind_ms"].fillna(3))
    return df, ["radiative_potential"]


def add_cooling_rate(df):
    # temp already dropped over last 3h (negative = cooling); we have temp_change_3h
    df = df.copy()
    df["cooling_rate_3h"] = -df["temp_change_3h"]  # positive = cooling
    return df, ["cooling_rate_3h"]


def add_daylength(df):
    # crude night-length proxy from day-of-year (longer nights near winter ends
    # of the windows). sin gives a smooth season signal already in doy, so use a
    # simple "distance from summer solstice" as hours-of-darkness proxy.
    df = df.copy()
    df["winter_proximity"] = np.cos(2 * np.pi * (df["doy"] - 172) / 365)
    return df, ["winter_proximity"]


EXPERIMENTS = {
    "baseline": (lambda d: (d, [])),
    "+radiative_proxy": add_radiative_proxy,
    "+radiative_potential": add_radiative_potential,
    "+cooling_rate": add_cooling_rate,
    "+daylength": add_daylength,
    "-drop_lon_slptend": (lambda d: (d, [])),  # handled via feature list below
}

# For the "drop" experiment, remove suspected-noise features.
DROP_FEATURES = {"lon", "slp_tendency_3h"}


def quick_model():
    # Fixed reasonable params (skip the search -- we compare features, not tuning,
    # and want speed + comparability).
    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
        min_samples_leaf=20, early_stopping=True, random_state=train.RANDOM_STATE,
    )


def score(df, features, held_years):
    maes, faos = [], []
    for held in held_years:
        tr = df[df["year"] != held]
        te = df[df["year"] == held]
        model = quick_model()
        model.fit(tr[features], tr[train.TARGET])
        pred = model.predict(te[features])
        maes.append(mean_absolute_error(te[train.TARGET], pred))
        faos.append(mean_absolute_error(te[train.TARGET],
                                        train.fao_predict(tr, te)))
    return float(np.mean(maes)), float(np.mean(faos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=1)
    args = ap.parse_args()

    df0 = pd.read_csv(train.DATA)
    df0["year"] = pd.to_datetime(df0["date"]).dt.year
    years = sorted(df0["year"].unique())[-args.folds:]
    print(f"scoring on held-out years {years} ({len(df0)} nights)\n", flush=True)

    t0 = time.monotonic()
    rows = []
    for name, fn in EXPERIMENTS.items():
        df, extra = fn(df0)
        if name == "-drop_lon_slptend":
            feats = [f for f in BASE if f not in DROP_FEATURES]
        else:
            feats = BASE + extra
        mae, fao = score(df, feats, years)
        rows.append((name, mae, fao, mae - fao, len(feats)))
        print(f"  {name:22s} MAE {mae:.3f}  (FAO {fao:.3f}, "
              f"model-FAO {mae-fao:+.3f})  [{len(feats)} feats]  "
              f"[{time.monotonic()-t0:.0f}s]", flush=True)

    print("\nranked by MAE (lower is better):", flush=True)
    for name, mae, fao, diff, nf in sorted(rows, key=lambda r: r[1]):
        beat = "beats FAO" if diff < 0 else "loses to FAO"
        print(f"  {mae:.3f}  {name:22s} ({beat} by {abs(diff):.3f})", flush=True)


if __name__ == "__main__":
    main()
