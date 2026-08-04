"""Train and honestly evaluate the overnight-minimum-temperature forecaster.

Reads ``data/nights.csv`` (one row per station frost-risk night) and predicts
``tmin_overnight_c`` -- tomorrow morning's minimum temperature -- from the 18:00
LST evening state. A frost alarm is then a threshold on the prediction, so one
model serves every grower and every crop.

Why regression (not frost/no-frost classification): the magnitude is the
decision (-0.5C you protect, -5C you may accept the loss), one model serves all
thresholds, and MAE in C is a self-interpreting, checkable metric.

Evaluation is honest by construction:

- **Leave-one-year-out (LOYO)** is the headline. Year-to-year variability is the
  dominant uncertainty for a weather target, so the error bar that matters is the
  SPREAD of MAE across held-out years, not a single pooled split.
- **Leave-one-station-out (LOSO)** holds out one station at a time -- but nearby
  stations share the same night's air mass, so this understates difficulty.
- **Leave-one-country-out (LOCO)** is the genuine transfer test: train on one
  region, predict the other, never having seen its air masses -- run with and
  without the geographic features (which are pure extrapolation across a border).
- Two REAL baselines, not a dummy:
  1. **Climatology** -- a per-station-month mean tmin (pooled-month fallback for
     an unseen station).
  2. **The FAO rule** tmin = a*temp + b*dewpoint + c (Snyder & de Melo-Abreu,
     2005), the operational evening-dewpoint estimator extension services teach.
  Both are FIT ON TRAINING data ONLY inside each fold -- fitting them over all
  data would leak the held-out period's climate.

Speed: HistGradientBoosting (fast, native NaN) and a small randomized search
per fold; progress + ETA printed throughout.

Usage:
    python3 train.py
"""

from __future__ import annotations

import os
import time

# Parallelise at one level only (the outer CV); cap OpenMP before sklearn so the
# native-threaded estimators don't oversubscribe the cores.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV

from frostlib import paths, physics
from frostlib.progress import Timer

DATA = paths.NIGHTS_CSV
OUT_DIR = paths.DATA_DIR

FEATURES = [
    "temp_c", "dewpoint_c", "dewpoint_depression_c", "slp_hpa", "wind_ms",
    "cloud_oktas", "radiative_potential", "temp_change_3h", "temp_change_24h",
    "slp_tendency_3h", "lat", "lon", "elev", "doy",
]
TARGET = "tmin_overnight_c"
# A frost night is defined by the actual overnight minimum dropping to <= 0 C.
FROST_TRUTH_C = 0.0
# The ALARM fires when the *predicted* minimum is at/below a threshold. Because
# the model is slightly conservative and a 2 m screen temperature runs warmer
# than ground frost (the standard ground-frost correction), alarming at exactly
# 0 C misses real frosts. We sweep alarm thresholds and report the precision/
# recall trade-off, so a grower (who cares far more about catching a frost than
# about a false alarm) can pick an operating point. The recommended default
# alarms early, at +1.5 C predicted.
ALARM_THRESHOLDS_C = [0.0, 0.5, 1.0, 1.5, 2.0]
RECOMMENDED_ALARM_C = 1.5
# Evening wind speed (m/s) separating the calm, radiative-cooling regime from
# the windier, advection-prone regime. Radiative frost (calm) is what surface
# data predicts well; the split makes the difference measurable.
CALM_WIND_MS = 2.0
RANDOM_STATE = 42

# A representative point from the search below, for the places that want ONE
# fixed comparable model instead of a per-fold search: predict.py's deployment
# artifact and tests/experiments.py's feature bench.
FIXED_MODEL_PARAMS = {
    "learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 31,
    "min_samples_leaf": 20, "early_stopping": True,
    "random_state": RANDOM_STATE,
}

assert RECOMMENDED_ALARM_C in ALARM_THRESHOLDS_C, (
    "RECOMMENDED_ALARM_C must be one of ALARM_THRESHOLDS_C"
)

_timer = Timer(style="clock")
step = _timer.log


def load_nights(path=None) -> pd.DataFrame:
    """Read nights.csv and add the grouping columns the CV folds hold out.

    Shared with the timing/experiment scripts under tests/ so every consumer of
    the dataset derives ``year`` and ``country`` the same way. ``path`` is read
    from DATA at call time, so a caller can point the module at another dataset.
    """
    df = pd.read_csv(DATA if path is None else path)
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["country"] = df["station"].str.slice(0, 2)  # pl / uk
    return df


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _clean_nan(obj):
    """Recursively replace NaN floats with None so the output is valid JSON.

    A bare NaN token is not valid JSON (jq, JS, R's jsonlite all reject it) even
    though Python's json.loads accepts it.
    """
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _dump_json(summary) -> str:
    # allow_nan=False so anything that slips through _clean_nan fails loudly.
    return json.dumps(_clean_nan(summary), indent=2, allow_nan=False)


# --- baselines (fit train-only inside each fold) ----------------------------


def climatology_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Per-station-month mean tmin (a stable seasonal climatology normal).

    Fit on train rows only. Falls back, in order, to the station's overall mean,
    then a pooled month mean across training stations (for an unseen station),
    then the global mean.
    """
    global_mean = train[TARGET].mean()
    by_station = train.groupby("station")[TARGET].mean()
    # day-of-year is noisy per station; use a per-station-month mean as a stable
    # climatological normal (a full doy curve would overfit with ~5 years).
    by_station_month = train.groupby(["station", "month"])[TARGET].mean()
    # Fallback for an UNSEEN station (leave-one-station-out): a pooled month mean
    # across the training stations -- still a real seasonal climatology. Without
    # this, an unseen station would collapse to the grand mean and make the
    # baseline artificially weak (inflating the model's margin).
    by_month = train.groupby("month")[TARGET].mean()
    preds = []
    for _, row in test.iterrows():
        key = (row["station"], row["month"])
        if key in by_station_month.index:
            preds.append(by_station_month[key])
        elif row["station"] in by_station.index:
            preds.append(by_station[row["station"]])
        elif row["month"] in by_month.index:
            preds.append(by_month[row["month"]])
        else:
            preds.append(global_mean)
    return np.array(preds, dtype=float)


def fao_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """FAO rule tmin = a*temp + b*dewpoint + c, fit per station on train rows.

    Uses the 18:00 evening temperature and dewpoint (our cutoff snapshot) as the
    operational inputs. Coefficients are site-specific, so we fit one linear
    model per station; unseen stations fall back to a pooled fit.
    """
    def fit(df):
        model = LinearRegression()
        model.fit(df[["temp_c", "dewpoint_c"]].to_numpy(), df[TARGET].to_numpy())
        return model

    pooled = fit(train)
    per_station = {}
    for st, df in train.groupby("station"):
        if len(df) >= 30:
            per_station[st] = fit(df)

    preds = np.empty(len(test), dtype=float)
    x = test[["temp_c", "dewpoint_c"]].to_numpy()
    for i, st in enumerate(test["station"].to_numpy()):
        model = per_station.get(st, pooled)
        preds[i] = model.predict(x[i : i + 1])[0]
    return preds


# --- the model --------------------------------------------------------------


def tuned_model(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingRegressor:
    """A HistGradientBoosting regressor with a small randomized search.

    HGB handles NaN natively, so no imputation (and no imputation-leak surface).
    The search is deliberately small to stay well inside the time budget.

    Known, accepted limitation: the inner RandomizedSearchCV splits the training
    rows randomly, not by station/year. So during hyperparameter *selection* a
    station can sit in both inner-train and inner-validation. This does NOT leak
    into the headline LOYO/LOSO metrics -- the final model is still scored only
    on the untouched held-out group -- but the chosen hyperparameters are picked
    on slightly optimistic inner validation. The effect is small (HGB is not very
    sensitive here) and fixing it (grouped inner CV) is not worth the complexity;
    documented rather than fixed.
    """
    search = RandomizedSearchCV(
        HistGradientBoostingRegressor(
            early_stopping=True, random_state=RANDOM_STATE
        ),
        {
            "learning_rate": [0.03, 0.05, 0.1],
            "max_iter": [300, 500],
            "max_leaf_nodes": [15, 31, 63],
            "min_samples_leaf": [20, 50],
            "l2_regularization": [0.0, 0.1, 1.0],
        },
        n_iter=10,
        scoring="neg_mean_absolute_error",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_


def evaluate_fold(train, test, label, features=FEATURES):
    """Fit model + baselines on train, score all three on test.

    Returns per-fold MAE/RMSE plus the raw prediction arrays, so the caller can
    pool predictions across folds (the honest way to compute frost recall/
    precision when fold base rates differ) and compute paired model-vs-FAO
    margins.
    """
    model = tuned_model(train[features], train[TARGET])
    y_true = test[TARGET].to_numpy()
    y_model = model.predict(test[features])
    y_clim = climatology_predict(train, test)
    y_fao = fao_predict(train, test)
    wind = test["wind_ms"].to_numpy()

    # Error sliced by wind regime. Radiative frost forms on calm nights (surface
    # data is the physics); advective frost is driven by an incoming air mass a
    # single station cannot see, so a surface-only model should be weaker there.
    # Nights with missing wind fall in NEITHER slice (NaN fails both comparisons),
    # so the two counts won't sum to n_test -- we report all three.
    calm = wind <= CALM_WIND_MS
    windy = wind > CALM_WIND_MS
    mae_calm = (mean_absolute_error(y_true[calm], y_model[calm])
                if calm.any() else float("nan"))
    mae_windy = (mean_absolute_error(y_true[windy], y_model[windy])
                 if windy.any() else float("nan"))

    return {
        "fold": label,
        "n_train": len(train),
        "n_test": len(test),
        "mae_model": mean_absolute_error(y_true, y_model),
        "rmse_model": rmse(y_true, y_model),
        "mae_climatology": mean_absolute_error(y_true, y_clim),
        "mae_fao": mean_absolute_error(y_true, y_fao),
        "mae_margin_vs_fao": (mean_absolute_error(y_true, y_model)
                              - mean_absolute_error(y_true, y_fao)),
        "mae_calm": mae_calm,
        "mae_windy": mae_windy,
        "n_calm": int(calm.sum()),
        "n_windy": int(windy.sum()),
        "n_wind_missing": int(np.isnan(wind).sum()),
        # raw arrays for pooled frost-alarm metrics and paired stats
        "_y_true": y_true,
        "_y_model": y_model,
        "_y_fao": y_fao,
    }


def _alarm_sweep(y_true, y_pred):
    """Recall/precision at each alarm threshold, pooled over the given arrays.

    Frost truth is tmin <= 0 C; the alarm fires when the PREDICTED tmin is at or
    below the threshold.
    """
    true_frost = y_true <= FROST_TRUTH_C
    out = {}
    for thr in ALARM_THRESHOLDS_C:
        pred = y_pred <= thr
        tp = int((true_frost & pred).sum())
        fn = int((true_frost & ~pred).sum())
        fp = int((~true_frost & pred).sum())
        out[thr] = {
            "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
            "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        }
    return out


def run_cv(df, group_col, name, features=FEATURES):
    """Leave-one-<group>-out: hold out each group value in turn."""
    groups = sorted(df[group_col].unique())
    step(f"{name}: {len(groups)} folds ...")
    results = []
    run_start = time.monotonic()  # per-run, so a later run's ETA excludes earlier ones
    for i, held in enumerate(groups, start=1):
        done = i - 1  # folds actually finished
        eta = ((time.monotonic() - run_start) / done * (len(groups) - done)) if done else 0
        step(f"  [{i}/{len(groups)}] hold out {group_col}={held} "
             f"(~{eta:.0f}s left) ...")
        train = df[df[group_col] != held]
        test = df[df[group_col] == held]
        results.append(evaluate_fold(train, test, str(held), features=features))
    return pd.DataFrame(results)


def summarise(res: pd.DataFrame, name: str) -> dict:
    mae = res["mae_model"]
    step(f"{name} summary:")
    for _, r in res.iterrows():
        step(f"    {r['fold']:>14s}  MAE {r['mae_model']:.2f}C  "
             f"(clim {r['mae_climatology']:.2f}, FAO {r['mae_fao']:.2f}, "
             f"vs-FAO {r['mae_margin_vs_fao']:+.2f})  "
             f"[train {r['n_train']}, test {r['n_test']}]")
    step(f"  {name} MAE: mean {mae.mean():.2f}C  spread {mae.min():.2f}-{mae.max():.2f}  "
         f"| clim {res['mae_climatology'].mean():.2f}  FAO {res['mae_fao'].mean():.2f}")

    # Paired margin vs FAO: mean and range of (model - FAO) per fold. A tight,
    # consistently-negative margin is the defensible "beats FAO" claim -- far
    # stronger than comparing two means against a wide fold spread.
    margin = res["mae_margin_vs_fao"]
    step(f"  {name} margin vs FAO (model-FAO, per fold): mean {margin.mean():+.3f}C  "
         f"range {margin.min():+.3f}..{margin.max():+.3f}")

    # Error by wind regime (calm=radiative easier; windy=advective harder).
    # Missing-wind nights are in neither slice, so the counts are shown too.
    mae_calm = float(np.nanmean(res["mae_calm"]))
    mae_windy = float(np.nanmean(res["mae_windy"]))
    n_calm, n_windy = int(res["n_calm"].sum()), int(res["n_windy"].sum())
    n_wmiss = int(res["n_wind_missing"].sum())
    step(f"  {name} MAE by regime: calm (<= {CALM_WIND_MS} m/s) {mae_calm:.2f}C "
         f"[n={n_calm}]  vs windy {mae_windy:.2f}C [n={n_windy}]  "
         f"(wind missing: {n_wmiss})")

    # Pool predictions across folds, then compute the frost-alarm sweep once on
    # the pooled set -- an unweighted per-fold mean is not a rate anyone
    # experiences when fold base rates differ. Sweep the FAO baseline too, so the
    # grower-facing comparison (model+threshold vs FAO+threshold) actually exists.
    y_true = np.concatenate(res["_y_true"].to_list())
    y_model = np.concatenate(res["_y_model"].to_list())
    y_fao = np.concatenate(res["_y_fao"].to_list())
    sweep_model = _alarm_sweep(y_true, y_model)
    sweep_fao = _alarm_sweep(y_true, y_fao)

    step(f"  {name} frost-alarm trade-off, pooled (alarm when predicted tmin <= thr):")
    step("      thr    model recall/prec     FAO recall/prec")
    for thr in ALARM_THRESHOLDS_C:
        m, f = sweep_model[thr], sweep_fao[thr]
        star = "  <-" if thr == RECOMMENDED_ALARM_C else "    "
        step(f"    {thr:+.1f}C   {m['recall']:.2f} / {m['precision']:.2f}"
             f"        {f['recall']:.2f} / {f['precision']:.2f}{star}")

    def round_sweep(s):
        return {thr: {"recall": round(v["recall"], 3),
                      "precision": round(v["precision"], 3)} for thr, v in s.items()}

    slim = res.drop(columns=["_y_true", "_y_model", "_y_fao"])
    rec = sweep_model[RECOMMENDED_ALARM_C]
    return {
        "mae_mean": round(float(mae.mean()), 3),
        "mae_min": round(float(mae.min()), 3),
        "mae_max": round(float(mae.max()), 3),
        "rmse_mean": round(float(res["rmse_model"].mean()), 3),
        "mae_climatology_mean": round(float(res["mae_climatology"].mean()), 3),
        "mae_fao_mean": round(float(res["mae_fao"].mean()), 3),
        "margin_vs_fao_mean": round(float(margin.mean()), 3),
        "margin_vs_fao_min": round(float(margin.min()), 3),
        "margin_vs_fao_max": round(float(margin.max()), 3),
        "mae_calm_mean": round(mae_calm, 3),
        "mae_windy_mean": round(mae_windy, 3),
        "recommended_alarm_c": RECOMMENDED_ALARM_C,
        "frost_recall_pooled": round(rec["recall"], 3),
        "frost_precision_pooled": round(rec["precision"], 3),
        "alarm_sweep_model": round_sweep(sweep_model),
        "alarm_sweep_fao": round_sweep(sweep_fao),
        "per_fold": slim.round(3).to_dict(orient="records"),
    }


def main() -> None:
    step("loading nights.csv ...")
    df = load_nights()
    step(f"loaded {len(df)} nights, {df['station'].nunique()} stations, "
         f"years {sorted(df['year'].unique())}")

    loyo = run_cv(df, "year", "leave-one-year-out")
    loso = run_cv(df, "station", "leave-one-station-out")
    # The honest deployment test: hold out a whole country cluster (train UK ->
    # predict PL, and vice versa). Nearby stations share the same night's air
    # mass, so leave-ONE-station-out understates difficulty when the other
    # stations sit ~40 km away; leaving out the whole region removes that.
    loco = run_cv(df, "country", "leave-one-country-out")
    # Across a country boundary, lat/lon/elev are pure extrapolation (held-out
    # region has no nearby training coordinate), so they can drag LOCO for a
    # non-physical reason. Re-run LOCO without them to separate "surface physics
    # doesn't transfer" from "the model is dragged by a coordinate it can't use".
    no_geo = [f for f in FEATURES if f not in ("lat", "lon", "elev")]
    loco_nogeo = run_cv(df, "country", "leave-one-country-out (no geo)",
                        features=no_geo)

    # The live-path accuracy. On the live service cloud_oktas is always missing
    # (no serviceable station supplies it in the training form), so re-run LOYO
    # with cloud blanked -- and radiative_potential recomputed from the blank --
    # to get the MAE growers actually receive, distinct from the with-cloud
    # headline. This is the number the deployed artifact should quote.
    df_live = df.copy()
    df_live["cloud_oktas"] = np.nan
    df_live["radiative_potential"] = physics.radiative_potential_col(
        df_live["cloud_oktas"], df_live["wind_ms"])
    loyo_live = run_cv(df_live, "year", "leave-one-year-out (cloud-blank / live)")

    summary = {
        "n_nights": len(df),
        "n_stations": int(df["station"].nunique()),
        "leave_one_year_out": summarise(loyo, "LOYO"),
        "leave_one_year_out_cloud_blank_live": summarise(loyo_live, "LOYO-live"),
        "leave_one_station_out": summarise(loso, "LOSO"),
        "leave_one_country_out": summarise(loco, "LOCO"),
        "leave_one_country_out_no_geo": summarise(loco_nogeo, "LOCO-nogeo"),
    }
    metrics_json = OUT_DIR / paths.METRICS_NAME
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(_dump_json(summary))
    step(f"wrote {metrics_json}")
    step("done.")


if __name__ == "__main__":
    main()
