# Frost Detector

**Overnight-frost forecasting for growers, from an ordinary weather station.**

Frost Detector predicts the overnight minimum temperature at a site from that
evening's surface weather, and turns it into a frost alarm a grower can act on —
run wind machines, light heaters, deploy sprinklers, or harvest early.

It uses only **free, global surface observations** (NOAA ISD), so it can run at
any of thousands of weather stations with no proprietary inputs. It is
benchmarked against the operational rule agricultural extension services
actually teach, and it beats it — modestly but consistently — across held-out
years and at stations it has never seen.

## Background

Spring frost after budbreak is the classic catastrophe for orchards and vines —
a single night can destroy a season's fruit. Growers need a few hours' warning
to protect the crop. The physics of the common case is favourable to a
surface-only model: **radiative frost** forms on calm, clear, dry nights when
the ground radiates heat away and cools toward the dew point — and evening
temperature, dew point, wind and cloud (all measured at the surface) are close
to the complete causal set. This is the opposite of predicting cloud cover
(which forms aloft, out of a surface station's reach); here the surface data
*is* the physics.

## What it does

- **Predicts** the overnight minimum temperature (°C) from the ~18:00 local
  evening state.
- **Alarms** at a configurable threshold on that prediction. Because a missed
  frost is far costlier to a grower than a false alarm — and a 2 m screen
  thermometer reads warmer than ground frost — the default alarm fires early, at
  a **predicted minimum of +1.5 °C** (the standard ground-frost correction).
- **Reports magnitude, not just yes/no.** −0.5 °C and −5 °C call for different
  responses; predicting the temperature lets the same model serve every crop's
  damage threshold without retraining.

## Results

Evaluated on **6,007 station-nights** across **5 Polish + 3 English** stations,
2019–2023, restricted to the spring and autumn frost-risk windows.

**Minimum-temperature accuracy (mean absolute error, °C)** on the two
deployment-relevant tests — a new season, and a new station, in a region the
model has trained on (the situations a real deployment is actually in):

| Evaluation | Frost Detector | FAO rule* | Climatology |
|---|---|---|---|
| **Leave-one-year-out** (unseen year, region trained on) | **1.80** (1.67–1.96) | 1.87 | 2.94 |
| **Leave-one-station-out** (unseen station, region trained on) | **1.65** (1.29–1.90) | 1.91 | 3.03 |

\* The FAO evening-temperature/dew-point rule (Snyder & de Melo-Abreu, 2005) —
the operational estimator extension services teach growers.

**The model beats both baselines on both tests.** Against the FAO rule it wins in
four of five year folds and ties in the fifth (a −0.07 °C mean margin, narrow but
consistent — it never loses within a region); across held-out stations the margin
is wider. Against climatology it wins comfortably throughout. The across-fold
*spread* (not sampling noise) is the error bar that matters.

**Frost-alarm trade-off.** A frost night is the actual overnight minimum ≤ 0 °C;
the alarm fires when the *predicted* minimum is at/below the threshold. Compared
against the FAO rule put through the *same* alarm logic (leave-one-year-out,
predictions pooled across folds):

| Alarm at predicted ≤ | Model recall / precision | FAO recall / precision |
|---|---|---|
| 0.0 °C | 0.54 / 0.82 | 0.53 / 0.80 |
| +1.0 °C | 0.70 / 0.71 | 0.64 / 0.70 |
| **+1.5 °C (default)** | **0.76 / 0.64** | 0.73 / 0.67 |
| +2.0 °C | 0.81 / 0.59 | 0.78 / 0.61 |

At the +1.5 °C default the model catches **about three frost nights in four**,
slightly ahead of FAO on recall at comparable precision. The +1.5 °C threshold
is the standard **ground-frost correction** (a 2 m thermometer reads warmer than
ground level on a stratified night), chosen *a priori* — not tuned on this table.
A grower dials it to their own cost of protection versus crop loss.

## Limitations

- **Train on local stations.** The learned relationship is partly
  region-specific: trained on one country and tested on the *other* — with no
  local data at all (leave-one-country-out) — accuracy falls to roughly the FAO
  rule's (1.97 vs 1.91 °C), and dropping the geographic features does not recover
  it. This is a stress test, not the deployment case (a real deployment includes
  the target region in training); it just quantifies *why* you train locally
  rather than assume the model transfers across climates.
- **Radiative vs. advective frost.** Radiative frost forms on calm, clear nights
  — the dominant orchard risk, and the case where surface data *is* the physics.
  Frost driven by an incoming cold air mass is set by upstream conditions a
  single station cannot see. (In these results the calm/windy MAE split is
  modest, but the mechanism is a real ceiling on any surface-only model.)
- **Airport siting.** ISD stations sit at airfields, which are typically warmer,
  well-drained sites; frost-vulnerable orchards often sit in cold-air-drainage
  pockets that run several degrees colder on a calm night. A real deployment
  would need a per-site offset calibrated from a season of on-site loggers. The
  model would otherwise **under-warn** for exactly the sites that need warning.
- **What it does not replace.** National met services (e.g. IMGW in Poland)
  issue frost forecasts backed by numerical weather prediction and upper-air
  data this project does not use.

## How it works (pipeline)

```
fetch_data.py  ->  prepare.py            ->  train.py
NOAA ISD CSVs      one row per            HistGradientBoosting regressor
(data_raw/)        station-night          + FAO & climatology baselines
                   (data/nights.csv)      + leave-one-year/station-out eval
                                          (data/metrics.json)
```

1. **`fetch_data.py`** — download hourly ISD observations for the configured
   stations and years (the one network step; parallel, resumable, robust to a
   failed station).
2. **`prepare.py`** — decode the packed ISD fields (honouring missing-value
   sentinels and quality flags), convert to Local Standard Time, and build one
   modelling row per frost-risk night: features from the 18:00 LST cutoff (plus
   3 h / 24 h trends and a radiative-cooling term), label = the minimum
   temperature 20:00 → 08:00 LST. An 18:00–20:00 gap keeps features and label
   from overlapping.
3. **`train.py`** — fit and evaluate the regressor: leave-one-year-out
   and leave-one-station-out, against the FAO and climatology baselines (both
   fit on training data only, inside each fold, to avoid leaking the held-out
   period's climate).

Shared definitions live in **`frostlib/`**: ISD field decoding and quality flags
(`isd`), the NCEI URLs (`net`, `isd_history`), data locations and the raw-file
naming convention (`paths`), and the frost feature formulas (`physics`). The
last matters most: `prepare.build_feature_row` (training rows) and `predict.py`
(a served forecast) compute `radiative_potential` and `dewpoint_depression_c`
from the same code, so the served row cannot drift from what the model was fit
on.

### Anti-leakage discipline

- Features use only observations up to the 18:00 LST cutoff; the label window
  starts at 20:00 LST, with the gap between excluded.
- Baselines (climatology, FAO coefficients) are fit per fold on training data
  only — never over the full dataset.
- Splits are by whole year or whole station, never a random shuffle: consecutive
  nights are correlated, so a random split would leak and inflate every score.

## Data

**NOAA / NCEI Integrated Surface Database (ISD)**, global-hourly access —
public, no API key. Stations were chosen in real frost-sensitive agricultural
regions and verified for temperature *and* dew-point coverage (dew point is the
single most important frost predictor). See the `tests/` scripts for the
station-selection and verification steps.

## Running it

System Python (no virtual environment); requires `numpy`, `pandas`,
`scikit-learn`, `requests`, `joblib`.

```
python3 fetch_data.py     # download ISD data (one-time; ~5 min, parallel)
python3 prepare.py        # build data/nights.csv
python3 train.py          # evaluate; writes data/metrics.json
```

**Forecasting a single night.** Fit the deployable model once, then forecast
from an evening's 18:00 LST observations (the frost alarm is derived at the
+1.5 °C ground-frost threshold):

```
python3 predict.py --fit
python3 predict.py --temp 3.0 --dewpoint 0.5 --wind 1.5 --cloud 1 \
                   --pressure 1022 --lat 51.7 --lon 19.4 --elev 180 --doy 110
# -> predicted overnight minimum: +0.1 C ; frost alarm: YES
```

`train.py --help` and the `tests/` scripts (timing probes, the feature-
improvement experiment bench) document how the pipeline was validated.

**Tests.** `tests/unit/` holds the unit suite (`pytest`, no network, no data
files needed — every fixture is synthetic); the other `tests/` scripts are one-off
investigation probes and are not collected.

```
python3 -m pytest                                     # unit suite
python3 -m pytest --cov=. --cov-report=term-missing    # with coverage
```

## Live service

The `service/` package runs the trained model as a seasonal forecast service:
each evening in the frost-risk window it fetches live weather, runs the frozen
model, stores the forecast, and can text subscribed growers. It is a portfolio
demonstration of deploying the model — subscribers are the author and consenting
friends, and SMS defaults to a no-cost stub that logs messages instead of sending
them.

**One model, frozen.** The service never refits on demand. `predict.py --fit`
saves a versioned artifact (`frostlib/model_io.py`) carrying the estimator, its
feature names, the training years and stations, and the measured MAE. The version
hashes the `(estimator, features)` pair, and the loader asserts the feature names
on load — a model scores by column position, so a silently reordered feature
vector would mis-predict, and this refuses it.

**Cloud is an accepted train/serve skew.** The model is trained *with* cloud, but
no serviceable station supplies cloud live, so on the live path `cloud_oktas` is
always missing and `radiative_potential` runs at its cloud-missing default. This
is a deliberate, measured trade — not a bug and not hidden. Cloud is a real frost
driver and is kept in the model; the cost of running without it live is small and
quantified: the evaluation re-runs leave-one-year-out with cloud blanked, and the
artifact stores both `mae_c_full` (with cloud, 1.80 °C) and `mae_c_live` (the
1.82 °C a grower actually receives). The quoted accuracy and the SMS uncertainty
band both use the live number. To *close* the skew later rather than accept it,
either retrain without cloud, or supply cloud from a separate source (e.g. a
cloud-cover model or a METAR feed).

**Coverage matches training, and is asserted.** The nightly job fetches through
the same `build_feature_row` the model was trained on, via the parity-validated
live path. The serviceable stations are an explicit list in `service/config.py`,
asserted at import to be a subset of what the parity gate validated — there is no
fallback that could silently widen coverage. The model is trained on spring *and*
autumn (autumn measurably helps and is when unharvested fruit is on the tree);
this build **serves spring only**, one config constant away from autumn.

**Delivery is a warning system, not a newsletter.** Sends are at-least-once with
bounded duplicates (a dropped frost warning costs a crop; a duplicate costs a mild
annoyance): each notification is claimed `pending` before the send and marked
`sent` after, so a crash never drops a message and a retry never double-texts. A
`nightly` subscriber whose station was skipped gets an explicit "no forecast"
message — silence is ambiguous between "clear" and "the system is down".

**Handling the parts that bite in production.** A warning system fails in ways a
demo doesn't, so the awkward cases are handled rather than deferred:

- **The message a grower receives can only change with a code.** A signup or
  settings change is *staged*; the confirmation code activates it. So knowing a
  phone number is not enough to alter — or silence — someone else's alerts. Codes
  are hashed, expire, and lock after repeated wrong guesses; phone numbers are
  normalised to E.164 so one phone is one identity; a per-phone cooldown and a
  global daily send cap keep no endpoint an open SMS relay.
- **Unsubscribe is the carrier STOP model.** It deactivates immediately and
  confirms by SMS; the confirmation is the receipt, so a subscription can't be
  silently disabled. If that confirmation can't be sent, the failure is recorded
  and retried, not swallowed.
- **A wrong timezone offset is caught, not trusted.** Serving the 17:00 or 19:00
  observation as the 18:00 cutoff would produce plausible, consistently wrong
  forecasts; a pinned per-station offset (asserted at startup) and a wall-clock
  sanity check on the newest observation both guard against it.
- **The nightly run reports whether it actually worked.** It records every
  station attempt and every send outcome, and exits non-zero when a station never
  ran, a send failed, or nobody was warned — so a silent partial failure pages
  someone instead of passing as green.
- **Runs on Postgres, tested on SQLite.** The schema and migrations target
  Postgres; the offline suite runs against in-memory SQLite (with foreign-key
  enforcement on), so a full test run needs no database and no network.

**Components** (`service/`): `config.py` (the single source of truth for stations,
labels, season, threshold), the nightly job (`nightly_job.py` — seasonal guard,
validate-never-impute, idempotent, snapshot-time asserted, one station's failure
never kills the night), the database (`db.py` + Alembic migrations — forecasts and
station-runs stored richly), the API (`api.py` — FastAPI: signup, phone
verification, STOP-model unsubscribe, forecast reads; serves the built React UI),
SMS (`sms.py`/`notify.py` — pluggable interface, log-stub default), and the React
signup UI (`web/`).

**Run it locally** (SQLite + log-stub SMS, no Postgres, no real texts):

```
python3 predict.py --fit                       # freeze the model artifact
python3 -m service.run_nightly --date 2026-04-15   # forecast -> store -> notify
FROST_BOOTSTRAP_DB=1 uvicorn service.api:create_app --factory --reload  # API+UI :8000
cd web && npm install && npm run build         # build the frontend it serves
```

(`FROST_BOOTSTRAP_DB=1` lets the app create the schema directly for local dev;
production applies Alembic migrations instead, so a boot never races them.)

**Or the full stack** (Postgres + app in one command):

```
docker compose up --build                              # runs `alembic upgrade head`
docker compose run --rm app python3 predict.py --fit   # one-off: fit the model
```
