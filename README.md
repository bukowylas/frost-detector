# Frost Detector

**Overnight-frost forecasting for growers, from an ordinary weather station.**

Frost Detector predicts *tomorrow morning's minimum temperature* at a site from
that evening's surface weather, and turns it into a frost alarm a grower can act
on — run wind machines, light heaters, deploy sprinklers, or harvest early.

It uses only **free, global surface observations** (NOAA ISD), so it can run at
any of thousands of weather stations with no proprietary inputs. It is
benchmarked against the operational rule agricultural extension services
actually teach, and it beats it — modestly but consistently — across held-out
years and at stations it has never seen.

## Why this problem

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

## Results (honest, leave-out evaluation)

Evaluated on **6,007 station-nights** across **5 Polish + 3 English** stations,
2019–2023, restricted to the spring and autumn frost-risk windows.

**Minimum-temperature accuracy (mean absolute error, °C):**

| Evaluation | Frost Detector | FAO rule* | Climatology |
|---|---|---|---|
| **Leave-one-year-out** | **1.80** (range 1.67–1.97) | 1.87 | 2.95 |
| **Leave-one-station-out** | **1.64** (range 1.21–1.88) | 1.91 | 3.69 |

\* The FAO evening-temperature/dew-point rule (Snyder & de Melo-Abreu, 2005) —
the operational estimator extension services teach growers.

The model beats both baselines in **every fold**, including at stations held
entirely out of training (the real deployment case: a grower whose nearest
station wasn't in the data). The across-year *spread* is the honest error bar:
year-to-year variability, not sampling noise, dominates.

**Frost-alarm trade-off (leave-one-year-out).** A frost night is defined by the
actual overnight minimum ≤ 0 °C; the alarm fires when the *predicted* minimum is
at/below the threshold:

| Alarm at predicted ≤ | Frosts caught (recall) | Precision |
|---|---|---|
| 0.0 °C | 0.53 | 0.81 |
| +1.0 °C | 0.69 | 0.70 |
| **+1.5 °C (default)** | **0.75** | 0.64 |
| +2.0 °C | 0.80 | 0.59 |

At the +1.5 °C default the model catches **three frost nights in four**. The
threshold is a deliberate, adjustable decision: a grower dials it to their own
cost of protection versus cost of crop loss.

## Honest limitations

- **Radiative vs. advective frost.** The model is strong on radiative frost
  (calm, clear nights — the dominant orchard risk). Frost driven by a cold air
  mass moving in is set by upstream conditions a single station cannot see, and
  is systematically harder; this shows up as lower recall at the mild, maritime
  English stations, where frost is both rarer and more often advective.
- **Airport siting.** ISD stations sit at airfields, which are typically warmer,
  well-drained sites; frost-vulnerable orchards often sit in cold-air-drainage
  pockets that run several degrees colder on a calm night. A real deployment
  would need a per-site offset calibrated from a season of on-site loggers. The
  model would otherwise **under-warn** for exactly the sites that need warning.
- **What it does not replace.** National met services (e.g. IMGW in Poland)
  issue frost forecasts backed by numerical weather prediction and upper-air
  data this project does not use. Frost Detector's niche is a cheap,
  site-specific, reproducible statistical model from free surface data alone.

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
3. **`train.py`** — fit and evaluate the regressor honestly: leave-one-year-out
   and leave-one-station-out, against the FAO and climatology baselines (both
   fit on training data only, inside each fold, to avoid leaking the held-out
   period's climate).

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
`scikit-learn`, `requests`.

```
python3 fetch_data.py     # download ISD data (one-time; ~5 min, parallel)
python3 prepare.py        # build data/nights.csv
python3 train.py          # evaluate; writes data/metrics.json
```

`train.py --help` and the `tests/` scripts (timing probes, the feature-
improvement experiment bench) document how the pipeline was validated.
