"""The nightly forecast job -- the ML heart of the service.

At the 18:00 LST cutoff, for each serviceable station: fetch the live window (the
parity-validated path), validate, build the feature vector through the SAME
``prepare.build_feature_row`` the model was trained on, predict with the frozen
artifact, and store the forecast richly.

Non-negotiables enforced here (each a documented production failure mode):

- **Seasonal guard.** The model is trained on the frost-risk windows (spring and
  autumn); outside them it would extrapolate. This build serves SPRING only
  (``config.SERVICE_RISK_WINDOWS``, a subset of what the model covers), enforced
  to never exceed the model's windows. Autumn is a one-constant change, not a
  model change. Out of the service season, the job exits without forecasting.
- **Validate, never impute.** A station-night is usable only if the cutoff
  snapshot exists with real temperature and dew point and the trends are
  computable (``build_feature_row`` returns None otherwise). An unusable station
  is logged and skipped, storing nothing. A confident wrong number is worse than
  an admitted gap.
- **Frozen artifact, feature names asserted.** The estimator is loaded, never
  refit, and its feature names are asserted against the code's -- a silently
  reordered feature vector is scored by column position and would mis-predict.
- **Clock sanity, and snapshot timing.** Two distinct guards. A mistimed *run*
  (cron drift) is caught by refusing to forecast before the cutoff and requiring
  the snapshot within a tight tolerance. A mistimed *clock* (a wrong
  ``isd.lst_offset``) is caught by comparing the provider's newest observation
  against wall-clock LST -- an offset error desynchronises the two. The
  snapshot-hour check is only a backstop for a widened tolerance, not an
  independent timezone check.
- **Idempotent.** Re-running for the same date updates that night's row in place
  rather than creating a duplicate (``unique(station, date)``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import pandas as pd

import prepare
from frostlib import live, model_io
from service import config
from service.db import Forecast, StationRun

log = logging.getLogger("frost.nightly")

CUTOFF_HOUR_LST = prepare.CUTOFF_HOUR   # 18:00 LST -- one source of truth

# The live path demands the cutoff snapshot be recent, not merely "within 90 min"
# (the training tolerance). A systematically-early snapshot -- from a mistimed run
# -- is warm-biased, the under-warning direction. 40 minutes is set from the
# measured cadence: the two serviceable stations report a median ~30 min apart
# near the cutoff, so 40 admits the normal gap while rejecting a snapshot from the
# previous hour. (Re-measure if the serviceable set changes.)
LIVE_CUTOFF_TOL_MIN = 40


def in_season(date) -> bool:
    """Whether the service forecasts on this evening date.

    Uses the SERVICE season (``config.SERVICE_RISK_WINDOWS``), which this build
    sets to spring only. The model is trained on autumn too and handles it fine,
    so autumn is enabled by widening that one constant -- no model change."""
    return config.service_in_season(date)


@dataclass
class StationResult:
    station: str
    stored: bool
    reason: str = ""
    predicted_tmin_c: float | None = None
    alarm: bool | None = None
    features: dict | None = None   # the built feature vector, when usable
    cutoff_ts: str | None = None   # the 18:00 LST cutoff, ISO, when usable


def _now_lst(station):
    """Current wall-clock time in the station's Local Standard Time."""
    from frostlib import isd
    return pd.Timestamp.now("UTC").tz_localize(None) + pd.Timedelta(
        hours=isd.lst_offset(station))


def _forecast_one(station, date, artifact, alarm_threshold_c,
                  now_lst=None, live_clock_check=True) -> StationResult:
    """Fetch, validate, predict and return a result for one station-night.

    Returns a StationResult describing what happened -- stored, or skipped with a
    reason -- without touching the database (the caller persists), so the decision
    logic is testable without a live network or a DB. The ENTIRE body is guarded:
    a single station's malformed payload must never kill the night for the others.

    ``live_clock_check`` gates the wall-clock guards (cutoff-not-reached, newest-obs
    lag). They assume a near-real-time run, so a deliberate BACKFILL of a past date
    turns them off -- for a backfill "now" is not a meaningful reference, and the
    snapshot-tolerance + parity gate still protect correctness.
    """
    cutoff = pd.Timestamp(date) + pd.Timedelta(hours=CUTOFF_HOUR_LST)

    # Refuse to forecast an evening that has not reached its cutoff yet -- there is
    # no 18:00 observation to use, so any snapshot would be from earlier and
    # warm-biased.
    if now_lst is None:
        now_lst = _now_lst(station)
    if live_clock_check and now_lst < cutoff:
        return StationResult(station, False,
                             f"cutoff {cutoff} not yet reached (now {now_lst} LST)")

    try:
        obs = live.fetch_window(station, cutoff)

        # Clock-sanity check: compare the provider's newest observation against
        # wall-clock LST. This is the ONE check that catches a wrong isd.lst_offset
        # (the Stage-2 timezone bug) -- an offset error desynchronises the two: a
        # +1h error puts the newest obs in the future (negative lag), a -1h error
        # makes it look staler than any real reporting cadence. (The snapshot-hour
        # check below cannot catch this, because it reads the same mis-stamped
        # times as the cutoff comparison.)
        if live_clock_check and not obs.empty:
            newest = pd.Timestamp(obs["lst"].max())
            lag = now_lst - newest
            if not (pd.Timedelta(0) <= lag <= pd.Timedelta(minutes=90)):
                return StationResult(
                    station, False,
                    f"newest obs {newest} is {lag} from now ({now_lst} LST) -- "
                    "provider clock or LST offset looks wrong")

        feats = prepare.build_feature_row(
            obs, cutoff, cutoff_tol_minutes=LIVE_CUTOFF_TOL_MIN)
        if feats is None:
            return StationResult(
                station, False,
                f"no usable snapshot within {LIVE_CUTOFF_TOL_MIN} min of cutoff")

        # Backstop only: with a 40-min tolerance a surviving snapshot is already
        # necessarily hour 17 or 18, so this can currently never fire. Kept so that
        # widening the tolerance past 60 min re-arms a floor -- it is NOT an
        # independent timezone check (see the clock-sanity check above for that).
        snap = pd.Timestamp(feats["snapshot_ts"])
        if snap.hour not in (CUTOFF_HOUR_LST - 1, CUTOFF_HOUR_LST):
            return StationResult(
                station, False,
                f"snapshot hour {snap.hour} LST outside the cutoff window")

        tmin = float(artifact.predict(pd.DataFrame([feats]))[0])
    except live.OgimetError as exc:
        return StationResult(station, False, f"provider declined: {exc}")
    except Exception as exc:  # noqa: BLE001 -- one station's failure must not kill the night
        return StationResult(station, False, f"error: {type(exc).__name__}: {exc}")

    alarm = tmin <= alarm_threshold_c
    return StationResult(station, True, "", predicted_tmin_c=tmin, alarm=alarm,
                         features=feats, cutoff_ts=cutoff.isoformat())


def _as_date(date):
    """Coerce a date-ish value to a datetime.date for the Date columns."""
    return pd.Timestamp(date).date()


def _store(session, date, result, model_version, alarm_threshold_c):
    """Upsert the forecast for (station, date) -- idempotent by construction."""
    d = _as_date(date)
    existing = (session.query(Forecast)
                .filter_by(station=result.station, date=d).one_or_none())
    payload = dict(
        predicted_tmin_c=result.predicted_tmin_c,
        alarm_fired=result.alarm,
        alarm_threshold_c=alarm_threshold_c,
        model_version=model_version,
        features_json=json.dumps(result.features, default=float),
        cutoff_ts=result.cutoff_ts,
        snapshot_ts=result.features.get("snapshot_ts"),
    )
    if existing is None:
        session.add(Forecast(station=result.station, date=d, **payload))
    else:
        for k, v in payload.items():
            setattr(existing, k, v)
    session.commit()


def _record_run(session, station, date, result):
    """Upsert the (station, date) run record, so 'did it run?' and 'why skipped?'
    are answerable and a monitor can alert on repeated skips."""
    d = _as_date(date)
    existing = (session.query(StationRun)
                .filter_by(station=station, date=d).one_or_none())
    payload = dict(forecast_stored=result.stored,
                   skip_reason=None if result.stored else result.reason)
    if existing is None:
        session.add(StationRun(station=station, date=d, **payload))
    else:
        for k, v in payload.items():
            setattr(existing, k, v)
    session.commit()


def run(session, dates=None, stations=None, alarm_threshold_c=None,
        artifact=None, live_clock_check=True, now_lst=None) -> list[StationResult]:
    """Run the nightly job for a date (default: today) across serviceable stations.

    Loads the frozen artifact (asserting feature names) unless one is injected.
    Applies the seasonal guard, then for each station fetches/validates/predicts
    and stores richly. Returns a result per station for logging/monitoring.

    ``live_clock_check`` (default True) enforces the wall-clock guards for a live
    run; a backfill of a past date passes False. ``now_lst`` overrides wall clock
    (tests / deterministic runs).
    """
    # FEATURES is imported from the training module because it IS the load-time
    # assertion (the deployed feature list must equal the trained one). The
    # operational alarm threshold comes from service config, not training.
    from train import FEATURES

    if artifact is None:
        artifact = model_io.load_model(expected_features=FEATURES)
    if alarm_threshold_c is None:
        alarm_threshold_c = config.RECOMMENDED_ALARM_C
    if dates is None:
        dates = [pd.Timestamp.now().normalize().date()]
    if stations is None:
        stations = list(config.SERVICEABLE_STATIONS)

    results: list[StationResult] = []
    for date in dates:
        if not in_season(date):
            log.info("out of season (%s); skipping", date)
            continue
        for station in stations:
            res = _forecast_one(station, date, artifact, alarm_threshold_c,
                                now_lst=now_lst, live_clock_check=live_clock_check)
            if res.stored:
                _store(session, date, res, artifact.model_version, alarm_threshold_c)
                log.info("%s %s: %.1f C alarm=%s", station, date,
                         res.predicted_tmin_c, res.alarm)
            else:
                log.info("%s %s: skipped (%s)", station, date, res.reason)
            # Record every attempt (stored or skipped) so notify can tell a skip
            # from a never-ran, and a monitor can alert on repeated skips.
            _record_run(session, station, date, res)
            results.append(res)
    return results
