"""Unit tests for the nightly forecast job -- offline, no network, no Postgres.

Live weather is faked (a synthetic observation frame) and the DB is in-memory
SQLite, so the job's decision logic -- seasonal guard, validate-never-impute,
idempotency, the cutoff-hour assertion -- is exercised deterministically.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from frostlib import live, model_io
from service import db, nightly_job


# --- a trivial frozen artifact over the real feature list ---------------------

@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    from train import FEATURES
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(50, len(FEATURES))), columns=FEATURES)
    y = x["temp_c"] - 4.0
    model = HistGradientBoostingRegressor(max_iter=20, random_state=0).fit(x, y)
    path = tmp_path_factory.mktemp("m") / "model.joblib"
    model_io.save_model(model, FEATURES, training_years=[2021],
                        training_stations=["uk_waddington"], n_training_nights=50,
                        path=path)
    return model_io.load_model(path=path)


@pytest.fixture
def session():
    engine = db.make_engine("sqlite://")   # in-memory
    db.create_all(engine)
    return db.make_session_factory(engine)()


def _obs_window(cutoff, temp=3.0):
    """A synthetic observation frame with hourly rows up to the cutoff, complete
    enough for build_feature_row to produce a usable row (temp+dew present, 3h/24h
    lookbacks available)."""
    times = pd.date_range(cutoff - pd.Timedelta(hours=26), cutoff, freq="h")
    return pd.DataFrame({
        "lst": times,
        "temp_c": np.linspace(temp + 5, temp, len(times)),
        "dewpoint_c": np.linspace(temp + 2, temp - 2, len(times)),
        "slp_hpa": 1015.0,
        "wind_ms": 1.5,
        "cloud_oktas": np.nan,
        "lat": 53.17, "lon": -0.52, "elev": 70.0,
    })


@pytest.fixture
def fake_fetch(monkeypatch):
    """Replace the live provider with a synthetic, always-usable window."""
    def _fetch(station, cutoff, hours=30):
        return _obs_window(cutoff)
    monkeypatch.setattr(live, "fetch_window", _fetch)
    return _fetch


IN_SEASON = pd.Timestamp("2021-04-15").date()
OUT_SEASON = pd.Timestamp("2021-07-15").date()
# A deterministic "now" 30 min after the 18:00 cutoff, so the live wall-clock
# guards (cutoff reached, newest obs recent) pass on the synthetic windows.
NOW_LST = pd.Timestamp("2021-04-15 18:30")


def _run(session, **kw):
    """nightly_job.run with the test's deterministic now_lst, unless overridden."""
    kw.setdefault("now_lst", NOW_LST)
    return nightly_job.run(session, **kw)


class TestSeasonalGuard:
    def test_in_season_date_is_forecast(self):
        assert nightly_job.in_season(IN_SEASON)

    def test_out_of_season_date_is_not(self):
        assert not nightly_job.in_season(OUT_SEASON)

    def test_out_of_season_run_stores_nothing(self, session, artifact, fake_fetch):
        results = _run(session, dates=[OUT_SEASON],
                                  stations=["uk_waddington"], artifact=artifact)
        assert results == []
        assert session.query(db.Forecast).count() == 0


class TestForecastAndStore:
    def test_in_season_run_stores_a_rich_forecast(self, session, artifact, fake_fetch):
        _run(session, dates=[IN_SEASON], stations=["uk_waddington"],
                        artifact=artifact, alarm_threshold_c=1.5)
        rows = session.query(db.Forecast).all()
        assert len(rows) == 1
        f = rows[0]
        assert f.station == "uk_waddington"
        assert f.model_version == artifact.model_version
        assert f.cutoff_ts.endswith("18:00:00")           # 18:00 LST cutoff
        # Stored richly: every model feature is kept (plus any extras like month).
        assert set(artifact.features) <= set(f.features)
        assert isinstance(f.predicted_tmin_c, float)

    def test_alarm_fires_when_prediction_at_or_below_threshold(
            self, session, artifact, monkeypatch):
        # Cold synthetic window -> low prediction -> alarm.
        monkeypatch.setattr(live, "fetch_window",
                            lambda s, c, hours=30: _obs_window(c, temp=-6.0))
        _run(session, dates=[IN_SEASON], stations=["uk_waddington"],
                        artifact=artifact, alarm_threshold_c=1.5)
        assert session.query(db.Forecast).one().alarm_fired is True


class TestValidateNeverImpute:
    def test_unusable_window_is_skipped_and_stores_nothing(
            self, session, artifact, monkeypatch):
        # A window with no temperature at the cutoff -> build_feature_row returns
        # None -> skip, store nothing (never impute).
        def _empty(station, cutoff, hours=30):
            return pd.DataFrame(columns=["lst", "temp_c", "dewpoint_c", "slp_hpa",
                                         "wind_ms", "cloud_oktas", "lat", "lon", "elev"])
        monkeypatch.setattr(live, "fetch_window", _empty)
        results = _run(session, dates=[IN_SEASON],
                                  stations=["uk_waddington"], artifact=artifact)
        assert results[0].stored is False
        assert "no usable snapshot" in results[0].reason
        assert session.query(db.Forecast).count() == 0

    def test_provider_error_is_skipped_not_fatal(self, session, artifact, monkeypatch):
        def _boom(station, cutoff, hours=30):
            raise live.OgimetError("rate limited")
        monkeypatch.setattr(live, "fetch_window", _boom)
        results = _run(session, dates=[IN_SEASON],
                                  stations=["uk_waddington"], artifact=artifact)
        assert results[0].stored is False
        assert "provider declined" in results[0].reason


class TestIdempotency:
    def test_rerun_updates_in_place_not_duplicate(self, session, artifact, fake_fetch):
        for _ in range(3):
            _run(session, dates=[IN_SEASON], stations=["uk_waddington"],
                            artifact=artifact)
        assert session.query(db.Forecast).count() == 1   # one row, not three
        assert session.query(db.StationRun).count() == 1


class TestCutoffAndTiming:
    def test_stores_the_snapshot_timestamp(self, session, artifact, fake_fetch):
        _run(session, dates=[IN_SEASON], stations=["uk_waddington"],
                        artifact=artifact)
        f = session.query(db.Forecast).one()
        # The snapshot time actually used is recorded, not just the intended cutoff.
        assert f.snapshot_ts is not None
        assert pd.Timestamp(f.snapshot_ts).hour in (17, 18)

    def test_refuses_a_future_cutoff(self, session, artifact, fake_fetch):
        # An evening whose 18:00 cutoff has not arrived yet must not be forecast:
        # any snapshot would be from earlier and warm-biased.
        res = nightly_job._forecast_one(
            "uk_waddington", IN_SEASON, artifact, 1.5,
            now_lst=pd.Timestamp(IN_SEASON) + pd.Timedelta(hours=12))  # noon < 18:00
        assert res.stored is False and "not yet reached" in res.reason

    def test_early_snapshot_is_rejected(self, session, artifact, monkeypatch):
        # A window whose latest observation is 16:00 (more than 40 min before the
        # 18:00 cutoff) must be rejected, not stored as an 18:00 forecast.
        def _early(station, cutoff, hours=30):
            times = pd.date_range(cutoff - pd.Timedelta(hours=6),
                                  cutoff - pd.Timedelta(hours=2), freq="h")
            return pd.DataFrame({
                "lst": times, "temp_c": 3.0, "dewpoint_c": 1.0, "slp_hpa": 1015.0,
                "wind_ms": 1.0, "cloud_oktas": float("nan"),
                "lat": 53.17, "lon": -0.52, "elev": 70.0})
        monkeypatch.setattr(live, "fetch_window", _early)
        # live_clock_check=False isolates the snapshot-tolerance rejection (the
        # subject here) from the wall-clock guards, which would also reject it.
        res = nightly_job._forecast_one(
            "uk_waddington", IN_SEASON, artifact, 1.5, live_clock_check=False)
        assert res.stored is False
        assert "snapshot" in res.reason


class TestStationRuns:
    def test_records_a_skip_with_reason(self, session, artifact, monkeypatch):
        monkeypatch.setattr(live, "fetch_window",
                            lambda s, c, hours=30: (_ for _ in ()).throw(
                                live.OgimetError("rate limited")))
        _run(session, dates=[IN_SEASON], stations=["uk_waddington"],
                        artifact=artifact)
        run = session.query(db.StationRun).one()
        assert run.forecast_stored is False and "declined" in run.skip_reason


class TestPinnedOffset:
    def test_pinned_offset_matches_isd(self):
        # T2: the pinned EXPECTED_LST_OFFSET must equal what isd.lst_offset returns,
        # or the guard would false-reject every real run. This is the assertion that
        # keeps the pin honest.
        from frostlib import isd
        for station, off in nightly_job.EXPECTED_LST_OFFSET.items():
            assert isd.lst_offset(station) == off

    def test_wrong_offset_is_rejected_in_backfill_too(self, session, artifact,
                                                      fake_fetch, monkeypatch):
        # The pinned-offset check runs in BOTH modes -- a mistimed clock is caught
        # even when the wall-clock guards are off (backfill).
        from frostlib import isd
        monkeypatch.setattr(isd, "lst_offset", lambda s: 1)   # wrong for a UK station
        res = nightly_job._forecast_one(
            "uk_waddington", IN_SEASON, artifact, 1.5, live_clock_check=False)
        assert res.stored is False and "lst_offset" in res.reason
