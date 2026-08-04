"""Unit tests for the database layer -- against in-memory SQLite.

Covers the constraints the service's correctness rests on: the (station, date)
uniqueness that makes the nightly job idempotent, and the (subscriber, date)
uniqueness that makes notification sending idempotent. Also checks the Alembic
migration builds the same tables the models declare, so migrations and code can't
silently drift.
"""

import datetime as dt
import json

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from service import db

DAY = dt.date(2021, 4, 15)


@pytest.fixture
def engine():
    e = db.make_engine("sqlite://")
    db.create_all(e)
    return e


@pytest.fixture
def session(engine):
    return db.make_session_factory(engine)()


def _forecast(station="uk_waddington", date=DAY, tmin=-1.0):
    return db.Forecast(
        station=station, date=date, predicted_tmin_c=tmin, alarm_fired=True,
        alarm_threshold_c=1.5, model_version="m-test",
        features_json=json.dumps({"temp_c": 3.0, "doy": 105}),
        cutoff_ts="2021-04-15T18:00:00", snapshot_ts="2021-04-15T18:00:00",
    )


class TestForecast:
    def test_round_trips_and_parses_features(self, session):
        session.add(_forecast())
        session.commit()
        f = session.query(db.Forecast).one()
        assert f.features["temp_c"] == 3.0
        assert f.alarm_fired is True

    def test_station_date_is_unique(self, session):
        session.add(_forecast())
        session.commit()
        session.add(_forecast())   # same (station, date)
        with pytest.raises(IntegrityError):
            session.commit()


class TestSubscriberAndSendLog:
    def test_subscriber_round_trips(self, session):
        session.add(db.Subscriber(station="uk_cranwell", phone="+44700900000",
                                  mode="frost", threshold_c=0.5))
        session.commit()
        s = session.query(db.Subscriber).one()
        assert s.verified is False and s.mode == "frost"

    def test_send_log_is_unique_per_subscriber_and_date(self, session):
        sub = db.Subscriber(station="uk_cranwell", phone="+44700900001")
        session.add(sub)
        session.commit()
        session.add(db.SentNotification(subscriber_id=sub.id, forecast_date=DAY))
        session.commit()
        session.add(db.SentNotification(subscriber_id=sub.id, forecast_date=DAY))
        with pytest.raises(IntegrityError):
            session.commit()


class TestMigrationMatchesModels:
    def test_alembic_head_builds_the_same_tables_as_the_models(self, tmp_path):
        import os
        from alembic import command
        from alembic.config import Config

        url = f"sqlite:///{tmp_path/'mig.db'}"
        cfg = Config(str(__import__("pathlib").Path(db.__file__).parents[1] / "alembic.ini"))
        os.environ["DATABASE_URL"] = url
        try:
            command.upgrade(cfg, "head")
            migrated = set(inspect(db.make_engine(url)).get_table_names())
        finally:
            os.environ.pop("DATABASE_URL", None)
        model_tables = set(db.Base.metadata.tables) | {"alembic_version"}
        assert model_tables <= migrated
