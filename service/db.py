"""Database layer: schema, engine and session management.

PostgreSQL in production (via ``DATABASE_URL``); an in-memory/file SQLite engine
in tests. The models below are written to run on both -- portable column types,
JSON stored as text where a native JSON type is not guaranteed -- so the fast
offline suite never needs a running Postgres.

Two tables carry the service's state:

- ``forecasts`` -- one row per (station, night), stored RICHLY: not just the
  prediction but the full feature vector, the model version, and the cutoff
  timestamp. Scoring forecasts later is out of scope, but the data to diagnose
  "which nights did it miss, and what did they share" must be captured now; it is
  free at write time and irreplaceable afterwards. ``unique(station, date)`` makes
  the nightly job idempotent.
- ``subscribers`` -- who to notify, how (nightly heartbeat vs frost-only), at what
  threshold, and whether their phone is verified. A separate ``sent_notifications``
  log with a uniqueness constraint makes the notify step idempotent: a retry after
  a partial failure can never double-text.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import datetime as _dt

from sqlalchemy import (
    Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text,
    UniqueConstraint, create_engine,
)
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker,
)

DEFAULT_SQLITE_URL = "sqlite:///./frost_service.db"


def database_url() -> str:
    """The configured database URL, defaulting to a local SQLite file so the
    service runs with no external database for development."""
    return os.environ.get("DATABASE_URL", DEFAULT_SQLITE_URL)


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Forecast(Base):
    """One stored forecast for a station-night, kept richly for later diagnosis."""

    __tablename__ = "forecasts"
    __table_args__ = (
        UniqueConstraint("station", "date", name="uq_forecast_station_date"),
        # The notify query filters on date alone; the unique index is station-first
        # so it does not serve that query. A dedicated date index does.
        Index("ix_forecasts_date", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    station: Mapped[str] = mapped_column(String(64), index=True)
    date: Mapped[_dt.date] = mapped_column(Date)          # evening date (LST)
    predicted_tmin_c: Mapped[float] = mapped_column(Float)
    alarm_fired: Mapped[bool] = mapped_column(Boolean)
    alarm_threshold_c: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(64))
    features_json: Mapped[str] = mapped_column(Text)        # the exact feature vector
    cutoff_ts: Mapped[str] = mapped_column(String(32))      # intended 18:00 LST cutoff, ISO
    snapshot_ts: Mapped[str] = mapped_column(String(32))    # LST of the obs actually used
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    @property
    def features(self) -> dict:
        return json.loads(self.features_json)


class Subscriber(Base):
    """A grower signed up for one station, with mode/threshold and phone status."""

    __tablename__ = "subscribers"
    __table_args__ = (UniqueConstraint("phone", "station", name="uq_sub_phone_station"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    station: Mapped[str] = mapped_column(String(64), index=True)
    phone: Mapped[str] = mapped_column(String(32), index=True)
    # "nightly" = a message every night (the heartbeat); "frost" = only when the
    # forecast is at/below the subscriber's threshold.
    mode: Mapped[str] = mapped_column(String(16), default="nightly")
    threshold_c: Mapped[float] = mapped_column(Float, default=0.0)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    # The verification code is stored HASHED (never plaintext), with an attempt
    # counter and an expiry so it cannot be brute-forced or replayed indefinitely.
    verify_code_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    verify_attempts: Mapped[int] = mapped_column(Integer, default=0)
    verify_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)
    # Soft-delete: unsubscribing sets active=False rather than deleting the row, so
    # the send audit trail survives and a same-night re-subscribe reuses the row
    # (no duplicate text under a fresh id).
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)
    unsubscribed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)

    sends: Mapped[list["SentNotification"]] = relationship(
        back_populates="subscriber", cascade="all, delete-orphan")


class SentNotification(Base):
    """A record of notifying a subscriber for a forecast date -- with delivery state.

    The uniqueness constraint on (subscriber, date) makes the claim idempotent: a
    concurrent or repeat attempt violates it, so a night is claimed once. ``status``
    then tracks delivery: a row is inserted 'pending' and committed BEFORE the send,
    so a crash cannot lose the claim; on success it becomes 'sent', on failure the
    attempt count rises and a retry pass can pick it up. This is at-least-once with
    bounded duplicates -- the right semantics for a warning system, where a dropped
    frost message costs a crop and a duplicate costs a mild annoyance.
    """

    __tablename__ = "sent_notifications"
    __table_args__ = (
        UniqueConstraint("subscriber_id", "forecast_date", name="uq_sent_sub_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subscriber_id: Mapped[int] = mapped_column(ForeignKey("subscribers.id"), index=True)
    forecast_date: Mapped[_dt.date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|sent|failed
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(String(256), nullable=True)
    claimed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)

    subscriber: Mapped[Subscriber] = relationship(back_populates="sends")


class StationRun(Base):
    """One row per (station, date) the nightly job attempted -- so the service can
    answer "did every station run last night?" and distinguish a skip (ran, no
    usable data) from never having run. Also the hook for a monitor: alert when a
    station skips two nights running. A nightly subscriber whose station skipped
    gets an explicit "no forecast" message rather than ambiguous silence.
    """

    __tablename__ = "station_runs"
    __table_args__ = (
        UniqueConstraint("station", "date", name="uq_run_station_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    station: Mapped[str] = mapped_column(String(64), index=True)
    date: Mapped[_dt.date] = mapped_column(Date, index=True)
    forecast_stored: Mapped[bool] = mapped_column(Boolean)
    skip_reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    ran_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


def make_engine(url: str | None = None, echo: bool = False):
    """An engine for ``url`` (default: the configured database).

    SQLite needs ``check_same_thread=False`` so a session created in one thread
    (a request handler) can be used from the app's thread pool; harmless and
    ignored for Postgres.
    """
    url = url or database_url()
    if not url.startswith("sqlite"):
        return create_engine(url, echo=echo, future=True)
    # SQLite: allow cross-thread use (request pool). A pure in-memory URL
    # (``sqlite://``) otherwise gives each connection its OWN empty database, so
    # pin it to one shared connection via StaticPool -- the schema created on it
    # is then visible to every session.
    kwargs = {"connect_args": {"check_same_thread": False}}
    if url in ("sqlite://", "sqlite:///:memory:"):
        kwargs["poolclass"] = StaticPool
    return create_engine(url, echo=echo, future=True, **kwargs)


def make_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False,
                        future=True)


def create_all(engine) -> None:
    """Create the schema directly from the models.

    Production uses Alembic migrations (``service/migrations``); this is for tests
    and for a first-run bootstrap where migrations are overkill.
    """
    Base.metadata.create_all(engine)
