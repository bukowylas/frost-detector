"""Database layer: schema, engine and session management.

PostgreSQL in production (via ``DATABASE_URL``); an in-memory/file SQLite engine
in tests. The models below are written to run on both -- portable column types,
JSON stored as text where a native JSON type is not guaranteed -- so the fast
offline suite never needs a running Postgres.

Four tables carry the service's state:

- ``forecasts`` -- one row per (station, night), stored RICHLY: the full feature
  vector, the model version, the intended cutoff and the actual snapshot time.
  Scoring forecasts later is out of scope, but the data to diagnose "which nights
  did it miss, and what did they share" must be captured now; it is free at write
  time and irreplaceable afterwards. ``unique(station, date)`` makes the nightly
  job idempotent.
- ``subscribers`` -- who to notify, how (nightly heartbeat vs frost-only), at what
  threshold, and their phone/verification state.
- ``sent_notifications`` -- the delivery log. Its ``unique(subscriber, date)``
  makes the CLAIM idempotent (a night is claimed once), and its ``status`` tracks
  delivery. Semantics are at-least-once with bounded duplicates -- claim before
  send, retry on failure -- the right trade for a warning system (a dropped frost
  message costs a crop; a duplicate costs a mild annoyance). See its own docstring.
- ``station_runs`` -- one row per (station, night) attempted, so the service can
  answer "did every station run?" and distinguish a skip from a never-ran.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import datetime as _dt

from sqlalchemy import (
    Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text,
    UniqueConstraint, create_engine, text,
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
    # LST of the obs actually used. Nullable so a migration can add it to a table
    # of pre-existing rows (which genuinely don't know their snapshot time) without
    # a dishonest backfill; new rows always set it.
    snapshot_ts: Mapped[str | None] = mapped_column(String(32), nullable=True)
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
    # Stored directly (not reconstructed from the expiry) so a change to the code
    # TTL cannot silently misread historical rows when computing the cooldown.
    verify_code_issued_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)
    verify_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)
    # Soft-delete: unsubscribing sets active=False rather than deleting the row, so
    # the send audit trail survives and a same-night re-subscribe reuses the row
    # (no duplicate text under a fresh id). server_default so a migration adding
    # this column to an existing table leaves every current subscriber ACTIVE --
    # without it the ALTER would default them to inactive, a fleet-wide silent
    # denial of warning delivered by the migration itself.
    active: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default=text("1"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)
    unsubscribed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)

    # No delete-orphan cascade: subscribers are soft-deleted (active=False), and
    # the send-history audit trail must survive. A stray hard delete should error
    # on the FK rather than silently destroy the log the soft-delete preserves.
    sends: Mapped[list["SentNotification"]] = relationship(
        back_populates="subscriber", passive_deletes=True)


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
