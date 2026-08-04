"""FastAPI backend: subscriptions, phone verification, and forecast reads.

Endpoints (all JSON, Pydantic-validated):

- ``POST /api/subscribe``        create a PENDING subscriber, OR (on a verified,
                                 active row) apply settings immediately and send a
                                 change-confirmation SMS -- never a silent change.
                                 A previously-unsubscribed row re-enters the
                                 pending/verify flow; it is never reactivated by an
                                 unauthenticated POST.
- ``POST /api/verify``           the ONLY path that activates a subscription
                                 (initial or reactivation). Always requires a fresh
                                 code. Rate-limited, attempt-capped, expiring.
- ``POST /api/unsubscribe``      deactivate immediately and confirm by SMS. The
                                 confirmation IS the authentication (STOP-keyword
                                 model): an attacker cannot silently un-warn a
                                 grower, and the grower resumes with one word.
- ``GET  /api/forecasts/{station}`` the latest stored forecast(s) for a station.
- ``GET  /api/stations``         the serviceable stations (for the signup dropdown).

Safety posture: a verified subscriber is never silently un-verified OR silently
reconfigured (settings changes are announced); an opted-out number is never
reactivated without a fresh code; every code-issuing path is throttled by a
per-phone cooldown, so no endpoint is an open SMS sender; the phone is normalised
to E.164 (per-station region); codes are hashed, expire, and lock after 5 tries.

The built React frontend is served as static files from this same app. Live
weather and real SMS are never touched here -- the SMS sender is injected (log-stub
by default), and forecasts are read from the DB the nightly job wrote.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from service import config, db
from service.phone import InvalidPhone, normalize_e164, region_for_station
from service.sms import (
    LogSmsSender, format_settings_summary, format_unsubscribe_sms,
    format_verification_sms,
)

# The serviceable set and labels come from service.config -- one source of truth,
# an explicit literal that never falls back to "every provider station".
SERVICEABLE = config.SERVICEABLE_STATIONS

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"

CODE_TTL = timedelta(minutes=10)
MAX_VERIFY_ATTEMPTS = 5
SUBSCRIBE_COOLDOWN = timedelta(minutes=1)   # per phone: throttle code re-issue


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def _now():
    return datetime.now(timezone.utc)


def _as_aware(dt):
    """Some DB backends (SQLite) return naive datetimes even for tz-aware columns;
    treat a naive value as UTC so comparisons never raise."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# --- request/response models -------------------------------------------------

class SubscribeIn(BaseModel):
    station: str
    phone: str = Field(min_length=5, max_length=32)
    mode: str = Field(default="nightly", pattern="^(nightly|frost)$")
    threshold_c: float = Field(default=0.0, ge=-10.0, le=5.0)


class StatusOut(BaseModel):
    # Deliberately opaque: subscribe returns this regardless of whether the phone
    # was known, its verification state, or its PK -- so the endpoint cannot be
    # used to enumerate subscribers or their state.
    status: str = "ok"


class VerifyOut(BaseModel):
    verified: bool


class VerifyIn(BaseModel):
    phone: str
    station: str
    code: str = Field(pattern=r"^\d{6}$")   # exactly 6 digits; also avoids a
    #                                          non-ASCII compare_digest TypeError


class UnsubscribeStartIn(BaseModel):
    phone: str
    station: str


class ForecastOut(BaseModel):
    station: str
    date: str
    predicted_tmin_c: float
    alarm_fired: bool
    model_version: str
    cutoff_ts: str
    snapshot_ts: str | None   # None on pre-Stage-7 rows


class StationOut(BaseModel):
    key: str
    label: str


def create_app(session_factory=None, sms_sender=None) -> FastAPI:
    """Build the app. ``session_factory`` and ``sms_sender`` are injectable so
    tests use SQLite + the log stub, and production wires Postgres + a real
    provider without changing this code."""
    if session_factory is None:
        engine = db.make_engine()
        # Bootstrap the schema only when explicitly asked (dev), so a production
        # boot before `alembic upgrade head` does not create tables that then
        # collide with the first migration.
        import os
        if os.environ.get("FROST_BOOTSTRAP_DB") == "1":
            db.create_all(engine)
        session_factory = db.make_session_factory(engine)
    sender = sms_sender or LogSmsSender()

    app = FastAPI(title="Frost Detector", docs_url="/api/docs")

    def get_session():
        s = session_factory()
        try:
            yield s
        finally:
            s.close()

    def _stage_code(sub, mode, threshold_c) -> str:
        """Stage requested settings and a fresh code on ``sub`` WITHOUT touching the
        active mode/threshold_c. Returns the plaintext code to send once (the DB
        only holds the hash). The staged settings are promoted only when the code
        is verified -- so an unauthenticated request can never change what a
        subscriber receives."""
        sub.pending_mode = mode
        sub.pending_threshold_c = threshold_c
        code = f"{secrets.randbelow(1_000_000):06d}"
        sub.verify_code_hash = _hash_code(code)
        sub.verify_attempts = 0
        sub.verify_expires_at = _now() + CODE_TTL
        return code

    def _phone_cooled_down(session, phone) -> bool:
        """True if an SMS was sent to THIS PHONE (any station) within the cooldown.
        Keyed on the phone, not (phone, station), so N stations can't multiply the
        rate and a first-contact number is still governed once a row exists."""
        last = (session.query(func.max(db.Subscriber.last_sms_at))
                .filter(db.Subscriber.phone == phone).scalar())
        last = _as_aware(last)
        return last is not None and last + SUBSCRIBE_COOLDOWN > _now()

    def _send_and_mark(session, sub, phone, message):
        """Send, then record last_sms_at only on success -- a failed send must not
        burn the cooldown (which would mislead 'code recently sent')."""
        try:
            sender.send(phone, message)
        except Exception:  # noqa: BLE001 -- staged state is committed; a resend recovers it
            raise HTTPException(503, "could not send an SMS; please try again")
        sub.last_sms_at = _now()
        session.commit()

    @app.get("/api/stations", response_model=list[StationOut])
    def stations():
        return [StationOut(key=k, label=config.station_label(k)) for k in SERVICEABLE]

    @app.post("/api/subscribe", response_model=StatusOut)
    def subscribe(body: SubscribeIn, session=Depends(get_session)):
        # Opaque by design (A3): every path returns the same StatusOut, so the
        # endpoint reveals nothing about whether the number is known or its state.
        if body.station not in SERVICEABLE:
            raise HTTPException(400, f"station {body.station!r} is not serviceable")
        try:
            phone = normalize_e164(body.phone, region_for_station(body.station))
        except InvalidPhone as exc:
            raise HTTPException(422, f"invalid phone: {exc}")

        # One 429 string regardless of state, so a cooled-down response is not an
        # oracle for whether the number is subscribed.
        if _phone_cooled_down(session, phone):
            raise HTTPException(429, "please wait before requesting another code")

        existing = (session.query(db.Subscriber)
                    .filter_by(phone=phone, station=body.station).one_or_none())
        if existing is None:
            sub = db.Subscriber(station=body.station, phone=phone,
                                verified=False, active=False)
            session.add(sub)
            try:
                session.flush()   # surface a concurrent-insert conflict now
            except IntegrityError:
                # Another request created the row first (A5): fall through to it.
                session.rollback()
                sub = (session.query(db.Subscriber)
                       .filter_by(phone=phone, station=body.station).one())
            existing = sub

        # EVERY path (new, unverified, verified, opted-out) only STAGES settings +
        # a code. Nothing that changes what the subscriber receives is applied here
        # -- verification promotes it. This is the single rule that makes the auth
        # safe (A1 + A2).
        code = _stage_code(existing, body.mode, body.threshold_c)
        session.commit()
        summary = format_settings_summary(config.station_label(body.station),
                                          body.mode, body.threshold_c)
        _send_and_mark(session, existing, phone, format_verification_sms(code, summary))
        return StatusOut()

    @app.post("/api/verify", response_model=VerifyOut)
    def verify(body: VerifyIn, session=Depends(get_session)):
        # Verify is the ONLY path that activates a subscription and the ONLY path
        # that promotes staged settings into effect. It always requires a fresh
        # code, so nothing a grower receives changes without one.
        #
        # DoS note (A6): the 5-attempt lock protects the code, but on its own it is
        # lockout-DoS-able (an attacker burns the cap so the grower's real code
        # 429s). The mitigation is a per-IP rate limit at the edge (reverse proxy /
        # slowapi) plus a global daily send budget -- deployment-layer concerns, not
        # wired here because the default sender is a no-cost log stub. The code path
        # must not be shaped so that adding a real provider is the dangerous step.
        try:
            phone = normalize_e164(body.phone, region_for_station(body.station))
        except InvalidPhone as exc:
            raise HTTPException(422, f"invalid phone: {exc}")
        sub = (session.query(db.Subscriber)
               .filter_by(phone=phone, station=body.station).one_or_none())
        # Same 400 whether unknown or wrong code, so verify cannot enumerate.
        bad = HTTPException(400, "invalid phone, station, or code")
        if sub is None or sub.verify_code_hash is None:
            raise bad
        if sub.verify_attempts >= MAX_VERIFY_ATTEMPTS:
            raise HTTPException(429, "too many attempts; request a new code")
        expires = _as_aware(sub.verify_expires_at)
        if expires is None or expires < _now():
            raise HTTPException(400, "code expired; request a new code")

        sub.verify_attempts += 1
        if not secrets.compare_digest(sub.verify_code_hash, _hash_code(body.code)):
            session.commit()
            raise bad
        # Correct code -> activate AND promote the staged settings (falling back to
        # the current values if nothing was staged).
        sub.verified = True
        sub.active = True
        if sub.pending_mode is not None:
            sub.mode = sub.pending_mode
        if sub.pending_threshold_c is not None:
            sub.threshold_c = sub.pending_threshold_c
        sub.pending_mode = None
        sub.pending_threshold_c = None
        sub.verify_code_hash = None
        sub.verify_expires_at = None
        sub.verified_at = _now()
        session.commit()
        return VerifyOut(verified=True)

    @app.post("/api/unsubscribe")
    def unsubscribe(body: UnsubscribeStartIn, session=Depends(get_session)):
        """Deactivate a subscription immediately and confirm by SMS.

        Modelled on the carrier STOP keyword: the confirmation SMS IS the
        authentication. An attacker who deactivates a grower cannot stop the grower
        being told, and the grower resumes with one word (or the re-subscribe +
        verify flow). This is safer AND lighter than a code round-trip, and it
        never leaves a grower silently un-warned without notice.

        Deactivation is a no-op if the subscription is already inactive/unknown,
        and the response is identical either way, so the endpoint cannot enumerate
        subscribers. Reactivation is NEVER done here -- only via re-verify.
        """
        try:
            phone = normalize_e164(body.phone, region_for_station(body.station))
        except InvalidPhone as exc:
            raise HTTPException(422, f"invalid phone: {exc}")
        sub = (session.query(db.Subscriber)
               .filter_by(phone=phone, station=body.station).one_or_none())
        if sub is not None:
            was_active = sub.active
            sub.active = False
            # Invalidate any outstanding code (A7) on ANY opt-out, active or pending:
            # a code issued before the opt-out must not reactivate after it. Also
            # drop staged settings. Done regardless of prior active state, so opting
            # out of a still-pending subscription cancels its code too.
            sub.verify_code_hash = None
            sub.verify_expires_at = None
            sub.pending_mode = None
            sub.pending_threshold_c = None
            if was_active:
                sub.unsubscribed_at = _now()
            session.commit()
            # Confirm by SMS only if they were actually active (nothing to confirm
            # for a pending row that never activated).
            if was_active:
                try:
                    sender.send(phone,
                                format_unsubscribe_sms(config.station_label(sub.station)))
                    sub.last_sms_at = _now()
                    session.commit()
                except Exception:  # noqa: BLE001 -- deactivation durable; SMS best-effort
                    pass
        return StatusOut()

    @app.get("/api/forecasts/{station}", response_model=list[ForecastOut])
    def forecasts(station: str, limit: int = 7, session=Depends(get_session)):
        limit = max(1, min(limit, 30))
        rows = (session.query(db.Forecast).filter_by(station=station)
                .order_by(db.Forecast.date.desc()).limit(limit).all())
        return [ForecastOut(station=r.station, date=r.date.isoformat(),
                            predicted_tmin_c=r.predicted_tmin_c,
                            alarm_fired=r.alarm_fired, model_version=r.model_version,
                            cutoff_ts=r.cutoff_ts, snapshot_ts=r.snapshot_ts)
                for r in rows]

    if WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")

    return app
