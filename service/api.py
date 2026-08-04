"""FastAPI backend: subscriptions, phone verification, and forecast reads.

Endpoints (all JSON, Pydantic-validated):

- ``POST /api/subscribe``        create a PENDING, unverified subscriber (or update
                                 settings on an existing one WITHOUT deactivating a
                                 verified subscription); a code is sent to the phone.
- ``POST /api/verify``           activate a subscriber by returning the code sent to
                                 their phone. Rate-limited, attempt-capped, expiring.
- ``POST /api/unsubscribe``      deactivate a subscriber -- REQUIRES a valid code,
                                 because silently disabling a grower's frost warnings
                                 is a bigger harm than a duplicate text.
- ``GET  /api/forecasts/{station}`` the latest stored forecast(s) for a station.
- ``GET  /api/stations``         the serviceable stations (for the signup dropdown).

Safety posture: a verified subscriber is never silently un-verified; the phone is
the identity, so it is normalised to E.164; the code is hashed, expires, and locks
after too many attempts.

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

from service import config, db
from service.phone import InvalidPhone, normalize_e164
from service.sms import LogSmsSender, format_verification_sms

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


class SubscribeOut(BaseModel):
    id: int
    station: str
    verified: bool


class VerifyIn(BaseModel):
    phone: str
    station: str
    code: str = Field(pattern=r"^\d{6}$")   # exactly 6 digits; also avoids a
    #                                          non-ASCII compare_digest TypeError


class UnsubscribeStartIn(BaseModel):
    phone: str
    station: str


class UnsubscribeIn(BaseModel):
    phone: str
    station: str
    code: str = Field(pattern=r"^\d{6}$")


class ForecastOut(BaseModel):
    station: str
    date: str
    predicted_tmin_c: float
    alarm_fired: bool
    model_version: str
    cutoff_ts: str
    snapshot_ts: str


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

    def _issue_code(sub) -> str:
        code = f"{secrets.randbelow(1_000_000):06d}"
        sub.verify_code_hash = _hash_code(code)
        sub.verify_attempts = 0
        sub.verify_expires_at = _now() + CODE_TTL
        return code

    @app.get("/api/stations", response_model=list[StationOut])
    def stations():
        return [StationOut(key=k, label=config.station_label(k)) for k in SERVICEABLE]

    @app.post("/api/subscribe", response_model=SubscribeOut)
    def subscribe(body: SubscribeIn, session=Depends(get_session)):
        if body.station not in SERVICEABLE:
            raise HTTPException(400, f"station {body.station!r} is not serviceable")
        try:
            phone = normalize_e164(body.phone)
        except InvalidPhone as exc:
            raise HTTPException(422, f"invalid phone: {exc}")

        existing = (session.query(db.Subscriber)
                    .filter_by(phone=phone, station=body.station).one_or_none())
        if existing is None:
            sub = db.Subscriber(station=body.station, phone=phone, mode=body.mode,
                                threshold_c=body.threshold_c, verified=False,
                                active=True)
            code = _issue_code(sub)
            session.add(sub)
        else:
            # Per-phone cooldown: don't let an open endpoint re-issue codes (and
            # send SMS) in a tight loop on someone else's number.
            last = _as_aware(existing.verify_expires_at)
            if (not existing.verified and last is not None
                    and last - CODE_TTL + SUBSCRIBE_COOLDOWN > _now()):
                raise HTTPException(429, "code recently sent; try again shortly")
            # Update settings. Crucially, a VERIFIED subscriber stays verified and
            # ACTIVE -- new settings apply immediately, we do not silently disable
            # their warnings. Only an unverified/ inactive row (re)enters pending.
            existing.mode = body.mode
            existing.threshold_c = body.threshold_c
            sub = existing
            if existing.verified:
                existing.active = True   # re-subscribe reactivates without a code
                code = None
            else:
                existing.active = True
                code = _issue_code(existing)
        session.commit()
        if code is not None:
            try:
                sender.send(phone, format_verification_sms(code))
            except Exception:  # noqa: BLE001 -- row is written; a resend recovers it
                raise HTTPException(503, "could not send the verification code; "
                                         "please try again")
        return SubscribeOut(id=sub.id, station=sub.station, verified=sub.verified)

    @app.post("/api/verify", response_model=SubscribeOut)
    def verify(body: VerifyIn, session=Depends(get_session)):
        try:
            phone = normalize_e164(body.phone)
        except InvalidPhone as exc:
            raise HTTPException(422, f"invalid phone: {exc}")
        sub = (session.query(db.Subscriber)
               .filter_by(phone=phone, station=body.station).one_or_none())
        # Return the SAME 400 whether the subscription is unknown or the code is
        # wrong, so the endpoint cannot be used to enumerate who is subscribed.
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
        sub.verified = True
        sub.active = True
        sub.verify_code_hash = None
        sub.verify_expires_at = None
        sub.verified_at = _now()
        session.commit()
        return SubscribeOut(id=sub.id, station=sub.station, verified=True)

    @app.post("/api/unsubscribe")
    def unsubscribe(body: UnsubscribeStartIn, session=Depends(get_session)):
        """Start unsubscribe: send a confirmation code to the phone.

        Unsubscribing must be authenticated -- an open endpoint that disables a
        grower's frost warnings by phone number is a real harm. This issues a code;
        the caller confirms via /api/unsubscribe/confirm. (A real provider's
        carrier-authenticated STOP keyword is the other legitimate path.) Always
        returns 202 regardless of whether the subscription exists, so the endpoint
        cannot enumerate subscribers.
        """
        try:
            phone = normalize_e164(body.phone)
        except InvalidPhone as exc:
            raise HTTPException(422, f"invalid phone: {exc}")
        sub = (session.query(db.Subscriber)
               .filter_by(phone=phone, station=body.station).one_or_none())
        if sub is not None and sub.active:
            code = _issue_code(sub)
            session.commit()
            sender.send(phone, format_verification_sms(code))
        return {"status": "confirmation code sent if the subscription exists"}

    @app.post("/api/unsubscribe/confirm")
    def unsubscribe_confirm(body: UnsubscribeIn, session=Depends(get_session)):
        try:
            phone = normalize_e164(body.phone)
        except InvalidPhone as exc:
            raise HTTPException(422, f"invalid phone: {exc}")
        sub = (session.query(db.Subscriber)
               .filter_by(phone=phone, station=body.station).one_or_none())
        bad = HTTPException(400, "invalid phone, station, or code")
        if sub is None or sub.verify_code_hash is None:
            raise bad
        expires = _as_aware(sub.verify_expires_at)
        if expires is None or expires < _now():
            raise HTTPException(400, "code expired; request unsubscribe again")
        if not secrets.compare_digest(sub.verify_code_hash, _hash_code(body.code)):
            raise bad
        # Soft-delete: keep the row (and its send history) but deactivate.
        sub.active = False
        sub.unsubscribed_at = _now()
        sub.verify_code_hash = None
        sub.verify_expires_at = None
        session.commit()
        return {"unsubscribed": True}

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
