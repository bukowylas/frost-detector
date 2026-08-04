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

from service import config, db
from service.phone import InvalidPhone, normalize_e164, region_for_station
from service.sms import (
    LogSmsSender, format_settings_changed_sms, format_unsubscribe_sms,
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

    def _issue_code(sub) -> str:
        """Issue a fresh code on ``sub``; returns the plaintext to send once.

        The DB only ever holds the hash; the caller sends the returned plaintext
        after the commit. ``verify_code_issued_at`` is stored directly rather than
        reconstructed from the expiry, so a change to CODE_TTL cannot silently
        misread historical rows."""
        code = f"{secrets.randbelow(1_000_000):06d}"
        sub.verify_code_hash = _hash_code(code)
        sub.verify_attempts = 0
        now = _now()
        sub.verify_code_issued_at = now
        sub.verify_expires_at = now + CODE_TTL
        return code

    def _send_or_503(sndr, to, message):
        try:
            sndr.send(to, message)
        except Exception:  # noqa: BLE001 -- state is committed; a resend recovers it
            raise HTTPException(503, "could not send an SMS; please try again")

    @app.get("/api/stations", response_model=list[StationOut])
    def stations():
        return [StationOut(key=k, label=config.station_label(k)) for k in SERVICEABLE]

    @app.post("/api/subscribe", response_model=SubscribeOut)
    def subscribe(body: SubscribeIn, session=Depends(get_session)):
        if body.station not in SERVICEABLE:
            raise HTTPException(400, f"station {body.station!r} is not serviceable")
        try:
            phone = normalize_e164(body.phone, region_for_station(body.station))
        except InvalidPhone as exc:
            raise HTTPException(422, f"invalid phone: {exc}")

        existing = (session.query(db.Subscriber)
                    .filter_by(phone=phone, station=body.station).one_or_none())

        # A per-phone cooldown throttles code/SMS issue on ANY path (verified or
        # not), so no endpoint is an unthrottled SMS sender pointed at a phone.
        def _cooled_down(sub):
            issued = _as_aware(sub.verify_code_issued_at) if sub else None
            return issued is not None and issued + SUBSCRIBE_COOLDOWN > _now()

        if existing is None:
            sub = db.Subscriber(station=body.station, phone=phone, mode=body.mode,
                                threshold_c=body.threshold_c, verified=False,
                                active=True)
            code = _issue_code(sub)
            session.add(sub)
            session.commit()
            _send_or_503(sender, phone, format_verification_sms(code))
        elif existing.verified and existing.active:
            # Verified + active: apply settings immediately (never disable warnings)
            # and ANNOUNCE the change, so an unauthorised edit is self-reporting.
            if _cooled_down(existing):
                raise HTTPException(429, "please wait before changing settings again")
            existing.mode = body.mode
            existing.threshold_c = body.threshold_c
            existing.verify_code_issued_at = _now()   # throttle the announce SMS too
            session.commit()
            summary = (f"{existing.mode}"
                       + (f" below {existing.threshold_c:+.1f} C"
                          if existing.mode == "frost" else ""))
            _send_or_503(sender, phone,
                         format_settings_changed_sms(config.station_label(existing.station), summary))
            sub = existing
        else:
            # Unverified, OR previously unsubscribed (active=False): (re)enter the
            # pending flow. Reactivation is NOT silent -- it requires a code, so an
            # opted-out number cannot be switched back on by an unauthenticated POST.
            if _cooled_down(existing):
                raise HTTPException(429, "code recently sent; try again shortly")
            existing.mode = body.mode
            existing.threshold_c = body.threshold_c
            existing.verified = False
            code = _issue_code(existing)
            session.commit()
            _send_or_503(sender, phone, format_verification_sms(code))
            sub = existing
        return SubscribeOut(id=sub.id, station=sub.station, verified=sub.verified)

    @app.post("/api/verify", response_model=SubscribeOut)
    def verify(body: VerifyIn, session=Depends(get_session)):
        # Verify is the ONLY path that activates a subscription (initial or a
        # reactivation after unsubscribe), and it always requires a fresh code --
        # so an opted-out number can never be switched back on unauthenticated.
        try:
            phone = normalize_e164(body.phone, region_for_station(body.station))
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
        # Clear the issue timestamp: the code is consumed, so it must no longer
        # count toward the cooldown a subsequent legitimate settings change checks.
        sub.verify_code_issued_at = None
        sub.verified_at = _now()
        session.commit()
        return SubscribeOut(id=sub.id, station=sub.station, verified=True)

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
        if sub is not None and sub.active:
            sub.active = False
            sub.unsubscribed_at = _now()
            session.commit()
            _send_or_503(sender, phone,
                         format_unsubscribe_sms(config.station_label(sub.station)))
        return {"status": "unsubscribed if the subscription existed"}

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
