"""Notification step: text the right subscribers about a night's forecast.

Runs after the nightly job has stored forecasts and recorded station runs. It
iterates over **(station, subscriber)** pairs, not merely over stored forecasts,
so a nightly subscriber whose station was skipped still hears something -- silence
is ambiguous between "clear" and "the system is down", and the heartbeat exists to
remove exactly that ambiguity.

Delivery is **at-least-once with bounded duplicates**, the correct semantics for a
warning system (a dropped frost message costs a crop; a duplicate costs a mild
annoyance). Each intended send is a two-phase claim:

1. Insert a ``SentNotification`` row as ``pending`` and commit. The uniqueness
   constraint on (subscriber, date) prevents concurrent/repeat duplicates.
2. Attempt the send. On success mark ``sent``; on failure record the error and
   leave it retryable. A crash between claim and send leaves a ``pending`` row a
   retry pass can pick up -- nothing is silently dropped.

Each send is wrapped, so one dead number never aborts the batch.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy.exc import IntegrityError

from service import config
from service.db import Forecast, SentNotification, StationRun, Subscriber
from service.sms import (
    LogSmsSender, format_forecast_sms, format_no_forecast_sms,
)

log = logging.getLogger("frost.notify")

MAX_SEND_ATTEMPTS = 3
# A frost warning has a shelf life of hours; after this window a pending claim is
# stale (the night is long past) and must never be sent -- a stale warning is
# actively misleading, worse than a drop. Expired rows stop counting as in-flight.
RETRY_WINDOW_DAYS = 1


def _as_date(date):
    import pandas as pd
    return pd.Timestamp(date).date()


def _should_notify(sub: Subscriber, forecast: Forecast | None) -> bool:
    """Whether this subscriber should get a message tonight.

    nightly: always (including a 'no forecast' heartbeat when the station skipped).
    frost:   only when a forecast exists and crosses the subscriber's threshold.

    Deliberate choice: a frost-mode subscriber gets NOTHING on a skipped night.
    They opted into silence-unless-frost, so a skip stays silent for them -- unlike
    a nightly subscriber, for whom silence would be the very ambiguity the
    heartbeat removes. (If skips ever become common enough that a frost subscriber
    should hear "no data tonight", revisit this -- it is a decision, not an
    oversight.)
    """
    if sub.mode == "nightly":
        return True
    return forecast is not None and forecast.predicted_tmin_c <= sub.threshold_c


def _message_for(sub: Subscriber, forecast: Forecast | None, date_label: str,
                 typical_error_c: float | None) -> str:
    label = config.station_label(sub.station)
    if forecast is None:
        return format_no_forecast_sms(label, date_label)
    # The verdict is per-subscriber: 'frost likely' iff the forecast crosses THIS
    # subscriber's threshold, not the global alarm threshold.
    frost_likely = forecast.predicted_tmin_c <= sub.threshold_c
    return format_forecast_sms(label, date_label, forecast.predicted_tmin_c,
                               frost_likely, typical_error_c=typical_error_c)


def _claim(session, subscriber_id: int, date) -> SentNotification | None:
    """Insert a pending claim for (subscriber, date). Returns the row if this call
    won the claim, or None if one already exists (already handled / in flight).

    This commits the session, so notify must own its session -- do not share one
    with an in-flight forecast write, or the claim would flush partial state."""
    row = SentNotification(subscriber_id=subscriber_id, forecast_date=_as_date(date),
                           status="pending", attempts=0)
    session.add(row)
    try:
        session.commit()
        return row
    except IntegrityError:
        session.rollback()
        return None


def _deliver(session, sender, sub: Subscriber, claim: SentNotification, message):
    """Attempt one send and record the outcome. Never raises out."""
    claim.attempts += 1
    try:
        sender.send(sub.phone, message)
        claim.status = "sent"
        claim.sent_at = datetime.now(timezone.utc)
        claim.last_error = None
        session.commit()
        return True
    except Exception as exc:  # noqa: BLE001 -- one dead number must not abort the batch
        claim.status = "failed" if claim.attempts >= MAX_SEND_ATTEMPTS else "pending"
        claim.last_error = f"{type(exc).__name__}: {exc}"[:256]
        session.commit()
        log.warning("send to sub %s failed (attempt %d): %s",
                    sub.id, claim.attempts, exc)
        return False


def notify_for_date(session, date, sender=None, typical_error_c=None):
    """Notify every eligible subscriber for ``date``. Returns (phone, message) sent.

    Iterates (station, subscriber) so skipped stations still heartbeat. Uses the
    log-stub sender by default, so this runs with no SMS account.
    """
    sender = sender or LogSmsSender()
    d = _as_date(date)
    date_label = d.isoformat()
    sent: list[tuple[str, str]] = []

    subs = (session.query(Subscriber)
            .filter_by(verified=True, active=True).all())
    for sub in subs:
        forecast = (session.query(Forecast)
                    .filter_by(station=sub.station, date=d).one_or_none())
        # If the station never ran (no forecast AND no run record), stay silent --
        # "never ran" is an operator problem, not a grower message. A recorded skip
        # DOES heartbeat a nightly subscriber.
        run = (session.query(StationRun)
               .filter_by(station=sub.station, date=d).one_or_none())
        if forecast is None and run is None:
            continue
        if not _should_notify(sub, forecast):
            continue

        claim = _claim(session, sub.id, d)
        if claim is None:
            log.info("sub %s already claimed for %s; skipping", sub.id, date_label)
            continue
        message = _message_for(sub, forecast, date_label, typical_error_c)
        if _deliver(session, sender, sub, claim, message):
            sent.append((sub.phone, message))
    return sent


def _expire_stale(session, today):
    """Mark pending claims older than the retry window 'expired', so a long-past
    night can never be re-sent and stops counting as in-flight."""
    horizon = today - timedelta(days=RETRY_WINDOW_DAYS)
    stale = (session.query(SentNotification)
             .filter(SentNotification.status == "pending",
                     SentNotification.forecast_date < horizon).all())
    for claim in stale:
        claim.status = "expired"
    if stale:
        session.commit()
    return len(stale)


def retry_pending(session, sender=None, typical_error_c=None, today=None):
    """Re-attempt recent pending claims that have not exhausted their attempts.

    Bounded to the retry window: a claim for a night older than RETRY_WINDOW_DAYS
    is expired, never sent -- so a claim left pending months ago cannot text a
    grower a stale, misleading forecast. Returns (phone, message) re-sent.
    """
    sender = sender or LogSmsSender()
    today = today or _as_date(datetime.now(timezone.utc))
    _expire_stale(session, today)

    horizon = today - timedelta(days=RETRY_WINDOW_DAYS)
    resent: list[tuple[str, str]] = []
    pending = (session.query(SentNotification)
               .filter(SentNotification.status == "pending",
                       SentNotification.attempts < MAX_SEND_ATTEMPTS,
                       SentNotification.forecast_date >= horizon).all())
    for claim in pending:
        sub = session.get(Subscriber, claim.subscriber_id)
        if sub is None or not (sub.verified and sub.active):
            continue
        forecast = (session.query(Forecast)
                    .filter_by(station=sub.station, date=claim.forecast_date)
                    .one_or_none())
        message = _message_for(sub, forecast, claim.forecast_date.isoformat(),
                               typical_error_c)
        if _deliver(session, sender, sub, claim, message):
            resent.append((sub.phone, message))
    return resent


def notify_health(session, date):
    """A one-glance answer to "did the system do its job for ``date``?"

    Returns a dict of send-status counts and the stations that skipped (with
    reasons). A non-empty ``failed`` or ``skips`` is the alert condition -- the
    caller (run_nightly) logs it and exits non-zero, so a silent cron becomes a
    watched one.
    """
    d = _as_date(date)
    counts: dict[str, int] = {}
    for (status,) in session.query(SentNotification.status).filter_by(forecast_date=d):
        counts[status] = counts.get(status, 0) + 1
    skips = [(r.station, r.skip_reason) for r in
             session.query(StationRun).filter_by(date=d, forecast_stored=False).all()]
    return {"date": d.isoformat(), "send_status": counts, "skips": skips,
            "healthy": not counts.get("failed") and not skips}
