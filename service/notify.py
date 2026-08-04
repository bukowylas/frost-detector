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

from service import config, db
from service.db import Forecast, SentNotification, StationRun, Subscriber
from service.sms import (
    LogSmsSender, format_forecast_sms, format_no_forecast_sms,
    format_unsubscribe_sms,
)

log = logging.getLogger("frost.notify")

MAX_SEND_ATTEMPTS = 3
# A frost warning's shelf life is HOURS, not days: a forecast for evening D warns
# about the morning of D+1, so a claim still pending the next day must never be
# sent (it would text "frost tonight" for a night already past -- actively
# misleading, worse than a drop). Bound the retry by the claim's age, not by date
# arithmetic. Expired rows stop counting as in-flight.
RETRY_WINDOW = timedelta(hours=12)
# Don't retry a claim younger than this: it may still be in flight from the
# notify pass, so a grace period removes the notify/retry overlap.
RETRY_GRACE = timedelta(minutes=5)


def _as_date(date):
    import pandas as pd
    return pd.Timestamp(date).date()


def _as_aware(dt):
    """Treat a naive datetime (SQLite returns these even for tz-aware columns) as
    UTC, so comparisons never raise. Passes None through."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


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
        claim.last_error = f"{type(exc).__name__}: {exc}"[:db.REASON_LEN]
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


def _expire_stale(session, now):
    """Mark pending claims older than the retry window 'expired' (by the claim's
    age), so a warning past its shelf life can never be re-sent and stops counting
    as in-flight."""
    horizon = _as_aware(now) - RETRY_WINDOW
    stale = [c for c in session.query(SentNotification)
             .filter(SentNotification.status == "pending").all()
             if _as_aware(c.claimed_at) < horizon]
    for claim in stale:
        claim.status = "expired"
    if stale:
        session.commit()
    return len(stale)


def retry_pending(session, sender=None, typical_error_c=None, now=None):
    """Re-attempt recent pending claims that have not exhausted their attempts.

    Bounded by the claim's AGE (RETRY_WINDOW hours): a claim older than that is
    expired, never sent -- so a stale, misleading forecast is never texted. A claim
    younger than RETRY_GRACE is left alone (it may still be in flight from the
    notify pass). Returns (phone, message) re-sent.

    Concurrency: this loads all pending rows and filters by ``claimed_at`` in
    Python (fine at portfolio scale; there is no query left to row-lock as written).
    To make concurrent retry passes safe on Postgres it would need restructuring so
    the age bound is in SQL with ``.with_for_update(skip_locked=True)`` -- not a
    one-line change. For the single-writer nightly design the RETRY_GRACE window is
    what keeps a retry from racing the notify pass.
    """
    sender = sender or LogSmsSender()
    now = _as_aware(now) or datetime.now(timezone.utc)
    _expire_stale(session, now)

    floor = now - RETRY_WINDOW
    ceil = now - RETRY_GRACE
    resent: list[tuple[str, str]] = []
    candidates = (session.query(SentNotification)
                  .filter(SentNotification.status == "pending",
                          SentNotification.attempts < MAX_SEND_ATTEMPTS).all())
    pending = [c for c in candidates if floor <= _as_aware(c.claimed_at) <= ceil]
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


def drain_confirmations(session, sender=None):
    """Re-send unsubscribe confirmations that failed at opt-out time.

    ``unsubscribe`` persists ``confirm_sms_pending`` when the confirmation SMS could
    not be sent (a provider blip). This drains that debt so the confirmation -- the
    STOP model's authentication -- is delivered eventually, never silently dropped.
    Returns the phones confirmed.
    """
    sender = sender or LogSmsSender()
    done = []
    owed = session.query(Subscriber).filter_by(confirm_sms_pending=True).all()
    for sub in owed:
        try:
            sender.send(sub.phone, format_unsubscribe_sms(config.station_label(sub.station)))
        except Exception as exc:  # noqa: BLE001 -- leave the debt for the next run
            log.warning("confirmation to sub %s still failing: %s", sub.id, exc)
            continue
        sub.confirm_sms_pending = False
        sub.last_sms_at = datetime.now(timezone.utc)
        session.commit()
        done.append(sub.phone)
    return done


def notify_health(session, date, stations=None):
    """A one-glance answer to "did the system do its job for ``date``?"

    Distinguishes conditions, only some of which are hard alerts, so an operator is
    not trained to ignore the exit code:

    - ``unwarned`` -- notifications that were NOT delivered: 'failed', but ALSO
                      'pending' and 'expired'. A night where the SMS provider was
                      down end-to-end leaves every claim pending (one attempt < the
                      3-attempt fail threshold), then expired 12 h later -- nobody
                      was warned, so this MUST count, or the total-failure night
                      reports green. An alert.
    - ``missing``  -- a station that was expected to run but produced NO run record
                      (never ran). An alert. ``stations`` bounds "expected" (defaults
                      to the serviceable set) so a partial run isn't falsely missing.
    - ``failed``   -- send failures. An alert.
    - ``skips``    -- a station that ran but had no usable data. Expected and
                      documented, so a single skip is a WARNING; it becomes an
                      alert only when the same station skipped the night before
                      too (the "two nights running" monitor hook).
    """
    import pandas as pd
    d = _as_date(date)
    expected = set(stations) if stations is not None else set(config.SERVICEABLE_STATIONS)
    counts: dict[str, int] = {}
    for (status,) in session.query(SentNotification.status).filter_by(forecast_date=d):
        counts[status] = counts.get(status, 0) + 1
    # Anything not 'sent' means the grower was not warned tonight.
    unwarned = (counts.get("failed", 0) + counts.get("pending", 0)
                + counts.get("expired", 0))

    ran = {r.station for r in session.query(StationRun).filter_by(date=d).all()}
    missing = sorted(expected - ran)

    skips = {r.station: r.skip_reason for r in
             session.query(StationRun).filter_by(date=d, forecast_stored=False).all()}
    # A skip is an alert only if the station also skipped the previous night.
    prev = _as_date(pd.Timestamp(d) - pd.Timedelta(days=1))
    prev_skipped = {r.station for r in
                    session.query(StationRun).filter_by(date=prev, forecast_stored=False).all()}
    repeated_skips = sorted(s for s in skips if s in prev_skipped)

    healthy = not unwarned and not missing and not repeated_skips
    return {"date": d.isoformat(), "send_status": counts, "unwarned": unwarned,
            "missing": missing, "skips": skips, "repeated_skips": repeated_skips,
            "healthy": healthy}
