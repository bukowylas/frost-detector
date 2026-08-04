"""Unit tests for SMS formatting and the notify step -- offline, log-stub sender.

Covers who gets texted under each mode/threshold, the heartbeat when a station
skipped, the per-subscriber frost verdict, and the two-phase delivery guarantees
(idempotent claim, retry of a failed send, no double-text).
"""

import datetime as dt
import json

import pytest

from service import db, notify
from service.sms import STOP_LINE, LogSmsSender, format_forecast_sms

DAY = dt.date(2021, 4, 15)


@pytest.fixture
def session():
    engine = db.make_engine("sqlite://")
    db.create_all(engine)
    return db.make_session_factory(engine)()


def _forecast(session, station="uk_waddington", tmin=-1.0, alarm=True):
    session.add(db.Forecast(
        station=station, date=DAY, predicted_tmin_c=tmin, alarm_fired=alarm,
        alarm_threshold_c=1.5, model_version="m-test",
        features_json=json.dumps({"temp_c": 2.0}),
        cutoff_ts=f"{DAY}T18:00:00", snapshot_ts=f"{DAY}T18:00:00"))
    session.commit()


def _run(session, station="uk_waddington", stored=True, reason=None):
    session.add(db.StationRun(station=station, date=DAY, forecast_stored=stored,
                              skip_reason=reason))
    session.commit()


def _sub(session, phone, station="uk_waddington", mode="nightly", threshold=0.0,
         verified=True, active=True):
    s = db.Subscriber(station=station, phone=phone, mode=mode,
                      threshold_c=threshold, verified=verified, active=active)
    session.add(s)
    session.commit()
    return s


class TestMessageFormat:
    def test_reports_magnitude_and_calls_itself_a_forecast(self):
        msg = format_forecast_sms("Waddington", "2021-04-15", -1.1, frost_likely=True,
                                  typical_error_c=1.8)
        assert "-1.1 C" in msg and "Predicted min" in msg
        assert "frost likely" in msg and STOP_LINE in msg
        assert "+/-1.8 C" in msg          # the uncertainty band

    def test_mild_night_omits_frost_likely(self):
        msg = format_forecast_sms("Cranwell", "2021-04-15", 6.2, frost_likely=False)
        assert "frost likely" not in msg and "+6.2 C" in msg


class TestNotifyLogic:
    def test_nightly_subscriber_notified_with_forecast(self, session):
        _forecast(session, tmin=6.0, alarm=False)
        _run(session)
        _sub(session, "+44700900001", mode="nightly")
        sent = notify.notify_for_date(session, DAY)
        assert len(sent) == 1

    def test_nightly_subscriber_heartbeats_when_station_skipped(self, session):
        # No forecast, but the station DID run (and skipped) -> explicit message.
        _run(session, stored=False, reason="provider unreachable")
        _sub(session, "+44700900002", mode="nightly")
        sent = notify.notify_for_date(session, DAY)
        assert len(sent) == 1
        assert "No forecast tonight" in sent[0][1]

    def test_silent_when_station_never_ran(self, session):
        # No forecast AND no run record -> operator problem, not a grower message.
        _sub(session, "+44700900003", mode="nightly")
        assert notify.notify_for_date(session, DAY) == []

    def test_frost_subscriber_silent_above_threshold(self, session):
        _forecast(session, tmin=6.0, alarm=False)
        _run(session)
        _sub(session, "+44700900004", mode="frost", threshold=0.0)
        assert notify.notify_for_date(session, DAY) == []

    def test_frost_subscriber_notified_at_or_below_threshold(self, session):
        _forecast(session, tmin=-1.0, alarm=True)
        _run(session)
        _sub(session, "+44700900005", mode="frost", threshold=0.0)
        assert len(notify.notify_for_date(session, DAY)) == 1

    def test_verdict_is_per_subscriber_threshold(self, session):
        # Forecast +2.0. A subscriber with threshold +3.0 is texted (nightly) and
        # SHOULD see 'frost likely' because it crossed THEIR threshold, even though
        # the global alarm (1.5) did not fire.
        _forecast(session, tmin=2.0, alarm=False)
        _run(session)
        _sub(session, "+44700900006", mode="nightly", threshold=3.0)
        msg = notify.notify_for_date(session, DAY)[0][1]
        assert "frost likely" in msg

    def test_unverified_and_inactive_never_notified(self, session):
        _forecast(session)
        _run(session)
        _sub(session, "+44700900007", verified=False)
        _sub(session, "+44700900008", verified=True, active=False)
        assert notify.notify_for_date(session, DAY) == []


class TestDeliveryGuarantees:
    def test_rerun_does_not_double_text(self, session):
        _forecast(session, tmin=-1.0)
        _run(session)
        _sub(session, "+44700900009", mode="nightly")
        sender = LogSmsSender()
        notify.notify_for_date(session, DAY, sender=sender)
        notify.notify_for_date(session, DAY, sender=sender)     # retry
        assert len(sender.sent) == 1
        assert session.query(db.SentNotification).filter_by(status="sent").count() == 1

    def test_failed_send_is_retryable_and_not_lost(self, session):
        _forecast(session, tmin=-1.0)
        _run(session)
        _sub(session, "+44700900010", mode="nightly")

        class FlakySender:
            def __init__(self):
                self.calls = 0
            def send(self, to, message):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("provider 503")

        flaky = FlakySender()
        notify.notify_for_date(session, DAY, sender=flaky)      # first attempt fails
        claim = session.query(db.SentNotification).one()
        assert claim.status == "pending" and claim.attempts == 1

        # today=DAY so the (fixed test date) claim is within the retry window.
        notify.retry_pending(session, sender=flaky, today=DAY)   # retry succeeds
        session.refresh(claim)
        assert claim.status == "sent" and flaky.calls == 2

    def test_stale_pending_claim_is_expired_not_resent(self, session):
        # A claim left pending for an old night must never be re-sent -- a stale
        # frost warning is actively misleading.
        _forecast(session, tmin=-1.0)
        _sub(session, "+44700900013", mode="nightly")
        session.add(db.SentNotification(subscriber_id=1, forecast_date=DAY,
                                        status="pending", attempts=1))
        session.commit()
        sender = LogSmsSender()
        # "today" is far after DAY -> the claim is outside the retry window.
        notify.retry_pending(session, sender=sender, today=dt.date(2026, 4, 15))
        assert sender.sent == []
        assert session.query(db.SentNotification).one().status == "expired"

    def test_send_failure_does_not_abort_the_batch(self, session):
        _forecast(session, tmin=-1.0)
        _run(session)
        _sub(session, "+44700900011", mode="nightly")
        _sub(session, "+44700900012", mode="nightly")

        class OneBadNumber:
            def send(self, to, message):
                if to.endswith("11"):
                    raise RuntimeError("dead number")

        # The bad number fails, but the loop still reaches the good one.
        notify.notify_for_date(session, DAY, sender=OneBadNumber())
        statuses = {n.status for n in session.query(db.SentNotification).all()}
        assert "sent" in statuses            # the good number was delivered


class TestNotifyHealth:
    def test_healthy_when_all_sent_and_no_skips(self, session):
        _forecast(session, tmin=-1.0)
        _run(session)
        _sub(session, "+44700900020", mode="nightly")
        notify.notify_for_date(session, DAY)
        health = notify.notify_health(session, DAY)
        assert health["healthy"] is True
        assert health["send_status"].get("sent") == 1

    def test_unhealthy_when_a_station_skipped(self, session):
        _run(session, stored=False, reason="provider unreachable")
        health = notify.notify_health(session, DAY)
        assert health["healthy"] is False
        assert health["skips"] == [("uk_waddington", "provider unreachable")]
