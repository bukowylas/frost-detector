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


def _now_utc():
    return dt.datetime.now(dt.timezone.utc)


def _aware(d):
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


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

        # now = 10 min after the claim: past the 5-min grace, inside the window.
        now = _aware(claim.claimed_at) + dt.timedelta(minutes=10)
        notify.retry_pending(session, sender=flaky, now=now)     # retry succeeds
        session.refresh(claim)
        assert claim.status == "sent" and flaky.calls == 2

    def test_stale_pending_claim_is_expired_not_resent(self, session):
        # A claim older than the retry WINDOW (by age, not date) must never be
        # re-sent -- a stale frost warning is actively misleading.
        _forecast(session, tmin=-1.0)
        _sub(session, "+44700900013", mode="nightly")
        old = _now_utc() - dt.timedelta(days=2)
        session.add(db.SentNotification(subscriber_id=1, forecast_date=DAY,
                                        status="pending", attempts=1, claimed_at=old))
        session.commit()
        sender = LogSmsSender()
        notify.retry_pending(session, sender=sender)
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
    def _run_all_serviceable(self, session, stored=True, reason=None):
        # A run record for EVERY serviceable station, so 'missing' is empty.
        from service import config
        for st in config.SERVICEABLE_STATIONS:
            session.add(db.StationRun(station=st, date=DAY, forecast_stored=stored,
                                      skip_reason=reason))
        session.commit()

    def test_healthy_when_all_ran_sent_and_no_repeat_skips(self, session):
        _forecast(session, tmin=-1.0)
        self._run_all_serviceable(session)
        _sub(session, "+44700900020", mode="nightly")
        notify.notify_for_date(session, DAY)
        health = notify.notify_health(session, DAY)
        assert health["healthy"] is True
        assert health["send_status"].get("sent") == 1
        assert health["missing"] == []

    def test_a_station_that_never_ran_is_unhealthy(self, session):
        # Only one of two serviceable stations produced a run record -> the other
        # is 'missing' (never ran), which is a hard alert.
        session.add(db.StationRun(station="uk_waddington", date=DAY,
                                  forecast_stored=True))
        session.commit()
        health = notify.notify_health(session, DAY)
        assert health["healthy"] is False
        assert "uk_cranwell" in health["missing"]

    def test_single_skip_is_a_warning_not_an_alert(self, session):
        # Both ran; one skipped once (no skip the night before) -> healthy (warning).
        self._run_all_serviceable(session)
        run = session.query(db.StationRun).filter_by(station="uk_cranwell").one()
        run.forecast_stored = False
        run.skip_reason = "provider unreachable"
        session.commit()
        health = notify.notify_health(session, DAY)
        assert health["healthy"] is True                 # single skip: warn, not alert
        assert "uk_cranwell" in health["skips"]
        assert health["repeated_skips"] == []

    def test_two_consecutive_skips_is_an_alert(self, session):
        self._run_all_serviceable(session)
        prev = DAY - dt.timedelta(days=1)
        for st in ("uk_waddington", "uk_cranwell"):
            session.add(db.StationRun(station=st, date=prev, forecast_stored=False,
                                      skip_reason="down"))
        run = session.query(db.StationRun).filter_by(station="uk_cranwell", date=DAY).one()
        run.forecast_stored = False
        run.skip_reason = "down again"
        session.commit()
        health = notify.notify_health(session, DAY)
        assert health["healthy"] is False
        assert "uk_cranwell" in health["repeated_skips"]


class TestRound4:
    def test_all_sends_pending_is_unhealthy(self, session):
        # R2: a night where every send is still 'pending' (provider down, one
        # attempt < the 3-attempt fail threshold) means nobody was warned -- health
        # must NOT report green.
        from service import config
        _forecast(session, tmin=-1.0)
        for st in config.SERVICEABLE_STATIONS:
            session.add(db.StationRun(station=st, date=DAY, forecast_stored=True))
        _sub(session, "+44700900030", mode="nightly")

        class DeadSender:
            def send(self, to, message):
                raise RuntimeError("provider down")

        notify.notify_for_date(session, DAY, sender=DeadSender())
        assert session.query(db.SentNotification).one().status == "pending"
        health = notify.notify_health(session, DAY)
        assert health["unwarned"] == 1 and health["healthy"] is False

    def test_drain_confirmations_resends_and_clears_flag(self, session):
        # R1: an unsubscribe confirmation that failed at opt-out time is persisted
        # (confirm_sms_pending) and re-sent by the drain, never silently dropped.
        sub = db.Subscriber(station="uk_waddington", phone="+44700900031",
                            verified=True, active=False, confirm_sms_pending=True)
        session.add(sub)
        session.commit()
        sender = LogSmsSender()
        done = notify.drain_confirmations(session, sender=sender)
        assert done == ["+44700900031"]
        assert len(sender.sent) == 1 and "unsubscribed" in sender.sent[0][1].lower()
        assert session.query(db.Subscriber).one().confirm_sms_pending is False

    def test_health_missing_respects_the_stations_subset(self, session):
        # Smaller: a partial run (one station) shouldn't report the others missing.
        session.add(db.StationRun(station="uk_waddington", date=DAY,
                                  forecast_stored=True))
        session.commit()
        health = notify.notify_health(session, DAY, stations=["uk_waddington"])
        assert health["missing"] == []
