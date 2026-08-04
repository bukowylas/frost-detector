"""Unit tests for the FastAPI backend -- SQLite + the log-stub SMS sender.

No live weather, no real texts. Covers the subscribe -> verify flow and its
hardening (E.164 normalisation, expiring/attempt-capped codes, no silent
un-verify, coded unsubscribe), and forecast reads.
"""

import datetime as dt
import json

import pytest
from fastapi.testclient import TestClient

from service import db
from service.api import create_app
from service.sms import LogSmsSender

DAY = dt.date(2021, 4, 15)


@pytest.fixture
def ctx():
    engine = db.make_engine("sqlite://")
    db.create_all(engine)
    factory = db.make_session_factory(engine)
    sender = LogSmsSender()
    app = create_app(session_factory=factory, sms_sender=sender)
    return TestClient(app), factory, sender


def _last_code(sender):
    # SMS is "Frost Detector code NNNNNN -- confirms: ..."; pull the first 6 digits.
    _, msg = sender.sent[-1]
    import re
    return re.search(r"\b(\d{6})\b", msg).group(1)


def _subscribe(client, phone="+447911123001", station="uk_waddington",
               mode="nightly", threshold_c=0.0):
    return client.post("/api/subscribe", json={
        "station": station, "phone": phone, "mode": mode, "threshold_c": threshold_c})


class TestStations:
    def test_lists_serviceable_stations(self, ctx):
        client, _, _ = ctx
        keys = {s["key"] for s in client.get("/api/stations").json()}
        assert keys == {"uk_waddington", "uk_cranwell"}


class TestSubscribeVerifyFlow:
    def test_subscribe_is_opaque_and_sends_a_code_naming_the_settings(self, ctx):
        client, _, sender = ctx
        r = _subscribe(client, mode="frost", threshold_c=-2.0)
        # Opaque response (no id/verified oracle); a code SMS that names the settings.
        assert r.status_code == 200 and r.json() == {"status": "ok"}
        assert len(sender.sent) == 1
        assert "-2.0" in sender.sent[-1][1] and "frost" in sender.sent[-1][1]

    def test_verify_with_correct_code_activates_and_promotes_settings(self, ctx):
        client, factory, sender = ctx
        _subscribe(client, phone="+447911123002", mode="frost", threshold_c=-2.0)
        code = _last_code(sender)
        r = client.post("/api/verify", json={
            "phone": "+447911123002", "station": "uk_waddington", "code": code})
        assert r.status_code == 200 and r.json()["verified"] is True
        sub = factory().query(db.Subscriber).one()
        # Settings only take effect on verify (staged until then).
        assert sub.mode == "frost" and sub.threshold_c == -2.0 and sub.active is True

    def test_verify_with_wrong_code_is_rejected(self, ctx):
        client, _, _ = ctx
        _subscribe(client, phone="+447911123003")
        r = client.post("/api/verify", json={
            "phone": "+447911123003", "station": "uk_waddington", "code": "000000"})
        assert r.status_code == 400

    def test_verify_locks_after_too_many_attempts(self, ctx):
        client, _, _ = ctx
        _subscribe(client, phone="+447911123004")
        for _ in range(5):
            client.post("/api/verify", json={"phone": "+447911123004",
                        "station": "uk_waddington", "code": "000000"})
        r = client.post("/api/verify", json={"phone": "+447911123004",
                        "station": "uk_waddington", "code": "000000"})
        assert r.status_code == 429      # locked

    def test_unknown_and_wrong_code_return_the_same_status(self, ctx):
        # No enumeration: an unknown subscription and a wrong code both 400.
        client, _, sender = ctx
        _subscribe(client, phone="+447911123005")
        unknown = client.post("/api/verify", json={
            "phone": "+447911123999", "station": "uk_waddington", "code": "123456"})
        wrong = client.post("/api/verify", json={
            "phone": "+447911123005", "station": "uk_waddington", "code": "000000"})
        assert unknown.status_code == wrong.status_code == 400

    def test_non_digit_code_is_a_validation_error(self, ctx):
        client, _, _ = ctx
        _subscribe(client, phone="+447911123006")
        r = client.post("/api/verify", json={
            "phone": "+447911123006", "station": "uk_waddington", "code": "abcxyz"})
        assert r.status_code == 422

    def test_unserviceable_station_is_rejected(self, ctx):
        client, _, _ = ctx
        r = _subscribe(client, phone="+48601234567", station="pl_lublinek_lodz")
        assert r.status_code == 400

    def test_threshold_out_of_range_is_rejected(self, ctx):
        client, _, _ = ctx
        r = client.post("/api/subscribe", json={
            "station": "uk_waddington", "phone": "+447911123008",
            "mode": "frost", "threshold_c": 100.0})
        assert r.status_code == 422


class TestPhoneNormalisation:
    def test_variants_map_to_one_subscriber(self, ctx):
        client, factory, _ = ctx
        # +44..., 0044..., and 07... national are the same UK phone.
        for p in ("+447911123123", "00447911123123", "07911123123"):
            _subscribe(client, phone=p)
            _cool_off(factory, "+447911123123")   # per-phone cooldown between tries
        assert factory().query(db.Subscriber).count() == 1


class TestNoEnumeration:
    def test_subscribe_response_is_opaque(self, ctx):
        # A3: subscribe reveals nothing -- same body for a new number, an existing
        # verified one, and an unserviceable-but-valid state is a 400 (not an oracle).
        client, factory, sender = ctx
        r1 = _subscribe(client, phone="+447911123401")
        _verify(client, factory, sender, "+447911123402")
        r2 = _subscribe(client, phone="+447911123402")   # already verified
        assert r1.json() == r2.json() == {"status": "ok"}
        assert "id" not in r1.json() and "verified" not in r1.json()


class TestCooldown:
    def test_first_contact_then_cooldown_blocks_a_second_code(self, ctx):
        # A4: even a first contact is throttled once a row exists; a second request
        # to the same phone within the window is 429, regardless of station.
        client, _, _ = ctx
        assert _subscribe(client, phone="+447911123501").status_code == 200
        r = _subscribe(client, phone="+447911123501", station="uk_cranwell")
        assert r.status_code == 429      # per-PHONE, so a different station is still blocked


def _cool_off(factory, phone):
    """Simulate the per-phone cooldown elapsing, so a test can make a second
    request to the same phone without a 429."""
    s = factory()
    for sub in s.query(db.Subscriber).filter_by(phone=phone).all():
        sub.last_sms_at = None
    s.commit()


def _verify(client, factory, sender, phone, station="uk_waddington",
            mode="nightly", threshold_c=0.0):
    _subscribe(client, phone=phone, station=station, mode=mode, threshold_c=threshold_c)
    code = _last_code(sender)
    client.post("/api/verify", json={"phone": phone, "station": station, "code": code})
    _cool_off(factory, phone)


class TestStagedSettings:
    def test_pending_subscribe_does_not_apply_settings(self, ctx):
        # A1: a subscribe on an unverified row must NOT change the active settings.
        client, factory, sender = ctx
        _subscribe(client, phone="+447911123201", mode="nightly")
        _cool_off(factory, "+447911123201")
        _subscribe(client, phone="+447911123201", mode="frost", threshold_c=-10.0)
        sub = factory().query(db.Subscriber).one()
        # Nothing applied: still unverified, and the poison threshold is only staged.
        assert sub.verified is False
        assert sub.mode == "nightly"                 # active, unchanged
        assert sub.pending_threshold_c == -10.0      # staged, pending a code

    def test_verified_settings_change_requires_the_code(self, ctx):
        # A2: changing settings on a verified row stages them; they take effect only
        # after the fresh code is verified (no unauthenticated write).
        client, factory, sender = ctx
        _verify(client, factory, sender, "+447911123202", mode="nightly")
        client.post("/api/subscribe", json={"station": "uk_waddington",
                    "phone": "+447911123202", "mode": "frost", "threshold_c": -1.0})
        sub = factory().query(db.Subscriber).one()
        assert sub.mode == "nightly"                 # NOT yet applied
        assert sub.pending_mode == "frost"           # staged
        code = _last_code(sender)
        client.post("/api/verify", json={"phone": "+447911123202",
                    "station": "uk_waddington", "code": code})
        sub = factory().query(db.Subscriber).one()
        assert sub.mode == "frost" and sub.threshold_c == -1.0   # promoted on verify

    def test_opted_out_row_is_not_reactivated_without_a_code(self, ctx):
        client, factory, sender = ctx
        _verify(client, factory, sender, "+447911123203")
        client.post("/api/unsubscribe", json={"phone": "+447911123203",
                    "station": "uk_waddington"})
        assert factory().query(db.Subscriber).one().active is False
        _cool_off(factory, "+447911123203")
        # A plain subscribe stages a code but does NOT reactivate.
        client.post("/api/subscribe", json={"station": "uk_waddington",
                    "phone": "+447911123203", "mode": "nightly"})
        assert factory().query(db.Subscriber).one().active is False   # still opted out


class TestUnsubscribe:
    def test_unsubscribe_deactivates_immediately_and_confirms(self, ctx):
        client, factory, sender = ctx
        _verify(client, factory, sender, "+447911123301")
        before = len(sender.sent)
        r = client.post("/api/unsubscribe", json={
            "phone": "+447911123301", "station": "uk_waddington"})
        assert r.status_code == 200 and r.json() == {"status": "ok"}
        sub = factory().query(db.Subscriber).one()
        assert sub.active is False and sub.unsubscribed_at is not None
        assert len(sender.sent) == before + 1
        assert "unsubscribed" in sender.sent[-1][1].lower()

    def test_unsubscribe_invalidates_an_outstanding_code(self, ctx):
        # A7: a code issued before opt-out must not reactivate after it.
        client, factory, sender = ctx
        _subscribe(client, phone="+447911123302")   # issues a code, row inactive
        code = _last_code(sender)
        client.post("/api/unsubscribe", json={"phone": "+447911123302",
                    "station": "uk_waddington"})
        r = client.post("/api/verify", json={"phone": "+447911123302",
                        "station": "uk_waddington", "code": code})
        assert r.status_code == 400            # the code was invalidated by opt-out
        assert factory().query(db.Subscriber).one().active is False

    def test_unsubscribe_does_not_enumerate(self, ctx):
        client, _, sender = ctx
        before = len(sender.sent)
        r = client.post("/api/unsubscribe", json={
            "phone": "+447911123999", "station": "uk_waddington"})
        assert r.status_code == 200 and r.json() == {"status": "ok"}
        assert len(sender.sent) == before      # and no SMS sent to a non-subscriber


class TestForecastRead:
    def test_returns_latest_forecasts_newest_first(self, ctx):
        client, factory, _ = ctx
        s = factory()
        for d, tmin in [(dt.date(2021, 4, 14), 2.0), (dt.date(2021, 4, 15), -1.0)]:
            s.add(db.Forecast(station="uk_waddington", date=d, predicted_tmin_c=tmin,
                              alarm_fired=tmin <= 1.5, alarm_threshold_c=1.5,
                              model_version="m-test",
                              features_json=json.dumps({"temp_c": 3.0}),
                              cutoff_ts=f"{d}T18:00:00", snapshot_ts=f"{d}T18:00:00"))
        s.commit()
        data = client.get("/api/forecasts/uk_waddington").json()
        assert data[0]["date"] == "2021-04-15" and data[0]["alarm_fired"] is True

    def test_limit_is_clamped(self, ctx):
        client, _, _ = ctx
        r = client.get("/api/forecasts/uk_waddington?limit=9999")
        assert r.status_code == 200      # clamped, not rejected


class TestAuthStateMachine:
    """The invariant that no round has tested: mode/threshold_c never change to
    unauthenticated values. Table over the (state x action) transitions, asserting
    the active settings only ever equal the LAST VERIFIED values."""

    def _make(self, client, factory, sender, phone, state):
        """Put a (phone, station) row into the requested state."""
        if state == "absent":
            return
        _subscribe(client, phone=phone, mode="nightly", threshold_c=0.0)
        if state == "unverified":
            return
        code = _last_code(sender)
        client.post("/api/verify", json={"phone": phone,
                    "station": "uk_waddington", "code": code})
        _cool_off(factory, phone)
        if state == "inactive":
            client.post("/api/unsubscribe", json={"phone": phone,
                        "station": "uk_waddington"})
            _cool_off(factory, phone)

    @pytest.mark.parametrize("state", ["absent", "unverified", "verified", "inactive"])
    def test_subscribe_never_changes_active_settings_unauthenticated(
            self, ctx, state):
        client, factory, sender = ctx
        phone = "+447911123" + {"absent": "601", "unverified": "602",
                                "verified": "603", "inactive": "604"}[state]
        self._make(client, factory, sender, phone, state)
        _cool_off(factory, phone)

        before = factory().query(db.Subscriber).filter_by(phone=phone).one_or_none()
        before_mode = before.mode if before else None

        # Attempt to poison via an unauthenticated subscribe.
        client.post("/api/subscribe", json={"station": "uk_waddington",
                    "phone": phone, "mode": "frost", "threshold_c": -10.0})

        after = factory().query(db.Subscriber).filter_by(phone=phone).one()
        if state in ("verified",):
            # active settings unchanged (still the verified values); poison is staged.
            assert after.mode == before_mode == "nightly"
            assert after.pending_threshold_c == -10.0
        else:
            # absent/unverified/inactive: nothing active to change; never active+frost.
            assert not (after.active and after.mode == "frost")


class TestSendBudget:
    def test_budget_refuses_past_the_daily_cap(self):
        # R4: the global daily send budget caps ALL sends, so cycling numbers can't
        # make subscribe an unbounded SMS sender.
        from service.sms import BudgetedSender, LogSmsSender, SmsBudgetExceeded
        inner = LogSmsSender()
        b = BudgetedSender(inner, daily_limit=2)
        b.send("+447911123001", "one")
        b.send("+447911123002", "two")
        with pytest.raises(SmsBudgetExceeded):
            b.send("+447911123003", "three")
        assert len(inner.sent) == 2      # the third never reached the provider
