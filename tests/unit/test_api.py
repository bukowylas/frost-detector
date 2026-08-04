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
    _, msg = sender.sent[-1]
    return "".join(ch for ch in msg.split("code is")[1][:8] if ch.isdigit())


def _subscribe(client, phone="+447700900001", station="uk_waddington", mode="nightly"):
    return client.post("/api/subscribe",
                       json={"station": station, "phone": phone, "mode": mode})


class TestStations:
    def test_lists_serviceable_stations(self, ctx):
        client, _, _ = ctx
        keys = {s["key"] for s in client.get("/api/stations").json()}
        assert keys == {"uk_waddington", "uk_cranwell"}


class TestSubscribeVerifyFlow:
    def test_subscribe_creates_unverified_and_sends_a_code(self, ctx):
        client, _, sender = ctx
        r = _subscribe(client)
        assert r.status_code == 200 and r.json()["verified"] is False
        assert len(sender.sent) == 1

    def test_verify_with_correct_code_activates(self, ctx):
        client, _, sender = ctx
        _subscribe(client, phone="+447700900002")
        code = _last_code(sender)
        r = client.post("/api/verify", json={
            "phone": "+447700900002", "station": "uk_waddington", "code": code})
        assert r.status_code == 200 and r.json()["verified"] is True

    def test_verify_with_wrong_code_is_rejected(self, ctx):
        client, _, _ = ctx
        _subscribe(client, phone="+447700900003")
        r = client.post("/api/verify", json={
            "phone": "+447700900003", "station": "uk_waddington", "code": "000000"})
        assert r.status_code == 400

    def test_verify_locks_after_too_many_attempts(self, ctx):
        client, _, _ = ctx
        _subscribe(client, phone="+447700900004")
        for _ in range(5):
            client.post("/api/verify", json={"phone": "+447700900004",
                        "station": "uk_waddington", "code": "000000"})
        r = client.post("/api/verify", json={"phone": "+447700900004",
                        "station": "uk_waddington", "code": "000000"})
        assert r.status_code == 429      # locked

    def test_unknown_and_wrong_code_return_the_same_status(self, ctx):
        # No enumeration: an unknown subscription and a wrong code both 400.
        client, _, sender = ctx
        _subscribe(client, phone="+447700900005")
        unknown = client.post("/api/verify", json={
            "phone": "+447700900999", "station": "uk_waddington", "code": "123456"})
        wrong = client.post("/api/verify", json={
            "phone": "+447700900005", "station": "uk_waddington", "code": "000000"})
        assert unknown.status_code == wrong.status_code == 400

    def test_non_digit_code_is_a_validation_error(self, ctx):
        client, _, _ = ctx
        _subscribe(client, phone="+447700900006")
        r = client.post("/api/verify", json={
            "phone": "+447700900006", "station": "uk_waddington", "code": "abcxyz"})
        assert r.status_code == 422

    def test_unserviceable_station_is_rejected(self, ctx):
        client, _, _ = ctx
        r = _subscribe(client, phone="+48700900007", station="pl_lublinek_lodz")
        assert r.status_code == 400

    def test_threshold_out_of_range_is_rejected(self, ctx):
        client, _, _ = ctx
        r = client.post("/api/subscribe", json={
            "station": "uk_waddington", "phone": "+447700900008",
            "mode": "frost", "threshold_c": 100.0})
        assert r.status_code == 422


class TestPhoneNormalisation:
    def test_variants_map_to_one_subscriber(self, ctx):
        client, factory, _ = ctx
        # +44..., 0044..., and 07... national are the same UK phone.
        for p in ("+447700900123", "00447700900123", "07700900123"):
            _subscribe(client, phone=p)
        assert factory().query(db.Subscriber).count() == 1


class TestNoSilentUnverify:
    def test_verified_subscriber_editing_settings_stays_verified(self, ctx):
        client, factory, sender = ctx
        _subscribe(client, phone="+447700900201")
        code = _last_code(sender)
        client.post("/api/verify", json={"phone": "+447700900201",
                    "station": "uk_waddington", "code": code})
        # Re-subscribe with new settings -- must NOT drop them to unverified.
        r = client.post("/api/subscribe", json={"station": "uk_waddington",
                        "phone": "+447700900201", "mode": "frost", "threshold_c": -1.0})
        assert r.json()["verified"] is True
        sub = factory().query(db.Subscriber).one()
        assert sub.verified is True and sub.mode == "frost" and sub.active is True


class TestUnsubscribe:
    def _verified(self, client, sender, phone="+447700900301"):
        _subscribe(client, phone=phone)
        code = _last_code(sender)
        client.post("/api/verify", json={"phone": phone,
                    "station": "uk_waddington", "code": code})
        return phone

    def test_unsubscribe_requires_a_code(self, ctx):
        client, factory, sender = ctx
        phone = self._verified(client, sender)
        # Start unsubscribe -> a code is sent; the subscriber is still active.
        r = client.post("/api/unsubscribe", json={
            "phone": phone, "station": "uk_waddington"})
        assert r.status_code == 200
        assert factory().query(db.Subscriber).one().active is True
        # Confirm with the code -> soft-deleted (inactive), row preserved.
        code = _last_code(sender)
        r = client.post("/api/unsubscribe/confirm", json={
            "phone": phone, "station": "uk_waddington", "code": code})
        assert r.status_code == 200
        sub = factory().query(db.Subscriber).one()
        assert sub.active is False and sub.unsubscribed_at is not None

    def test_unsubscribe_start_does_not_enumerate(self, ctx):
        client, _, _ = ctx
        r = client.post("/api/unsubscribe", json={
            "phone": "+447700900999", "station": "uk_waddington"})
        assert r.status_code == 200     # same response whether or not it exists


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
