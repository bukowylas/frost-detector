"""Unit tests for phone normalisation and the service config invariants."""

import pytest

from service import config
from service.phone import InvalidPhone, normalize_e164


class TestPhoneNormalisation:
    @pytest.mark.parametrize("raw,expected", [
        ("+447700900123", "+447700900123"),
        ("00447700900123", "+447700900123"),
        ("07700900123", "+447700900123"),        # UK national -> +44
        ("+48 601 234 567", "+48601234567"),      # spaces stripped
    ])
    def test_variants_normalise_to_e164(self, raw, expected):
        assert normalize_e164(raw) == expected

    def test_polish_national_uses_pl_region(self):
        assert normalize_e164("0601234567", default_region="PL") == "+48601234567"

    @pytest.mark.parametrize("bad", ["", "hello", "12345", "+++"])
    def test_unparseable_raises(self, bad):
        with pytest.raises(InvalidPhone):
            normalize_e164(bad)


class TestServiceConfig:
    def test_serviceable_stations_are_known_to_the_provider(self):
        from frostlib import live
        for s in config.SERVICEABLE_STATIONS:
            assert s in live.PROVIDER

    def test_service_season_is_a_subset_of_the_model_windows(self):
        for w in config.SERVICE_RISK_WINDOWS:
            assert w in config.MODEL_RISK_WINDOWS

    def test_service_serves_spring_not_autumn(self):
        assert config.service_in_season("2026-04-15")       # spring
        assert not config.service_in_season("2026-10-15")   # autumn (model covers it)
        assert config.model_covers("2026-10-15")

    def test_every_serviceable_station_has_a_label(self):
        for s in config.SERVICEABLE_STATIONS:
            assert config.station_label(s) != s   # a real label, not the key
