"""Unit tests for phone normalisation and the service config invariants."""

import pytest

from service import config
from service.phone import InvalidPhone, normalize_e164, region_for_station


class TestPhoneNormalisation:
    # Real, valid UK mobile format (07911 xxxxxx); the phonenumbers library
    # rejects Ofcom's fictional 07700 900xxx range as invalid, which is the point.
    @pytest.mark.parametrize("raw,expected", [
        ("+447911123456", "+447911123456"),
        ("00447911123456", "+447911123456"),
        ("07911123456", "+447911123456"),            # UK national -> +44
        ("+44 (0)7911 123456", "+447911123456"),      # the (0) business-card format
    ])
    def test_variants_normalise_to_e164(self, raw, expected):
        assert normalize_e164(raw, "GB") == expected

    def test_polish_national_uses_pl_region(self):
        assert normalize_e164("601234567", region="PL") == "+48601234567"

    @pytest.mark.parametrize("raw", [
        "02079460958",     # London landline
        "09001234567",     # premium rate (billable to us)
        "03001234567",     # UAN (03xx) -- not SMS-capable
    ])
    def test_non_mobile_numbers_are_rejected(self, raw):
        # Only SMS-capable (mobile) numbers pass -- texting these fails silently or
        # bills us, so they're caught at signup, not at 2am on a frost night.
        with pytest.raises(InvalidPhone):
            normalize_e164(raw, "GB")

    @pytest.mark.parametrize("bad", ["", "hello", "12345", "+++"])
    def test_unparseable_or_invalid_raises(self, bad):
        with pytest.raises(InvalidPhone):
            normalize_e164(bad, "GB")

    def test_region_derived_from_station_key(self):
        assert region_for_station("uk_waddington") == "GB"
        assert region_for_station("pl_lublinek_lodz") == "PL"

    def test_unknown_station_prefix_fails_closed(self):
        # P2: an unconfigured region raises rather than silently defaulting to GB
        # (which could misroute a foreign number to the wrong country).
        with pytest.raises(InvalidPhone):
            region_for_station("no_oslo")


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
