"""Offline unit tests for frostlib.live -- the SYNOP decoder and config.

The decoder is the most bug-prone code in the live path, so it is exercised here
over hand-built FM-12 reports (no network), covering the cases a two-date network
parity run does not reach: sub-1000 hPa pressure, the negative sign digit (frost),
a frost-boundary 0.0 C and a calm 0.0 wind, and the mis-decode traps the
plausibility guard must catch.

Every fixture is built on the verified real-report skeleton
    AAXX ddhhiw IIiii iRiXhVV Nddff 1sTTT 2sTdTdTd (3PPPP) 4PPPP ... <section>
and the expected values are what the decoder actually produces for it.
"""

import math

import pandas as pd
import pytest

import prepare
from frostlib import live, paths


class TestDecodeSynop:
    # Real Waddington report (verified against nights.csv): temp 9.0, dew 4.7,
    # SLP 1003.2, 12 kt wind (iw=4 knots). The 333 section carries max/min-temp
    # 1.../2... groups that must NOT overwrite the section-1 instantaneous values.
    REAL = ("AAXX 10184 03377 07689 12712 10090 20047 39949 40032 "
            "51013 60042 333 10141 20086==")

    def test_core_fields_match_real_report(self):
        d = live.decode_synop(self.REAL)
        assert d["temp_c"] == pytest.approx(9.0)
        assert d["dewpoint_c"] == pytest.approx(4.7)
        assert d["slp_hpa"] == pytest.approx(1003.2)
        assert d["wind_ms"] == pytest.approx(12 * live.KNOTS_TO_MS, abs=0.06)

    def test_ignores_section_333_groups(self):
        # The 10141 / 20086 after 333 (max/min temps) must not become temp/dew.
        d = live.decode_synop(self.REAL)
        assert d["temp_c"] == pytest.approx(9.0)   # not 14.1
        assert d["dewpoint_c"] == pytest.approx(4.7)  # not 8.6

    def test_negative_temperature_frost_case(self):
        # The whole point of the product: 11002 -> -0.2 C, 21015 -> -1.5 C.
        r = "AAXX 15184 03377 07689 00000 11002 21015 30231 40231 333=="
        d = live.decode_synop(r)
        assert d["temp_c"] == pytest.approx(-0.2)
        assert d["dewpoint_c"] == pytest.approx(-1.5)

    def test_zero_temperature_and_calm_wind_survive(self):
        # A frost-boundary 0.0 C and a calm 0.0 m/s wind are real, peak-frost
        # signals -- they must not be turned into NaN by a truthiness test.
        r = "AAXX 10184 03377 07689 00000 10000 21005 30021 40137 333=="
        d = live.decode_synop(r)
        assert d["temp_c"] == pytest.approx(0.0)
        assert d["wind_ms"] == pytest.approx(0.0)

    def test_sub_1000_hpa_pressure(self):
        # 49765 -> 976.5 hPa (a cyclonic night), NOT 1876.5.
        r = "AAXX 10184 03377 07689 00000 10035 20021 30111 49765 333=="
        assert live.decode_synop(r)["slp_hpa"] == pytest.approx(976.5)

    def test_over_1000_hpa_pressure(self):
        r = "AAXX 10184 03377 07689 00000 10098 20058 30021 40137 333=="
        assert live.decode_synop(r)["slp_hpa"] == pytest.approx(1013.7)

    def test_geopotential_4a3hhh_not_read_as_slp(self):
        # 48520 (850 hPa geopotential, leading digit 8) must NOT become 1752.0.
        r = "AAXX 10184 03377 07689 00000 10098 20058 30021 48520 333=="
        assert math.isnan(live.decode_synop(r)["slp_hpa"])

    def test_relative_humidity_not_read_as_dewpoint(self):
        # 29085 is RH 85% (sign digit 9), not a dew point of 8.5.
        r = "AAXX 10184 03377 07689 00000 10032 29085 30021 40231 333=="
        d = live.decode_synop(r)
        assert math.isnan(d["dewpoint_c"])
        assert d["temp_c"] == pytest.approx(3.2)

    def test_wind_unit_ms_when_iw_is_1(self):
        # ddhhiw ends in 1 -> wind already m/s; ff=06 -> 6.0 m/s, no knots factor.
        r = "AAXX 10181 12105 07689 12706 10098 20058 30021 40137 333=="
        assert live.decode_synop(r)["wind_ms"] == pytest.approx(6.0)

    def test_wind_rounded_to_tenth(self):
        # 12 kt * 0.514444 = 6.173 -> rounded to 6.2 to match ISD's stored tenths.
        assert live.decode_synop(self.REAL)["wind_ms"] == pytest.approx(6.2)

    def test_section_444_boundary(self):
        # A 1... group after 444 must not overwrite the section-1 temperature.
        r = "AAXX 10184 03377 07689 00000 10098 20058 40137 444 10250=="
        assert live.decode_synop(r)["temp_c"] == pytest.approx(9.8)

    def test_dewpoint_above_temp_is_dropped(self):
        # temp 3.2, dew 8.5 is physically impossible -> both NaN.
        r = "AAXX 10184 03377 07689 00000 10032 20085 30021 40231 333=="
        d = live.decode_synop(r)
        assert math.isnan(d["temp_c"]) and math.isnan(d["dewpoint_c"])

    def test_nil_or_headerless_report_is_all_nan(self):
        d = live.decode_synop("NIL")
        assert all(math.isnan(v) for v in d.values())

    def test_anchors_on_aaxx_despite_bulletin_prefix(self):
        r = ("SIXX01 EGRR 101800 AAXX 10184 03377 07689 00000 "
             "10098 20058 40137 333==")
        assert live.decode_synop(r)["temp_c"] == pytest.approx(9.8)


class TestConfig:
    def test_serviceable_and_unserviced_cover_the_training_stations(self):
        # PROVIDER u UNSERVICED must equal exactly the training station set -- a
        # typo or a stray ninth station is caught here, offline.
        nights = pd.read_csv(paths.NIGHTS_CSV)
        training = set(nights["station"].unique())
        configured = set(live.PROVIDER) | set(live.UNSERVICED_STATIONS)
        assert configured == training, (
            f"config {configured} != training stations {training}")

    def test_station_meta_matches_training_geography(self):
        # Hand-copied lat/lon/elev must equal the training values (a silent drift
        # channel otherwise).
        nights = pd.read_csv(paths.NIGHTS_CSV)
        for st, meta in live.STATION_META.items():
            row = nights[nights.station == st].iloc[0]
            assert meta["lat"] == pytest.approx(float(row["lat"]), abs=1e-4)
            assert meta["lon"] == pytest.approx(float(row["lon"]), abs=1e-4)
            assert meta["elev"] == pytest.approx(float(row["elev"]), abs=1e-2)

    def test_station_meta_covers_every_serviceable_station(self):
        # Every station the service will fetch must have geography to attach.
        assert set(live.STATION_META) == set(live.PROVIDER)

    def test_fetch_window_rejects_unserviced_station(self):
        with pytest.raises(ValueError, match="no live provider"):
            live.fetch_window("pl_krzesiny_poznan", pd.Timestamp("2023-04-10T18"))
