"""Unit tests for prepare.py's ISD field decoding and station/window helpers."""

from __future__ import annotations

import math

import pandas as pd
import pytest

import prepare


class TestScaledNum:
    def test_decodes_and_scales(self):
        assert prepare._scaled_num("+0125,1", "+9999", 10.0) == 12.5

    def test_negative_value(self):
        assert prepare._scaled_num("-0043,1", "+9999", 10.0) == pytest.approx(-4.3)

    def test_sentinel_is_missing(self):
        assert math.isnan(prepare._scaled_num("+9999,1", "+9999", 10.0))

    @pytest.mark.parametrize("flag", sorted(prepare._BAD_QUALITY_FLAGS))
    def test_bad_quality_flag_is_missing(self, flag):
        assert math.isnan(prepare._scaled_num(f"+0125,{flag}", "+9999", 10.0))

    @pytest.mark.parametrize("flag", ["0", "1", "4", "5", "9", "A", "C"])
    def test_other_flags_are_accepted(self, flag):
        assert prepare._scaled_num(f"+0125,{flag}", "+9999", 10.0) == 12.5

    def test_non_numeric_token_is_missing(self):
        assert math.isnan(prepare._scaled_num("abcd,1", "+9999", 10.0))

    @pytest.mark.parametrize("raw", ["", None, float("nan"), 42])
    def test_unusable_raw_is_missing(self, raw):
        assert math.isnan(prepare._scaled_num(raw, "+9999", 10.0))

    def test_missing_flag_field_still_decodes(self):
        assert prepare._scaled_num("+0125", "+9999", 10.0) == 12.5

    def test_custom_indices(self):
        # WND: direction,dir-flag,type,speed,speed-flag
        assert prepare._scaled_num("270,1,N,0031,1", "9999", 10.0,
                                   value_idx=3, flag_idx=4) == pytest.approx(3.1)


class TestFieldWrappers:
    def test_temp_c(self):
        assert prepare._temp_c("-0035,1") == pytest.approx(-3.5)
        assert math.isnan(prepare._temp_c("+9999,1"))

    def test_slp_hpa(self):
        assert prepare._slp_hpa("10223,1") == pytest.approx(1022.3)
        assert math.isnan(prepare._slp_hpa("99999,1"))

    def test_wind_speed_ms(self):
        assert prepare._wind_speed_ms("180,1,N,0015,1") == pytest.approx(1.5)
        assert math.isnan(prepare._wind_speed_ms("180,1,N,9999,1"))

    def test_wind_speed_ignores_direction_flag(self):
        # A bad DIRECTION flag must not discard an otherwise good speed.
        assert prepare._wind_speed_ms("180,2,N,0015,1") == pytest.approx(1.5)

    def test_wind_speed_missing_subfields(self):
        assert math.isnan(prepare._wind_speed_ms("180,1,N"))


class TestCloudOktas:
    @pytest.mark.parametrize("code,expected", [("00", 0.0), ("04", 4.0), ("08", 8.0)])
    def test_valid_coverage_codes(self, code, expected):
        assert prepare._cloud_oktas(f"{code},1,99,+00450,1,99,9") == expected

    @pytest.mark.parametrize("code", ["09", "10", "99"])
    def test_out_of_range_codes_are_missing(self, code):
        assert math.isnan(prepare._cloud_oktas(f"{code},1"))

    def test_bad_quality_flag_is_missing(self):
        assert math.isnan(prepare._cloud_oktas("04,3"))

    def test_non_numeric_is_missing(self):
        assert math.isnan(prepare._cloud_oktas("xx,1"))

    @pytest.mark.parametrize("raw", ["", None, float("nan")])
    def test_absent_group_is_missing(self, raw):
        assert math.isnan(prepare._cloud_oktas(raw))


class TestPacked:
    def test_splits_on_commas(self):
        assert prepare._packed("a,b,c") == ["a", "b", "c"]

    @pytest.mark.parametrize("raw", ["", None, 3.0])
    def test_non_string_or_empty_gives_empty_list(self, raw):
        assert prepare._packed(raw) == []


class TestStationName:
    def test_strips_prefix_id_and_year(self):
        from pathlib import Path
        assert prepare._station_name(
            Path("data_raw/isd_pl_tomaszow_12105499999_2021.csv")
        ) == "pl_tomaszow"

    def test_only_leading_prefix_is_removed(self):
        from pathlib import Path
        # An "isd_" occurring inside the name must survive.
        assert prepare._station_name(
            Path("isd_uk_isd_town_03377099999_2020.csv")
        ) == "uk_isd_town"


class TestLstOffset:
    def test_poland_is_utc_plus_one(self):
        assert prepare._lst_offset("pl_tomaszow") == 1

    def test_uk_is_utc(self):
        assert prepare._lst_offset("uk_cranwell") == 0

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="no LST offset"):
            prepare._lst_offset("de_berlin")


class TestInRiskWindow:
    @pytest.mark.parametrize("month,day", [(3, 1), (4, 15), (5, 31), (9, 15),
                                           (10, 20), (11, 15)])
    def test_inside_windows_including_edges(self, month, day):
        assert prepare._in_risk_window(month, day)

    @pytest.mark.parametrize("month,day", [(2, 28), (6, 1), (7, 15), (9, 14),
                                           (11, 16), (12, 25), (1, 10)])
    def test_outside_windows(self, month, day):
        assert not prepare._in_risk_window(month, day)


class TestNearestAtOrBefore:
    @staticmethod
    def frame(pairs):
        return pd.DataFrame({
            "lst": pd.to_datetime([t for t, _ in pairs]),
            "temp_c": pd.Series([v for _, v in pairs], dtype=float),
        })

    def test_picks_last_observation_at_or_before_target(self):
        df = self.frame([("2021-04-01 16:50", 8.0), ("2021-04-01 17:50", 6.0),
                         ("2021-04-01 18:20", 5.0)])
        row = prepare._nearest_at_or_before(df, pd.Timestamp("2021-04-01 18:00"))
        assert row["temp_c"] == 6.0

    def test_exact_match_is_included(self):
        df = self.frame([("2021-04-01 18:00", 6.0)])
        row = prepare._nearest_at_or_before(df, pd.Timestamp("2021-04-01 18:00"))
        assert row["temp_c"] == 6.0

    def test_none_when_nothing_before_target(self):
        df = self.frame([("2021-04-01 19:00", 6.0)])
        assert prepare._nearest_at_or_before(
            df, pd.Timestamp("2021-04-01 18:00")) is None

    def test_none_when_only_stale_observation(self):
        df = self.frame([("2021-04-01 16:00", 9.0)])  # 120 min before, tol 90
        assert prepare._nearest_at_or_before(
            df, pd.Timestamp("2021-04-01 18:00")) is None

    def test_tolerance_is_configurable(self):
        df = self.frame([("2021-04-01 16:00", 9.0)])
        row = prepare._nearest_at_or_before(
            df, pd.Timestamp("2021-04-01 18:00"), tol_minutes=180)
        assert row["temp_c"] == 9.0

    def test_require_col_skips_valueless_later_report(self):
        df = self.frame([("2021-04-01 17:30", 7.0), ("2021-04-01 17:55", math.nan)])
        row = prepare._nearest_at_or_before(
            df, pd.Timestamp("2021-04-01 18:00"), require_col="temp_c")
        assert row["temp_c"] == 7.0

    def test_require_col_tolerance_measured_from_the_usable_row(self):
        # A valueless report inside tolerance cannot rescue a stale good reading.
        df = self.frame([("2021-04-01 16:00", 7.0), ("2021-04-01 17:55", math.nan)])
        assert prepare._nearest_at_or_before(
            df, pd.Timestamp("2021-04-01 18:00"), require_col="temp_c") is None

    def test_empty_frame(self):
        assert prepare._nearest_at_or_before(
            self.frame([]), pd.Timestamp("2021-04-01 18:00")) is None
