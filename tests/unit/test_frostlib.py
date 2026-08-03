"""Unit tests for frostlib: ISD field decoding, station naming, physics."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from frostlib import isd, paths, physics


class TestScaledNum:
    def test_decodes_and_scales(self):
        assert isd.scaled_num("+0125,1", "+9999", 10.0) == 12.5

    def test_negative_value(self):
        assert isd.scaled_num("-0043,1", "+9999", 10.0) == pytest.approx(-4.3)

    def test_sentinel_is_missing(self):
        assert math.isnan(isd.scaled_num("+9999,1", "+9999", 10.0))

    @pytest.mark.parametrize("flag", sorted(isd.BAD_QUALITY_FLAGS))
    def test_bad_quality_flag_is_missing(self, flag):
        assert math.isnan(isd.scaled_num(f"+0125,{flag}", "+9999", 10.0))

    @pytest.mark.parametrize("flag", ["0", "1", "4", "5", "9", "A", "C"])
    def test_other_flags_are_accepted(self, flag):
        assert isd.scaled_num(f"+0125,{flag}", "+9999", 10.0) == 12.5

    def test_non_numeric_token_is_missing(self):
        assert math.isnan(isd.scaled_num("abcd,1", "+9999", 10.0))

    @pytest.mark.parametrize("raw", ["", None, float("nan"), 42])
    def test_unusable_raw_is_missing(self, raw):
        assert math.isnan(isd.scaled_num(raw, "+9999", 10.0))

    def test_missing_flag_field_still_decodes(self):
        assert isd.scaled_num("+0125", "+9999", 10.0) == 12.5

    def test_custom_indices(self):
        # WND: direction,dir-flag,type,speed,speed-flag
        assert isd.scaled_num("270,1,N,0031,1", "9999", 10.0,
                              value_idx=3, flag_idx=4) == pytest.approx(3.1)


class TestFieldWrappers:
    def test_temp_c(self):
        assert isd.temp_c("-0035,1") == pytest.approx(-3.5)
        assert math.isnan(isd.temp_c("+9999,1"))

    def test_slp_hpa(self):
        assert isd.slp_hpa("10223,1") == pytest.approx(1022.3)
        assert math.isnan(isd.slp_hpa("99999,1"))

    def test_wind_speed_ms(self):
        assert isd.wind_speed_ms("180,1,N,0015,1") == pytest.approx(1.5)
        assert math.isnan(isd.wind_speed_ms("180,1,N,9999,1"))

    def test_wind_speed_ignores_direction_flag(self):
        # A bad DIRECTION flag must not discard an otherwise good speed.
        assert isd.wind_speed_ms("180,2,N,0015,1") == pytest.approx(1.5)

    def test_wind_speed_missing_subfields(self):
        assert math.isnan(isd.wind_speed_ms("180,1,N"))

    def test_has_temp(self):
        assert isd.has_temp("-0035,1")
        assert not isd.has_temp("+9999,1")


class TestCloudOktas:
    @pytest.mark.parametrize("code,expected", [("00", 0.0), ("04", 4.0), ("08", 8.0)])
    def test_valid_coverage_codes(self, code, expected):
        assert isd.cloud_oktas(f"{code},1,99,+00450,1,99,9") == expected

    @pytest.mark.parametrize("code", ["09", "10", "99"])
    def test_out_of_range_codes_are_missing(self, code):
        assert math.isnan(isd.cloud_oktas(f"{code},1"))

    def test_bad_quality_flag_is_missing(self):
        assert math.isnan(isd.cloud_oktas("04,3"))

    def test_non_numeric_is_missing(self):
        assert math.isnan(isd.cloud_oktas("xx,1"))

    @pytest.mark.parametrize("raw", ["", None, float("nan")])
    def test_absent_group_is_missing(self, raw):
        assert math.isnan(isd.cloud_oktas(raw))


class TestPacked:
    def test_splits_on_commas(self):
        assert isd.packed("a,b,c") == ["a", "b", "c"]

    @pytest.mark.parametrize("raw", ["", None, 3.0])
    def test_non_string_or_empty_gives_empty_list(self, raw):
        assert isd.packed(raw) == []


class TestLstOffset:
    def test_poland_is_utc_plus_one(self):
        assert isd.lst_offset("pl_tomaszow") == 1

    def test_uk_is_utc(self):
        assert isd.lst_offset("uk_cranwell") == 0

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="no LST offset"):
            isd.lst_offset("de_berlin")


class TestRawPathNaming:
    def test_write_then_read_round_trips_the_station_name(self):
        path = paths.raw_csv_path("pl_tomaszow", "12105499999", 2021)
        assert path.name == "isd_pl_tomaszow_12105499999_2021.csv"
        assert paths.station_name_from_path(path) == "pl_tomaszow"

    def test_honours_an_alternative_raw_dir(self, tmp_path):
        assert paths.raw_csv_path("pl_x", "1", 2021, tmp_path).parent == tmp_path

    def test_strips_prefix_id_and_year(self):
        assert paths.station_name_from_path(
            Path("data_raw/isd_pl_tomaszow_12105499999_2021.csv")
        ) == "pl_tomaszow"

    def test_only_leading_prefix_is_removed(self):
        # An "isd_" occurring inside the name must survive.
        assert paths.station_name_from_path(
            Path("isd_uk_isd_town_03377099999_2020.csv")
        ) == "uk_isd_town"

    def test_lists_raw_station_years_in_a_stable_order(self, tmp_path):
        for name in ["isd_uk_b_2_2021.csv", "isd_pl_a_1_2021.csv", "other.csv"]:
            (tmp_path / name).write_text("")
        assert [p.name for p in paths.raw_station_years(tmp_path)] == [
            "isd_pl_a_1_2021.csv", "isd_uk_b_2_2021.csv"]


class TestPhysics:
    def test_dewpoint_depression(self):
        assert physics.dewpoint_depression_c(3.0, 0.5) == pytest.approx(2.5)

    def test_radiative_potential_is_clear_fraction_over_wind(self):
        assert physics.radiative_potential(0.0, 1.0) == pytest.approx(0.5)
        assert physics.radiative_potential(8.0, 0.0) == pytest.approx(0.0)

    def test_clear_calm_beats_cloudy_windy(self):
        assert (physics.radiative_potential(0.0, 0.5)
                > physics.radiative_potential(8.0, 6.0))

    @pytest.mark.parametrize("cloud,wind", [(None, None), (float("nan"), float("nan"))])
    def test_missing_inputs_use_the_documented_defaults(self, cloud, wind):
        # (1 - 4/8) / (1 + 3): mid cloud, light wind.
        assert physics.radiative_potential(cloud, wind) == pytest.approx(0.125)

    def test_column_form_matches_the_scalar_form(self):
        import pandas as pd
        cloud = pd.Series([0.0, 4.0, 8.0, None])
        wind = pd.Series([1.0, 3.0, None, 0.0])
        expected = [physics.radiative_potential(c, w) for c, w in zip(cloud, wind)]
        assert physics.radiative_potential_col(cloud, wind).tolist() == pytest.approx(
            expected)
