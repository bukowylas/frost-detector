"""Unit tests for prepare.build_feature_row -- the one shared feature builder.

The training path and the future live path both call this function, so these
tests pin the exact numbers it produces: if the live service ever fed the model
a different vector, the published accuracy would no longer describe it.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

import prepare

from test_prepare_pipeline import good_night, obs_frame

CUTOFF = pd.Timestamp("2021-04-01 18:00")


def evening(rows, **kw):
    """Observations running up to the 18:00 cutoff (no label window at all)."""
    return obs_frame(rows, **kw)


def full_window(temp=5.0, **snapshot):
    """A well-formed 30 h window: 24 h, 3 h and cutoff observations."""
    rows = [
        {"lst": "2021-03-31 18:00", "temp_c": temp + 4.0, "slp_hpa": 1010.0},
        {"lst": "2021-04-01 15:00", "temp_c": temp + 2.0, "slp_hpa": 1017.0},
        {"lst": "2021-04-01 18:00", "temp_c": temp, "slp_hpa": 1020.0,
         "dewpoint_c": 1.0, "wind_ms": 1.0, "cloud_oktas": 2.0},
    ]
    rows[-1].update(snapshot)
    return evening(rows)


class TestWellFormedWindow:
    @pytest.fixture
    def row(self):
        return prepare.build_feature_row(full_window(), CUTOFF)

    def test_returns_the_snapshot_values(self, row):
        assert row["temp_c"] == pytest.approx(5.0)
        assert row["dewpoint_c"] == pytest.approx(1.0)
        assert row["slp_hpa"] == pytest.approx(1020.0)
        assert row["wind_ms"] == pytest.approx(1.0)
        assert row["cloud_oktas"] == pytest.approx(2.0)

    def test_dewpoint_depression_is_derived(self, row):
        assert row["dewpoint_depression_c"] == pytest.approx(4.0)

    def test_radiative_potential_formula(self, row):
        # (1 - 2/8) / (1 + 1) = 0.375
        assert row["radiative_potential"] == pytest.approx(0.375)

    def test_trend_arithmetic_is_now_minus_past(self, row):
        assert row["temp_change_3h"] == pytest.approx(-2.0)
        assert row["temp_change_24h"] == pytest.approx(-4.0)
        assert row["slp_tendency_3h"] == pytest.approx(3.0)

    def test_carries_station_geography(self, row):
        assert (row["lat"], row["lon"], row["elev"]) == (51.7, 19.4, 180.0)

    def test_calendar_features_come_from_the_cutoff(self, row):
        assert row["month"] == 4
        assert row["doy"] == 91

    def test_produces_exactly_the_model_features_and_nothing_label_shaped(self, row):
        import train
        assert set(row) == set(train.FEATURES) | {"month"}
        assert "tmin_overnight_c" not in row

    def test_unsorted_input_still_picks_the_cutoff_observation(self):
        obs = pd.concat([full_window(), evening([
            {"lst": "2021-04-01 17:50", "temp_c": 6.0},
            # A post-cutoff observation must never influence the features.
            {"lst": "2021-04-01 18:30", "temp_c": -40.0},
        ])], ignore_index=True)
        row = prepare.build_feature_row(obs, CUTOFF)
        assert row["temp_c"] == pytest.approx(5.0)

    def test_duplicate_cutoff_timestamps_keep_input_order(self):
        # ISD sometimes reports twice at one timestamp; the sort must be stable,
        # so "the last observation" stays the last one as supplied.
        obs = evening([
            {"lst": "2021-04-01 15:00", "temp_c": 7.0},
            {"lst": "2021-04-01 18:00", "temp_c": 5.0},
            {"lst": "2021-04-01 18:00", "temp_c": 5.5},
        ])
        assert prepare.build_feature_row(obs, CUTOFF)["temp_c"] == pytest.approx(5.5)

    def test_a_wider_frame_gives_the_same_result(self):
        narrow = full_window()
        wide = pd.concat([narrow, evening([
            {"lst": "2021-03-20 18:00", "temp_c": -99.0},
            {"lst": "2021-04-05 18:00", "temp_c": 99.0},
        ])], ignore_index=True)
        assert (prepare.build_feature_row(wide, CUTOFF)
                == prepare.build_feature_row(narrow, CUTOFF))


class TestUnusableWindow:
    def test_no_cutoff_snapshot(self):
        obs = evening([{"lst": "2021-04-01 12:00", "temp_c": 9.0}])  # stale
        assert prepare.build_feature_row(obs, CUTOFF) is None

    def test_snapshot_missing_temperature(self):
        obs = evening([{"lst": "2021-04-01 18:00", "temp_c": math.nan}])
        assert prepare.build_feature_row(obs, CUTOFF) is None

    def test_snapshot_missing_dewpoint(self):
        assert prepare.build_feature_row(
            full_window(dewpoint_c=math.nan), CUTOFF) is None

    def test_only_post_cutoff_observations(self):
        obs = evening([{"lst": "2021-04-01 19:00", "temp_c": 5.0}])
        assert prepare.build_feature_row(obs, CUTOFF) is None

    def test_empty_frame(self):
        obs = full_window().iloc[0:0]
        assert prepare.build_feature_row(obs, CUTOFF) is None


class TestMissingValueDefaults:
    def test_missing_cloud_uses_four_oktas_but_stays_missing(self):
        row = prepare.build_feature_row(
            full_window(cloud_oktas=math.nan, wind_ms=1.0), CUTOFF)
        # (1 - 4/8) / (1 + 1)
        assert row["radiative_potential"] == pytest.approx(0.25)
        assert math.isnan(row["cloud_oktas"])

    def test_missing_wind_uses_three_ms_but_stays_missing(self):
        row = prepare.build_feature_row(
            full_window(cloud_oktas=0.0, wind_ms=math.nan), CUTOFF)
        # (1 - 0/8) / (1 + 3)
        assert row["radiative_potential"] == pytest.approx(0.25)
        assert math.isnan(row["wind_ms"])

    def test_both_missing(self):
        row = prepare.build_feature_row(
            full_window(cloud_oktas=math.nan, wind_ms=math.nan), CUTOFF)
        assert row["radiative_potential"] == pytest.approx(0.125)

    def test_missing_pressure_is_reported_missing_not_imputed(self):
        row = prepare.build_feature_row(full_window(slp_hpa=math.nan), CUTOFF)
        assert math.isnan(row["slp_hpa"])
        assert math.isnan(row["slp_tendency_3h"])


class TestAbsentLookbacks:
    def test_trends_are_nan_without_any_lookback(self):
        obs = evening([{"lst": "2021-04-01 18:00", "temp_c": 5.0}])
        row = prepare.build_feature_row(obs, CUTOFF)
        assert math.isnan(row["temp_change_3h"])
        assert math.isnan(row["temp_change_24h"])
        assert math.isnan(row["slp_tendency_3h"])

    def test_a_present_3h_lookback_does_not_supply_the_24h_trend(self):
        obs = evening([
            {"lst": "2021-04-01 15:00", "temp_c": 7.0},
            {"lst": "2021-04-01 18:00", "temp_c": 5.0},
        ])
        row = prepare.build_feature_row(obs, CUTOFF)
        assert row["temp_change_3h"] == pytest.approx(-2.0)
        assert math.isnan(row["temp_change_24h"])

    def test_lookback_outside_the_feature_window_is_ignored(self):
        # 31 h before the cutoff: beyond FEATURE_WINDOW_HOURS, so no 24 h trend.
        obs = evening([
            {"lst": "2021-03-31 11:00", "temp_c": 9.0},
            {"lst": "2021-04-01 18:00", "temp_c": 5.0},
        ])
        assert math.isnan(prepare.build_feature_row(obs, CUTOFF)["temp_change_24h"])

    def test_pressure_trend_uses_its_own_lookback_row(self):
        # The 15:00 report has no pressure, so the 14:30 one supplies the pressure
        # trend while the temperature trend still comes from 15:00.
        obs = evening([
            {"lst": "2021-04-01 14:30", "temp_c": math.nan, "slp_hpa": 1016.0},
            {"lst": "2021-04-01 15:00", "temp_c": 7.0, "slp_hpa": math.nan},
            {"lst": "2021-04-01 18:00", "temp_c": 5.0, "slp_hpa": 1020.0},
        ])
        row = prepare.build_feature_row(obs, CUTOFF)
        assert row["temp_change_3h"] == pytest.approx(-2.0)
        assert row["slp_tendency_3h"] == pytest.approx(4.0)


class TestAgreesWithBuildNights:
    """The regression guard for the extraction: one code path, one answer."""

    def test_night_row_features_equal_the_builders_output(self):
        obs = good_night()
        nights, _ = prepare.build_nights(obs)
        built = prepare.build_feature_row(obs, CUTOFF)
        row = nights.iloc[0]
        for key, value in built.items():
            assert row[key] == pytest.approx(value, nan_ok=True), key

    def test_the_night_row_adds_only_station_date_and_the_label(self):
        obs = good_night()
        nights, _ = prepare.build_nights(obs)
        extra = set(nights.columns) - set(prepare.build_feature_row(obs, CUTOFF))
        assert extra == {"station", "date", "tmin_overnight_c"}

    def test_agreement_holds_with_missing_cloud_and_wind(self):
        obs = good_night()
        obs.loc[obs["lst"] == CUTOFF, ["cloud_oktas", "wind_ms"]] = [math.nan, math.nan]
        nights, _ = prepare.build_nights(obs)
        built = prepare.build_feature_row(obs, CUTOFF)
        assert nights.iloc[0]["radiative_potential"] == pytest.approx(
            built["radiative_potential"])

    def test_unusable_window_is_the_no_evening_snapshot_rejection(self):
        obs = good_night()
        obs.loc[obs["lst"] == CUTOFF, "dewpoint_c"] = math.nan
        nights, rejected = prepare.build_nights(obs)
        assert prepare.build_feature_row(obs, CUTOFF) is None
        assert nights.empty
        assert rejected["no_evening_snapshot"] >= 1
