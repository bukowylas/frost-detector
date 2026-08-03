"""Unit tests for prepare.py's risk-window and observation-lookup helpers.

The ISD field decoders these used to sit beside now live in frostlib.isd; their
tests moved to test_frostlib.py with them.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

import prepare


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
