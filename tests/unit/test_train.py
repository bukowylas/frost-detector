"""Unit tests for train.py's baselines, metrics and fold plumbing."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

import train


def nights(records):
    return pd.DataFrame(records)


def synthetic_nights(n_per_station=40, stations=("pl_a", "pl_b", "uk_c"),
                     years=(2020, 2021)):
    """A small deterministic dataset with the columns train.py needs."""
    rng = np.random.default_rng(train.RANDOM_STATE)
    rows = []
    for si, station in enumerate(stations):
        for year in years:
            for i in range(n_per_station):
                temp = 2.0 + si + rng.normal(0, 2)
                dew = temp - abs(rng.normal(2, 1))
                rows.append({
                    "station": station,
                    "date": f"{year}-04-{1 + i % 28:02d}",
                    "month": 4,
                    "doy": 91 + i % 28,
                    "lat": 51.0 + si, "lon": 19.0 + si, "elev": 100.0 + si,
                    "temp_c": temp,
                    "dewpoint_c": dew,
                    "dewpoint_depression_c": temp - dew,
                    "slp_hpa": 1015.0 + rng.normal(0, 5),
                    "wind_ms": abs(rng.normal(2, 1)),
                    "cloud_oktas": float(rng.integers(0, 9)),
                    "radiative_potential": rng.random() * 0.5,
                    "temp_change_3h": rng.normal(0, 1),
                    "temp_change_24h": rng.normal(0, 2),
                    "slp_tendency_3h": rng.normal(0, 1),
                    "tmin_overnight_c": 0.6 * temp + 0.3 * dew - 2.0 + rng.normal(0, 1),
                })
    df = nights(rows)
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["country"] = df["station"].str.slice(0, 2)
    return df


class TestRmse:
    def test_zero_when_exact(self):
        assert train.rmse([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_known_value(self):
        assert train.rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(math.sqrt(12.5))

    def test_accepts_pandas_and_lists(self):
        assert train.rmse(pd.Series([1.0, 3.0]), [2.0, 2.0]) == pytest.approx(1.0)

    def test_penalises_large_errors_more_than_mae(self):
        y_true = [0.0, 0.0, 0.0, 0.0]
        y_pred = [0.0, 0.0, 0.0, 4.0]
        assert train.rmse(y_true, y_pred) > np.mean(np.abs(np.array(y_pred)))


class TestCleanNan:
    def test_replaces_nan_with_none(self):
        assert train._clean_nan(float("nan")) is None

    def test_keeps_finite_floats_and_other_scalars(self):
        assert train._clean_nan(1.5) == 1.5
        assert train._clean_nan("a") == "a"
        assert train._clean_nan(None) is None
        assert train._clean_nan(3) == 3

    def test_recurses_into_dicts_and_lists(self):
        out = train._clean_nan({"a": [1.0, float("nan")], "b": {"c": float("nan")}})
        assert out == {"a": [1.0, None], "b": {"c": None}}

    def test_infinities_are_left_alone(self):
        assert train._clean_nan(float("inf")) == float("inf")


class TestDumpJson:
    def test_produces_valid_json_without_nan_tokens(self):
        text = train._dump_json({"mae": float("nan"), "n": 3})
        assert "NaN" not in text
        assert json.loads(text) == {"mae": None, "n": 3}

    def test_raises_on_values_clean_nan_cannot_reach(self):
        with pytest.raises(ValueError):
            train._dump_json({"x": float("inf")})


class TestClimatologyPredict:
    def test_uses_per_station_month_mean(self):
        train_df = nights([
            {"station": "pl_a", "month": 4, "tmin_overnight_c": 1.0},
            {"station": "pl_a", "month": 4, "tmin_overnight_c": 3.0},
            {"station": "pl_a", "month": 5, "tmin_overnight_c": 9.0},
        ])
        test_df = nights([{"station": "pl_a", "month": 4, "tmin_overnight_c": 0.0}])
        assert train.climatology_predict(train_df, test_df) == pytest.approx([2.0])

    def test_falls_back_to_station_mean_for_an_unseen_month(self):
        train_df = nights([
            {"station": "pl_a", "month": 4, "tmin_overnight_c": 1.0},
            {"station": "pl_a", "month": 4, "tmin_overnight_c": 3.0},
        ])
        test_df = nights([{"station": "pl_a", "month": 9, "tmin_overnight_c": 0.0}])
        assert train.climatology_predict(train_df, test_df) == pytest.approx([2.0])

    def test_falls_back_to_pooled_month_mean_for_an_unseen_station(self):
        train_df = nights([
            {"station": "pl_a", "month": 4, "tmin_overnight_c": 1.0},
            {"station": "pl_b", "month": 4, "tmin_overnight_c": 5.0},
            {"station": "pl_b", "month": 9, "tmin_overnight_c": 20.0},
        ])
        test_df = nights([{"station": "uk_c", "month": 4, "tmin_overnight_c": 0.0}])
        # The pooled April mean, not the grand mean over all months.
        assert train.climatology_predict(train_df, test_df) == pytest.approx([3.0])

    def test_falls_back_to_global_mean_for_unseen_station_and_month(self):
        train_df = nights([
            {"station": "pl_a", "month": 4, "tmin_overnight_c": 1.0},
            {"station": "pl_b", "month": 4, "tmin_overnight_c": 5.0},
        ])
        test_df = nights([{"station": "uk_c", "month": 11, "tmin_overnight_c": 0.0}])
        assert train.climatology_predict(train_df, test_df) == pytest.approx([3.0])

    def test_returns_one_float_prediction_per_test_row(self):
        train_df = nights([{"station": "pl_a", "month": 4, "tmin_overnight_c": 2.0}])
        test_df = nights([{"station": "pl_a", "month": 4, "tmin_overnight_c": 0.0}] * 4)
        preds = train.climatology_predict(train_df, test_df)
        assert preds.shape == (4,)
        assert preds.dtype == float


def fao_frame(station, n, slope=1.0, intercept=-2.0):
    rows = []
    for i in range(n):
        temp = 1.0 + 0.5 * i
        dew = temp - 2.0
        rows.append({"station": station, "temp_c": temp, "dewpoint_c": dew,
                     "tmin_overnight_c": slope * temp + intercept})
    return nights(rows)


class TestFaoPredict:
    def test_recovers_an_exact_linear_relationship(self):
        train_df = fao_frame("pl_a", 40)
        test_df = nights([{"station": "pl_a", "temp_c": 10.0, "dewpoint_c": 8.0,
                           "tmin_overnight_c": 0.0}])
        assert train.fao_predict(train_df, test_df) == pytest.approx([8.0], abs=1e-6)

    def test_per_station_fit_is_used_when_the_station_has_enough_rows(self):
        train_df = pd.concat([fao_frame("pl_a", 40, intercept=-2.0),
                              fao_frame("pl_b", 40, intercept=+10.0)],
                             ignore_index=True)
        test_df = nights([
            {"station": "pl_a", "temp_c": 5.0, "dewpoint_c": 3.0, "tmin_overnight_c": 0},
            {"station": "pl_b", "temp_c": 5.0, "dewpoint_c": 3.0, "tmin_overnight_c": 0},
        ])
        preds = train.fao_predict(train_df, test_df)
        assert preds[0] == pytest.approx(3.0, abs=1e-6)
        assert preds[1] == pytest.approx(15.0, abs=1e-6)

    def test_station_with_fewer_than_30_rows_uses_the_pooled_fit(self):
        # pl_b has only 5 rows, so its own (offset) relationship must be ignored.
        train_df = pd.concat([fao_frame("pl_a", 40, intercept=-2.0),
                              fao_frame("pl_b", 5, intercept=+40.0)],
                             ignore_index=True)
        test_df = nights([{"station": "pl_b", "temp_c": 5.0, "dewpoint_c": 3.0,
                           "tmin_overnight_c": 0.0}])
        pred = train.fao_predict(train_df, test_df)[0]
        own_fit = 5.0 + 40.0  # what pl_b's own relationship would give
        assert abs(pred - own_fit) > 10.0

    def test_unseen_station_uses_the_pooled_fit(self):
        train_df = fao_frame("pl_a", 40)
        test_df = nights([{"station": "uk_new", "temp_c": 10.0, "dewpoint_c": 8.0,
                           "tmin_overnight_c": 0.0}])
        assert train.fao_predict(train_df, test_df) == pytest.approx([8.0], abs=1e-6)

    def test_returns_one_prediction_per_test_row(self):
        train_df = fao_frame("pl_a", 40)
        test_df = nights([{"station": "pl_a", "temp_c": 3.0, "dewpoint_c": 1.0,
                           "tmin_overnight_c": 0.0}] * 3)
        assert train.fao_predict(train_df, test_df).shape == (3,)


class TestAlarmSweep:
    def test_perfect_predictions_give_perfect_recall_and_precision(self):
        y_true = np.array([-2.0, -0.5, 5.0, 8.0])
        out = train._alarm_sweep(y_true, y_true.copy())
        assert out[0.0]["recall"] == 1.0
        assert out[0.0]["precision"] == 1.0

    def test_a_higher_threshold_never_lowers_recall(self):
        y_true = np.array([-2.0, -0.5, 0.5, 3.0, 6.0])
        y_pred = np.array([-1.0, 0.7, 1.2, 2.0, 5.0])
        out = train._alarm_sweep(y_true, y_pred)
        recalls = [out[t]["recall"] for t in train.ALARM_THRESHOLDS_C]
        assert recalls == sorted(recalls)

    def test_frost_truth_is_at_or_below_zero(self):
        # A night at exactly 0 C counts as frost; predicting 0 C alarms at thr 0.
        out = train._alarm_sweep(np.array([0.0]), np.array([0.0]))
        assert out[0.0]["recall"] == 1.0

    def test_counts_false_positives_in_precision(self):
        y_true = np.array([-1.0, 5.0])
        y_pred = np.array([-1.0, -1.0])
        out = train._alarm_sweep(y_true, y_pred)
        assert out[0.0]["recall"] == 1.0
        assert out[0.0]["precision"] == pytest.approx(0.5)

    def test_nan_when_no_frost_nights_exist(self):
        out = train._alarm_sweep(np.array([5.0, 6.0]), np.array([5.0, 6.0]))
        assert math.isnan(out[0.0]["recall"])

    def test_nan_precision_when_the_alarm_never_fires(self):
        out = train._alarm_sweep(np.array([-1.0, -2.0]), np.array([9.0, 9.0]))
        assert out[0.0]["recall"] == 0.0
        assert math.isnan(out[0.0]["precision"])

    def test_covers_every_configured_threshold(self):
        out = train._alarm_sweep(np.array([-1.0, 4.0]), np.array([0.2, 3.0]))
        assert set(out) == set(train.ALARM_THRESHOLDS_C)


class TestEvaluateFold:
    @pytest.fixture(scope="class")
    @classmethod
    def fold(cls):
        df = synthetic_nights()
        tr = df[df["year"] == 2020]
        te = df[df["year"] == 2021]
        return train.evaluate_fold(tr, te, "2021"), tr, te

    def test_reports_label_and_sizes(self, fold):
        res, tr, te = fold
        assert res["fold"] == "2021"
        assert (res["n_train"], res["n_test"]) == (len(tr), len(te))

    def test_scores_model_and_both_baselines(self, fold):
        res, _, _ = fold
        for key in ("mae_model", "rmse_model", "mae_climatology", "mae_fao"):
            assert res[key] > 0

    def test_margin_is_model_minus_fao(self, fold):
        res, _, _ = fold
        assert res["mae_margin_vs_fao"] == pytest.approx(
            res["mae_model"] - res["mae_fao"])

    def test_raw_arrays_are_returned_for_pooling(self, fold):
        res, _, te = fold
        for key in ("_y_true", "_y_model", "_y_fao"):
            assert len(res[key]) == len(te)

    def test_wind_slices_partition_the_non_missing_nights(self, fold):
        res, _, te = fold
        assert res["n_calm"] + res["n_windy"] + res["n_wind_missing"] == len(te)

    def test_missing_wind_falls_in_neither_slice(self):
        df = synthetic_nights(n_per_station=25)
        tr = df[df["year"] == 2020]
        te = df[df["year"] == 2021].copy()
        te.iloc[0, te.columns.get_loc("wind_ms")] = math.nan
        res = train.evaluate_fold(tr, te, "2021")
        assert res["n_wind_missing"] == 1
        assert res["n_calm"] + res["n_windy"] == len(te) - 1

    def test_calm_mae_is_nan_when_no_calm_nights(self):
        df = synthetic_nights(n_per_station=25)
        tr = df[df["year"] == 2020]
        te = df[df["year"] == 2021].copy()
        te["wind_ms"] = train.CALM_WIND_MS + 5.0
        res = train.evaluate_fold(tr, te, "2021")
        assert math.isnan(res["mae_calm"])
        assert res["n_calm"] == 0
        assert not math.isnan(res["mae_windy"])

    def test_restricted_feature_list_is_honoured(self):
        df = synthetic_nights(n_per_station=25)
        no_geo = [f for f in train.FEATURES if f not in ("lat", "lon", "elev")]
        res = train.evaluate_fold(df[df["year"] == 2020], df[df["year"] == 2021],
                                  "nogeo", features=no_geo)
        assert res["mae_model"] > 0


class TestRunCvAndSummarise:
    @pytest.fixture(scope="class")
    @classmethod
    def res(cls):
        return train.run_cv(synthetic_nights(n_per_station=25), "year", "LOYO-test")

    def test_one_row_per_group(self, res):
        assert len(res) == 2
        assert sorted(res["fold"]) == ["2020", "2021"]

    def test_each_fold_holds_out_exactly_its_group(self, res):
        df = synthetic_nights(n_per_station=25)
        for _, row in res.iterrows():
            assert row["n_test"] == (df["year"] == int(row["fold"])).sum()
            assert row["n_train"] == len(df) - row["n_test"]

    def test_summarise_reports_the_fold_spread(self, res):
        out = train.summarise(res, "LOYO-test")
        assert out["mae_min"] <= out["mae_mean"] <= out["mae_max"]

    def test_summarise_margin_matches_the_per_fold_margins(self, res):
        out = train.summarise(res, "LOYO-test")
        assert out["margin_vs_fao_mean"] == pytest.approx(
            res["mae_margin_vs_fao"].mean(), abs=5e-4)
        assert out["margin_vs_fao_min"] <= out["margin_vs_fao_max"]

    def test_summarise_drops_the_raw_arrays_from_per_fold_output(self, res):
        out = train.summarise(res, "LOYO-test")
        assert len(out["per_fold"]) == len(res)
        assert not any(k.startswith("_") for k in out["per_fold"][0])

    def test_summarise_recommended_alarm_matches_its_sweep_entry(self, res):
        out = train.summarise(res, "LOYO-test")
        assert out["recommended_alarm_c"] == train.RECOMMENDED_ALARM_C
        sweep = out["alarm_sweep_model"][train.RECOMMENDED_ALARM_C]
        assert out["frost_recall_pooled"] == sweep["recall"]
        assert out["frost_precision_pooled"] == sweep["precision"]

    def test_summarise_output_is_json_serialisable(self, res):
        text = train._dump_json({"loyo": train.summarise(res, "LOYO-test")})
        assert json.loads(text)["loyo"]["mae_mean"] > 0


class TestMain:
    def test_writes_metrics_json_for_every_evaluation(self, tmp_path, monkeypatch):
        csv = tmp_path / "nights.csv"
        df = synthetic_nights(n_per_station=12).drop(columns=["year", "country"])
        df.to_csv(csv, index=False)
        monkeypatch.setattr(train, "DATA", csv)
        monkeypatch.setattr(train, "OUT_DIR", tmp_path)
        train.main()
        summary = json.loads((tmp_path / "metrics.json").read_text())
        assert summary["n_nights"] == len(df)
        assert summary["n_stations"] == df["station"].nunique()
        for key in ("leave_one_year_out", "leave_one_station_out",
                    "leave_one_country_out", "leave_one_country_out_no_geo"):
            assert summary[key]["mae_mean"] > 0


class TestConfiguration:
    def test_recommended_alarm_is_a_swept_threshold(self):
        assert train.RECOMMENDED_ALARM_C in train.ALARM_THRESHOLDS_C

    def test_target_is_not_a_feature(self):
        assert train.TARGET not in train.FEATURES

    def test_features_are_unique(self):
        assert len(set(train.FEATURES)) == len(train.FEATURES)

    def test_omp_threads_capped_before_sklearn_import(self):
        import os
        assert os.environ["OMP_NUM_THREADS"]
