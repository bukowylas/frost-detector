"""Unit tests for predict.py: feature assembly, fitting, and the alarm output."""

from __future__ import annotations

import argparse
import math

import pandas as pd
import pytest

import predict
import train
from frostlib import model_io


def args(**over):
    base = {"temp": 3.0, "dewpoint": 0.5, "wind": 1.5, "cloud": 1.0,
            "pressure": 1022.0, "temp_change_3h": 0.0, "temp_change_24h": 0.0,
            "slp_tendency_3h": 0.0, "lat": 51.7, "lon": 19.4, "elev": 180.0,
            "doy": 110, "fit": False}
    base.update(over)
    return argparse.Namespace(**base)


class TestDerived:
    def test_covers_exactly_the_training_features(self):
        assert set(predict._derived(args())) == set(train.FEATURES)

    def test_dewpoint_depression_is_temp_minus_dewpoint(self):
        row = predict._derived(args(temp=3.0, dewpoint=0.5))
        assert row["dewpoint_depression_c"] == pytest.approx(2.5)

    def test_radiative_potential_matches_the_prepare_formula(self):
        row = predict._derived(args(cloud=0.0, wind=1.0))
        assert row["radiative_potential"] == pytest.approx(0.5)

    def test_clear_calm_night_scores_higher_than_cloudy_windy(self):
        clear = predict._derived(args(cloud=0.0, wind=0.5))["radiative_potential"]
        cloudy = predict._derived(args(cloud=8.0, wind=6.0))["radiative_potential"]
        assert clear > cloudy

    def test_missing_cloud_and_wind_use_the_documented_defaults(self):
        row = predict._derived(args(cloud=None, wind=None))
        # (1 - 4/8) / (1 + 3): mid cloud, light wind -- same as prepare.py
        assert row["radiative_potential"] == pytest.approx(0.125)
        # ...but the raw features stay missing, which the model handles natively.
        assert row["cloud_oktas"] is None
        assert row["wind_ms"] is None

    def test_passes_the_snapshot_inputs_through_unchanged(self):
        row = predict._derived(args(pressure=1005.0, lat=52.0, doy=91))
        assert row["slp_hpa"] == 1005.0
        assert row["lat"] == 52.0
        assert row["doy"] == 91


def tiny_nights(n=60):
    rows = []
    for i in range(n):
        temp = -2.0 + 0.2 * i
        dew = temp - 1.5
        doy = 91 + i % 30
        rows.append({
            "station": "uk_test", "date": f"2021-{3 + doy // 31:02d}-{1 + doy % 28:02d}",
            "temp_c": temp, "dewpoint_c": dew, "dewpoint_depression_c": 1.5,
            "slp_hpa": 1015.0, "wind_ms": 1.0, "cloud_oktas": 2.0,
            "radiative_potential": 0.25, "temp_change_3h": -0.5,
            "temp_change_24h": 0.0, "slp_tendency_3h": 0.1,
            "lat": 51.7, "lon": 19.4, "elev": 180.0, "doy": doy,
            "tmin_overnight_c": temp - 4.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def fitted_model(tmp_path, monkeypatch):
    csv = tmp_path / "nights.csv"
    tiny_nights().to_csv(csv, index=False)
    model_path = tmp_path / "model.joblib"
    monkeypatch.setattr(predict, "DATA", csv)
    monkeypatch.setattr(predict, "MODEL_PATH", model_path)
    predict.fit_and_save()
    return model_path


class TestFitAndSave:
    def test_freezes_a_versioned_artifact_with_the_feature_order(self, fitted_model):
        artifact = model_io.load_model(path=fitted_model)
        assert artifact.features == train.FEATURES
        assert hasattr(artifact.model, "predict")
        assert artifact.model_version.startswith("m-")

    def test_embeds_the_training_metadata(self, fitted_model):
        artifact = model_io.load_model(path=fitted_model)
        # tiny_nights spans doy 91.. within one year, one synthetic station.
        assert artifact.n_training_nights == 60
        assert artifact.training_years  # at least one year recorded
        assert artifact.training_stations  # at least one station recorded

    def test_creates_the_output_directory(self, tmp_path, monkeypatch):
        csv = tmp_path / "nights.csv"
        tiny_nights().to_csv(csv, index=False)
        monkeypatch.setattr(predict, "DATA", csv)
        monkeypatch.setattr(predict, "MODEL_PATH", tmp_path / "sub" / "model.joblib")
        predict.fit_and_save()
        assert (tmp_path / "sub" / "model.joblib").exists()

    def test_saved_model_predicts_a_finite_temperature(self, fitted_model):
        artifact = model_io.load_model(path=fitted_model)
        row = pd.DataFrame([predict._derived(args())])
        assert math.isfinite(float(artifact.predict(row)[0]))


class TestForecast:
    def test_missing_model_file_exits_with_guidance(self, tmp_path, monkeypatch):
        monkeypatch.setattr(predict, "MODEL_PATH", tmp_path / "absent.joblib")
        with pytest.raises(SystemExit, match="--fit"):
            predict.forecast(args())

    def test_prints_a_prediction_and_an_alarm_line(self, fitted_model, capsys):
        predict.forecast(args(temp=-3.0, dewpoint=-4.0))
        out = capsys.readouterr().out
        assert "predicted overnight minimum" in out
        assert "frost alarm" in out

    def test_cold_evening_raises_the_alarm(self, fitted_model, capsys):
        predict.forecast(args(temp=-3.0, dewpoint=-4.0))
        assert "YES" in capsys.readouterr().out

    def test_mild_evening_does_not_raise_the_alarm(self, fitted_model, capsys):
        predict.forecast(args(temp=9.5, dewpoint=8.0))
        assert ": no" in capsys.readouterr().out

    def test_alarm_threshold_is_the_recommended_one(self, fitted_model, capsys):
        predict.forecast(args())
        assert f"{predict.RECOMMENDED_ALARM_C:+.1f} C" in capsys.readouterr().out

    def test_artifact_selects_its_own_feature_order(self, fitted_model):
        # artifact.predict picks its stored feature columns, so the caller's dict
        # order cannot mis-order the model input.
        artifact = model_io.load_model(path=fitted_model)
        assert artifact.features == train.FEATURES


class TestMainCli:
    def test_fit_flag_calls_fit_and_save(self, monkeypatch):
        called = []
        monkeypatch.setattr(predict, "fit_and_save", lambda: called.append(True))
        monkeypatch.setattr("sys.argv", ["predict.py", "--fit"])
        predict.main()
        assert called == [True]

    # A forecast requires geography + day-of-year (never invented defaults).
    _FORECAST_ARGV = ["--temp", "3", "--dewpoint", "0.5",
                      "--lat", "53.2", "--lon", "-0.5", "--doy", "105"]

    def test_temp_and_dewpoint_trigger_a_forecast(self, monkeypatch):
        seen = []
        monkeypatch.setattr(predict, "forecast", lambda a: seen.append(a))
        monkeypatch.setattr("sys.argv", ["predict.py", *self._FORECAST_ARGV])
        predict.main()
        assert seen[0].temp == 3.0 and seen[0].dewpoint == 0.5

    def test_cli_defaults_leave_wind_cloud_pressure_and_trends_missing(self, monkeypatch):
        import math
        seen = []
        monkeypatch.setattr(predict, "forecast", lambda a: seen.append(a))
        monkeypatch.setattr("sys.argv", ["predict.py", *self._FORECAST_ARGV])
        predict.main()
        assert (seen[0].wind, seen[0].cloud, seen[0].pressure) == (None, None, None)
        # Trends default to NaN, not 0.0: an absent trend is missing, not "no change".
        assert math.isnan(seen[0].temp_change_3h)
        assert seen[0].elev is None

    @pytest.mark.parametrize("argv", [
        [],
        ["--temp", "3"],
        ["--dewpoint", "0.5"],
        # temp+dewpoint but no geography/doy is now also an error.
        ["--temp", "3", "--dewpoint", "0.5"],
    ])
    def test_incomplete_input_is_an_argparse_error(self, monkeypatch, argv):
        monkeypatch.setattr("sys.argv", ["predict.py", *argv])
        with pytest.raises(SystemExit):
            predict.main()


class TestModelParams:
    def test_early_stopping_and_seed_are_pinned(self):
        assert train.FIXED_MODEL_PARAMS["early_stopping"] is True
        assert train.FIXED_MODEL_PARAMS["random_state"] == train.RANDOM_STATE

    def test_hyperparameters_are_within_the_searched_ranges(self):
        assert 0.0 < train.FIXED_MODEL_PARAMS["learning_rate"] <= 0.1
        assert train.FIXED_MODEL_PARAMS["max_iter"] >= 100
