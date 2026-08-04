"""Unit tests for frostlib.model_io -- the freeze/load contract.

The load-time feature-name assertion is the guard against the classic production
failure (a silently reordered feature vector scored by column position), so it is
tested directly here, offline, over a trivial fitted model.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from frostlib import model_io

FEATURES = ["a", "b", "c"]


def _fit():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(40, len(FEATURES))), columns=FEATURES)
    y = x["a"] * 2 - x["b"]
    m = HistGradientBoostingRegressor(max_iter=20, random_state=0)
    m.fit(x, y)
    return m


@pytest.fixture
def frozen(tmp_path):
    path = tmp_path / "model.joblib"
    version = model_io.save_model(
        _fit(), FEATURES,
        training_years=[2021, 2019, 2020],
        training_stations=["uk_b", "uk_a"],
        n_training_nights=40,
        mae_c=1.8,
        path=path,
    )
    return path, version


class TestRoundTrip:
    def test_load_returns_a_usable_artifact(self, frozen):
        path, version = frozen
        art = model_io.load_model(path=path)
        assert art.features == FEATURES
        assert art.model_version == version
        assert hasattr(art.model, "predict")

    def test_metadata_is_preserved_and_normalised(self, frozen):
        art = model_io.load_model(path=frozen[0])
        assert art.training_years == [2019, 2020, 2021]      # sorted
        assert art.training_stations == ["uk_a", "uk_b"]      # sorted
        assert art.n_training_nights == 40
        assert art.mae_c == pytest.approx(1.8)
        assert art.trained_at  # ISO timestamp recorded

    def test_version_is_stable_across_freezes_of_the_same_fit(self, tmp_path):
        model = _fit()
        p1, p2 = tmp_path / "a.joblib", tmp_path / "b.joblib"
        common = dict(training_years=[2021], training_stations=["x"],
                      n_training_nights=1)
        v1 = model_io.save_model(model, FEATURES, path=p1, **common)
        v2 = model_io.save_model(model, FEATURES, path=p2, **common)
        assert v1 == v2

    def test_version_covers_the_feature_list(self, tmp_path):
        # Reordering the feature list changes the version -- the version identifies
        # the full (estimator, features) contract, not just the trees.
        model = _fit()
        common = dict(training_years=[2021], training_stations=["x"],
                      n_training_nights=1)
        v1 = model_io.save_model(model, ["a", "b", "c"], path=tmp_path / "a.joblib", **common)
        v2 = model_io.save_model(model, ["c", "b", "a"], path=tmp_path / "b.joblib", **common)
        assert v1 != v2

    def test_predict_selects_the_models_feature_columns(self, frozen):
        art = model_io.load_model(path=frozen[0])
        # Columns supplied in a DIFFERENT order + an extra column; predict must
        # still select a,b,c in the model's order.
        row = pd.DataFrame([{"c": 0.0, "extra": 9.0, "b": 1.0, "a": 2.0}])
        assert np.isfinite(art.predict(row)[0])


class TestFeatureAssertion:
    def test_matching_features_pass(self, frozen):
        art = model_io.load_model(expected_features=FEATURES, path=frozen[0])
        assert art.features == FEATURES

    def test_reordered_features_are_rejected(self, frozen):
        with pytest.raises(model_io.ModelContractError, match="position"):
            model_io.load_model(expected_features=["b", "a", "c"], path=frozen[0])

    def test_added_or_removed_feature_is_rejected(self, frozen):
        with pytest.raises(model_io.ModelContractError):
            model_io.load_model(expected_features=FEATURES + ["d"], path=frozen[0])


class TestRefusals:
    def test_missing_file_is_rejected_with_guidance(self, tmp_path):
        with pytest.raises(model_io.ModelContractError, match="--fit"):
            model_io.load_model(path=tmp_path / "absent.joblib")

    def test_legacy_unversioned_bundle_is_rejected(self, tmp_path):
        # The old {"model", "features"} dict must not load as an artifact -- it
        # carries no version/metadata the service relies on.
        path = tmp_path / "legacy.joblib"
        joblib.dump({"model": _fit(), "features": FEATURES}, path)
        with pytest.raises(model_io.ModelContractError, match="versioned"):
            model_io.load_model(path=path)

    def test_newer_schema_is_rejected(self, frozen, monkeypatch):
        # A bundle whose schema is newer than this loader must be refused, not
        # silently half-read.
        path = frozen[0]
        art = joblib.load(path)
        object.__setattr__(art, "schema_version", model_io.ARTIFACT_SCHEMA_VERSION + 1)
        joblib.dump(art, path)
        with pytest.raises(model_io.ModelContractError, match="newer"):
            model_io.load_model(path=path)
