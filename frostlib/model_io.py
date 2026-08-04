"""Freeze, load and version the deployed model artifact.

The evaluation pipeline measures how well the approach works; the deployed
service must run *one* fixed model whose accuracy is the published number, never
refitting on demand. This module is the boundary between the two: it saves the
fitted estimator together with the metadata that identifies it, and loads it back
with the checks a production loader needs.

Two guarantees it exists to provide:

- **Identity.** A loaded model can say which model it is -- a content-hash
  version, the feature names it was fit on, the training years and stations, and
  the measured accuracy. Every forecast the service stores carries that version,
  so a bad forecast months later is attributable to an exact artifact.
- **Feature-order safety.** A HistGradientBoosting model predicts on a bare
  array: it matches feature *position*, not name. A live feature vector built in
  a different order than training would be scored silently and wrongly (dew point
  read as pressure) with no error. ``load_model`` asserts the saved feature names
  equal the caller's expected list, in order, and refuses the model otherwise.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib

from frostlib.paths import MODEL_PATH

# Bump when the bundle layout changes in a way older loaders cannot read.
ARTIFACT_SCHEMA_VERSION = 1


class ModelContractError(RuntimeError):
    """The loaded artifact does not match what the caller requires (feature
    names/order, or a schema the loader cannot read). Raised instead of letting
    a mismatched model score silently."""


@dataclass(frozen=True)
class ModelArtifact:
    """A fitted estimator plus the metadata that identifies and validates it."""

    model: object
    features: list[str]
    model_version: str            # content hash of the fitted estimator
    trained_at: str               # ISO-8601 UTC, when it was frozen
    training_years: list[int]     # years present in the training data
    training_stations: list[str]  # stations present in the training data
    n_training_nights: int
    mae_c: float | None           # published leave-one-year-out MAE, if known
    schema_version: int = ARTIFACT_SCHEMA_VERSION
    extra: dict = field(default_factory=dict)

    def predict(self, frame):
        """Predict on a frame, selecting the model's own feature columns in the
        model's own order -- so a caller cannot mis-order the input."""
        return self.model.predict(frame[self.features])


def _artifact_hash(model, features) -> str:
    """A stable content hash of the (fitted estimator, feature list), the version.

    Hashing the feature list alongside the estimator means hand-editing the saved
    feature order changes the version too -- the version identifies the full
    contract, not just the trees. A retrain hashes differently by design.

    The hash is stable WITHIN an environment; a Python/sklearn/numpy upgrade
    changes the pickle bytes and therefore the version of an unchanged model, so
    do not treat a version change across environments as evidence of a retrain.
    """
    buf = io.BytesIO()
    joblib.dump((model, list(features)), buf)
    return "m-" + hashlib.sha256(buf.getvalue()).hexdigest()[:16]


def save_model(model, features, *, training_years, training_stations,
               n_training_nights, mae_c=None, path=MODEL_PATH, extra=None):
    """Freeze ``model`` to ``path`` as a versioned bundle, and return its version.

    ``features`` is the exact ordered feature list the model was fit on; it is
    stored so the loader can assert it. The version is derived from the fitted
    estimator's bytes, so it identifies this exact artifact.
    """
    version = _artifact_hash(model, features)
    artifact = ModelArtifact(
        model=model,
        features=list(features),
        model_version=version,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        training_years=sorted(int(y) for y in training_years),
        training_stations=sorted(map(str, training_stations)),
        n_training_nights=int(n_training_nights),
        mae_c=None if mae_c is None else float(mae_c),
        extra=dict(extra or {}),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return version


def load_model(*, expected_features=None, path=MODEL_PATH) -> ModelArtifact:
    """Load the frozen artifact, asserting it is one this code can safely use.

    If ``expected_features`` is given, the saved feature names must equal it
    exactly and in order, or ``ModelContractError`` is raised -- the guard against
    a silently reordered feature vector. Also refuses a bundle whose schema
    version this loader does not understand.
    """
    if not path.exists():
        raise ModelContractError(
            f"no model artifact at {path} -- run `python3 predict.py --fit` first")
    artifact = joblib.load(path)

    if not isinstance(artifact, ModelArtifact):
        raise ModelContractError(
            f"{path} is not a versioned model artifact (got {type(artifact).__name__}); "
            "re-freeze it with `python3 predict.py --fit`")
    if artifact.schema_version > ARTIFACT_SCHEMA_VERSION:
        raise ModelContractError(
            f"artifact schema v{artifact.schema_version} is newer than this loader "
            f"(v{ARTIFACT_SCHEMA_VERSION}); upgrade the code before loading it")
    if expected_features is not None and list(expected_features) != artifact.features:
        raise ModelContractError(
            "feature names/order do not match the deployed model. A model scores "
            "by column position, so this would silently mis-predict.\n"
            f"  expected: {list(expected_features)}\n"
            f"  artifact: {artifact.features}")
    return artifact
