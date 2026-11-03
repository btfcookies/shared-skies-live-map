"""The deployed bundle loads under the pinned stack and predicts every tract."""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pytest

from backend import config, model, weather_feed

# The app fixture swaps model.load_bundle for the session's cached bundle;
# keep the real loader, captured at collection time, for the failure test.
_real_load_bundle = model.load_bundle

FEATURE_NAMES_PATH = os.path.join(config.MODELS_DIR, "feature_names.json")
N_TRACTS = 6896

# 2026-10-21 UTC, the evening of Oct 20 in Texas.
FIXED_UTC = datetime(2026, 10, 21, 2, 30, tzinfo=timezone.utc)

# Climatology-fallback predictions for FIXED_UTC (training-median weather,
# neighbor features from models/sensor_recent_pm.json, dist_to_nearest_sensor
# from models/training_sensors.json), identical on Python 3.11 and 3.12.
# A dependency bump that changes how the pickled models predict moves these.
# If a retrain or one of those model files changes on purpose, regenerate them
# with run_predictions_batch exactly as test_fallback_predictions_golden does.
GOLDEN_MEAN = 9.729093759368514
GOLDEN_TRACTS = {
    "48001950100": 10.728111791914891,
    "48085030303": 8.920841151793509,
    "48157672303": 9.978000101961655,
    "48203020105": 12.85201535037875,
    "48507950302": 9.704405760496517,
}


def test_bundle_matches_feature_names(bundle):
    with open(FEATURE_NAMES_PATH) as f:
        expected = json.load(f)
    assert bundle["feature_names"] == expected
    assert len(expected) == 30


def test_bundle_blend(bundle):
    weights = bundle["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-3
    assert weights.get("xgb", 0.0) == 0.0
    # Zero-weight XGBoost is dropped from RAM at load; the rest predict.
    assert set(bundle["models"]) == {"rf", "lgbm", "cat"}
    for name in ("rf", "lgbm"):
        assert bundle["models"][name].get_params()["n_jobs"] == 1


def test_unloadable_bundle_degrades(monkeypatch, tmp_path):
    bad = tmp_path / "ensemble.joblib"
    bad.write_bytes(b"not a pickle")
    monkeypatch.setattr(config, "MODEL_PATH", str(bad))
    assert _real_load_bundle() is None
    monkeypatch.setattr(config, "MODEL_PATH", str(tmp_path / "missing.joblib"))
    assert _real_load_bundle() is None


def _fallback_predict(lookup):
    return model.run_predictions_batch(
        lookup.copy(), weather_feed.weather_fallback(), model.get_temporal(FIXED_UTC))


def test_full_lookup_fallback_predictions(with_bundle, tract_lookup):
    assert len(tract_lookup) == N_TRACTS
    preds = _fallback_predict(tract_lookup)
    assert preds.shape == (N_TRACTS,)
    assert np.isfinite(preds).all()
    assert preds.min() >= 0.0
    assert preds.max() < 60.0
    assert 3.0 < float(np.median(preds)) < 20.0
    # Deterministic: a second run is bit-identical.
    assert np.array_equal(preds, _fallback_predict(tract_lookup))


def test_fallback_predictions_golden(with_bundle, tract_lookup):
    preds = _fallback_predict(tract_lookup)
    assert float(preds.mean()) == pytest.approx(GOLDEN_MEAN, abs=1e-6)
    by_geoid = dict(zip(tract_lookup["GEOID"], preds))
    for geoid, want in GOLDEN_TRACTS.items():
        assert float(by_geoid[geoid]) == pytest.approx(want, abs=1e-6), geoid


def test_missing_features_use_training_fill(with_bundle, tract_lookup):
    # Without per-tract weather columns the shared scalars and training
    # medians apply; the result must stay finite (no zero-filled features).
    lean = tract_lookup[["GEOID", "lat", "lon"]].head(50).copy()
    preds = model.run_predictions_batch(lean, {}, model.get_temporal(FIXED_UTC))
    assert np.isfinite(preds).all() and (preds > 0).all()
