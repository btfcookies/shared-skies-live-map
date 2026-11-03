"""Model bundle loading and batch inference: the 30-feature ensemble, calendar and spatial features."""

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from backend import config
from backend.config import DIST_TO_SENSOR_MAX_KM, ROOT
from backend.state import state
from backend.weather_feed import WEATHER_FALLBACK


# ── Spatial-context reference points (kept in sync with pipeline/03_train_enhanced.py) ──
TX_COAST_POINTS = [
    (25.97, -97.50), (27.80, -97.40), (28.93, -95.97),
    (29.30, -94.79), (29.70, -93.90),
]


TX_URBAN_POINTS = [
    (32.78, -96.80), (29.76, -95.37), (30.27, -97.74),
    (29.42, -98.49), (32.75, -97.33), (31.76, -106.49),
]


def _haversine_km_np(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _min_dist_to_points(lats, lons, points):
    out = np.full(len(lats), np.inf)
    for (plat, plon) in points:
        out = np.minimum(out, _haversine_km_np(lats, lons, plat, plon))
    return out


def load_training_sensors() -> list[dict]:
    """[{sensor_id, lat, lon}] from models/training_sensors.json, or [] if the
    file is missing or unreadable."""
    try:
        with open(config.TRAINING_SENSORS_PATH) as f:
            rows = json.load(f)
        return [{"sensor_id": r.get("sensor_id"), "lat": float(r["lat"]), "lon": float(r["lon"])}
                for r in rows]
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"  could not read {config.TRAINING_SENSORS_PATH}: {e}")
        return []


def enrich_tract_lookup_with_distances(df: pd.DataFrame) -> None:
    """Add dist_to_nearest_sensor, dist_to_coast, dist_to_urban to tract_lookup in place.
    The model uses these features at inference. Sensor locations come ONLY from
    models/training_sensors.json, the exact pre-QC set training used. Idempotent
    — no-op if columns already exist."""
    if {"dist_to_nearest_sensor", "dist_to_coast", "dist_to_urban"} <= set(df.columns):
        return
    lats = df["lat"].values
    lons = df["lon"].values
    df["dist_to_coast"] = _min_dist_to_points(lats, lons, TX_COAST_POINTS)
    df["dist_to_urban"] = _min_dist_to_points(lats, lons, TX_URBAN_POINTS)

    # No fallback file on purpose: the legacy p2_processed.xls is a 240-sensor
    # subset, and silently using it moved the tract median from 7.8 to 11.0 km.
    sensors = load_training_sensors()
    if not sensors:
        print("!" * 70)
        print(f"WARNING: {os.path.relpath(config.TRAINING_SENSORS_PATH, ROOT)} is missing. "
              "dist_to_nearest_sensor is set to the training max "
              f"({DIST_TO_SENSOR_MAX_KM} km) for every tract, which does NOT match "
              "training. Run pipeline/14_export_training_sensors.py and commit it.")
        print("!" * 70)
        df["dist_to_nearest_sensor"] = DIST_TO_SENSOR_MAX_KM
    else:
        sensor_lats = np.array([s["lat"] for s in sensors])
        sensor_lons = np.array([s["lon"] for s in sensors])
        nearest = np.full(len(df), np.inf)
        for sl, slo in zip(sensor_lats, sensor_lons):
            nearest = np.minimum(nearest, _haversine_km_np(lats, lons, sl, slo))
        # Clip to the training-network max. Training computed sensor->nearest-OTHER-
        # sensor (max ~164km); tracts can be ~195km from any sensor. Clipping keeps
        # rural tracts in the tree models' training distribution instead of
        # extrapolating into never-seen feature space.
        nearest = np.clip(nearest, 0.0, DIST_TO_SENSOR_MAX_KM)
        df["dist_to_nearest_sensor"] = nearest
        print(f"  enriched lookup from {len(sensors)} training sensors: "
              f"dist_to_nearest_sensor median={np.median(nearest):.2f} km, "
              f"p90={np.percentile(nearest, 90):.1f} km "
              f"(clipped to {DIST_TO_SENSOR_MAX_KM}), "
              f"dist_to_coast median={np.median(df['dist_to_coast'].values):.1f} km, "
              f"dist_to_urban median={np.median(df['dist_to_urban'].values):.1f} km")

    # ── Neighbor-PM features: compute the SAME multi-radius (25/50/100km) neighbor
    # PM2.5 the model trains on, here from the per-sensor CLIMATOLOGICAL means.
    # This is the inference-time FALLBACK; the live path overrides these columns
    # each cycle with same-day PurpleAir. Both paths call the SHARED
    # compute_neighbor_features so they are byte-identical and match training.
    nbr_path = os.path.join(ROOT, "models", "sensor_recent_pm.json")
    if os.path.exists(nbr_path):
        try:
            from backend.purpleair import compute_neighbor_features
            with open(nbr_path) as f:
                sensors_meta = json.load(f)
            static_sensors = [
                {"lat": float(s["lat"]), "lon": float(s["lon"]),
                 "pm25": float(s["recent_mean_pm25"])}
                for s in sensors_meta
            ]
            nbr_feats = compute_neighbor_features(lats, lons, static_sensors)
            if nbr_feats is None:
                raise ValueError("too few sensors for neighbor features")
            for fname, arr in nbr_feats.items():
                df[fname] = arr
            print(f"  neighbor features (climatological fallback): nbr_pm25_50km "
                  f"median={np.median(nbr_feats['nbr_pm25_50km']):.2f} µg/m³, "
                  f"50km coverage={(nbr_feats['nbr_count_50km']>0).sum()/len(df)*100:.1f}%")
        except Exception as e:
            print(f"  WARNING: failed to build neighbor features from {nbr_path}: {e}; filling 0")
            for r_km in (25, 50, 100):
                df[f"nbr_pm25_{r_km}km"] = 0.0
                df[f"nbr_count_{r_km}km"] = 0
            df["nbr_std_50km"] = 0.0
    else:
        # Pre-v3 deployment — model doesn't use these features. Backend's
        # run_predictions_batch only picks features the bundle requests, so
        # leaving them off costs nothing.
        print(f"  no {os.path.basename(nbr_path)} (pre-v3 model); skipping neighbor features")


def load_bundle() -> dict | None:
    """Load the model bundle, drop zero-weight models and cap thread pools.
    Any failure returns None so the API still starts: endpoints that need the
    model answer 503 and /api/health reports degraded, instead of the process
    dying before it binds the port (which Render shows only as a failed deploy)."""
    if not os.path.exists(config.MODEL_PATH):
        print("WARNING: models/ensemble.joblib not found. Run pipeline/03_train_enhanced.py")
        return None
    try:
        # No mmap_mode: the bundle is compressed, and joblib ignores mmap for
        # compressed files (it only emitted a warning).
        b = joblib.load(config.MODEL_PATH)
    except Exception as e:
        print(f"ERROR: failed to load {config.MODEL_PATH}: {type(e).__name__}: {e}")
        return None
    try:
        # Free the RAM of any model that carries zero blend weight. XGBoost
        # sits at weight 0.0 in the deployed v6 blend, so its booster is dead
        # weight; _active_models would skip it anyway, so no output changes.
        # The xgboost library stays imported (it was needed to unpickle).
        weights = b.get("weights", {})
        dropped = [n for n in list(b.get("models", {}))
                   if abs(weights.get(n, 0.0)) <= _ZERO_WEIGHT_EPS]
        for n in dropped:
            b["models"].pop(n, None)
        if dropped:
            print(f"Dropped zero-weight models from RAM: {dropped}")
    except Exception as e:
        print(f"Zero-weight model prune skipped: {e}")
    # Training saved n_jobs=-1, which sizes predict-time pools to the host's
    # cores. One thread is right for a 0.1-CPU instance; outputs are unchanged
    # (measured max |diff| 4e-15 on RF, 0 on LightGBM). CatBoost's fitted
    # params are frozen, so its thread_count goes on each predict call instead.
    for name, model in b.get("models", {}).items():
        try:
            if "n_jobs" in model.get_params():
                model.set_params(n_jobs=1)
        except Exception as e:
            print(f"Could not cap threads for {name}: {e}")
    return b


def get_temporal(now: datetime | None = None) -> dict:
    """Calendar model features for the current UTC day. Training defines a day
    as the UTC day (PurpleAir daily buckets start at UTC midnight, and the
    weather/CAMS archives were pulled with timezone=UTC), so local time here
    would shift dow/day_of_year for the evening hours in Texas. Local time is
    for display only."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return {
        "month":       now.month,
        "dow":         now.weekday(),
        "day_of_year": now.timetuple().tm_yday,
    }


# One threshold for "this model carries no blend weight", used both to drop
# models at load and to skip them at predict time.
_ZERO_WEIGHT_EPS = 1e-9


def _active_models(weights: dict, models) -> list:
    """Model names with non-negligible ensemble weight. Skips zero-weight models
    (e.g. XGBoost at weight 0.0 in the v6 simplex blend) so we don't pay for a
    .predict() call + its RAM that contributes nothing on the 512MB free tier.
    The weighted sum is numerically identical with the zero-weight terms removed."""
    active = [n for n in models if abs(weights.get(n, 0.0)) > _ZERO_WEIGHT_EPS]
    return active or list(models)  # defensive: never return an empty model set


def _predict_one(model, X: np.ndarray) -> np.ndarray:
    # CatBoost ignores OMP_NUM_THREADS and its fitted params can't be changed,
    # so its thread count is set per call.
    if type(model).__name__.startswith("CatBoost"):
        return model.predict(X, thread_count=1)
    return model.predict(X)


def _compute_v3_shared(weather: dict, temporal: dict) -> dict:
    """Compute all derived features that are shared across tracts (weather + temporal).
    Defaults are the training medians in TRAINING UNITS (°C, %, surface hPa, m/s)."""
    month = temporal.get("month", 1)
    dow = temporal.get("dow", 0)
    doy = temporal.get("day_of_year", 1)
    temp = weather.get("temperature", WEATHER_FALLBACK["temperature"])
    hum = weather.get("humidity", WEATHER_FALLBACK["humidity"])
    ws = weather.get("wind_speed", WEATHER_FALLBACK["wind_speed"])
    precip = weather.get("precipitation", 0)

    shared = {**weather, **temporal}
    shared.pop("display", None)  # UI-only sub-dict, never a model feature
    shared["precipitation"] = precip
    # Cyclical
    shared["month_sin"] = np.sin(2 * np.pi * month / 12)
    shared["month_cos"] = np.cos(2 * np.pi * month / 12)
    shared["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    shared["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    shared["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    shared["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    # Interactions: ONLY the two the deployed model actually uses (features 36-37
    # in feature_names.json). Formulas mirror pipeline/03_train_enhanced.py exactly.
    # The cold-start single-tract path reads these from `shared`;
    # without them the model would silently receive 0.0 → train/serve skew. The
    # three previously-computed extras (wind_x_season / humidity_x_season /
    # precip_x_temp) were never model features — dead compute — so they stay gone.
    shared["temp_x_humidity"] = temp * hum / 100.0
    shared["wind_x_temp"] = ws * temp / 100.0
    return shared


def run_predictions_batch(df: pd.DataFrame, weather: dict, temporal: dict) -> np.ndarray:
    """
    Vectorized batch prediction. Handles v1/v2/v3 feature sets automatically
    based on what the loaded model expects.
    """
    bundle   = state["bundle"]
    features = bundle["feature_names"]
    weights  = bundle["weights"]
    models   = bundle["models"]

    shared = _compute_v3_shared(weather, temporal)

    # dist_to_coast normally comes from the startup enrichment. If a caller
    # passes a frame without it, compute it with the training formula
    # (haversine km to the nearest coast point); the old fallback here used
    # |lon + 94| in degrees, a different feature entirely.
    if "dist_to_coast" in features and "dist_to_coast" not in df.columns:
        df = df.copy()
        df["dist_to_coast"] = _min_dist_to_points(
            pd.to_numeric(df["lat"]).values, pd.to_numeric(df["lon"]).values, TX_COAST_POINTS)

    # Lookup uses lat/lon; v2 model was trained with latitude/longitude column names
    feat_set = set(features)
    if "latitude" in feat_set and "lat" in df.columns and "latitude" not in df.columns:
        df = df.copy()
        df["latitude"] = df["lat"]
    if "longitude" in feat_set and "lon" in df.columns and "longitude" not in df.columns:
        df = df.copy() if "latitude" not in df.columns else df
        df["longitude"] = df["lon"]

    n = len(df)
    X = np.zeros((n, len(features)), dtype=np.float64)

    # Use the EXACT training-time fill values for any feature missing/NaN at
    # inference, so the model never sees a value distribution it wasn't trained
    # on. bundle['feature_fill'] holds the training medians (and hms_smoke=0).
    feature_fill = bundle.get("feature_fill", {})

    # Precedence: per-tract df column FIRST (live gridded weather / neighbors /
    # HMS / AOD vary by tract), then the shared scalar (single-point weather +
    # temporal), then the training-median fill. df-first matters: the texas path
    # now writes per-tract weather columns that must not be shadowed by the
    # single shared scalar.
    # Vectorized DataFrame cast
    df_cols = [f for f in features if f in df.columns]
    if df_cols:
        df[df_cols] = df[df_cols].apply(pd.to_numeric, errors="coerce")

    for i, feat in enumerate(features):
        fill = float(feature_fill.get(feat, 0.0))
        if feat in df_cols:
            col = df[feat]
            if feat in shared:
                col = col.fillna(float(shared[feat]))
            X[:, i] = col.fillna(fill).values
        elif feat in shared:
            X[:, i] = shared[feat]
        else:
            X[:, i] = fill  # feature entirely absent -> training fill, not 0

    preds = sum(weights[name] * _predict_one(models[name], X) for name in _active_models(weights, models))
    # v3 models fit on log1p(pm25); invert before returning. Pre-v3 bundles
    # don't have this key so this is a no-op for them.
    if bundle.get("target_transform") == "log1p":
        preds = np.expm1(preds)
    return np.maximum(0.0, preds)
