"""Paths, environment-driven settings and the city table shared across the backend."""

import os


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(ROOT, "backend", "static")
MODEL_PATH = os.path.join(ROOT, "models", "ensemble.joblib")
LOOKUP_PATH = os.path.join(STATIC_DIR, "tract_lookup.parquet")
TEXAS_GEOJSON_PATH = os.path.join(STATIC_DIR, "texas_all_tracts.geojson")
VISITS_DB = os.path.join(ROOT, "backend", "visits.sqlite")
# Overridable so tests and parallel dev servers don't share (or clobber) the
# snapshot files a local dev server hydrates from.
SNAPSHOTS_DIR = os.environ.get("SSI_SNAPSHOTS_DIR") or os.path.join(STATIC_DIR, "snapshots")
TEXAS_SNAPSHOT_PATH = os.path.join(SNAPSHOTS_DIR, "texas_predictions_latest.json")
QUANTUM_SNAPSHOT_PATH = os.path.join(SNAPSHOTS_DIR, "quantum_latest.json")
HMS_SNAPSHOT_PATH = os.path.join(SNAPSHOTS_DIR, "hms_smoke_latest.json")
PURPLEAIR_SNAPSHOT_PATH = os.path.join(SNAPSHOTS_DIR, "purpleair_sensors_latest.json")
WEATHER_GRID_SNAPSHOT_PATH = os.path.join(SNAPSHOTS_DIR, "weather_grid_latest.json")
AQ_SNAPSHOT_PATH = os.path.join(SNAPSHOTS_DIR, "airquality_latest.json")
CONDITIONS_SNAPSHOT_PATH = os.path.join(SNAPSHOTS_DIR, "conditions_latest.json")
MODELS_DIR = os.path.join(ROOT, "models")
# The exact sensor set training built dist_to_nearest_sensor from, exported by
# pipeline/14_export_training_sensors.py. Tracked, so prod and a clean clone
# serve the same feature as a dev machine that has the training .xls.
TRAINING_SENSORS_PATH = os.path.join(MODELS_DIR, "training_sensors.json")

# Live PurpleAir: the model's dominant feature (nbr_pm25_*) must be computed
# from same-day live readings to match training. Set PURPLEAIR_API_KEY in Render
# env to supply the key.
#
# Cost, measured on 2026-10-20: one statewide pull returned 497 sensors and
# cost 3,484 points (~7 points per sensor). A 15-min refresh is therefore
# ~10M points/month; the organization balance was 400M at the time. The
# refresh runs on this schedule whether or not anyone is viewing the map, so
# the cost is predictable. Raise PURPLEAIR_CACHE_TTL_MIN in Render to stretch
# the balance further (the feature is a 24-hour rolling mean, so even a 60-min
# refresh tracks it closely).
PURPLEAIR_CACHE_TTL_MIN = int(os.environ.get("PURPLEAIR_CACHE_TTL_MIN", "15"))
# How old live PurpleAir data may get (failed refreshes, or a snapshot from
# before a restart) before predictions stop treating it as live and fall back
# to climatology. The 24-hour rolling mean drifts slowly, so a few hours of
# outage is better served by the last real readings.
PURPLEAIR_MAX_AGE_MIN = int(os.environ.get("PURPLEAIR_MAX_AGE_MIN", "180"))
# First retry delay after a failed PurpleAir refresh; doubles up to the TTL.
PURPLEAIR_BACKOFF_MIN = 5
# Clip dist_to_nearest_sensor to the training-network max so rural tracts (which
# can be 195km from any sensor vs the 164km training max) don't push the tree
# models out of distribution.
DIST_TO_SENSOR_MAX_KM = 164.4
# A disk snapshot of predictions older than this is not served at startup.
TEXAS_SNAPSHOT_MAX_AGE_MIN = 24 * 60


# ── City Configuration (Generic, extensible design) ─────────────────────────
CITIES = {
    "dallas": {
        "fips": "48113",
        "center": (32.7767, -96.7970),
        "tz": "America/Chicago",
        "geojson": os.path.join(STATIC_DIR, "dallas_tracts.geojson"),
        "display_name": "Dallas County"
    },
    "austin": {
        "fips": "48453",
        "center": (30.2672, -97.7431),
        "tz": "America/Chicago",
        "geojson": os.path.join(STATIC_DIR, "austin_tracts.geojson"),
        "display_name": "Travis County"
    },
    "houston": {
        "fips": "48201",
        "center": (29.7604, -95.3698),
        "tz": "America/Chicago",
        "geojson": os.path.join(STATIC_DIR, "houston_tracts.geojson"),
        "display_name": "Harris County"
    },
    "san_antonio": {
        "fips": "48029",
        "center": (29.4241, -98.4936),
        "tz": "America/Chicago",
        "geojson": os.path.join(STATIC_DIR, "san_antonio_tracts.geojson"),
        "display_name": "Bexar County"
    }
}
DEFAULT_CITY = "dallas"
