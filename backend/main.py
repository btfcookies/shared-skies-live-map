"""
Shared Skies — FastAPI Backend
Serves model-predicted PM2.5 for every Texas census tract, plus per-county
slices (Dallas, Austin, Houston, San Antonio) of the same statewide result.

Start with:
    uvicorn backend.main:app --reload --port 8000
(run from the project root)
"""

import os

# Cap native thread pools BEFORE numpy/sklearn/lightgbm/catboost load their
# OpenMP/BLAS runtimes; they size pools from the host's core count, which on
# Render's 0.1-CPU instance means dozens of idle threads and their stacks.
# setdefault so an operator's explicit setting still wins.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import asyncio
import ctypes
import gzip
import hashlib
import json
import math
import sys
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Literal

import httpx
import joblib
import numpy as np
import pandas as pd
import sqlite3
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.ratelimit import FixedWindowLimiter, RecentKeys, client_ip

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

# Upstash Redis REST — set these env vars in Render for persistent visit counts.
# If not set, falls back to SQLite (resets on every server restart).
UPSTASH_URL   = os.environ.get("UPSTASH_REDIS_REST_URL", "").rstrip("/")
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")

# How long the *publicly-shown* visit count is cached before refreshing.
# This only governs what users see, so the number doesn't obviously tick up
# when someone refreshes.
PUBLIC_VISITS_TTL_MIN = 10
# One counted visit per client IP per window; the frontend also sends at most
# one POST per browser session.
VISIT_DEDUPE_MIN = 30

# Module-level cache for the displayed visit count
_public_visits_cache: dict = {"count": None, "expires": datetime.min.replace(tzinfo=timezone.utc)}

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

# ── Shared state ──────────────────────────────────────────────────────────────
state: dict = {}

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)

# Single-flight: one shared asyncio.Task per resource. Every caller (the
# refresh loop, a cold-start request, the quantum solve) awaits the same task,
# so two computes of the same thing can never overlap on the 0.1-CPU instance.
_inflight: dict[str, asyncio.Task] = {}


def _consume_task_exception(task: asyncio.Task) -> None:
    # Callers that were cancelled (client disconnect) leave the shared task's
    # exception unobserved; read it here so asyncio doesn't log it as lost.
    if not task.cancelled():
        task.exception()


async def _single_flight(key: str, factory):
    """Run factory() as the one in-flight task for `key`, or join the task
    already running. shield() keeps a cancelled caller from cancelling the
    work other callers are waiting on."""
    task = _inflight.get(key)
    if task is None or task.done():
        task = asyncio.create_task(factory(), name=f"single-flight:{key}")
        task.add_done_callback(_consume_task_exception)
        _inflight[key] = task
    return await asyncio.shield(task)


def _in_flight(key: str) -> bool:
    task = _inflight.get(key)
    return task is not None and not task.done()


def _retry_after(due: datetime, now: datetime) -> str:
    """Retry-After header value (whole seconds, at least 1) until `due`."""
    return str(max(1, int((due - now).total_seconds() + 0.999)))


def _parse_ts(raw) -> datetime | None:
    """ISO timestamp -> aware UTC datetime, or None."""
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _age_min(raw, now: datetime | None = None) -> float | None:
    ts = _parse_ts(raw)
    if ts is None:
        return None
    return ((now or datetime.now(timezone.utc)) - ts).total_seconds() / 60.0


def _malloc_trim() -> None:
    """Hand freed heap pages back to the OS (glibc only). After the model load
    and after each compute this is the difference between RSS settling and
    RSS ratcheting up toward the 512 MB limit."""
    if not sys.platform.startswith("linux"):
        return
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _rss_mb() -> float | None:
    """Current resident set size, cheaply (no subprocess, no compute)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / 2**20, 1)
    except Exception:
        pass
    try:
        import psutil  # dev machines only; not a deploy dependency
        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except Exception:
        return None


# NOAA HMS smoke polygon cache, refreshed by the background loop.
_hms_cache: dict = {"data": None, "expires": _EPOCH}
HMS_CACHE_TTL_MIN = 60  # HMS analysts publish daily and re-draw a few times/day

# Live PurpleAir sensor cache (list of {lat, lon, pm25} for current TX air).
# next_attempt drives the refresh loop's schedule and failure backoff.
_purpleair_cache: dict = {"data": None, "next_attempt": _EPOCH, "failures": 0}

# Open-Meteo grid caches (daily means — refreshing a few times a day is plenty).
# QUOTA: Open-Meteo's free tier is ~10,000 call-weights/day per API, and a
# multi-location call weighs ~1 PER LOCATION. Refetching a 254-cell grid every
# 30-min prediction cycle (48×/day ≈ 12k weights) exhausted the quota and 429'd
# (observed on Render). With a 6h TTL + disk snapshot + stale-on-failure, the
# steady-state cost is a few hundred weights/day per API.
OPENMETEO_GRID_TTL_MIN = int(os.environ.get("OPENMETEO_GRID_TTL_MIN", "360"))
# Both grids hold one UTC day's daily values (forecast_days=1, timezone=UTC),
# the day get_temporal() and HMS use, so they also expire just after the next
# UTC midnight. Stale-on-failure reuse stops after a day: past that the
# training-median fill is a better guess than an old day's weather or smoke.
OPENMETEO_DAY_ROLLOVER_MIN = 5
OPENMETEO_STALE_MAX_H = 24
_wgrid_cache: dict = {"data": None, "expires": _EPOCH}
_aq_cache: dict = {"data": None, "expires": _EPOCH}


def _daily_grid_expiry(fetched: datetime, ttl_min: int = OPENMETEO_GRID_TTL_MIN) -> datetime:
    """The earlier of fetched + TTL and just after the next UTC midnight."""
    fetched = fetched.astimezone(timezone.utc)
    rollover = (datetime.combine(fetched.date() + timedelta(days=1), datetime.min.time(),
                                 tzinfo=timezone.utc)
                + timedelta(minutes=OPENMETEO_DAY_ROLLOVER_MIN))
    return min(fetched + timedelta(minutes=ttl_min), rollover)


def _grid_recent(snap: dict | None, now: datetime) -> bool:
    """A cached daily grid is still worth serving stale (<= a day old)."""
    age = _age_min((snap or {}).get("fetched_at"), now)
    return age is not None and age <= OPENMETEO_STALE_MAX_H * 60

# CURRENT-CONDITIONS grid (display-only): one batched 81-cell call per hour is
# the always-available baseline for every tract/city/overview panel.
CONDITIONS_TTL_MIN = int(os.environ.get("CONDITIONS_TTL_MIN", "60"))
_conditions_cache: dict = {"data": None, "expires": _EPOCH}

# EXACT-location current weather for a CLICKED tract (display only). The grid
# above is coarse (1.0° ≈ a cell center up to ~70km away → a few °F off vs a
# user's local weather); a clicked tract gets an exact single-location fetch
# instead, cached per ~0.1° cell (~11km) and SHARED across users so the unique-
# cell count is bounded by geography, not traffic. Hard guarantees against the
# past failures: (1) any fetch error/429 trips a cooldown and falls back to the
# conditions GRID (location-varying, never a frozen statewide constant);
# (2) the cache is size-capped, above the ~1,500 occupied cells so walking
# every tract cannot keep evicting and refetching.
CLICK_WEATHER_TTL_MIN = int(os.environ.get("CLICK_WEATHER_TTL_MIN", "60"))
CLICK_WEATHER_CACHE_MAX = 2000
_click_weather_cache = OrderedDict()       # "lat,lon" 0.1° key -> {weather, expires}
_click_weather_cooldown = {"until": _EPOCH}


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


def _load_training_sensors() -> list[dict]:
    """[{sensor_id, lat, lon}] from models/training_sensors.json, or [] if the
    file is missing or unreadable."""
    try:
        with open(TRAINING_SENSORS_PATH) as f:
            rows = json.load(f)
        return [{"sensor_id": r.get("sensor_id"), "lat": float(r["lat"]), "lon": float(r["lon"])}
                for r in rows]
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"  could not read {TRAINING_SENSORS_PATH}: {e}")
        return []


def _enrich_tract_lookup_with_distances(df: pd.DataFrame) -> None:
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
    sensors = _load_training_sensors()
    if not sensors:
        print("!" * 70)
        print(f"WARNING: {os.path.relpath(TRAINING_SENSORS_PATH, ROOT)} is missing. "
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


def _save_snapshot(path: str, data: dict | bytes) -> None:
    """Write a cached API response (dict, or already-encoded JSON bytes) to
    disk so a restart on a persistent disk can serve it instantly. Blocking:
    callers on the event loop go through _save_snapshot_async."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        if isinstance(data, bytes):
            with open(tmp, "wb") as f:
                f.write(data)
        else:
            with open(tmp, "w") as f:
                json.dump(data, f, default=str)
        os.replace(tmp, path)
    except Exception as e:
        print(f"Snapshot save failed ({path}): {e}")


async def _save_snapshot_async(path: str, data: dict | bytes) -> None:
    await asyncio.to_thread(_save_snapshot, path, data)


def _load_snapshot(path: str) -> dict | None:
    """Load a previously persisted API response (returns None if missing/corrupt)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Snapshot load failed ({path}): {e}")
        return None


# Pre-gzipped statewide tract GeoJSON (~24.6 MB raw → ~3 MB gz), built once and
# held in memory so /api/texas/tracts/geojson costs zero CPU per request.
_texas_geojson_gz: dict = {"bytes": None, "mtime": None}


async def _get_texas_geojson_gz() -> bytes | None:
    """Return gzip bytes of the statewide GeoJSON, (re)building if the source
    file is new or changed. Compression runs in a thread (~0.5 s, once)."""
    try:
        mtime = os.path.getmtime(TEXAS_GEOJSON_PATH)
        if _texas_geojson_gz["bytes"] is None or _texas_geojson_gz["mtime"] != mtime:
            import gzip as _gzip

            def _build() -> bytes:
                with open(TEXAS_GEOJSON_PATH, "rb") as f:
                    return _gzip.compress(f.read(), compresslevel=6)

            _texas_geojson_gz["bytes"] = await asyncio.to_thread(_build)
            _texas_geojson_gz["mtime"] = mtime
            print(f"GeoJSON gz cache built: {len(_texas_geojson_gz['bytes'])/1e6:.1f} MB "
                  f"(from {os.path.getsize(TEXAS_GEOJSON_PATH)/1e6:.1f} MB)")
        return _texas_geojson_gz["bytes"]
    except Exception as e:
        print(f"GeoJSON gz cache failed: {e}")
        return None


def _load_bundle() -> dict | None:
    """Load the model bundle, drop zero-weight models and cap thread pools.
    Any failure returns None so the API still starts: endpoints that need the
    model answer 503 and /api/health reports degraded, instead of the process
    dying before it binds the port (which Render shows only as a failed deploy)."""
    if not os.path.exists(MODEL_PATH):
        print("WARNING: models/ensemble.joblib not found. Run pipeline/03_train_enhanced.py")
        return None
    try:
        # No mmap_mode: the bundle is compressed, and joblib ignores mmap for
        # compressed files (it only emitted a warning).
        b = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"ERROR: failed to load {MODEL_PATH}: {type(e).__name__}: {e}")
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


def _hydrate_from_snapshots() -> None:
    """Restore caches from disk snapshots, honouring their real age. Render's
    disk is ephemeral so this mostly helps local dev; it must never present
    old live data as current."""
    global _quantum_cache, _hms_cache, _purpleair_cache, _wgrid_cache, _aq_cache, _conditions_cache
    now = datetime.now(timezone.utc)

    texas_snap = _load_snapshot(TEXAS_SNAPSHOT_PATH)
    age = _age_min((texas_snap or {}).get("generated_at"), now)
    if texas_snap and texas_snap.get("tracts") and age is not None and age < TEXAS_SNAPSHOT_MAX_AGE_MIN:
        try:
            _set_texas_result(texas_snap, _encode_payload(texas_snap))
            print(f"Loaded Texas predictions snapshot ({len(texas_snap['tracts'])} tracts, "
                  f"{age:.0f} min old); serving it until the first compute finishes")
        except Exception as e:
            print(f"Ignoring Texas predictions snapshot ({type(e).__name__}: {e})")
    elif texas_snap:
        print("Ignoring Texas predictions snapshot (too old or unreadable)")

    quantum_snap = _load_snapshot(QUANTUM_SNAPSHOT_PATH)
    q_ts = _parse_ts((quantum_snap or {}).get("generated_at"))
    if quantum_snap and q_ts:
        _quantum_cache = {"data": quantum_snap,
                          "expires": q_ts + timedelta(minutes=QUANTUM_CACHE_TTL_MIN)}
        print("Loaded quantum sensor-placement snapshot")

    # HMS: only a snapshot whose data_date is still today or yesterday (UTC).
    # The old hydrate served a June snapshot as live smoke in October.
    from backend.hms import hms_currency
    hms_snap = _load_snapshot(HMS_SNAPSHOT_PATH)
    h_ts = _parse_ts((hms_snap or {}).get("fetched_at"))
    if hms_snap and h_ts and hms_currency(hms_snap, now)[0]:
        _hms_cache = {"data": hms_snap, "expires": h_ts + timedelta(minutes=HMS_CACHE_TTL_MIN)}
        print(f"Loaded HMS snapshot ({hms_snap.get('count', 0)} polygons, "
              f"data_date={hms_snap.get('data_date')})")
    elif hms_snap:
        print(f"Ignoring HMS snapshot (data_date={hms_snap.get('data_date')} is not "
              f"today or yesterday UTC)")

    # PurpleAir: loaded with its real fetch time. _live_purpleair() refuses it
    # once it is older than PURPLEAIR_MAX_AGE_MIN, and the loop skips the
    # startup fetch only while it is younger than the TTL, so a dev server on
    # --reload doesn't bill a full call per file save.
    pa_snap = _load_snapshot(PURPLEAIR_SNAPSHOT_PATH)
    p_ts = _parse_ts((pa_snap or {}).get("fetched_at"))
    if pa_snap and pa_snap.get("usable") and p_ts:
        _purpleair_cache = {"data": pa_snap, "failures": 0,
                            "next_attempt": p_ts + timedelta(minutes=PURPLEAIR_CACHE_TTL_MIN)}
        age = _age_min(pa_snap["fetched_at"], now)
        print(f"Loaded PurpleAir snapshot ({pa_snap.get('count', 0)} live sensors, "
              f"{age:.0f} min old" + ("; too old to use as live)" if age > PURPLEAIR_MAX_AGE_MIN else ")"))

    # Open-Meteo snapshots carry their real expiry; an expired one is still
    # the stale-on-failure fallback if the quota is exhausted after a redeploy.
    for path, label, ttl in ((WEATHER_GRID_SNAPSHOT_PATH, "weather-grid", OPENMETEO_GRID_TTL_MIN),
                             (AQ_SNAPSHOT_PATH, "air-quality", OPENMETEO_GRID_TTL_MIN),
                             (CONDITIONS_SNAPSHOT_PATH, "conditions", CONDITIONS_TTL_MIN)):
        snap = _load_snapshot(path)
        if not snap:
            continue
        ts = _parse_ts(snap.get("fetched_at"))
        if ts is None:
            expires = _EPOCH
        elif path == CONDITIONS_SNAPSHOT_PATH:
            expires = ts + timedelta(minutes=ttl)
        else:
            expires = _daily_grid_expiry(ts, ttl)
        cache = {"data": snap, "expires": expires}
        if path == WEATHER_GRID_SNAPSHOT_PATH:
            _wgrid_cache = cache
        elif path == AQ_SNAPSHOT_PATH:
            _aq_cache = cache
        else:
            _conditions_cache = cache
        print(f"Loaded {label} snapshot ({snap.get('n_cells', 0)} cells)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model in a thread so the event loop stays responsive
    state["bundle"] = await asyncio.to_thread(_load_bundle)
    if state["bundle"] is not None:
        print(f"Model loaded. Features: {state['bundle']['feature_names']}")
    _malloc_trim()

    # Load tract lookup in a thread
    if os.path.exists(LOOKUP_PATH):
        df = await asyncio.to_thread(pd.read_parquet, LOOKUP_PATH)
        df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)
        # Enrich with spatial-context features the model expects. These are
        # static once sensor locations are known, so we compute them once at
        # startup rather than per-prediction.
        await asyncio.to_thread(_enrich_tract_lookup_with_distances, df)
        state["tract_lookup"] = df
        state["geoid_set"] = frozenset(df["GEOID"])
        print(f"Tract lookup loaded: {len(df)} total tracts")
        for city, config in CITIES.items():
            count = df["GEOID"].str.startswith(config["fips"]).sum()
            print(f"  {city}: {count} tracts")
    else:
        state["tract_lookup"] = None
        state["geoid_set"] = frozenset()
        print("WARNING: backend/static/tract_lookup.parquet not found. Run pipeline/01_build_tract_lookup.py")

    state["cache_texas"] = {"data": None, "expires": _EPOCH}
    state["cache_texas_by_geoid"] = {}
    state["texas_payload"] = None
    state["city_cache"] = {}
    state["texas_first_attempt"] = asyncio.Event()
    try:
        await asyncio.to_thread(_hydrate_from_snapshots)
    except Exception as e:
        # A corrupt snapshot must never keep the API from starting.
        print(f"Snapshot hydrate failed ({type(e).__name__}: {e}); starting cold")

    if _purpleair_key_configured():
        print(f"PurpleAir: using PURPLEAIR_API_KEY from environment; refreshing "
              f"every {PURPLEAIR_CACHE_TTL_MIN} min.")
    else:
        print("PurpleAir: PURPLEAIR_API_KEY not set. Live neighbor features are "
              "DISABLED and predictions use climatology. Set the key in the "
              "Render dashboard to enable them.")

    state["refresh_task"] = asyncio.create_task(_refresh_loop(), name="refresh-loop")

    # Keepalive loop: pings an external URL every ~14 min so Render's free tier
    # never spins us down. Internal health() calls do NOT count as external traffic
    # to Render — we must actually hit the public URL.
    async def _keepalive_loop():
        # Prefer explicit override, otherwise auto-detect Render's public URL.
        url = (
            os.environ.get("KEEPALIVE_URL")
            or os.environ.get("RENDER_EXTERNAL_URL")
        )
        if url:
            url = url.rstrip("/") + "/api/health"
        try:
            interval = int(os.environ.get("KEEPALIVE_INTERVAL", "840"))  # 14 min
        except Exception:
            interval = 840

        if not url:
            print("Keepalive: no KEEPALIVE_URL or RENDER_EXTERNAL_URL set — disabled. "
                  "Cold starts will happen after 15 min of inactivity.")
            return

        print(f"Keepalive: will self-ping {url} every {interval}s")
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                try:
                    try:
                        resp = await client.get(url)
                        print(f"Keepalive: pinged {url} status={resp.status_code}")
                    except Exception as e:
                        print(f"Keepalive HTTP error pinging {url}: {e}")
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"Keepalive loop error: {e}")
                    await asyncio.sleep(60)

    state["keepalive_task"] = asyncio.create_task(_keepalive_loop(), name="keepalive")

    # SQLite visit counter for local dev (and the fallback when Upstash is not
    # configured at all).
    if not (UPSTASH_URL and UPSTASH_TOKEN):
        try:
            def _init_visits_db():
                conn = sqlite3.connect(VISITS_DB, timeout=5)
                cur = conn.cursor()
                cur.execute("CREATE TABLE IF NOT EXISTS visits (id INTEGER PRIMARY KEY CHECK (id = 1), count INTEGER)")
                cur.execute("INSERT OR IGNORE INTO visits (id, count) VALUES (1, 0)")
                conn.commit()
                conn.close()
            await asyncio.to_thread(_init_visits_db)
            print(f"Visits DB initialized at {VISITS_DB}")
        except Exception as e:
            print(f"Failed to initialize visits DB: {e}")

    # Pre-warm the gzipped statewide GeoJSON so even the first visitor after a
    # restart gets the ~3 MB compressed payload instantly.
    if os.path.exists(TEXAS_GEOJSON_PATH):
        asyncio.create_task(_get_texas_geojson_gz())

    yield

    for key in ("refresh_task", "keepalive_task"):
        task = state.get(key)
        if task is not None:
            task.cancel()
    if _http_client["client"] is not None:
        await _http_client["client"].aclose()


app = FastAPI(title="Shared Skies API", lifespan=lifespan)

# CORS. This is a PUBLIC, read-only data API that the map frontend (Vercel) and
# any embedder fetch cross-origin, so the default stays permissive ("*"). To lock
# it down WITHOUT a code change, set ALLOWED_ORIGINS in the environment to a
# comma-separated allowlist, e.g.
#   ALLOWED_ORIGINS=https://sharedskiesinitiative.org,https://shared-skies-initiative.vercel.app
# NOTE: CORS is a browser-enforced READ guard — it does NOT stop a script/curl
# from POSTing /api/visit, so it is not what protects the visit counter (the
# per-IP dedupe in record_visit does that). Methods are limited to the verbs the
# API actually serves.
_origins_env = os.environ.get("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

# Gzip every sizable response. compresslevel=6 is the speed/ratio sweet spot;
# level 9 costs ~3x CPU for ~1% smaller output. The two payloads that dominate
# map load, /api/texas/predictions and the statewide GeoJSON, are compressed
# once and served with Content-Encoding already set, which this middleware
# passes through untouched, so they cost no per-request CPU.
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Utilities ─────────────────────────────────────────────────────────────────

# Weather fallback = exact TRAINING MEDIANS (and training units: °C, %, surface
# hPa, m/s, mm). The old fallback {72, 55, 1013, 8} was °F/MSL-shaped — values
# the model never saw in training (temperature median is 20.2 °C; pressure is
# SURFACE pressure ~991 hPa, not sea-level 1013).
WEATHER_FALLBACK = {
    "temperature": 20.2, "humidity": 65.0, "pressure": 991.3,
    "wind_speed": 5.5, "precipitation": 0.0,
}


DISPLAY_WEATHER_FALLBACK = {
    "temperature": round(WEATHER_FALLBACK["temperature"] * 9 / 5 + 32, 1),  # 68.4 °F
    "humidity":    WEATHER_FALLBACK["humidity"],
    "pressure":    1013.0,                                                  # MSL-style
    "wind_speed":  round(WEATHER_FALLBACK["wind_speed"] * 2.23694, 1),      # 12.3 mph
}


def _weather_fallback() -> dict:
    """Model-unit fallback weather (training medians) WITH a converted display
    dict. Display fallbacks must NEVER expose bare model units — that once
    rendered the 20.2 °C training median as '20 °F' on every tract."""
    w = dict(WEATHER_FALLBACK)
    w["display"] = dict(DISPLAY_WEATHER_FALLBACK)
    return w


async def _fetch_conditions_grid() -> dict | None:
    """One conditions-grid fetch (run single-flight by _get_conditions_grid)."""
    global _conditions_cache
    now = datetime.now(timezone.utc)
    cached = _conditions_cache.get("data")
    lookup = state.get("tract_lookup")
    if lookup is None:
        return cached if cached and cached.get("usable") else None
    try:
        from backend.airquality import fetch_conditions_grid
        snap = await fetch_conditions_grid(lookup["lat"].values, lookup["lon"].values)
    except Exception as e:
        print(f"[conditions] fetch raised: {e}")
        snap = None
    if snap and snap.get("usable"):
        _conditions_cache = {"data": snap, "expires": now + timedelta(minutes=CONDITIONS_TTL_MIN)}
        await _save_snapshot_async(CONDITIONS_SNAPSHOT_PATH, snap)
        return snap
    if cached and cached.get("usable"):
        print("[conditions] refetch failed — serving stale conditions (backoff 15 min)")
        _conditions_cache["expires"] = now + timedelta(minutes=15)
        return cached
    return None


async def _get_conditions_grid(allow_fetch: bool = True) -> dict | None:
    """Cached CURRENT-CONDITIONS cell grid (display units, TTL 60 min + disk
    snapshot, stale-on-failure with 15-min backoff). The ONLY source of
    'Current Conditions' shown in the UI. The refresh loop keeps it current
    through each Texas compute; request handlers pass allow_fetch=False so
    traffic never triggers the fetch."""
    cached = _conditions_cache.get("data")
    if cached and cached.get("usable") and (
            not allow_fetch or datetime.now(timezone.utc) < _conditions_cache["expires"]):
        return cached
    if not allow_fetch:
        return None
    return await _single_flight("conditions", _fetch_conditions_grid)


async def get_display_weather(lat: float, lon: float, allow_fetch: bool = True) -> dict:
    """Current conditions for a location (display units), from the cached cell
    grid. Falls back: exact cell -> nearest cell -> static converted medians.
    Never calls the weather API directly for a single location."""
    grid = await _get_conditions_grid(allow_fetch)
    if grid and grid.get("by_key"):
        from backend.airquality import conditions_cell_key
        by_key = grid["by_key"]
        v = by_key.get(conditions_cell_key(lat, lon))
        if v is None:
            # Nearest occupied cell (e.g. coastal/border rounding) — ~81 keys.
            try:
                best, best_d = None, float("inf")
                for k, vals in by_key.items():
                    kla, klo = (float(x) for x in k.split(","))
                    d = (kla - lat) ** 2 + (klo - lon) ** 2
                    if d < best_d:
                        best, best_d = vals, d
                v = best
            except Exception:
                v = None
        if v is not None:
            return dict(v)
    return dict(DISPLAY_WEATHER_FALLBACK)


async def get_click_weather(lat: float, lon: float) -> dict:
    """EXACT-location current weather (display units) for a clicked tract, with
    a per-0.1°-cell shared cache. On cooldown or any failure, falls back to the
    coarse conditions grid (location-varying) — so a quota outage degrades to
    'slightly coarse' instead of the old 'frozen statewide constant'."""
    key = f"{round(float(lat), 1):.1f},{round(float(lon), 1):.1f}"
    now = datetime.now(timezone.utc)
    hit = _click_weather_cache.get(key)
    if hit:
        _click_weather_cache.move_to_end(key)
        if now < hit["expires"]:
            return hit["weather"]
    if now < _click_weather_cooldown["until"]:
        return await get_display_weather(lat, lon, allow_fetch=False)  # API recently failed
    if not _click_weather_budget.check(_GLOBAL)[0]:
        return await get_display_weather(lat, lon, allow_fetch=False)  # hourly budget spent
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": "UTC",
            })
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        cur = r.json().get("current", {})
        if cur.get("temperature_2m") is None:
            raise RuntimeError("no current data")
        weather = {
            "temperature": round(float(cur["temperature_2m"]), 1),
            "humidity":    round(float(cur.get("relative_humidity_2m") or 0), 1),
            "pressure":    round(float(cur.get("pressure_msl") or 1013.0), 1),
            "wind_speed":  round(float(cur.get("wind_speed_10m") or 0), 1),
        }
        _click_weather_cache[key] = {"weather": weather, "expires": now + timedelta(minutes=CLICK_WEATHER_TTL_MIN)}
        if len(_click_weather_cache) > CLICK_WEATHER_CACHE_MAX:
            _click_weather_cache.popitem(last=False)
        return weather
    except Exception as e:
        # Trip a 30-min cooldown so we don't hammer a quota-exhausted API on
        # every click; serve the grid (still varies by location) meanwhile.
        _click_weather_cooldown["until"] = now + timedelta(minutes=30)
        print(f"[click-weather] exact fetch failed ({e}); cooldown 30m, using conditions grid")
        return await get_display_weather(lat, lon, allow_fetch=False)


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


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color. Uses round-half-up (int(c+0.5)) — matching
    JS Math.round for non-negative values — so the backend's per-tract `color`
    is BYTE-IDENTICAL to the frontend legend's pm25Color (avoids Python's
    banker's rounding diverging by 1 LSB at exact .5 midpoints)."""
    return '#{:02x}{:02x}{:02x}'.format(
        max(0, min(255, int(rgb[0] + 0.5))),
        max(0, min(255, int(rgb[1] + 0.5))),
        max(0, min(255, int(rgb[2] + 0.5))),
    )


def interpolate_color(color1, color2, factor):
    """Interpolate between two colors based on factor (0-1)."""
    factor = max(0, min(1, factor))
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)

    rgb = tuple(rgb1[i] + (rgb2[i] - rgb1[i]) * factor for i in range(3))
    return rgb_to_hex(rgb)


def pm25_color_gradient(pm25: float) -> str:
    """Get gradient color based on PM2.5 value.

    Bands: Good 0-9 (green), Moderate 9-13 (yellow), Elevated 13-17 (orange),
    High 17+ (red→dark red). Each band darkens toward its upper boundary, and
    the High band keeps darkening with concentration so wildfire-smoke days read
    dramatically dark. 9 = U.S. EPA annual NAAQS (2024).
    """
    if pm25 < 0:
        pm25 = 0

    # Green: 0.0-9.0  (within the EPA annual standard)
    if pm25 <= 9.0:
        factor = pm25 / 9.0
        return interpolate_color("#90EE90", "#00b894", factor)

    # Yellow: 9.0-13.0
    elif pm25 <= 13.0:
        factor = (pm25 - 9.0) / 4.0
        return interpolate_color("#FFFF99", "#FFD700", factor)

    # Orange: 13.0-17.0
    elif pm25 <= 17.0:
        factor = (pm25 - 13.0) / 4.0
        return interpolate_color("#FFB347", "#E8590C", factor)

    # Red → dark red: 17.0+  (darkens with concentration; saturates ~55, the EPA
    # 24-hr Unhealthy threshold, so smoke/dust days read dramatically dark).
    else:
        factor = min(1.0, (pm25 - 17.0) / 38.0)
        return interpolate_color("#FF6B6B", "#800000", factor)


# U.S. EPA PM2.5 AQI breakpoints (May 2024 revision). Used to show an
# AQI-equivalent next to our µg/m³ so users can compare directly with apps that
# display AQI (PurpleAir map, AirNow). (C_lo, C_hi, AQI_lo, AQI_hi)
_EPA_AQI_BREAKPOINTS = [
    (0.0,   9.0,   0,   50),
    (9.1,   35.4,  51,  100),
    (35.5,  55.4,  101, 150),
    (55.5,  125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
]


def pm25_to_epa_aqi(pm25: float) -> int:
    """Convert PM2.5 (µg/m³) to the U.S. EPA AQI (2024 breakpoints)."""
    c = max(0.0, math.floor(float(pm25) * 10) / 10)  # EPA truncates to 0.1
    for c_lo, c_hi, a_lo, a_hi in _EPA_AQI_BREAKPOINTS:
        if c <= c_hi:
            c_lo_eff = c_lo if c >= c_lo else 0.0
            return int(round((a_hi - a_lo) / (c_hi - c_lo_eff) * (c - c_lo_eff) + a_lo))
    return 500


def pm25_info(pm25: float) -> dict:
    """Get category info. Bands: Good 0-9 / Moderate 9-13 / Elevated 13-17 /
    High 17+. 9 µg/m³ = U.S. EPA annual NAAQS; 15 µg/m³ = WHO 24-hr guideline."""
    color = pm25_color_gradient(pm25)

    if pm25 <= 9.0:
        return {
            "category": "Good",
            "color": color,
            "aqi_range": "0–9",
            "health_msg": "Air quality is good — within the U.S. EPA annual PM2.5 standard (9 µg/m³).",
        }
    elif pm25 <= 13.0:
        return {
            "category": "Moderate",
            "color": color,
            "aqi_range": "9–13",
            "health_msg": "Moderate — above the EPA annual standard. Unusually sensitive people may want to limit prolonged outdoor exertion.",
        }
    elif pm25 <= 17.0:
        return {
            "category": "Elevated",
            "color": color,
            "aqi_range": "13–17",
            "health_msg": "Elevated — above the WHO 24-hour guideline (15 µg/m³). Sensitive groups should limit prolonged outdoor activity.",
        }
    else:
        return {
            "category": "High",
            "color": color,
            "aqi_range": "17+",
            "health_msg": "⚠️ High — everyone may begin to feel effects; sensitive groups are at greater risk. Often driven by wildfire smoke or dust.",
        }


def _safe_float(val) -> float:
    """Convert to float, handle NaN and None."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if np.isnan(f) else round(f, 1)
    except (TypeError, ValueError):
        return 0.0


async def get_city_predictions(city: str):
    """PM2.5 predictions for one county: a slice of the current statewide
    result, cached per Texas generated_at so it can never lag the map."""
    if city not in CITIES:
        raise HTTPException(404, f"Unknown city: {city}. Available: {', '.join(CITIES.keys())}")

    texas = await _current_texas()
    generated_at = texas.get("generated_at")
    hit = state["city_cache"].get(city)
    if hit is not None and hit["generated_at"] == generated_at:
        return hit["data"]

    # CONSISTENCY: slice the texas-wide batch (live neighbors + HMS + AOD +
    # gridded weather) instead of recomputing here, so city views always
    # match the statewide map.
    city_config = CITIES[city]
    fips = city_config["fips"]
    tracts = [t for t in texas["tracts"] if str(t["geoid"]).startswith(fips)]
    if not tracts:
        raise HTTPException(404, f"No tracts found for city: {city}")

    # City-center CURRENT conditions (display only, from the cached cell grid —
    # no direct API call).
    weather_display = await get_display_weather(*city_config["center"], allow_fetch=False)

    avg_pm25 = round(float(np.mean([t["pm25"] for t in tracts])), 2)

    result = {
        "city":         city,
        "display_name": city_config["display_name"],
        "generated_at": generated_at,
        "expires_at":   texas.get("expires_at"),
        "weather":      weather_display,
        "avg_pm25":     avg_pm25,
        "avg_epa_aqi":  pm25_to_epa_aqi(avg_pm25),
        "avg_info":     pm25_info(avg_pm25),
        "tract_count":  len(tracts),
        "data_sources": texas.get("data_sources", {}),
        "tracts":       tracts,
    }

    state["city_cache"][city] = {"generated_at": generated_at, "data": result}
    return result


# ── Persistent visits counter helpers ────────────────────────────────────────
# Priority: Upstash Redis (truly persistent across restarts) > SQLite (ephemeral)

def _get_visit_count_sync():
    conn = sqlite3.connect(VISITS_DB, timeout=5)
    cur = conn.cursor()
    cur.execute("SELECT count FROM visits WHERE id=1")
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else 0


def _inc_visit_sync():
    conn = sqlite3.connect(VISITS_DB, timeout=5)
    cur = conn.cursor()
    cur.execute("UPDATE visits SET count = count + 1 WHERE id = 1")
    conn.commit()
    cur.execute("SELECT count FROM visits WHERE id=1")
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else 0


# One pooled client for the small per-request upstream calls (Upstash,
# geocoding), so each call doesn't pay a fresh TLS handshake.
_http_client: dict = {"client": None}


def _shared_http() -> httpx.AsyncClient:
    if _http_client["client"] is None:
        _http_client["client"] = httpx.AsyncClient(timeout=8.0)
    return _http_client["client"]


async def _upstash_incr() -> int:
    """Atomically increment the visit counter in Upstash Redis and return the new value."""
    resp = await _shared_http().post(
        f"{UPSTASH_URL}/incr/shared_skies_visits",
        headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
        timeout=5.0,
    )
    resp.raise_for_status()
    return int(resp.json()["result"])


async def _upstash_get() -> int:
    """Get the current visit count from Upstash Redis."""
    resp = await _shared_http().get(
        f"{UPSTASH_URL}/get/shared_skies_visits",
        headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
        timeout=5.0,
    )
    resp.raise_for_status()
    val = resp.json().get("result")
    return int(val) if val is not None else 0


async def _get_public_visit_count() -> int:
    """Return the 10-min-cached visit count shown to users.
    The stored count moves on every counted /api/visit; this layer only
    controls what's displayed so the number can't be watched ticking up.
    All clients in the same 10-min window see the same value."""
    global _public_visits_cache
    now = datetime.now(timezone.utc)
    cached_count = _public_visits_cache.get("count")
    expires_at = _public_visits_cache.get("expires")

    if cached_count is not None and expires_at is not None and now < expires_at:
        return cached_count

    # Cache miss or expired — read the true count and refresh the cache.
    try:
        if UPSTASH_URL and UPSTASH_TOKEN:
            count = await _upstash_get()
        else:
            count = await asyncio.to_thread(_get_visit_count_sync)
    except Exception as e:
        print(f"Public visit cache refresh failed: {e}")
        # If reading fails, keep showing the last good value rather than zero.
        return cached_count if cached_count is not None else 0

    _public_visits_cache = {
        "count": count,
        "expires": now + timedelta(minutes=PUBLIC_VISITS_TTL_MIN),
    }
    return count


# ── Routes ────────────────────────────────────────────────────────────────────

def _purpleair_key_configured() -> bool:
    return bool(os.environ.get("PURPLEAIR_API_KEY", "").strip())


@app.get("/api/health")
async def health():
    """Health + live-data-source status. The data_sources block makes silent
    degradation visible: if PurpleAir/HMS/weather quietly fail, predictions
    fall back to climatology/medians — operators need to SEE that. Reads
    in-memory state only, so it is safe as Render's health check. HTTP status
    is always 200: a degraded map is still worth serving, and a non-200 would
    make Render restart the instance."""
    now = datetime.now(timezone.utc)
    pa = _purpleair_cache.get("data") or {}
    texas = state.get("cache_texas", {}).get("data") or {}
    bundle = state.get("bundle")
    sources = texas.get("data_sources", {})
    live_neighbors = bool(sources.get("using_live_neighbors"))
    predictions_age = _age_min(texas.get("generated_at"), now)
    purpleair_age = _age_min(pa.get("fetched_at"), now) if pa.get("usable") else None
    degraded = (
        bundle is None
        or not live_neighbors
        or predictions_age is None
        or predictions_age > 3 * PURPLEAIR_CACHE_TTL_MIN
    )
    return {
        "status": "degraded" if degraded else "ok",
        "model_loaded": bundle is not None,
        "model_version": (bundle or {}).get("version"),
        # The version string is unchanged across retrains, so it cannot tell you
        # WHICH bundle is live. The feature count can: the EJScreen removal took
        # the model from 38 to 30, so this is the field to check after a deploy.
        "model_features": len((bundle or {}).get("feature_names", [])),
        "lookup_loaded": state.get("tract_lookup") is not None,
        "available_cities": list(CITIES.keys()),
        "live_purpleair_sensors": len(pa.get("sensors", [])) if pa.get("usable") else 0,
        "purpleair_fetched_at": pa.get("fetched_at"),
        "purpleair_key_configured": _purpleair_key_configured(),
        "live_neighbors": live_neighbors,
        "predictions_generated_at": texas.get("generated_at"),
        "predictions_age_min": round(predictions_age, 1) if predictions_age is not None else None,
        "purpleair_age_min": round(purpleair_age, 1) if purpleair_age is not None else None,
        "refresh_interval_min": PURPLEAIR_CACHE_TTL_MIN,
        "memory_rss_mb": _rss_mb(),
        "conditions_fetched_at": (_conditions_cache.get("data") or {}).get("fetched_at"),
        "data_sources": sources,
    }


@app.get("/api/cities")
async def list_cities():
    """List available cities with metadata."""
    return {
        "cities": [
            {
                "id": city_id,
                "name": config["display_name"],
                "center": config["center"],
                "fips": config["fips"]
            }
            for city_id, config in CITIES.items()
        ]
    }


@app.post("/api/visit")
async def record_visit(request: Request):
    """Count a visit at most once per client IP per VISIT_DEDUPE_MIN, then
    return the *publicly-cached* total so users can't watch the number tick up
    by refreshing."""
    if _visit_dedupe.first_time(client_ip(request)) and _visit_budget.check(_GLOBAL)[0]:
        try:
            if UPSTASH_URL and UPSTASH_TOKEN:
                await _upstash_incr()
            else:
                await asyncio.to_thread(_inc_visit_sync)
        except Exception as e:
            # With Upstash configured, a local SQLite write would never be read
            # back (the public count reads Upstash), so the visit is dropped and
            # logged rather than silently diverted.
            print(f"Visit increment failed: {type(e).__name__}: {e}")
    # Return the display-cached count (refreshes every PUBLIC_VISITS_TTL_MIN)
    try:
        return {"visits": await _get_public_visit_count()}
    except Exception:
        return {"visits": 0}


@app.get("/api/metrics")
async def get_metrics():
    """Return the publicly-cached visit count (refreshes every ~10 min).
    For the raw real-time count, read Upstash/SQLite directly."""
    try:
        return {"visits": await _get_public_visit_count()}
    except Exception as e:
        print(f"Metrics read error: {e}")
        return {"visits": 0}


def _grid_matches(snap: dict | None, n: int, schema: int | None = None) -> bool:
    """A cached per-tract grid is reusable only if it was built for the same
    tract list length (the tract lookup is static, so length is a sufficient key)
    and, when given, the same value schema."""
    if not snap or not snap.get("usable"):
        return False
    if schema is not None and snap.get("schema") != schema:
        return False
    feats = snap.get("features") or {}
    arr = next(iter(feats.values()), None)
    return arr is not None and len(arr) == n


async def _fetch_weather_grid(lats, lons) -> dict | None:
    """One weather-grid fetch (run single-flight by _get_weather_grid)."""
    global _wgrid_cache
    from backend.airquality import fetch_weather_grid, WEATHER_GRID_SCHEMA
    now = datetime.now(timezone.utc)
    cached = _wgrid_cache.get("data")
    try:
        snap = await fetch_weather_grid(lats, lons)
    except Exception as e:
        print(f"[weather-grid] fetch raised: {e}")
        snap = None
    if snap and snap.get("usable"):
        _wgrid_cache = {"data": snap, "expires": _daily_grid_expiry(now)}
        await _save_snapshot_async(WEATHER_GRID_SNAPSHOT_PATH, snap)
        return snap
    if _grid_matches(cached, len(lats), WEATHER_GRID_SCHEMA):
        _wgrid_cache["expires"] = now + timedelta(minutes=30)
        if _grid_recent(cached, now):
            print("[weather-grid] refetch failed — serving stale cached grid (backoff 30 min)")
            return cached
        print(f"[weather-grid] refetch failed and the cached grid is over "
              f"{OPENMETEO_STALE_MAX_H} h old — using training-median weather (backoff 30 min)")
    return None


async def _get_weather_grid(lats, lons) -> dict | None:
    """Cached per-tract daily weather grid (TTL 6h, capped at the UTC day, +
    disk snapshot). On refetch failure, serves the stale grid for up to a day
    and backs off 30 min — never hammers a quota-exhausted API every cycle."""
    from backend.airquality import WEATHER_GRID_SCHEMA
    cached = _wgrid_cache.get("data")
    now = datetime.now(timezone.utc)
    if _grid_matches(cached, len(lats), WEATHER_GRID_SCHEMA) and now < _wgrid_cache["expires"]:
        # Inside a failure backoff the cached grid may be too old to use.
        return cached if _grid_recent(cached, now) else None
    return await _single_flight("weather_grid", lambda: _fetch_weather_grid(lats, lons))


async def _fetch_aq_snapshot(lats, lons, include_met: bool) -> dict | None:
    """One CAMS air-quality fetch (run single-flight by _get_aq_snapshot)."""
    global _aq_cache
    now = datetime.now(timezone.utc)
    cached = _aq_cache.get("data")
    try:
        from backend.airquality import fetch_airquality_snapshot
        snap = await fetch_airquality_snapshot(lats, lons, include_met=include_met)
    except Exception as e:
        print(f"[airquality] fetch raised: {e}")
        snap = None
    if snap and snap.get("usable"):
        _aq_cache = {"data": snap, "expires": _daily_grid_expiry(now)}
        await _save_snapshot_async(AQ_SNAPSHOT_PATH, snap)
        return snap
    if _grid_matches(cached, len(lats)):
        _aq_cache["expires"] = now + timedelta(minutes=30)
        if _grid_recent(cached, now):
            print("[airquality] refetch failed — serving stale cached snapshot (backoff 30 min)")
            return cached
        print(f"[airquality] refetch failed and the cached snapshot is over "
              f"{OPENMETEO_STALE_MAX_H} h old — using training-median fill (backoff 30 min)")
    return None


async def _get_aq_snapshot(lats, lons, include_met: bool) -> dict | None:
    """Cached per-tract CAMS air-quality snapshot (TTL 6h, capped at the UTC
    day, + disk snapshot, stale-on-failure). Was refetched EVERY 30-min cycle before — the main
    Open-Meteo quota burner."""
    cached = _aq_cache.get("data")
    now = datetime.now(timezone.utc)
    if _grid_matches(cached, len(lats)) and now < _aq_cache["expires"]:
        return cached if _grid_recent(cached, now) else None
    return await _single_flight("airquality", lambda: _fetch_aq_snapshot(lats, lons, include_met))


def _live_purpleair(now: datetime | None = None) -> dict | None:
    """The cached PurpleAir snapshot when it is usable and young enough to
    count as live (PURPLEAIR_MAX_AGE_MIN), else None (climatology fallback)."""
    snap = _purpleair_cache.get("data")
    if not snap or not snap.get("usable"):
        return None
    age = _age_min(snap.get("fetched_at"), now)
    if age is None or age > PURPLEAIR_MAX_AGE_MIN:
        return None
    return snap


# Per-tract keys copied from the lookup into every tract dict (EJScreen
# context for the UI and the sensor-placement solver, never model inputs).
_TRACT_CONTEXT_COLS = (
    "ejf_score", "pct_people_of_color", "pct_low_income", "traffic_proximity",
    "superfund_proximity", "diesel_pm_proximity", "pct_ling_isolated",
)


def _build_tracts(lookup: pd.DataFrame, pm25_array: np.ndarray) -> list[dict]:
    """Per-tract response dicts, in lookup order. Column lists instead of
    iterrows: same values, a fraction of the time."""
    n = len(lookup)
    geoids = lookup["GEOID"].tolist()
    lats = lookup["lat"].tolist()
    lons = lookup["lon"].tolist()
    ctx = {c: (lookup[c].tolist() if c in lookup.columns else [None] * n)
           for c in _TRACT_CONTEXT_COLS}
    counties = lookup["CNTY_NAME"].tolist() if "CNTY_NAME" in lookup.columns else ["Texas"] * n
    tracts = []
    for i in range(n):
        pm25 = round(float(pm25_array[i]), 2)
        info = pm25_info(pm25)
        tracts.append({
            "geoid":               geoids[i],
            "lat":                 round(float(lats[i]), 6),
            "lon":                 round(float(lons[i]), 6),
            "pm25":                pm25,
            "epa_aqi":             pm25_to_epa_aqi(pm25),
            "category":            info["category"],
            "color":               info["color"],
            "aqi_range":           info["aqi_range"],
            "health_msg":          info["health_msg"],
            **{c: _safe_float(ctx[c][i]) for c in _TRACT_CONTEXT_COLS},
            "county":              str(counties[i]),
        })
    return tracts


def _encode_payload(result: dict) -> dict:
    """Serialize a predictions result once: raw JSON, gzip, and an ETag.
    Same encoding FastAPI's JSONResponse uses, so clients see identical JSON;
    allow_nan=False turns a stray NaN into a failed compute (the previous
    result stays live) instead of invalid JSON on the wire."""
    raw = json.dumps(result, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
    return {
        "raw": raw,
        "gz": gzip.compress(raw, compresslevel=6),
        "etag": '"' + hashlib.sha1(raw).hexdigest()[:20] + '"',
    }


def _set_texas_result(result: dict, payload: dict) -> None:
    """Install a statewide result everywhere it is read from. The by-GEOID
    index and the encoded bytes are rebuilt every time, so /api/tract and the
    map can never disagree (the old snapshot hydrate skipped the index)."""
    generated = _parse_ts(result.get("generated_at")) or datetime.now(timezone.utc)
    state["cache_texas"] = {"data": result,
                            "expires": generated + timedelta(minutes=PURPLEAIR_CACHE_TTL_MIN)}
    state["cache_texas_by_geoid"] = {t["geoid"]: t for t in result.get("tracts", [])}
    state["texas_payload"] = payload
    state["city_cache"] = {}


def _predict_and_package(lookup_reset: pd.DataFrame, weather: dict, temporal: dict,
                         header: dict) -> tuple[dict, dict]:
    """Worker-thread half of a compute: model predict, tract dicts, JSON +
    gzip encoding. Nothing here touches the event loop."""
    pm25_array = run_predictions_batch(lookup_reset, weather, temporal)
    tracts = _build_tracts(lookup_reset, pm25_array)
    avg_pm25 = round(float(np.mean([t["pm25"] for t in tracts])), 2)
    result = {
        **header,
        "avg_pm25":     avg_pm25,
        "avg_epa_aqi":  pm25_to_epa_aqi(avg_pm25),
        "avg_info":     pm25_info(avg_pm25),
        "tract_count":  len(tracts),
    }
    # Keep the public key order: data_sources/value_semantics/tracts last.
    for k in ("data_sources", "value_semantics"):
        result[k] = result.pop(k)
    result["tracts"] = tracts
    return result, _encode_payload(result)


async def _compute_texas_predictions() -> dict:
    """Heavy path: actually compute predictions for every TX tract, update
    cache + snapshot. Only ever run through _refresh_texas (single-flight)."""
    if state.get("bundle") is None:
        raise HTTPException(503, "Model not loaded.")
    if state.get("tract_lookup") is None:
        raise HTTPException(503, "Tract lookup not loaded.")

    from backend.hms import hms_currency

    lookup = state["tract_lookup"]
    now = datetime.now(timezone.utc)
    t0 = time.monotonic()

    print(f"Generating predictions for all {len(lookup)} Texas tracts (vectorized)...")
    temporal = get_temporal(now)

    # Work on a COPY so the shared lookup is never half-updated.
    lookup_reset = lookup.reset_index(drop=True)

    # ── Per-tract DAILY weather (training-consistent, CACHED 6h) ────────────
    # Training weather is per-sensor-location daily aggregates (°C, %, surface
    # hPa, daily-max m/s, mm). One batched Open-Meteo call over a coarse grid,
    # cached with a 6h TTL + disk snapshot (daily values don't change every
    # cycle, and refetching per cycle exhausted the free API quota).
    weather_grid_cells = 0
    weather = _weather_fallback()  # model-unit statewide backfill scalars
    wsnap = await _get_weather_grid(
        lookup_reset["lat"].values, lookup_reset["lon"].values)
    if wsnap is not None:
        lookup_reset = lookup_reset.copy()
        for f in ("temperature", "humidity", "pressure", "wind_speed", "precipitation"):
            vals = wsnap["features"].get(f)
            if vals is not None:
                # None -> NaN; run_predictions_batch backfills with the
                # statewide scalar, then the training median.
                lookup_reset[f] = pd.Series(vals, dtype="float64").values
                # Statewide backfill scalar = today's median across the grid
                # (model units) — better than the all-time training median.
                med = float(np.nanmedian(lookup_reset[f].values))
                if np.isfinite(med):
                    weather[f] = med
        # Per-tract interactions (must mirror training formulas exactly).
        lookup_reset["temp_x_humidity"] = (
            lookup_reset["temperature"] * lookup_reset["humidity"] / 100.0)
        lookup_reset["wind_x_temp"] = (
            lookup_reset["wind_speed"] * lookup_reset["temperature"] / 100.0)
        weather_grid_cells = int(wsnap.get("n_cells", 0))
        print(f"Gridded daily weather applied ({weather_grid_cells} cells): "
              f"T median={lookup_reset['temperature'].median():.1f}°C, "
              f"daily-max wind median={lookup_reset['wind_speed'].median():.1f} m/s")
    else:
        print("Weather grid unavailable — training-median weather backfill in effect.")

    # ── Live neighbor-feature recompute (the fix for scrambled predictions) ──
    # The multi-radius neighbor PM features (~dominant model signal) MUST reflect
    # same-day air to match training. Recompute them from LIVE PurpleAir sensors
    # here, every prediction cycle. If live data is unavailable (no key / API
    # down / too old), we leave the static climatological columns from
    # _enrich_tract_lookup_with_distances in place — stable and non-scrambled.
    live_sensor_count = 0
    pa_snap = _live_purpleair(now)
    try:
        if pa_snap is not None:
            from backend.purpleair import compute_neighbor_features
            live_sensors = pa_snap["sensors"]
            nbr_feats = await asyncio.to_thread(
                compute_neighbor_features,
                lookup_reset["lat"].values, lookup_reset["lon"].values, live_sensors,
            )
            if nbr_feats is not None:
                lookup_reset = lookup_reset.copy()
                for fname, arr in nbr_feats.items():
                    lookup_reset[fname] = arr
                live_sensor_count = len(live_sensors)
                nbr_mean = nbr_feats["nbr_pm25_50km"]
                nbr_count = nbr_feats["nbr_count_50km"]
                print(f"Live multi-radius neighbor features applied: {live_sensor_count} sensors "
                      f"({_age_min(pa_snap['fetched_at'], now):.0f} min old), "
                      f"nbr_pm25_50km mean={float(np.mean(nbr_mean)):.2f}, "
                      f"50km coverage={(nbr_count>0).sum()/len(nbr_count)*100:.1f}%")
        else:
            print("No live PurpleAir data — using static climatological neighbor features.")
    except Exception as e:
        print(f"Live neighbor recompute failed ({e}); using static neighbor features.")

    # ── Live HMS smoke feature (hms_smoke) ──────────────────────────────────
    # Compute the SAME point-in-polygon smoke density the model trained on, for
    # every tract centroid, using today's HMS polygons (yesterday's as a flagged
    # proxy before today's is published, never older). Uses the shared
    # backend.hms helpers so train/inference are byte-identical. Tracts under no
    # smoke polygon get 0 (the training fill). Only runs if the model uses it.
    features_needed = set(state["bundle"].get("feature_names", []))
    hms_data = _hms_cache.get("data") or {}
    hms_usable, hms_is_proxy = hms_currency(hms_data, now)
    if not hms_usable:
        hms_data = {}
    if "hms_smoke" in features_needed:
        try:
            polys = []
            if hms_data.get("features"):
                from backend.hms import build_density_polygons
                polys = await asyncio.to_thread(build_density_polygons, hms_data["features"])
            if polys:
                from backend.hms import smoke_density_at
                def _smoke_all(lats, lons):
                    return np.array([smoke_density_at(polys, float(lo), float(la))
                                     for la, lo in zip(lats, lons)], dtype=np.int16)
                hms_vals = await asyncio.to_thread(
                    _smoke_all, lookup_reset["lat"].values, lookup_reset["lon"].values)
                lookup_reset = lookup_reset.copy()
                lookup_reset["hms_smoke"] = hms_vals
                print(f"HMS smoke feature applied ({hms_data.get('data_date')}"
                      f"{', proxy' if hms_is_proxy else ''}): {(hms_vals>0).sum()} tracts "
                      f"under smoke (max tier {int(hms_vals.max()) if len(hms_vals) else 0})")
            else:
                lookup_reset = lookup_reset.copy()
                lookup_reset["hms_smoke"] = 0  # no current polygons = no smoke
        except Exception as e:
            print(f"HMS smoke feature failed ({e}); defaulting to 0.")
            lookup_reset = lookup_reset.copy()
            lookup_reset["hms_smoke"] = 0

    # ── Live CAMS air-quality (+ met PBL proxies only if the model uses them) ──
    # aod/cams_pm25 per tract, computed the SAME way as training (daily-mean of
    # today's hourly AOD), CACHED 6h + disk snapshot. The met call (shortwave/
    # et0/cloud_cover) is skipped for v6 — those features were dropped, and the
    # 254-location met call every cycle was the main api.open-meteo.com quota
    # burner. Missing values fall back to the training median fill.
    aq_feats = {"aod", "cams_pm25", "dust", "shortwave", "et0", "cloud_cover"}
    if aq_feats & features_needed:
        met_needed = bool({"shortwave", "et0", "cloud_cover"} & features_needed)
        snap = await _get_aq_snapshot(
            lookup_reset["lat"].values, lookup_reset["lon"].values,
            include_met=met_needed)
        if snap is not None:
            lookup_reset = lookup_reset.copy()
            applied = []
            for f in sorted(aq_feats & features_needed):
                vals = snap["features"].get(f)
                if vals is not None:
                    # leave None -> NaN so run_predictions_batch fills with training median
                    lookup_reset[f] = pd.Series(vals, dtype="float64").values
                    applied.append(f)
            print(f"Air-quality features applied ({snap['n_cells']} cells): {applied}")
        else:
            print("No usable live air-quality data — features will use training-median fill.")

    header = {
        "region":       "texas",
        "display_name": "All of Texas",
        "generated_at": now.isoformat(),
        # When the background loop's next scheduled refresh is due.
        "expires_at":   (now + timedelta(minutes=PURPLEAIR_CACHE_TTL_MIN)).isoformat(),
        # Display weather (current conditions at Austin, °F/mph/MSL) for the
        # statewide overview panel; the model consumed per-tract gridded daily
        # values in training units.
        "weather":      await get_display_weather(30.2672, -97.7431),
        # Data-source transparency: lets the UI (and operators) see whether the
        # dominant live signals were actually live for THIS prediction cycle.
        "data_sources": {
            "live_purpleair_sensors":   live_sensor_count,
            "using_live_neighbors":     live_sensor_count > 0,
            "hms_polygons":             int(hms_data.get("count") or 0),
            "hms_data_date":            hms_data.get("data_date"),
            "weather_grid_cells":       weather_grid_cells,
            "purpleair_fetched_at":     pa_snap["fetched_at"] if live_sensor_count else None,
            "purpleair_key_configured": _purpleair_key_configured(),
            "hms_is_proxy":             hms_is_proxy,
            "refresh_interval_min":     PURPLEAIR_CACHE_TTL_MIN,
        },
        "value_semantics": "Model-predicted 24-hour-average PM2.5 (µg/m³) per census tract",
    }

    # Vectorized batch prediction, tract dicts and the JSON/gzip encoding all
    # run in one worker thread; the event loop only swaps references.
    result, payload = await asyncio.to_thread(
        _predict_and_package, lookup_reset, weather, temporal, header)
    del lookup_reset

    _set_texas_result(result, payload)
    await _save_snapshot_async(TEXAS_SNAPSHOT_PATH, payload["raw"])
    await asyncio.to_thread(_malloc_trim)
    print(f"✓ Texas predictions complete in {time.monotonic() - t0:.1f}s: "
          f"{result['tract_count']} tracts (live sensors={live_sensor_count}, "
          f"hms_polys={hms_data.get('count', 0)}, weather_cells={weather_grid_cells}, "
          f"payload {len(payload['raw'])/1e6:.1f} MB / {len(payload['gz'])/1e3:.0f} KB gz)")
    return result


async def _refresh_texas() -> dict:
    """Run (or join) the single in-flight statewide compute."""
    try:
        result = await _single_flight("texas", _compute_texas_predictions)
    except Exception:
        # Retry sooner than a full interval, but never in a tight loop.
        state["texas_next_due"] = datetime.now(timezone.utc) + timedelta(minutes=5)
        raise
    state["texas_next_due"] = state["cache_texas"]["expires"]
    return result


# How long a request made before the first result waits for the startup
# compute (PurpleAir + HMS + weather fetches + model run on 0.1 CPU).
TEXAS_COLD_WAIT_S = 240


async def _current_texas() -> dict:
    """The current statewide result for request handlers. Never starts a
    compute while a result exists (the refresh loop keeps it current). Before
    the first result exists, waits for the startup compute instead of racing
    it with a climatology-only one."""
    data = state["cache_texas"]["data"]
    if data is not None:
        return data
    if state.get("bundle") is None:
        raise HTTPException(503, "Model not loaded.")
    if state.get("tract_lookup") is None:
        raise HTTPException(503, "Tract lookup not loaded.")
    try:
        await asyncio.wait_for(state["texas_first_attempt"].wait(), timeout=TEXAS_COLD_WAIT_S)
    except asyncio.TimeoutError:
        raise HTTPException(503, "Predictions are still being computed.",
                            headers={"Retry-After": "30"})
    data = state["cache_texas"]["data"]
    if data is not None:
        return data
    # The startup compute failed. Join one in flight, or start one once the
    # failure backoff has passed; never one per request in the meantime (the
    # refresh loop retries on its own schedule).
    now = datetime.now(timezone.utc)
    due = state.get("texas_next_due", _EPOCH)
    if not _in_flight("texas") and now < due:
        raise HTTPException(503, "Predictions are temporarily unavailable.",
                            headers={"Retry-After": _retry_after(due, now)})
    try:
        return await _refresh_texas()
    except HTTPException:
        raise
    except Exception as e:
        print(f"Texas compute failed: {type(e).__name__}: {e}")
        raise HTTPException(503, "Predictions are temporarily unavailable.",
                            headers={"Retry-After": "60"})


def _etag_matches(if_none_match: str | None, etag: str) -> bool:
    """Weak comparison (RFC 9110): ignores W/ and the -gz variant suffix, so a
    proxy that re-encodes or weakens the tag still gets its 304."""
    if not if_none_match:
        return False
    base = etag.strip('"')
    for tag in if_none_match.split(","):
        tag = tag.strip()
        if tag == "*":
            return True
        if tag.startswith("W/"):
            tag = tag[2:]
        tag = tag.strip('"')
        if tag.endswith("-gz"):
            tag = tag[:-3]
        if tag == base:
            return True
    return False


@app.get("/api/texas/predictions")
async def texas_predictions(request: Request):
    """
    Returns PM2.5 predictions for all Texas census tracts.
    Served from bytes encoded once per compute (gzip when accepted), with an
    ETag; If-None-Match gets a 304. no-cache makes browsers revalidate on
    every poll, which costs a 304 unless a new compute has landed. The
    background loop refreshes the result every PURPLEAIR_CACHE_TTL_MIN; only
    requests before the very first result wait.
    """
    await _current_texas()
    payload = state["texas_payload"]
    headers = {"Cache-Control": "no-cache", "Vary": "Accept-Encoding"}
    if "gzip" in request.headers.get("accept-encoding", "").lower():
        headers["ETag"] = payload["etag"][:-1] + '-gz"'
        body, headers["Content-Encoding"] = payload["gz"], "gzip"
    else:
        headers["ETag"] = payload["etag"]
        body = payload["raw"]
    if _etag_matches(request.headers.get("if-none-match"), payload["etag"]):
        headers.pop("Content-Encoding", None)
        return Response(status_code=304, headers=headers)
    return Response(body, media_type="application/json", headers=headers)


@app.get("/api/{city}/predictions")
async def city_predictions(city: str):
    """
    Returns PM2.5 predictions for all census tracts in the specified city:
    a slice of the current statewide result.
    """
    return await get_city_predictions(city)


@app.get("/api/tract/{geoid}")
async def get_tract(geoid: str, request: Request):
    """Detailed view for a single census tract."""
    allowed, retry_after = _tract_limiter.check(client_ip(request))
    if not allowed:
        raise HTTPException(429, "Too many requests.", headers={"Retry-After": str(retry_after)})

    geoid = geoid.zfill(11)
    if state.get("tract_lookup") is None:
        raise HTTPException(503, "Tract lookup not loaded.")
    if geoid not in state["geoid_set"]:
        raise HTTPException(404, f"Tract {geoid} not found.")

    # CONSISTENCY: the tract's PM2.5 and context come from the SAME texas-wide
    # batch the map shows, so clicking a tract can never show a different
    # number than its color. Before the first batch exists this waits for it.
    await _current_texas()
    t = state["cache_texas_by_geoid"].get(geoid)
    if t is None:
        raise HTTPException(503, "Prediction not available yet.", headers={"Retry-After": "30"})

    # Local CURRENT conditions — display only (°F/mph). A clicked tract gets an
    # EXACT-location fetch (cached per ~11km, shared) so it matches the user's
    # local weather closely; on any failure it falls back to the coarse hourly
    # conditions grid (location-varying — never the old frozen statewide value).
    display_weather = await get_click_weather(float(t["lat"]), float(t["lon"]))

    return {
        "geoid":               geoid,
        "lat":                 t["lat"],
        "lon":                 t["lon"],
        "pm25":                t["pm25"],
        "epa_aqi":             t["epa_aqi"],
        "category":            t["category"],
        "color":               t["color"],
        "aqi_range":           t["aqi_range"],
        "health_msg":          t["health_msg"],
        "weather":             display_weather,
        "from_map_batch":      True,
        **{c: t[c] for c in _TRACT_CONTEXT_COLS},
        "county":              t["county"],
    }


@app.get("/api/texas/tracts/geojson")
async def texas_tracts_geojson(request: Request):
    """Serve all Texas census tract GeoJSON. MUST be before /{city}/ route.

    The raw file is ~24.6 MB — far too heavy to re-send uncompressed (or to
    re-gzip per request) over a residential uplink. We gzip it ONCE into
    memory (~3 MB) and serve those bytes directly with Content-Encoding: gzip,
    plus a 1-day browser cache (tract geometry only changes when the pipeline
    regenerates the file, i.e. on redeploy)."""
    if not os.path.exists(TEXAS_GEOJSON_PATH):
        raise HTTPException(503, "Statewide GeoJSON not found. Run pipeline/01_build_tract_lookup.py first.")
    cache_headers = {"Cache-Control": "public, max-age=86400", "Vary": "Accept-Encoding"}
    if "gzip" in request.headers.get("accept-encoding", "").lower():
        gz = await _get_texas_geojson_gz()
        if gz is not None:
            return Response(
                gz,
                media_type="application/geo+json",
                headers={"Content-Encoding": "gzip", **cache_headers},
            )
    # Client doesn't accept gzip (rare) or compression failed — plain file.
    return FileResponse(TEXAS_GEOJSON_PATH, media_type="application/geo+json",
                        headers=cache_headers)


@app.get("/api/tracts/geojson")
async def tracts_geojson_legacy():
    """Legacy endpoint - returns Dallas GeoJSON for backward compatibility."""
    geojson_path = CITIES["dallas"]["geojson"]
    if not os.path.exists(geojson_path):
        raise HTTPException(503, "GeoJSON not found. Run pipeline/01_build_tract_lookup.py first.")
    return FileResponse(geojson_path, media_type="application/geo+json")


@app.get("/api/{city}/tracts/geojson")
async def city_tracts_geojson(city: str):
    """Serve census tract GeoJSON for the specified city."""
    if city not in CITIES:
        raise HTTPException(404, f"Unknown city: {city}")

    geojson_path = CITIES[city]["geojson"]
    if not os.path.exists(geojson_path):
        raise HTTPException(503, f"GeoJSON not found for {city}. Run pipeline/01_build_tract_lookup.py first.")
    return FileResponse(geojson_path, media_type="application/geo+json")


# ── Quantum Sensor Placement ────────────────────────────────────────────────
_quantum_cache: dict = {"data": None, "expires": _EPOCH}
# Placements change slowly (they depend on the sensor network and the
# tract-level picture, not on hour-to-hour air), and one solve is tens of
# seconds of CPU on the free instance, so refresh a few times a day.
QUANTUM_CACHE_TTL_MIN = int(os.environ.get("QUANTUM_CACHE_TTL_MIN", "360"))
# Sites requested from the solver. The response's top-level num_sensors is
# this request; methods.quantum_annealing.num_sensors is how many it placed,
# which is fewer when too few spaced-out candidates exist.
QUANTUM_NUM_SENSORS = 25

LOSO_RESIDUALS_PATH = os.path.join(ROOT, "models", "loso_residuals.json")


def _existing_sensors_for_solver() -> tuple[list[dict], str]:
    """The current sensor network as [{lat, lon}]: the live PurpleAir list
    when it is usable, else the training sensor set. (The old source was the
    legacy 240-sensor, four-metro p2_processed.xls, which made El Paso and
    the Valley look unmonitored.)"""
    live = _live_purpleair()
    if live is not None:
        return [{"lat": s["lat"], "lon": s["lon"]} for s in live["sensors"]], "live PurpleAir"
    return ([{"lat": s["lat"], "lon": s["lon"]} for s in _load_training_sensors()],
            "models/training_sensors.json")


def _compute_model_disagreement(df: pd.DataFrame, weather: dict, temporal: dict) -> np.ndarray:
    """
    Compute per-tract prediction uncertainty across weather perturbations.
    Since the ensemble was retrained on full data, individual models may agree
    closely. Instead, we measure sensitivity: how much does the prediction
    change when weather varies? Tracts that are highly sensitive to weather
    inputs are harder to predict accurately without a local sensor.
    """
    bundle = state.get("bundle")
    if bundle is None:
        raise ValueError("Model bundle not loaded")
    def _predict_with_weather(w):
        # Delegate to the one construction the map itself uses. The previous
        # inline builder used `{**w, **temporal}` instead of _compute_v3_shared(),
        # so month_sin/cos, dow_sin/cos, doy_sin/cos, temp_x_humidity and
        # wind_x_temp were absent from `shared` AND from df.columns and stayed at
        # the np.zeros() initialisation; it also skipped the lat/lon ->
        # latitude/longitude rename and filled gaps with 0.0 rather than the
        # training medians. That made the sensitivity probe read the model at a
        # point it was never trained on.
        return run_predictions_batch(df, w, temporal)

    # Predict under several weather perturbations
    base_pred = _predict_with_weather(weather)
    perturbations = [
        {**weather, "temperature": weather["temperature"] + 15},
        {**weather, "temperature": weather["temperature"] - 15},
        {**weather, "humidity": min(100, weather["humidity"] + 25)},
        {**weather, "humidity": max(0, weather["humidity"] - 25)},
        {**weather, "wind_speed": weather["wind_speed"] + 10},
        {**weather, "pressure": weather["pressure"] + 15},
    ]
    all_preds = [base_pred] + [_predict_with_weather(w) for w in perturbations]
    pred_stack = np.array(all_preds)
    disagreement = np.std(pred_stack, axis=0)

    return disagreement


def _load_loso_residuals():
    """Load per-GEOID LOSO-CV residuals for quantum solver."""
    if not os.path.exists(LOSO_RESIDUALS_PATH):
        return None
    try:
        with open(LOSO_RESIDUALS_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load LOSO residuals: {e}")
        return None


async def _run_quantum_placement():
    """Run the sensor-placement solver on the current predictions, the current
    sensor network and the LOSO residuals."""
    from backend.quantum.qubo_solver import solve_quantum, compute_coverage

    texas_data = await _current_texas()
    tracts = texas_data.get("tracts", [])

    if not tracts:
        raise HTTPException(503, "No tract predictions available. Wait for predictions to load.")

    existing_sensors, sensors_source = await asyncio.to_thread(_existing_sensors_for_solver)
    print(f"Sensor placement: {len(existing_sensors)} existing sensors from {sensors_source}")

    # Load LOSO-CV residuals (true spatial prediction errors)
    loso_residuals = await asyncio.to_thread(_load_loso_residuals)

    # Per-tract prediction error from LOSO residuals. Residuals exist only for
    # tracts that hosted a training sensor; every other tract is UNKNOWN, so it
    # is NaN (the solver treats NaN as unknown). The old 0.0 fill told the
    # solver that unmonitored tracts were perfectly predicted.
    model_disagreement = None
    if loso_residuals:
        model_disagreement = np.array([
            loso_residuals.get(str(t["geoid"]), np.nan) for t in tracts
        ], dtype=np.float64)
        known = np.isfinite(model_disagreement)
        print(f"LOSO residuals loaded for {int(known.sum())}/{len(tracts)} tracts: "
              f"mean={np.nanmean(model_disagreement):.3f}, "
              f"max={np.nanmax(model_disagreement):.3f}")

    # Fallback to weather sensitivity if no LOSO residuals
    if model_disagreement is None:
        if state.get("bundle") and state.get("tract_lookup") is not None:
            try:
                # Model-unit fallback (the payload's `weather` is the DISPLAY
                # dict in °F/mph — never feed that to the model).
                weather = dict(WEATHER_FALLBACK)
                temporal = get_temporal()
                model_disagreement = await asyncio.to_thread(
                    _compute_model_disagreement, state["tract_lookup"],
                    weather, temporal)
            except Exception as e:
                print(f"Disagreement computation failed: {e}")

    quantum_result = await asyncio.to_thread(
        solve_quantum,
        tracts,
        k=QUANTUM_NUM_SENSORS,
        num_reads=500,
        top_candidates=120,
        proximity_threshold_miles=8.0,
        existing_sensors=existing_sensors,
        model_disagreement=model_disagreement,
    )

    coverage = await asyncio.to_thread(
        compute_coverage,
        tracts, quantum_result["selected_tracts"],
        radius_miles=10.0,
        existing_sensors=existing_sensors,
    )

    avg_ej = float(np.mean([
        t.get("ejf_score", 0.0) or 0.0
        for t in quantum_result["selected_tracts"]
    ])) if quantum_result["selected_tracts"] else 0.0

    avg_composite = float(np.mean([
        t["composite_score"]
        for t in quantum_result["selected_tracts"]
    ])) if quantum_result["selected_tracts"] else 0.0

    now = datetime.now(timezone.utc)
    result = {
        "generated_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=QUANTUM_CACHE_TTL_MIN)).isoformat(),
        "num_total_tracts": len(tracts),
        "num_sensors": QUANTUM_NUM_SENSORS,
        "num_existing_sensors": len(existing_sensors),
        "proximity_threshold_miles": 8.0,
        "coverage_radius_miles": 10.0,
        "existing_sensors": existing_sensors,
        "methods": {
            "quantum_annealing": {
                # The solver describes its own method and reports the weights
                # it actually used, so the UI copy can't drift from the code.
                "method_display": quantum_result["method_display"],
                "weights": quantum_result.get("weights"),
                "feasible": bool(quantum_result.get(
                    "feasible", quantum_result["num_sensors"] == QUANTUM_NUM_SENSORS)),
                "num_sensors": quantum_result["num_sensors"],
                "coverage": coverage,
                "avg_ej_score": round(avg_ej, 1),
                "avg_composite_score": round(avg_composite, 4),
                "timing": quantum_result["timing"],
                "selected_tracts": quantum_result["selected_tracts"],
                "best_energy": quantum_result.get("best_energy"),
                "num_reads": quantum_result.get("num_reads"),
                "num_candidates": quantum_result.get("num_candidates"),
            },
        },
    }

    return result


async def _solve_quantum_and_cache() -> dict:
    """One placement solve (run single-flight); updates cache + disk snapshot."""
    global _quantum_cache
    try:
        result = await _run_quantum_placement()
    except HTTPException:
        # No predictions yet: nothing expensive ran, so no backoff; the loop
        # solves as soon as predictions exist.
        raise
    except Exception:
        # Back off so a failing solve doesn't rerun every loop tick.
        _quantum_cache["expires"] = datetime.now(timezone.utc) + timedelta(minutes=30)
        raise
    _quantum_cache = {
        "data": result,
        "expires": datetime.now(timezone.utc) + timedelta(minutes=QUANTUM_CACHE_TTL_MIN),
    }
    await _save_snapshot_async(QUANTUM_SNAPSHOT_PATH, result)
    print("✓ Sensor placement solved")
    return result


@app.get("/api/quantum/sensor-placement")
async def quantum_sensor_placement():
    """
    Returns optimized sensor placement recommendations. The background loop
    solves after the first predictions and refreshes every
    QUANTUM_CACHE_TTL_MIN; a request only waits when no result exists yet,
    and then it joins the single in-flight solve.
    """
    cached_data = _quantum_cache.get("data")
    if cached_data is not None:
        return cached_data
    # After a failed solve, wait out its backoff instead of re-solving per
    # request (seconds of CPU each on the free instance).
    now = datetime.now(timezone.utc)
    if not _in_flight("quantum") and now < _quantum_cache["expires"]:
        raise HTTPException(503, "Sensor placement is temporarily unavailable.",
                            headers={"Retry-After": _retry_after(_quantum_cache["expires"], now)})
    try:
        return await _single_flight("quantum", _solve_quantum_and_cache)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Quantum placement error: {type(e).__name__}: {e}")
        raise HTTPException(500, "Sensor placement is temporarily unavailable.")


# ── NOAA HMS smoke polygon live layer ─────────────────────────────────────────

async def _fetch_hms() -> None:
    """Fetch today's (or, before it is published, yesterday's) HMS smoke
    polygons from NOAA, clipped to the TX bbox; cache + snapshot. Run
    single-flight by the refresh loop and by a cold-start overlay request."""
    global _hms_cache
    from backend.hms import fetch_latest_hms, hms_currency
    now = datetime.now(timezone.utc)
    try:
        result = await fetch_latest_hms()
    except Exception as e:
        print(f"HMS refresh failed: {type(e).__name__}: {e}")
        _hms_cache["expires"] = now + timedelta(minutes=15)
        return
    old = _hms_cache.get("data")
    old_usable = hms_currency(old, now)[0]
    if old_usable and (result.get("data_date") is None
                       or result["data_date"] < old["data_date"]):
        # NOAA unreachable (or today's file vanished mid-outage): a still-
        # current cached day beats an older or empty answer.
        print(f"HMS refresh found nothing newer than {old['data_date']}; keeping it")
        _hms_cache["expires"] = now + timedelta(minutes=15)
        return
    if result.get("data_date") is None:
        # Neither today's nor yesterday's file could be read (NOAA down, or not
        # published). That is a failure, not a "no smoke" day: retry in 15 min
        # instead of caching an empty day for the full TTL.
        print("HMS refresh failed: no file for today or yesterday; retry in 15 min")
        _hms_cache["expires"] = now + timedelta(minutes=15)
        return
    _hms_cache = {"data": result, "expires": now + timedelta(minutes=HMS_CACHE_TTL_MIN)}
    await _save_snapshot_async(HMS_SNAPSHOT_PATH, result)
    print(f"✓ HMS refreshed: {result.get('count', 0)} polygons "
          f"(data_date={result.get('data_date')}"
          f"{', proxy for today' if result.get('is_proxy') else ''})")


async def _refresh_hms() -> None:
    await _single_flight("hms", _fetch_hms)


@app.get("/api/live/hms-smoke")
async def hms_smoke():
    """Returns today's NOAA HMS smoke polygons clipped to Texas, as a GeoJSON
    FeatureCollection, with data_date and is_proxy (true when yesterday's
    file stands in because today's is not published yet). Smoke older than
    yesterday is never served."""
    from backend.hms import hms_currency
    now = datetime.now(timezone.utc)
    try:
        # Before the first good fetch, join or start one, but only once the
        # failure backoff has passed; the refresh loop retries on its own.
        if _hms_cache.get("data") is None and (_in_flight("hms") or now >= _hms_cache["expires"]):
            await _refresh_hms()
        data = _hms_cache.get("data")
        usable, is_proxy = hms_currency(data, now)
        if usable:
            return {**data, "is_proxy": is_proxy}
        error = None
    except Exception as e:
        print(f"HMS endpoint error: {type(e).__name__}: {e}")
        error = "unavailable"
    # Don't 500 the frontend just because NOAA is down — return empty FC.
    empty = {
        "type": "FeatureCollection",
        "features": [],
        "fetched_at": now.isoformat(),
        "data_date": None,
        "is_proxy": False,
        "count": 0,
        "density_counts": {},
    }
    if error:
        empty["error"] = error
    return empty


# ── Live PurpleAir layer (feeds the model's nbr_pm25_* features) ──────────────

async def _fetch_purpleair() -> bool:
    """One statewide PurpleAir fetch. Schedules the next attempt: the TTL on
    success, a doubling backoff (5 min up to the TTL) on failure. A failed
    fetch never wipes a good prior snapshot. Returns True on success."""
    global _purpleair_cache
    from backend.purpleair import fetch_live_snapshot

    try:
        snap = await fetch_live_snapshot()
    except Exception as e:
        # An unexpected parse error must take the same backoff as an HTTP
        # error; otherwise next_attempt stays in the past and the loop bills
        # a full call about once a minute.
        print(f"[purpleair] snapshot raised: {type(e).__name__}: {e}")
        snap = {"usable": False, "count": 0, "error": type(e).__name__}
    now = datetime.now(timezone.utc)
    if snap.get("usable"):
        _purpleair_cache = {
            "data": snap,
            "failures": 0,
            "next_attempt": now + timedelta(minutes=PURPLEAIR_CACHE_TTL_MIN),
        }
        await _save_snapshot_async(PURPLEAIR_SNAPSHOT_PATH, snap)
        print(f"✓ PurpleAir refreshed: {snap['count']} live sensors, "
              f"statewide mean={snap.get('statewide_mean'):.2f}")
        return True
    failures = _purpleair_cache.get("failures", 0) + 1
    backoff = min(PURPLEAIR_BACKOFF_MIN * 2 ** (failures - 1), PURPLEAIR_CACHE_TTL_MIN)
    _purpleair_cache["failures"] = failures
    _purpleair_cache["next_attempt"] = now + timedelta(minutes=backoff)
    print(f"PurpleAir refresh failed ({snap.get('error') or 'too few sensors'}, "
          f"{snap.get('count', 0)} usable sensors); keeping prior data, "
          f"retry in {backoff} min")
    return False


async def _refresh_purpleair() -> bool:
    return await _single_flight("purpleair", _fetch_purpleair)


# ── Background refresh loop ──────────────────────────────────────────────────
# ONE loop owns every scheduled refresh, independent of traffic: requests only
# read caches (or, before the first result exists, wait for it). Render's free
# instance stays up through the keepalive, so the map stays current even when
# nobody is looking, and PurpleAir spend is a fixed schedule rather than a
# function of traffic.

async def _ensure_quantum() -> None:
    # Before expires there is either a current result or a failure backoff.
    if datetime.now(timezone.utc) < _quantum_cache["expires"]:
        return
    if not _in_flight("quantum"):
        print("Solving sensor placement...")
    try:
        await _single_flight("quantum", _solve_quantum_and_cache)
    except Exception as e:
        print(f"Sensor placement solve failed: {type(e).__name__}: {e}")


async def _startup_refresh() -> None:
    """Startup order matters: live inputs first, so the first compute is a
    real one rather than a climatology placeholder."""
    now = datetime.now(timezone.utc)
    if _purpleair_key_configured():
        if now >= _purpleair_cache["next_attempt"]:
            print("Startup: fetching live PurpleAir sensors...")
            await _refresh_purpleair()
        else:
            age = _age_min(_purpleair_cache["data"]["fetched_at"], now)
            print(f"Startup: PurpleAir snapshot is {age:.0f} min old; skipping the fetch")
    if now >= _hms_cache["expires"]:
        print("Startup: fetching NOAA HMS smoke polygons...")
        await _refresh_hms()
    if state.get("bundle") is not None and state.get("tract_lookup") is not None:
        print("Startup: computing Texas predictions...")
        try:
            await _refresh_texas()
        except Exception as e:
            print(f"Startup Texas compute failed: {type(e).__name__}: {e}")


async def _refresh_cycle() -> None:
    now = datetime.now(timezone.utc)
    pa_refreshed = False
    # Each input step is isolated: one raising must not skip the compute.
    if _purpleair_key_configured() and now >= _purpleair_cache["next_attempt"]:
        try:
            pa_refreshed = await _refresh_purpleair()
        except Exception as e:
            print(f"PurpleAir refresh error: {type(e).__name__}: {e}")
    if now >= _hms_cache["expires"]:
        try:
            await _refresh_hms()
        except Exception as e:
            print(f"HMS refresh error: {type(e).__name__}: {e}")
    if state.get("bundle") is not None and state.get("tract_lookup") is not None:
        if pa_refreshed or now >= state.get("texas_next_due", _EPOCH):
            try:
                await _refresh_texas()
            except Exception as e:
                print(f"Texas compute failed: {type(e).__name__}: {e}")
    # Lowest priority: only after predictions exist.
    if state["cache_texas"]["data"] is not None:
        await _ensure_quantum()


def _next_wakeup_s() -> float:
    due = [_hms_cache["expires"]]
    if _purpleair_key_configured():
        due.append(_purpleair_cache["next_attempt"])
    if state.get("bundle") is not None and state.get("tract_lookup") is not None:
        due.append(state.get("texas_next_due", _EPOCH))
    if state["cache_texas"]["data"] is not None:
        due.append(_quantum_cache["expires"])
    delay = (min(due) - datetime.now(timezone.utc)).total_seconds()
    # Re-check at least once a minute; never spin.
    return min(max(delay, 1.0), 60.0)


async def _refresh_loop() -> None:
    try:
        await _startup_refresh()
    except Exception as e:
        print(f"Startup refresh failed: {type(e).__name__}: {e}")
    finally:
        state["texas_first_attempt"].set()
    if state["cache_texas"]["data"] is not None:
        await _ensure_quantum()
    while True:
        try:
            await asyncio.sleep(_next_wakeup_s())
            await _refresh_cycle()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"Refresh loop error: {type(e).__name__}: {e}")
            await asyncio.sleep(60)


# ── Geocoding (typo-tolerant autocomplete) ───────────────────────────────────
# Primary: Photon (photon.komoot.io) — fuzzy/typo-tolerant, built for
# search-as-you-type ("hueston" still finds Houston). Results are biased to the
# Texas centroid and HARD-filtered to a Texas bounding box.
# Fallback: Nominatim (exact-match), only for an explicit search (mode=submit)
# that Photon found nothing for — scoped to Texas via viewbox+bounded instead
# of appending ", Texas" to the query text. Nominatim's usage policy allows at
# most 1 request/second and forbids autocomplete use, so it is never called
# while typing and is globally throttled.
# Both are normalized to the same shape the frontend expects:
#   [{name, display_name, lat, lon, type}]
_TX_BBOX = (-107.0, 25.5, -93.2, 36.8)  # minLon, minLat, maxLon, maxLat (+margin)
_GEOCODE_UA = "SharedSkiesInitiative/1.0 (+https://sharedskiesinitiative.org; chebrolusaketh@gmail.com)"
_geocode_cache: OrderedDict = OrderedDict()   # (q, lang, mode) -> {results, expires}
_GEOCODE_CACHE_TTL_MIN = 24 * 60        # places don't move
_GEOCODE_EMPTY_CACHE_TTL_MIN = 15       # a miss may be a typo the user fixes
_GEOCODE_CACHE_MAX = 500
_GEOCODE_MAX_LEN = 100
_NOMINATIM_MIN_INTERVAL_S = 1.1
_nominatim_state = {"last": 0.0, "lock": None}

# Per-client-IP limits (see backend/ratelimit.py).
_geocode_limiter = FixedWindowLimiter(60, 60.0)
_tract_limiter = FixedWindowLimiter(120, 60.0)
_visit_dedupe = RecentKeys(VISIT_DEDUPE_MIN * 60.0)

# Global budgets that hold whatever X-Forwarded-For says (a direct caller can
# rotate it past every per-IP limit). Each is one shared key.
_GLOBAL = "*"
_photon_budget = FixedWindowLimiter(120, 60.0, max_keys=1)        # uncached Photon calls
_visit_budget = FixedWindowLimiter(30, 60.0, max_keys=1)          # visit increments
# Uncached click-weather fetches share the api.open-meteo.com daily quota with
# the weather and conditions grids; 200/h caps them at 4,800 of ~10,000/day.
_click_weather_budget = FixedWindowLimiter(200, 3600.0, max_keys=1)

# "Did you mean" spell-correction layer: Photon's fuzziness handles swapped /
# extra letters but misses dropped letters mid-word ("corpus cristi"). We
# correct the query against the big Texas place names with stdlib difflib and
# re-query. Names only — coordinates always come from the geocoder.
_TX_CITIES = [
    "Houston", "San Antonio", "Dallas", "Austin", "Fort Worth", "El Paso",
    "Arlington", "Corpus Christi", "Plano", "Laredo", "Lubbock", "Garland",
    "Irving", "Amarillo", "Grand Prairie", "Brownsville", "Pasadena",
    "McKinney", "Mesquite", "Killeen", "Frisco", "McAllen", "Waco",
    "Carrollton", "Denton", "Midland", "Abilene", "Beaumont", "Round Rock",
    "Odessa", "Wichita Falls", "Richardson", "Lewisville", "Tyler",
    "College Station", "Pearland", "San Angelo", "Allen", "League City",
    "Sugar Land", "Longview", "Edinburg", "Mission", "Bryan", "Baytown",
    "Pharr", "Temple", "Missouri City", "Flower Mound", "Harlingen",
    "Victoria", "Conroe", "New Braunfels", "Mansfield", "Cedar Park",
    "Rowlett", "Port Arthur", "Euless", "Georgetown", "Pflugerville",
    "DeSoto", "San Marcos", "Grapevine", "Bedford", "Galveston",
    "Cedar Hill", "Texas City", "Wylie", "Keller", "Coppell", "Rockwall",
    "Huntsville", "Duncanville", "Sherman", "The Woodlands", "Texarkana",
    "Friendswood", "Weslaco", "Lufkin", "Nacogdoches", "Del Rio", "Eagle Pass",
    "Katy", "Spring", "Humble", "Cypress", "Kingwood", "Tomball", "Sealy",
    "Boerne", "Kerrville", "Seguin", "Schertz", "Alamo Heights", "Uvalde",
    "Stephenville", "Granbury", "Weatherford", "Cleburne", "Waxahachie",
    "Corsicana", "Ennis", "Athens", "Palestine", "Marshall", "Paris",
    "Greenville", "Denison", "Gainesville", "Mineral Wells", "Big Spring",
    "Snyder", "Sweetwater", "Brownwood", "Kingsville", "Alice", "Beeville",
    "Port Lavaca", "Bay City", "Angleton", "Lake Jackson", "Freeport",
    "Orange", "Vidor", "Silsbee", "Jasper", "Livingston", "Bastrop",
    "Lockhart", "Taylor", "Brenham", "Navasota", "Hearne", "Cameron",
]


def _spell_correct_tx(query: str) -> str | None:
    """Return a corrected Texas place name if the query looks like a typo of
    one, else None. Trailing 'tx'/'texas' tokens are ignored for matching."""
    import difflib
    qc = query.lower().strip()
    for suf in (" texas", " tx", ",texas", ",tx"):
        if qc.endswith(suf):
            qc = qc[: -len(suf)].rstrip(", ")
    if len(qc) < 4:
        return None
    match = difflib.get_close_matches(qc, [c.lower() for c in _TX_CITIES], n=1, cutoff=0.78)
    if not match:
        return None
    corrected = next(c for c in _TX_CITIES if c.lower() == match[0])
    return corrected if corrected.lower() != qc else None


def _photon_to_result(feat: dict) -> dict | None:
    """Normalize one Photon GeoJSON feature to the frontend's result shape."""
    try:
        p = feat.get("properties", {})
        # The TX bbox has margin that clips border-state slivers — hard-reject
        # anything Photon labels with a non-Texas state.
        st = p.get("state")
        if st and st not in ("Texas", "TX"):
            return None
        lon, lat = feat["geometry"]["coordinates"][:2]
        name = p.get("name") or " ".join(
            x for x in (p.get("housenumber"), p.get("street")) if x
        ) or p.get("city") or p.get("postcode")
        if not name:
            return None
        parts = []
        street = " ".join(x for x in (p.get("housenumber"), p.get("street")) if x)
        if street and street != name:
            parts.append(street)
        for k in ("district", "city", "county", "state"):
            v = p.get(k)
            if v and v != name and v not in parts:
                parts.append(v)
        if p.get("postcode"):
            parts.append(p["postcode"])
        display = name + (", " + ", ".join(parts) if parts else "")
        return {
            "name": str(name),
            "display_name": display,
            "lat": str(lat),
            "lon": str(lon),
            "type": p.get("osm_value") or p.get("type") or "place",
        }
    except Exception:
        return None


def _geocode_lang(accept_language: str) -> str:
    """Primary language subtag of Accept-Language ('es-MX,es;q=0.9' -> 'es'),
    so the cache key stays small and bounded."""
    tag = (accept_language or "").split(",")[0].split(";")[0].strip().lower()
    tag = tag.split("-")[0]
    return tag if tag.isalpha() and len(tag) <= 3 else "en"


async def _nominatim_search(query: str, lang: str) -> list | None:
    """Nominatim fallback, globally throttled to one call per 1.1 s. Returns
    None when skipped (throttled, or another call in flight) or on error."""
    if _nominatim_state["lock"] is None:
        _nominatim_state["lock"] = asyncio.Lock()
    lock = _nominatim_state["lock"]
    if lock.locked() or time.monotonic() - _nominatim_state["last"] < _NOMINATIM_MIN_INTERVAL_S:
        return None
    minlon, minlat, maxlon, maxlat = _TX_BBOX
    async with lock:
        _nominatim_state["last"] = time.monotonic()
        try:
            r = await _shared_http().get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": query, "format": "json", "countrycodes": "us",
                    "limit": 6, "viewbox": f"{minlon},{maxlat},{maxlon},{minlat}",
                    "bounded": 1,
                },
                headers={"User-Agent": _GEOCODE_UA, "Accept-Language": lang},
                timeout=8.0,
            )
            r.raise_for_status()
            return [{
                "name": (it.get("display_name") or "").split(",")[0],
                "display_name": it.get("display_name", ""),
                "lat": it.get("lat"), "lon": it.get("lon"),
                "type": it.get("type", "place"),
            } for it in r.json()]
        except Exception as e:
            print(f"[geocode] nominatim failed: {type(e).__name__}: {e}")
            return None


@app.get("/api/geocode")
async def proxy_geocode(
    request: Request,
    q: str,
    mode: Literal["suggest", "submit"] = "suggest",
    accept_language: str = Header(default="en"),
):
    """Texas place search. mode=suggest (while typing) uses Photon only;
    mode=submit (Enter / explicit search) adds the Nominatim fallback when
    Photon finds nothing. 60 requests/min per client IP."""
    allowed, retry_after = _geocode_limiter.check(client_ip(request))
    if not allowed:
        raise HTTPException(429, "Too many requests.", headers={"Retry-After": str(retry_after)})

    query = q.strip()
    if len(query) > _GEOCODE_MAX_LEN:
        raise HTTPException(422, f"q must be at most {_GEOCODE_MAX_LEN} characters.")
    # The old frontend appended ", Texas" to every query; strip it so fuzzy
    # matching sees the user's actual words (Texas scoping is done by bbox).
    if query.lower().endswith(", texas"):
        query = query[:-7].rstrip(", ")
    if not query:
        return []

    # Serve recent identical queries from cache (autocomplete hammers this).
    lang = _geocode_lang(accept_language)
    ck = (query.lower(), lang, mode)
    now = datetime.now(timezone.utc)
    hit = _geocode_cache.get(ck)
    if hit and now < hit["expires"]:
        _geocode_cache.move_to_end(ck)
        return hit["results"]

    results: list = []
    minlon, minlat, maxlon, maxlat = _TX_BBOX
    upstream_failed = False
    throttled = 0  # Retry-After seconds once the global Photon budget is spent

    async def _photon_search(q_text: str) -> list:
        nonlocal upstream_failed, throttled
        allowed, wait = _photon_budget.check(_GLOBAL)
        if not allowed:
            throttled = wait
            return []
        try:
            r = await _shared_http().get(
                "https://photon.komoot.io/api",
                params={
                    "q": q_text,
                    "limit": 10,
                    "lat": 31.0, "lon": -99.3,      # bias: Texas centroid
                    "bbox": f"{minlon},{minlat},{maxlon},{maxlat}",
                },
                headers={"User-Agent": _GEOCODE_UA},
                timeout=6.0,
            )
            r.raise_for_status()
            return [res for f in r.json().get("features", [])
                    if (res := _photon_to_result(f))]
        except Exception as e:
            upstream_failed = True
            print(f"[geocode] photon failed: {type(e).__name__}: {e}")
            return []

    def _merge(*lists) -> list:
        """Concatenate, dedupe near-identical entries, cap at 6."""
        out, seen = [], set()
        for lst in lists:
            for res in lst:
                key = (res["name"].lower(),
                       round(float(res["lat"]), 3), round(float(res["lon"]), 3))
                if key in seen:
                    continue
                seen.add(key)
                out.append(res)
                if len(out) >= 6:
                    return out
        return out

    # 1) Photon as typed — typo-tolerant, autocomplete-grade.
    raw = await _photon_search(query)

    # 2) "Did you mean": if the query looks like a misspelled Texas place,
    #    re-query with the corrected name. Empty as-typed results -> the
    #    correction leads; otherwise the as-typed top hit stays first and
    #    the corrected city is injected right behind it (Google-style mix).
    corrected_name = _spell_correct_tx(query)
    if corrected_name:
        fixed = await _photon_search(corrected_name)
        if not raw:
            results = _merge(fixed, raw)
        else:
            results = _merge(raw[:1], fixed[:2], raw[1:])
    else:
        results = _merge(raw)

    # 3) Nominatim fallback (exact match), explicit searches only.
    if not results and mode == "submit":
        nominatim = await _nominatim_search(query, lang)
        if nominatim is None:
            upstream_failed = True
        else:
            results = nominatim

    if not results and throttled:
        # Global Photon budget spent: not an answer, so not cached either.
        raise HTTPException(429, "Too many requests.", headers={"Retry-After": str(throttled)})
    if not results and upstream_failed:
        # Don't cache an outage as "no such place".
        raise HTTPException(status_code=502, detail="Geocoding service unavailable.")

    # Partial answers (one Photon call failed) get the short TTL too.
    ttl = (_GEOCODE_CACHE_TTL_MIN if results and not (upstream_failed or throttled)
           else _GEOCODE_EMPTY_CACHE_TTL_MIN)
    _geocode_cache[ck] = {"results": results, "expires": now + timedelta(minutes=ttl)}
    if len(_geocode_cache) > _GEOCODE_CACHE_MAX:
        _geocode_cache.popitem(last=False)
    return results
