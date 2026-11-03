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
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from backend import geocode, health, hms_feed, model, placement, predictions, purpleair_feed, visits, weather_feed
from backend.config import (
    CITIES,
    LOOKUP_PATH,
    PURPLEAIR_CACHE_TTL_MIN,
    STATIC_DIR,
    TEXAS_GEOJSON_PATH,
)
from backend.http_client import close_shared_http
from backend.model import enrich_tract_lookup_with_distances
from backend.predictions import get_texas_geojson_gz
from backend.purpleair_feed import purpleair_key_configured
from backend.refresh import refresh_loop
from backend.state import EPOCH, state
from backend.util import malloc_trim


def _hydrate_from_snapshots() -> None:
    """Restore caches from disk snapshots, honouring their real age. Render's
    disk is ephemeral so this mostly helps local dev; it must never present
    old live data as current."""
    now = datetime.now(timezone.utc)
    predictions.hydrate(now)
    placement.hydrate(now)
    hms_feed.hydrate(now)
    purpleair_feed.hydrate(now)
    weather_feed.hydrate(now)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model in a thread so the event loop stays responsive
    state["bundle"] = await asyncio.to_thread(model.load_bundle)
    if state["bundle"] is not None:
        print(f"Model loaded. Features: {state['bundle']['feature_names']}")
    malloc_trim()

    # Load tract lookup in a thread
    if os.path.exists(LOOKUP_PATH):
        df = await asyncio.to_thread(pd.read_parquet, LOOKUP_PATH)
        df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)
        # Enrich with spatial-context features the model expects. These are
        # static once sensor locations are known, so we compute them once at
        # startup rather than per-prediction.
        await asyncio.to_thread(enrich_tract_lookup_with_distances, df)
        state["tract_lookup"] = df
        state["geoid_set"] = frozenset(df["GEOID"])
        print(f"Tract lookup loaded: {len(df)} total tracts")
        for city, city_cfg in CITIES.items():
            count = df["GEOID"].str.startswith(city_cfg["fips"]).sum()
            print(f"  {city}: {count} tracts")
    else:
        state["tract_lookup"] = None
        state["geoid_set"] = frozenset()
        print("WARNING: backend/static/tract_lookup.parquet not found. Run pipeline/01_build_tract_lookup.py")

    state["cache_texas"] = {"data": None, "expires": EPOCH}
    state["cache_texas_by_geoid"] = {}
    state["texas_payload"] = None
    state["city_cache"] = {}
    state["texas_first_attempt"] = asyncio.Event()
    try:
        await asyncio.to_thread(_hydrate_from_snapshots)
    except Exception as e:
        # A corrupt snapshot must never keep the API from starting.
        print(f"Snapshot hydrate failed ({type(e).__name__}: {e}); starting cold")

    if purpleair_key_configured():
        print(f"PurpleAir: using PURPLEAIR_API_KEY from environment; refreshing "
              f"every {PURPLEAIR_CACHE_TTL_MIN} min.")
    else:
        print("PurpleAir: PURPLEAIR_API_KEY not set. Live neighbor features are "
              "DISABLED and predictions use climatology. Set the key in the "
              "Render dashboard to enable them.")

    state["refresh_task"] = asyncio.create_task(refresh_loop(), name="refresh-loop")

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
    if not (visits.UPSTASH_URL and visits.UPSTASH_TOKEN):
        try:
            await asyncio.to_thread(visits.init_db)
            print(f"Visits DB initialized at {visits.VISITS_DB}")
        except Exception as e:
            print(f"Failed to initialize visits DB: {e}")

    # Pre-warm the gzipped statewide GeoJSON so even the first visitor after a
    # restart gets the ~3 MB compressed payload instantly.
    if os.path.exists(TEXAS_GEOJSON_PATH):
        asyncio.create_task(get_texas_geojson_gz())

    yield

    for key in ("refresh_task", "keepalive_task"):
        task = state.get(key)
        if task is not None:
            task.cancel()
    await close_shared_http()


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


# Routers, in the order the routes were originally registered. The first match
# wins, so the wildcard /api/{city}/... routes must stay behind the specific ones.
app.include_router(health.router)
app.include_router(visits.router)
app.include_router(predictions.router)
app.include_router(placement.router)
app.include_router(hms_feed.router)
app.include_router(geocode.router)
