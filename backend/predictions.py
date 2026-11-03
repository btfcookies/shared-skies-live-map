"""Statewide tract predictions: compute, package, cache, and the prediction / tract / GeoJSON routes."""

import asyncio
import gzip
import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from backend import config, hms_feed, limits, snapshots
from backend.aqi import pm25_info, pm25_to_epa_aqi, safe_float
from backend.config import (
    CITIES,
    PURPLEAIR_CACHE_TTL_MIN,
    TEXAS_GEOJSON_PATH,
    TEXAS_SNAPSHOT_MAX_AGE_MIN,
)
from backend.model import get_temporal, run_predictions_batch
from backend.purpleair_feed import live_purpleair, purpleair_key_configured
from backend.ratelimit import client_ip
from backend.state import EPOCH, state
from backend.util import (
    age_min,
    in_flight,
    malloc_trim,
    parse_ts,
    retry_after,
    single_flight,
)
from backend.weather_feed import (
    get_aq_snapshot,
    get_click_weather,
    get_display_weather,
    get_weather_grid,
    weather_fallback,
)


router = APIRouter()


# Pre-gzipped statewide tract GeoJSON (~24.6 MB raw → ~3 MB gz), built once and
# held in memory so /api/texas/tracts/geojson costs zero CPU per request.
_texas_geojson_gz: dict = {"bytes": None, "mtime": None}


async def get_texas_geojson_gz() -> bytes | None:
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


def hydrate(now: datetime) -> None:
    """Texas predictions from the last snapshot, if it is young enough."""
    texas_snap = snapshots.load(config.TEXAS_SNAPSHOT_PATH)
    age = age_min((texas_snap or {}).get("generated_at"), now)
    if texas_snap and texas_snap.get("tracts") and age is not None and age < TEXAS_SNAPSHOT_MAX_AGE_MIN:
        try:
            set_texas_result(texas_snap, encode_payload(texas_snap))
            print(f"Loaded Texas predictions snapshot ({len(texas_snap['tracts'])} tracts, "
                  f"{age:.0f} min old); serving it until the first compute finishes")
        except Exception as e:
            print(f"Ignoring Texas predictions snapshot ({type(e).__name__}: {e})")
    elif texas_snap:
        print("Ignoring Texas predictions snapshot (too old or unreadable)")


async def get_city_predictions(city: str):
    """PM2.5 predictions for one county: a slice of the current statewide
    result, cached per Texas generated_at so it can never lag the map."""
    if city not in CITIES:
        raise HTTPException(404, f"Unknown city: {city}. Available: {', '.join(CITIES.keys())}")

    texas = await current_texas()
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
            **{c: safe_float(ctx[c][i]) for c in _TRACT_CONTEXT_COLS},
            "county":              str(counties[i]),
        })
    return tracts


def encode_payload(result: dict) -> dict:
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


def set_texas_result(result: dict, payload: dict) -> None:
    """Install a statewide result everywhere it is read from. The by-GEOID
    index and the encoded bytes are rebuilt every time, so /api/tract and the
    map can never disagree (the old snapshot hydrate skipped the index)."""
    generated = parse_ts(result.get("generated_at")) or datetime.now(timezone.utc)
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
    return result, encode_payload(result)


async def _compute_texas_predictions() -> dict:
    """Heavy path: actually compute predictions for every TX tract, update
    cache + snapshot. Only ever run through refresh_texas (single-flight)."""
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
    weather = weather_fallback()  # model-unit statewide backfill scalars
    wsnap = await get_weather_grid(
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
    # enrich_tract_lookup_with_distances in place — stable and non-scrambled.
    live_sensor_count = 0
    pa_snap = live_purpleair(now)
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
                      f"({age_min(pa_snap['fetched_at'], now):.0f} min old), "
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
    hms_data = hms_feed.hms_cache.get("data") or {}
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
        snap = await get_aq_snapshot(
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
            "purpleair_key_configured": purpleair_key_configured(),
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

    set_texas_result(result, payload)
    await snapshots.save_async(config.TEXAS_SNAPSHOT_PATH, payload["raw"])
    await asyncio.to_thread(malloc_trim)
    print(f"✓ Texas predictions complete in {time.monotonic() - t0:.1f}s: "
          f"{result['tract_count']} tracts (live sensors={live_sensor_count}, "
          f"hms_polys={hms_data.get('count', 0)}, weather_cells={weather_grid_cells}, "
          f"payload {len(payload['raw'])/1e6:.1f} MB / {len(payload['gz'])/1e3:.0f} KB gz)")
    return result


async def refresh_texas() -> dict:
    """Run (or join) the single in-flight statewide compute."""
    try:
        result = await single_flight("texas", _compute_texas_predictions)
    except Exception:
        # Retry sooner than a full interval, but never in a tight loop.
        state["texas_next_due"] = datetime.now(timezone.utc) + timedelta(minutes=5)
        raise
    state["texas_next_due"] = state["cache_texas"]["expires"]
    return result


# How long a request made before the first result waits for the startup
# compute (PurpleAir + HMS + weather fetches + model run on 0.1 CPU).
TEXAS_COLD_WAIT_S = 240


async def current_texas() -> dict:
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
    due = state.get("texas_next_due", EPOCH)
    if not in_flight("texas") and now < due:
        raise HTTPException(503, "Predictions are temporarily unavailable.",
                            headers={"Retry-After": retry_after(due, now)})
    try:
        return await refresh_texas()
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


@router.get("/api/texas/predictions")
async def texas_predictions(request: Request):
    """
    Returns PM2.5 predictions for all Texas census tracts.
    Served from bytes encoded once per compute (gzip when accepted), with an
    ETag; If-None-Match gets a 304. no-cache makes browsers revalidate on
    every poll, which costs a 304 unless a new compute has landed. The
    background loop refreshes the result every PURPLEAIR_CACHE_TTL_MIN; only
    requests before the very first result wait.
    """
    await current_texas()
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


@router.get("/api/{city}/predictions")
async def city_predictions(city: str):
    """
    Returns PM2.5 predictions for all census tracts in the specified city:
    a slice of the current statewide result.
    """
    return await get_city_predictions(city)


@router.get("/api/tract/{geoid}")
async def get_tract(geoid: str, request: Request):
    """Detailed view for a single census tract."""
    allowed, retry_after = limits.tract_limiter.check(client_ip(request))
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
    await current_texas()
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


@router.get("/api/texas/tracts/geojson")
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
        gz = await get_texas_geojson_gz()
        if gz is not None:
            return Response(
                gz,
                media_type="application/geo+json",
                headers={"Content-Encoding": "gzip", **cache_headers},
            )
    # Client doesn't accept gzip (rare) or compression failed — plain file.
    return FileResponse(TEXAS_GEOJSON_PATH, media_type="application/geo+json",
                        headers=cache_headers)


@router.get("/api/tracts/geojson")
async def tracts_geojson_legacy():
    """Legacy endpoint - returns Dallas GeoJSON for backward compatibility."""
    geojson_path = CITIES["dallas"]["geojson"]
    if not os.path.exists(geojson_path):
        raise HTTPException(503, "GeoJSON not found. Run pipeline/01_build_tract_lookup.py first.")
    return FileResponse(geojson_path, media_type="application/geo+json")


@router.get("/api/{city}/tracts/geojson")
async def city_tracts_geojson(city: str):
    """Serve census tract GeoJSON for the specified city."""
    if city not in CITIES:
        raise HTTPException(404, f"Unknown city: {city}")

    geojson_path = CITIES[city]["geojson"]
    if not os.path.exists(geojson_path):
        raise HTTPException(503, f"GeoJSON not found for {city}. Run pipeline/01_build_tract_lookup.py first.")
    return FileResponse(geojson_path, media_type="application/geo+json")
