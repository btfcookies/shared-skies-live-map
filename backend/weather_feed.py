"""Open-Meteo weather grid, CAMS air-quality grid, conditions and click-weather lookups."""

import os
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

import httpx

from backend import config, limits, snapshots
from backend.state import EPOCH, state
from backend.util import age_min, parse_ts, single_flight


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


_wgrid_cache: dict = {"data": None, "expires": EPOCH}


_aq_cache: dict = {"data": None, "expires": EPOCH}


def _daily_grid_expiry(fetched: datetime, ttl_min: int = OPENMETEO_GRID_TTL_MIN) -> datetime:
    """The earlier of fetched + TTL and just after the next UTC midnight."""
    fetched = fetched.astimezone(timezone.utc)
    rollover = (datetime.combine(fetched.date() + timedelta(days=1), datetime.min.time(),
                                 tzinfo=timezone.utc)
                + timedelta(minutes=OPENMETEO_DAY_ROLLOVER_MIN))
    return min(fetched + timedelta(minutes=ttl_min), rollover)


def _grid_recent(snap: dict | None, now: datetime) -> bool:
    """A cached daily grid is still worth serving stale (<= a day old)."""
    age = age_min((snap or {}).get("fetched_at"), now)
    return age is not None and age <= OPENMETEO_STALE_MAX_H * 60


# CURRENT-CONDITIONS grid (display-only): one batched 81-cell call per hour is
# the always-available baseline for every tract/city/overview panel.
CONDITIONS_TTL_MIN = int(os.environ.get("CONDITIONS_TTL_MIN", "60"))


conditions_cache: dict = {"data": None, "expires": EPOCH}


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


_click_weather_cooldown = {"until": EPOCH}


def hydrate(now: datetime) -> None:
    """Open-Meteo weather, air-quality and conditions grids."""
    global _wgrid_cache, _aq_cache, conditions_cache
    # Open-Meteo snapshots carry their real expiry; an expired one is still
    # the stale-on-failure fallback if the quota is exhausted after a redeploy.
    for path, label, ttl in ((config.WEATHER_GRID_SNAPSHOT_PATH, "weather-grid", OPENMETEO_GRID_TTL_MIN),
                             (config.AQ_SNAPSHOT_PATH, "air-quality", OPENMETEO_GRID_TTL_MIN),
                             (config.CONDITIONS_SNAPSHOT_PATH, "conditions", CONDITIONS_TTL_MIN)):
        snap = snapshots.load(path)
        if not snap:
            continue
        ts = parse_ts(snap.get("fetched_at"))
        if ts is None:
            expires = EPOCH
        elif path == config.CONDITIONS_SNAPSHOT_PATH:
            expires = ts + timedelta(minutes=ttl)
        else:
            expires = _daily_grid_expiry(ts, ttl)
        cache = {"data": snap, "expires": expires}
        if path == config.WEATHER_GRID_SNAPSHOT_PATH:
            _wgrid_cache = cache
        elif path == config.AQ_SNAPSHOT_PATH:
            _aq_cache = cache
        else:
            conditions_cache = cache
        print(f"Loaded {label} snapshot ({snap.get('n_cells', 0)} cells)")


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


def weather_fallback() -> dict:
    """Model-unit fallback weather (training medians) WITH a converted display
    dict. Display fallbacks must NEVER expose bare model units — that once
    rendered the 20.2 °C training median as '20 °F' on every tract."""
    w = dict(WEATHER_FALLBACK)
    w["display"] = dict(DISPLAY_WEATHER_FALLBACK)
    return w


async def _fetch_conditions_grid() -> dict | None:
    """One conditions-grid fetch (run single-flight by _get_conditions_grid)."""
    global conditions_cache
    now = datetime.now(timezone.utc)
    cached = conditions_cache.get("data")
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
        conditions_cache = {"data": snap, "expires": now + timedelta(minutes=CONDITIONS_TTL_MIN)}
        await snapshots.save_async(config.CONDITIONS_SNAPSHOT_PATH, snap)
        return snap
    if cached and cached.get("usable"):
        print("[conditions] refetch failed — serving stale conditions (backoff 15 min)")
        conditions_cache["expires"] = now + timedelta(minutes=15)
        return cached
    return None


async def _get_conditions_grid(allow_fetch: bool = True) -> dict | None:
    """Cached CURRENT-CONDITIONS cell grid (display units, TTL 60 min + disk
    snapshot, stale-on-failure with 15-min backoff). The ONLY source of
    'Current Conditions' shown in the UI. The refresh loop keeps it current
    through each Texas compute; request handlers pass allow_fetch=False so
    traffic never triggers the fetch."""
    cached = conditions_cache.get("data")
    if cached and cached.get("usable") and (
            not allow_fetch or datetime.now(timezone.utc) < conditions_cache["expires"]):
        return cached
    if not allow_fetch:
        return None
    return await single_flight("conditions", _fetch_conditions_grid)


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
    if not limits.click_weather_budget.check(limits.GLOBAL_KEY)[0]:
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
    """One weather-grid fetch (run single-flight by get_weather_grid)."""
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
        await snapshots.save_async(config.WEATHER_GRID_SNAPSHOT_PATH, snap)
        return snap
    if _grid_matches(cached, len(lats), WEATHER_GRID_SCHEMA):
        _wgrid_cache["expires"] = now + timedelta(minutes=30)
        if _grid_recent(cached, now):
            print("[weather-grid] refetch failed — serving stale cached grid (backoff 30 min)")
            return cached
        print(f"[weather-grid] refetch failed and the cached grid is over "
              f"{OPENMETEO_STALE_MAX_H} h old — using training-median weather (backoff 30 min)")
    return None


async def get_weather_grid(lats, lons) -> dict | None:
    """Cached per-tract daily weather grid (TTL 6h, capped at the UTC day, +
    disk snapshot). On refetch failure, serves the stale grid for up to a day
    and backs off 30 min — never hammers a quota-exhausted API every cycle."""
    from backend.airquality import WEATHER_GRID_SCHEMA
    cached = _wgrid_cache.get("data")
    now = datetime.now(timezone.utc)
    if _grid_matches(cached, len(lats), WEATHER_GRID_SCHEMA) and now < _wgrid_cache["expires"]:
        # Inside a failure backoff the cached grid may be too old to use.
        return cached if _grid_recent(cached, now) else None
    return await single_flight("weather_grid", lambda: _fetch_weather_grid(lats, lons))


async def _fetch_aq_snapshot(lats, lons, include_met: bool) -> dict | None:
    """One CAMS air-quality fetch (run single-flight by get_aq_snapshot)."""
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
        await snapshots.save_async(config.AQ_SNAPSHOT_PATH, snap)
        return snap
    if _grid_matches(cached, len(lats)):
        _aq_cache["expires"] = now + timedelta(minutes=30)
        if _grid_recent(cached, now):
            print("[airquality] refetch failed — serving stale cached snapshot (backoff 30 min)")
            return cached
        print(f"[airquality] refetch failed and the cached snapshot is over "
              f"{OPENMETEO_STALE_MAX_H} h old — using training-median fill (backoff 30 min)")
    return None


async def get_aq_snapshot(lats, lons, include_met: bool) -> dict | None:
    """Cached per-tract CAMS air-quality snapshot (TTL 6h, capped at the UTC
    day, + disk snapshot, stale-on-failure). Was refetched EVERY 30-min cycle before — the main
    Open-Meteo quota burner."""
    cached = _aq_cache.get("data")
    now = datetime.now(timezone.utc)
    if _grid_matches(cached, len(lats)) and now < _aq_cache["expires"]:
        return cached if _grid_recent(cached, now) else None
    return await single_flight("airquality", lambda: _fetch_aq_snapshot(lats, lons, include_met))
