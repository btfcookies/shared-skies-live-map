"""Live PurpleAir sensor fetcher + same-day neighbor-feature computation.

The v6 model's single most important feature group is the same-day MULTI-RADIUS
neighbor PM2.5 (nbr_pm25_25km / nbr_pm25_50km / nbr_pm25_100km; the 50km anchor
alone is ~37% of total feature importance) — the mean PM2.5 of OTHER sensors
within 25 / 50 / 100 km ON THE SAME DAY. At training (pipeline/03_train_enhanced.py)
this is computed per (sensor, date) with a BallTree over that day's readings. To
reproduce it at inference WITHOUT train/serve skew, the backend must use LIVE
same-day PurpleAir readings — not a static historical mean.

This module:
  1. fetch_live_sensors(): one bounding-box call to PurpleAir's /v1/sensors
     live endpoint for all outdoor Texas sensors, QC'd to fresh + trustworthy.
  2. compute_neighbor_features(): replicates the training BallTree computation
     with census-tract centroids as query points and live sensors as the
     reference set, returning the full multi-radius feature dict (nbr_pm25_ +
     nbr_count_ at 25/50/100km, plus nbr_std_50km).

CRITICAL correctness notes (these MUST track pipeline/03_train_enhanced.py):
  - PM2.5 field: use pm2.5_24hour (the 24-hour rolling average of the raw ATM
    channel), falling back to instantaneous pm2.5_atm only when it is missing.
    Training pulls daily means (PurpleAir history average=1440), so the 24h
    rolling average is the train-consistent semantics and it smooths single-
    reading sensor glitches. Do NOT apply the EPA Barkjohn correction — the
    training target was the raw ATM mass concentration.
  - MULTI-RADIUS: NBR_RADII_KM = [25, 50, 100], each r/6371.0 radians on a
    haversine BallTree, byte-identical to training's RADII_KM (pipeline/03).
    Query ONCE at the 100km max radius WITH distances, then mask per radius.
  - Zero-neighbor fallback: the 50km anchor falls back to the live statewide
    same-day mean; the 25km/100km radii fall back to the (filled) 50km value.
    (Training's analogous fallback is the same-day statewide LEAVE-ONE-OUT mean;
    inference can't do true leave-one-out because the query points are tracts,
    not sensors, so the plain statewide same-day mean is the closest match.)
"""
import os
from datetime import datetime, timezone

import httpx
import numpy as np

# Texas + buffer bounding box (matches pipeline/06_pull_purpleair_full.py).
# PurpleAir bbox semantics: nw = NORTHWEST corner (max lat, min lon),
# se = SOUTHEAST corner (min lat, max lon).
TX_BBOX = {"nwlng": -106.65, "nwlat": 36.5, "selng": -93.5, "selat": 25.84}

PA_LIVE_URL = "https://api.purpleair.com/v1/sensors"

# QC thresholds (mirror training + live-only quality gates).
PM25_MIN_VALID = 0.0
PM25_MAX_VALID = 200.0       # same hard cap as the training pull
MAX_AGE_SEC = 3600           # drop sensors that haven't reported in 1 hour
MIN_CONFIDENCE = 50          # PurpleAir A/B channel agreement score
MIN_LIVE_SENSORS = 20        # below this we don't trust the live snapshot

EARTH_R_KM = 6371.0


# The READ key is supplied ONLY through the environment. A fallback key used to
# be hardcoded here so the live path worked without a dashboard step; that key
# sat in a public repository, where credential scrapers find keys within minutes
# of a push, and PurpleAir bills per call against the account that owns it. Any
# key committed here is compromised the moment it lands, so there is no safe
# fallback value — the correct failure mode is to degrade to the climatological
# neighbor features and say so loudly.
_MISSING_KEY_MSG = (
    "PURPLEAIR_API_KEY is not set. Live same-day neighbor features are DISABLED "
    "and predictions will fall back to climatology. Set it in the Render "
    "dashboard (Environment -> Add Environment Variable). Never commit the key."
)
_warned_missing_key = False


def _get_api_key() -> str:
    """The PurpleAir READ key from the environment, or "" if unset."""
    global _warned_missing_key
    key = os.environ.get("PURPLEAIR_API_KEY", "").strip()
    if not key and not _warned_missing_key:
        print(f"[purpleair] {_MISSING_KEY_MSG}")
        _warned_missing_key = True
    return key


# Status codes that will not fix themselves on retry, with what to do about them.
_AUTH_MESSAGES = {
    401: "PurpleAir rejected the API key (HTTP 401). Check PURPLEAIR_API_KEY "
         "in the Render dashboard; it may have been rotated or revoked.",
    402: "PurpleAir reports the account is out of points (HTTP 402). Live "
         "neighbor features fall back to climatology until points are added.",
    403: "PurpleAir refused the request (HTTP 403). The key may be a WRITE key "
         "or lack access to /v1/sensors; a READ key is required.",
}


async def _fetch_live(key: str) -> tuple[list[dict], int | None, str | None]:
    """One statewide /v1/sensors call. Returns (sensors, http_status, error)."""
    # Billing, measured on 2026-10-20: one statewide pull returned 497 sensors
    # and cost 3,484 points, about 7 points per sensor with these 5 fields. A
    # 15-min refresh is therefore ~10M points/month against an organization
    # balance that stood at 400M. last_seen and location_type are not
    # requested because the location_type=0 and max_age params already filter
    # outdoor + fresh sensors server-side; confidence has no server-side
    # filter, so it stays.
    params = {
        "fields": "latitude,longitude,pm2.5_24hour,pm2.5_atm,confidence",
        "location_type": 0,        # outdoor only (sole outdoor filter now)
        "max_age": MAX_AGE_SEC,    # sole freshness filter (last_seen field dropped)
        **TX_BBOX,
    }
    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            resp = await client.get(PA_LIVE_URL, headers={"X-API-Key": key}, params=params)
        if resp.status_code != 200:
            msg = _AUTH_MESSAGES.get(resp.status_code)
            if msg:
                print(f"[purpleair] {msg}")
            else:
                print(f"[purpleair] HTTP {resp.status_code}: {resp.text[:200]}")
            return [], resp.status_code, f"HTTP {resp.status_code}"
        payload = resp.json()
    except Exception as e:
        print(f"[purpleair] fetch error: {type(e).__name__}: {e}")
        return [], None, type(e).__name__

    if not isinstance(payload, dict):
        return [], 200, "unexpected response"
    cols = payload.get("fields")
    rows = payload.get("data")
    if not cols or not rows:
        return [], 200, "empty response"

    idx = {name: i for i, name in enumerate(cols)}
    needed = ("latitude", "longitude", "pm2.5_atm", "confidence")
    if not all(n in idx for n in needed):
        print(f"[purpleair] unexpected fields: {cols}")
        return [], 200, "unexpected fields"
    i24 = idx.get("pm2.5_24hour")

    sensors = []
    for row in rows:
        try:
            lat = row[idx["latitude"]]
            lon = row[idx["longitude"]]
            pm24 = row[i24] if i24 is not None else None
            pm = pm24 if pm24 is not None else row[idx["pm2.5_atm"]]
            conf = row[idx["confidence"]]
        except (IndexError, KeyError, TypeError):
            continue
        # QC: trustworthy + valid PM. Outdoor-only and freshness are already
        # enforced SERVER-SIDE by the location_type=0 and max_age params, so we
        # no longer download (or bill for) last_seen / location_type to re-check.
        if lat is None or lon is None or pm is None:
            continue
        # One malformed value (a string confidence, say) skips that row; it
        # must not abort the whole billed pull.
        try:
            if conf is None or conf < MIN_CONFIDENCE:
                continue
            if not (PM25_MIN_VALID <= pm <= PM25_MAX_VALID):
                continue
            sensors.append({"lat": float(lat), "lon": float(lon), "pm25": float(pm)})
        except (TypeError, ValueError):
            continue

    return sensors, 200, None


async def fetch_live_sensors() -> list[dict]:
    """Fetch current outdoor PurpleAir readings across Texas, QC'd.

    Returns a list of {lat, lon, pm25} dicts. pm25 = **pm2.5_24hour** (24-hour
    rolling average of the raw ATM channel), falling back to instantaneous
    pm2.5_atm only when the rolling average is missing.

    WHY the 24-hour field: the model trains on DAILY MEANS (PurpleAir history
    pulled with average=1440), so the rolling 24h average is the train-consistent
    semantics for the dominant neighbor features. Instantaneous pm2.5_atm both
    (a) injects diurnal swings the model never saw and (b) passes through
    single-reading sensor glitches — e.g. a Dallas sensor was observed reading
    atm=3326 µg/m³ while its 24h average was a sane 33.8. The 24h field smooths
    glitches AND keeps genuine smoke-event signal.

    Returns an empty list on any failure (no key, HTTP error, too few sensors)
    so the caller can cleanly fall back to the static climatology.
    """
    key = _get_api_key()
    if not key:
        return []
    sensors, _, _ = await _fetch_live(key)
    return sensors


# MULTI-RADIUS neighbor radii — MUST match pipeline/03_train_enhanced.py exactly
# (RADII_KM there). Changing one side without the other scrambles predictions.
NBR_RADII_KM = [25.0, 50.0, 100.0]


def compute_neighbor_features(tract_lats, tract_lons, sensors):
    """Replicate the training MULTI-RADIUS same-day neighbor computation for
    tract centroids — the single source of truth shared by the live and the
    climatological-fallback inference paths.

    For each tract: mean + count of PM2.5 from sensors within 25 / 50 / 100 km,
    plus population-std at the 50km anchor. Mirrors the training BallTree
    (haversine, query once at the 100km max radius WITH distances, then mask to
    each radius) byte-identically. Fallbacks match training: the 50km anchor
    falls back to the statewide same-day mean; 25km and 100km fall back to the
    (filled) 50km value.

    Returns a dict {feature_name: np.ndarray aligned to inputs}, or None if there
    aren't enough sensors to trust.
    """
    from sklearn.neighbors import BallTree

    if not sensors or len(sensors) < MIN_LIVE_SENSORS:
        return None

    s_lat = np.array([s["lat"] for s in sensors], dtype=np.float64)
    s_lon = np.array([s["lon"] for s in sensors], dtype=np.float64)
    s_pm = np.array([s["pm25"] for s in sensors], dtype=np.float64)

    s_coords = np.radians(np.column_stack([s_lat, s_lon]))
    t_coords = np.radians(np.column_stack([
        np.asarray(tract_lats, dtype=np.float64),
        np.asarray(tract_lons, dtype=np.float64),
    ]))
    tree = BallTree(s_coords, metric="haversine")
    max_rad = max(NBR_RADII_KM) / EARTH_R_KM
    ind, dist = tree.query_radius(t_coords, r=max_rad, return_distance=True)

    n = len(t_coords)
    mean = {r: np.full(n, np.nan) for r in NBR_RADII_KM}
    cnt = {r: np.zeros(n, dtype=np.int32) for r in NBR_RADII_KM}
    std50 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        nbrs = ind[i]
        if len(nbrs) == 0:
            continue
        d_km = dist[i] * EARTH_R_KM
        vals_all = s_pm[nbrs]
        for r in NBR_RADII_KM:
            m = d_km <= r
            if not m.any():
                continue
            vr = vals_all[m]
            mean[r][i] = float(vr.mean())
            cnt[r][i] = int(len(vr))
            if r == 50.0 and len(vr) > 1:
                std50[i] = float(vr.std())

    # Fallbacks (match training): 50km anchor -> statewide same-day mean; the
    # finer/coarser radii fall back to the (filled) 50km value.
    statewide_mean = float(s_pm.mean())
    m50 = np.where(cnt[50.0] > 0, mean[50.0], statewide_mean)
    m25 = np.where(cnt[25.0] > 0, mean[25.0], m50)
    m100 = np.where(cnt[100.0] > 0, mean[100.0], m50)

    return {
        "nbr_pm25_25km": m25, "nbr_count_25km": cnt[25.0],
        "nbr_pm25_50km": m50, "nbr_count_50km": cnt[50.0], "nbr_std_50km": std50,
        "nbr_pm25_100km": m100, "nbr_count_100km": cnt[100.0],
    }


async def fetch_live_snapshot() -> dict:
    """Fetch + package live sensors into a snapshot dict (cache/disk friendly).
    http_status is None when no request was made (no key) or it failed in
    transit; the caller uses it to choose a retry backoff."""
    key = _get_api_key()
    if key:
        sensors, status, error = await _fetch_live(key)
    else:
        sensors, status, error = [], None, "no key"
    now = datetime.now(timezone.utc)
    pm_vals = [s["pm25"] for s in sensors]
    return {
        "sensors": sensors,
        "count": len(sensors),
        "statewide_mean": float(np.mean(pm_vals)) if pm_vals else None,
        "fetched_at": now.isoformat(),
        "usable": len(sensors) >= MIN_LIVE_SENSORS,
        "http_status": status,
        "error": error,
    }
