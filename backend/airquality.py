"""Live CAMS air-quality + met PBL-proxy fetcher for inference.

Provides the same 6 features the v4 model trains on:
  - aod, cams_pm25, dust   (Open-Meteo Air-Quality forecast, daily mean of today)
  - shortwave, et0, cloud_cover  (Open-Meteo forecast, today's daily values)

CONSISTENCY: training used the DAILY MEAN of hourly archive AOD/pm2_5/dust, so
inference averages today's hourly forecast to a daily mean (NOT a single current
hour) — otherwise the diurnal cycle would create train/serve skew. Met features
are daily aggregates in both phases.

To stay within rate limits and memory, tracts are binned to a coarse ~0.5deg
grid (~50-100 occupied cells over Texas); we make 2 batched multi-location calls
(air-quality + met) covering all cells, then map each tract to its cell value.
"""
import os
from datetime import datetime, timezone

import httpx
import numpy as np

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
MET_URL = "https://api.open-meteo.com/v1/forecast"
GRID_DEG = 0.5  # cell size for binning tracts (CAMS air-quality — matches training)
# Weather grid is COARSER (1.0° ≈ ~70 cells vs 254): daily-mean weather varies
# smoothly, and Open-Meteo weighs multi-location calls per location — the free
# quota (~10k/day) cannot afford 254-cell refetches.
WEATHER_GRID_DEG = 1.0


def _cell_key(lat: float, lon: float, grid_deg: float = GRID_DEG) -> tuple:
    return (round(lat / grid_deg) * grid_deg, round(lon / grid_deg) * grid_deg)


def build_cells(lats, lons, grid_deg: float = GRID_DEG):
    """Map tracts to coarse grid cells. Returns (cell_list, tract_cell_index)
    where cell_list is unique (lat, lon) cell centers and tract_cell_index[i] is
    the index into cell_list for tract i."""
    keys = [_cell_key(float(la), float(lo), grid_deg) for la, lo in zip(lats, lons)]
    uniq = {}
    order = []
    idx = np.empty(len(keys), dtype=np.int32)
    for i, k in enumerate(keys):
        if k not in uniq:
            uniq[k] = len(order)
            order.append(k)
        idx[i] = uniq[k]
    return order, idx


def _daily_mean_today(times, values):
    """Mean of today's hourly values (skip NaN). times are ISO hour strings."""
    if not times or not values:
        return None
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    vals = [v for t, v in zip(times, values) if v is not None and str(t).startswith(today)]
    if not vals:
        vals = [v for v in values if v is not None]  # fallback: whole window
    return float(np.mean(vals)) if vals else None


async def fetch_cell_features(cell_centers, include_met: bool = True) -> dict:
    """Fetch air-quality (and optionally met) for each grid cell. Returns
    {cell_index: {aod, cams_pm25, dust[, shortwave, et0, cloud_cover]}} (None
    where unavailable). Batched multi-location calls to stay within rate limits.

    include_met=False skips the api.open-meteo.com met call entirely — the v6
    model dropped shortwave/et0/cloud_cover, and that 254-location call per
    cycle was the main consumer of the free daily quota.
    """
    if not cell_centers:
        return {}
    lats = ",".join(f"{c[0]:.4f}" for c in cell_centers)
    lons = ",".join(f"{c[1]:.4f}" for c in cell_centers)
    out = {i: {} for i in range(len(cell_centers))}

    # 1. Air-quality forecast (hourly today -> daily mean).
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(AQ_URL, params={
                "latitude": lats, "longitude": lons,
                "hourly": "aerosol_optical_depth,pm2_5,dust",
                "forecast_days": 1, "timezone": "UTC",
            })
            if r.status_code == 200:
                payload = r.json()
                if isinstance(payload, dict):
                    payload = [payload]
                for i, loc in enumerate(payload):
                    h = loc.get("hourly", {})
                    t = h.get("time", [])
                    out[i]["aod"] = _daily_mean_today(t, h.get("aerosol_optical_depth", []))
                    out[i]["cams_pm25"] = _daily_mean_today(t, h.get("pm2_5", []))
                    out[i]["dust"] = _daily_mean_today(t, h.get("dust", []))
            else:
                print(f"[airquality] AQ HTTP {r.status_code}: {r.text[:160]}")
    except Exception as e:
        print(f"[airquality] AQ fetch error: {e}")

    # 2. Met forecast (today's daily aggregates) — only when the model uses them.
    if include_met:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.get(MET_URL, params={
                    "latitude": lats, "longitude": lons,
                    "daily": "shortwave_radiation_sum,et0_fao_evapotranspiration,cloud_cover_mean",
                    "forecast_days": 1, "timezone": "UTC",
                })
                if r.status_code == 200:
                    payload = r.json()
                    if isinstance(payload, dict):
                        if payload.get("error"):
                            print(f"[airquality] met API error: {payload.get('reason', payload)}")
                            payload = []
                        else:
                            payload = [payload]
                    for i, loc in enumerate(payload):
                        d = loc.get("daily", {})
                        sw = d.get("shortwave_radiation_sum", [None])
                        et = d.get("et0_fao_evapotranspiration", [None])
                        cc = d.get("cloud_cover_mean", [None])
                        out[i]["shortwave"] = sw[0] if sw else None
                        out[i]["et0"] = et[0] if et else None
                        out[i]["cloud_cover"] = cc[0] if cc else None
                else:
                    print(f"[airquality] met HTTP {r.status_code}: {r.text[:160]}")
        except Exception as e:
            print(f"[airquality] met fetch error: {e}")

    return out


# Bumped whenever the meaning of a weather-grid value changes, so a snapshot
# written by older code (e.g. daily-MEAN wind) is never reused as model input.
WEATHER_GRID_SCHEMA = 2


async def fetch_weather_grid(lats, lons) -> dict:
    """Fetch TODAY'S (UTC day) daily weather per 1.0° grid cell, in TRAINING UNITS.

    The model trains on daily aggregates: mean temperature °C, mean humidity %,
    mean SURFACE pressure hPa, MAXIMUM 10 m wind m/s, precipitation mm/day, per
    sensor location. Training wind is the daily max on both of its sources
    (Open-Meteo archive wind_speed_10m_max converted from km/h, NASA POWER
    WS10M_MAX in m/s), so serving requests wind_speed_10m_max in m/s too; the
    daily mean runs about 31% lower and would bias the model.

    Returns {'features': {feat: [per-tract values or None]}, 'usable': bool,
    'n_cells', 'fetched_at', 'schema'}.
    """
    cells, idx = build_cells(lats, lons, grid_deg=WEATHER_GRID_DEG)
    feat_names = ["temperature", "humidity", "pressure", "wind_speed", "precipitation"]
    per_tract = {f: [None] * len(lats) for f in feat_names}
    fetched_at = datetime.now(timezone.utc).isoformat()

    def _result(usable: bool) -> dict:
        return {"features": per_tract, "usable": usable, "n_cells": len(cells),
                "fetched_at": fetched_at, "schema": WEATHER_GRID_SCHEMA}

    if not cells:
        return _result(False)

    lat_s = ",".join(f"{c[0]:.4f}" for c in cells)
    lon_s = ",".join(f"{c[1]:.4f}" for c in cells)
    cell_vals = {}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(MET_URL, params={
                "latitude": lat_s, "longitude": lon_s,
                "daily": ("temperature_2m_mean,relative_humidity_2m_mean,"
                          "surface_pressure_mean,wind_speed_10m_max,precipitation_sum"),
                "wind_speed_unit": "ms",   # training wind is m/s
                "forecast_days": 1, "timezone": "UTC",
            })
            if r.status_code != 200:
                print(f"[weather-grid] HTTP {r.status_code}: {r.text[:160]}")
                return _result(False)
            payload = r.json()
            if isinstance(payload, dict):
                if payload.get("error"):
                    print(f"[weather-grid] met API error: {payload.get('reason', payload)}")
                    payload = []
                else:
                    payload = [payload]
            for i, loc in enumerate(payload):
                d = loc.get("daily", {})
                def first(key):
                    v = d.get(key) or [None]
                    return v[0] if v else None
                wind = first("wind_speed_10m_max")
                # Guard the unit rather than trust the request param: the
                # model expects m/s, and km/h would read 3.6x too windy.
                wind_unit = (loc.get("daily_units") or {}).get("wind_speed_10m_max", "m/s")
                if wind is not None and wind_unit == "km/h":
                    wind = wind / 3.6
                cell_vals[i] = {
                    "temperature":   first("temperature_2m_mean"),       # °C
                    "humidity":      first("relative_humidity_2m_mean"), # %
                    "pressure":      first("surface_pressure_mean"),     # hPa (surface)
                    "wind_speed":    wind,                               # m/s, daily max
                    "precipitation": first("precipitation_sum"),         # mm/day
                }
            units = (payload[0].get("daily_units") or {}) if payload else {}
            print(f"[weather-grid] {len(cell_vals)} cells, wind unit "
                  f"{units.get('wind_speed_10m_max')!r} (daily max)")
    except Exception as e:
        print(f"[weather-grid] fetch error: {e}")
        return _result(False)

    usable = False
    for i in range(len(lats)):
        cv = cell_vals.get(int(idx[i]), {})
        for f in feat_names:
            v = cv.get(f)
            per_tract[f][i] = v
            if v is not None:
                usable = True
    return _result(usable)


def conditions_cell_key(lat: float, lon: float) -> str:
    """String cell key for the CURRENT-CONDITIONS grid (1.0°). String keys keep
    the snapshot JSON-serializable and let arbitrary points (tract clicks, city
    centers) look up their cell directly."""
    k = _cell_key(float(lat), float(lon), WEATHER_GRID_DEG)
    return f"{k[0]:.1f},{k[1]:.1f}"


async def fetch_conditions_grid(lats, lons) -> dict:
    """Fetch CURRENT conditions per 1.0° cell, in DISPLAY UNITS (°F, mph,
    sea-level hPa) — UI-only, never model input.

    One batched multi-location call per refresh (hourly TTL upstream) replaces
    the old one-API-call-per-tract-click design, which both burned quota and
    froze to a single static fallback during the quota outage (the
    '68°F everywhere' report).

    Returns {'by_key': {cell_key: {temperature, humidity, pressure, wind_speed}},
             'n_cells', 'usable', 'fetched_at'}.
    """
    cells, _ = build_cells(lats, lons, grid_deg=WEATHER_GRID_DEG)
    out = {"by_key": {}, "n_cells": len(cells), "usable": False,
           "fetched_at": datetime.now(timezone.utc).isoformat()}
    if not cells:
        return out
    lat_s = ",".join(f"{c[0]:.4f}" for c in cells)
    lon_s = ",".join(f"{c[1]:.4f}" for c in cells)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(MET_URL, params={
                "latitude": lat_s, "longitude": lon_s,
                "current": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "UTC",
            })
            if r.status_code != 200:
                print(f"[conditions] HTTP {r.status_code}: {r.text[:160]}")
                return out
            payload = r.json()
            if isinstance(payload, dict):
                payload = [payload]
            for i, loc in enumerate(payload):
                cur = loc.get("current", {})
                if cur.get("temperature_2m") is None:
                    continue
                key = conditions_cell_key(cells[i][0], cells[i][1])
                out["by_key"][key] = {
                    "temperature": round(float(cur["temperature_2m"]), 1),
                    "humidity":    round(float(cur.get("relative_humidity_2m") or 0), 1),
                    "pressure":    round(float(cur.get("pressure_msl") or 1013.0), 1),
                    "wind_speed":  round(float(cur.get("wind_speed_10m") or 0), 1),
                }
            out["usable"] = len(out["by_key"]) > 0
    except Exception as e:
        print(f"[conditions] fetch error: {e}")
    return out


async def fetch_airquality_snapshot(lats, lons, include_met: bool = True) -> dict:
    """Build a per-tract feature snapshot for all tracts. Returns
    {'features': {feat: [per-tract values]}, 'fetched_at', 'n_cells', 'usable'}.
    Values are None where the API didn't return them (backend fills with the
    training median). include_met=False skips the met call (v6 dropped those
    features) to conserve the api.open-meteo.com daily quota."""
    cells, idx = build_cells(lats, lons)
    cell_feats = await fetch_cell_features(cells, include_met=include_met)
    feat_names = ["aod", "cams_pm25", "dust", "shortwave", "et0", "cloud_cover"]
    per_tract = {f: [None] * len(lats) for f in feat_names}
    usable = False
    for i in range(len(lats)):
        cf = cell_feats.get(int(idx[i]), {})
        for f in feat_names:
            v = cf.get(f)
            per_tract[f][i] = v
            if v is not None:
                usable = True
    return {
        "features": per_tract,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_cells": len(cells),
        "usable": usable,
    }
