"""
Maximize 1M PurpleAir API points → multi-year daily PM2.5 for every active outdoor
TX sensor, joined with Open-Meteo (free) weather, EJScreen, population, tract
centroids, and elevations. Output: a training-ready CSV with maximum row count
within budget, with QC filtering applied.

Strategy (per the project doc "Maximizing 1M PurpleAir API Points"):
  - Pull pm2.5_atm only from PurpleAir (one field keeps the per-row cost
    down; see the cost note below). Source weather from Open-Meteo
    historical archive at training time so train-serve distribution matches the
    backend's live inference path (which already calls Open-Meteo).
  - 1-day-average history is capped at 2-year windows per call → chunk pulls.
  - Resume-safe: every successful (HTTP 200) response is cached to disk; reruns
    skip cached chunks. Failures are never cached, so a rerun retries them.
  - Budget-aware: track points consumed; stop pulling new sensors when remaining
    budget falls below a per-sensor reserve.

Run:
    cd real-time-map
    python3 pipeline/06_pull_purpleair_full.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

# Read key comes from the environment only. It was previously hardcoded here and
# in backend/purpleair.py, in a public repository — scrapers harvest keys from
# public pushes within minutes, and PurpleAir bills per call against the owning
# account. Export PURPLEAIR_API_KEY before running a pull.
API_KEY = os.environ.get("PURPLEAIR_API_KEY", "").strip()
if not API_KEY:
    raise SystemExit(
        "PURPLEAIR_API_KEY is not set. Export your PurpleAir READ key before "
        "running this pull, e.g.  export PURPLEAIR_API_KEY=... "
        "(PowerShell: $env:PURPLEAIR_API_KEY='...')"
    )
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "pipeline" / "data_pull_cache"
PA_CACHE = CACHE_DIR / "purpleair"
OM_CACHE = CACHE_DIR / "openmeteo"
PA_CACHE.mkdir(parents=True, exist_ok=True)
OM_CACHE.mkdir(parents=True, exist_ok=True)

SENSOR_META_FILE = CACHE_DIR / "tx_sensors_meta.json"
BUDGET_LEDGER = CACHE_DIR / "budget_ledger.json"

# Texas bounding box (covers the whole state; we filter by FIPS later).
TX_BBOX = dict(nwlng=-106.65, nwlat=36.5, selng=-93.5, selat=25.84)

# Pull window. TODAY is the actual run date so re-running always advances the
# window to the present (it used to be hardcoded to 2026-05-03, which silently
# froze the dataset 6 weeks behind no matter when you re-ran). Earliest possible
# chunk start is anchored to each sensor's date_created. Latest chunk ends
# "yesterday" so we don't pull a partial day. Override TODAY with the
# PURPLEAIR_PULL_TODAY env var (YYYY-MM-DD) for reproducible/backfill runs.
_pull_today_env = os.environ.get("PURPLEAIR_PULL_TODAY", "").strip()
TODAY = date.fromisoformat(_pull_today_env) if _pull_today_env else date.today()
EARLIEST_START = date(2021, 1, 1)
LATEST_END = TODAY - timedelta(days=1)

# Per-run spending cap for the ledger below, not the account balance (PurpleAir
# points are bought per organization; check the dashboard). Reserve 50k for safety.
TOTAL_BUDGET = 1_000_000
SAFETY_RESERVE = 50_000
EFFECTIVE_BUDGET = TOTAL_BUDGET - SAFETY_RESERVE

# Cost model. The ledger below charges 2 + 2 * fields_count * rows per call, an
# older reading of PurpleAir's docs that was never checked against the account.
# The one measured figure (2026-10-20, the live map's statewide /v1/sensors
# call) is 3,484 points for 497 rows with 6 fields: about 7 points per row per
# call. That is empirical, not a published rate, and history calls were not
# measured, so treat the ledger as an estimate and check the real balance on the
# PurpleAir dashboard before a large pull.
PA_HISTORY_FIELDS = ["pm2.5_atm"]

# Retry schedule (seconds) for 429, 5xx and network errors on history calls. A
# window that still fails is skipped for this run and NOT cached, so a rerun
# retries it.
PA_RETRY_DELAYS = (10, 30, 60, 120, 300)
# Key rejected (401), out of points (402) or forbidden (403): every later call
# would fail the same way, so the run halts rather than marching on.
PA_HALT_STATUSES = {401, 402, 403}
SKIPPED_WINDOWS: list[str] = []

# QC thresholds: PM2.5 outside this range is sensor noise / fault.
# 200 µg/m³ is past EPA "Hazardous" (250.4+ AQI) and at the upper edge of
# PurpleAir PA-II's well-calibrated regime; values above 200 are mostly nonlinear
# sensor saturation, not real ambient PM. Keep wildfire/dust/fireworks events.
PM25_MIN_VALID = 0.0
PM25_MAX_VALID = 200.0
# Per-sensor robust-z MAD threshold (see pipeline/08_finish_pull.py for context).
PM25_MAD_Z_MAX = 20.0

# Open-Meteo archive endpoint
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_DAILY_FIELDS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "precipitation_sum",
]

# Politeness
PA_SLEEP = 0.05   # 20 req/s ceiling; PurpleAir hard cap = 30/min for history
OM_SLEEP = 0.6    # ~100/min, well under 600/hr

# --------------------------------------------------------------------------------------
# Budget ledger
# --------------------------------------------------------------------------------------

@dataclass
class BudgetLedger:
    points_used: int = 0
    history_calls: int = 0
    metadata_calls: int = 0
    rows_pulled: int = 0

    def save(self):
        BUDGET_LEDGER.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls) -> "BudgetLedger":
        if BUDGET_LEDGER.exists():
            return cls(**json.loads(BUDGET_LEDGER.read_text()))
        return cls()

    def remaining(self) -> int:
        return EFFECTIVE_BUDGET - self.points_used

LEDGER = BudgetLedger.load()

def charge(call_kind: str, fields_count: int, rows_count: int):
    cost = 2 + 2 * fields_count * rows_count
    LEDGER.points_used += cost
    if call_kind == "history":
        LEDGER.history_calls += 1
        LEDGER.rows_pulled += rows_count
    elif call_kind == "metadata":
        LEDGER.metadata_calls += 1
    LEDGER.save()
    return cost


# --------------------------------------------------------------------------------------
# Sensor metadata
# --------------------------------------------------------------------------------------

def fetch_sensor_metadata() -> list[dict]:
    """One-time pull of all TX outdoor public sensors."""
    if SENSOR_META_FILE.exists():
        meta = json.loads(SENSOR_META_FILE.read_text())
        print(f"[meta] using cached metadata: {len(meta['data'])} sensors")
        return meta["data"]

    fields = [
        "latitude", "longitude", "date_created", "last_seen",
        "location_type", "private", "name", "altitude", "model",
    ]
    params = {
        "fields": ",".join(fields),
        "location_type": 0,  # outdoor
        **TX_BBOX,
    }
    print("[meta] fetching TX sensor list…")
    r = requests.get(
        "https://api.purpleair.com/v1/sensors",
        headers={"X-API-Key": API_KEY},
        params=params,
        timeout=60,
    )
    r.raise_for_status()
    payload = r.json()
    rows = payload["data"]
    out_fields = payload["fields"]
    sensors = [dict(zip(out_fields, row)) for row in rows]
    # Charge: metadata pull cost = 2 + 2 * fields * sensors
    cost = charge("metadata", len(out_fields), len(sensors))
    payload["sensors"] = sensors
    SENSOR_META_FILE.write_text(json.dumps(payload, indent=2))
    print(f"[meta] {len(sensors)} sensors, cost {cost} pts (total used {LEDGER.points_used})")
    return sensors


def filter_sensors(sensors: list[dict]) -> list[dict]:
    """Drop bad sensors that won't help model accuracy:
        - private (already filtered server-side)
        - inactive > 30 days (last_seen)
        - too new (< 60 days of history)
        - not in TX (lat/lon sanity)
    """
    now_ts = int(time.time())
    keep = []
    for s in sensors:
        if s.get("private"):
            continue
        if s.get("location_type") != 0:
            continue
        last = s.get("last_seen") or 0
        created = s.get("date_created") or 0
        if (now_ts - last) > 30 * 86400:
            continue
        if (now_ts - created) < 60 * 86400:
            continue
        lat, lon = s.get("latitude"), s.get("longitude")
        if lat is None or lon is None:
            continue
        if not (25.5 <= lat <= 36.6 and -107 <= lon <= -93):
            continue
        keep.append(s)
    print(f"[filter] {len(keep)}/{len(sensors)} sensors pass quality filters")
    return keep


# --------------------------------------------------------------------------------------
# PurpleAir history pull (chunked, cached)
# --------------------------------------------------------------------------------------

def history_chunk_path(sensor_id: int, start: date, end: date) -> Path:
    return PA_CACHE / f"{sensor_id}_{start.isoformat()}_{end.isoformat()}.json"


def chunked_windows(start: date, end: date, max_days: int = 730) -> list[tuple[date, date]]:
    """Split [start, end) into windows ≤ max_days (PurpleAir 1-day-avg cap = 2 yrs)."""
    windows = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=max_days), end)
        windows.append((cur, nxt))
        cur = nxt
    return windows


def _retry_after_s(r: requests.Response, default: float) -> float:
    try:
        return max(default, float(r.headers.get("Retry-After", "")))
    except ValueError:
        return default


def fetch_history_chunk(sid: int, w_start: date, w_end: date) -> Optional[dict]:
    """One history call with retries. Returns the 200 payload, or None when the
    window should be skipped for this run. Only a 200 is ever cached: an empty
    chunk written for a failure would make every rerun skip that window for good."""
    params = dict(
        start_timestamp=int(datetime.combine(w_start, datetime.min.time(), tzinfo=timezone.utc).timestamp()),
        end_timestamp=int(datetime.combine(w_end, datetime.min.time(), tzinfo=timezone.utc).timestamp()),
        average=1440,
        fields=",".join(PA_HISTORY_FIELDS),
    )
    label = f"sensor {sid} {w_start}..{w_end}"
    for attempt in range(len(PA_RETRY_DELAYS) + 1):
        delay = PA_RETRY_DELAYS[min(attempt, len(PA_RETRY_DELAYS) - 1)]
        try:
            r = requests.get(
                f"https://api.purpleair.com/v1/sensors/{sid}/history",
                headers={"X-API-Key": API_KEY},
                params=params,
                timeout=60,
            )
        except requests.RequestException as e:
            reason = f"network error: {e}"
        else:
            if r.status_code == 200:
                return r.json()
            if r.status_code in PA_HALT_STATUSES:
                raise SystemExit(
                    f"[halt] {label} → HTTP {r.status_code}: {r.text[:200]}\n"
                    f"Nothing was cached for this window; fix the key or add points, "
                    f"then rerun (finished windows are read from the cache)."
                )
            if r.status_code != 429 and r.status_code < 500:
                # Other 4xx (e.g. a removed sensor): retrying will not help.
                print(f"[err] {label} → HTTP {r.status_code}: {r.text[:200]} (skipped, not cached)")
                SKIPPED_WINDOWS.append(label)
                return None
            reason = f"HTTP {r.status_code}"
            if r.status_code == 429:
                delay = _retry_after_s(r, delay)
        if attempt == len(PA_RETRY_DELAYS):
            break
        print(f"[retry] {label}: {reason}; retry {attempt + 1}/{len(PA_RETRY_DELAYS)} in {delay:.0f}s")
        time.sleep(delay)
    print(f"[err] {label}: still failing after {len(PA_RETRY_DELAYS)} retries ({reason}); skipped, not cached")
    SKIPPED_WINDOWS.append(label)
    return None


def pull_sensor_history(sensor: dict, sensor_start: date, end: date) -> Optional[pd.DataFrame]:
    """Pull pm2.5_atm daily averages for one sensor across [sensor_start, end).
    Returns concatenated DataFrame or None if pull fully failed."""
    sid = sensor["sensor_index"]
    frames = []
    for w_start, w_end in chunked_windows(sensor_start, end, max_days=720):
        cache = history_chunk_path(sid, w_start, w_end)
        payload = json.loads(cache.read_text()) if cache.exists() else None
        # Older versions of this script cached failures as empty "_error" chunks;
        # those are re-fetched rather than trusted as real empty windows.
        if payload is not None and "_error" in payload:
            payload = None
        if payload is None:
            # Budget gate before each call: assume worst-case 2-year chunk = ~1462 pts
            est_cost = 2 + 2 * len(PA_HISTORY_FIELDS) * (w_end - w_start).days
            if LEDGER.remaining() < est_cost + 5_000:
                print(f"[budget] STOP — {LEDGER.remaining()} pts left, would not fit {est_cost} for sensor {sid} {w_start}..{w_end}")
                return None
            payload = fetch_history_chunk(sid, w_start, w_end)
            if payload is None:
                continue
            cache.write_text(json.dumps(payload))
            rows = payload.get("data", [])
            charge("history", len(PA_HISTORY_FIELDS), len(rows))
            time.sleep(PA_SLEEP)
        # Parse
        rows = payload.get("data", [])
        if not rows:
            continue
        cols = payload["fields"]
        df = pd.DataFrame(rows, columns=cols)
        df["sensor_id"] = sid
        df["latitude"] = sensor["latitude"]
        df["longitude"] = sensor["longitude"]
        df["altitude"] = sensor.get("altitude")
        frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["time_stamp"], unit="s", utc=True).dt.tz_convert(None).dt.normalize()
    out = out.drop(columns=["time_stamp"])
    return out


# --------------------------------------------------------------------------------------
# Open-Meteo historical weather (free)
# --------------------------------------------------------------------------------------

def om_cache_path(lat: float, lon: float, start: date, end: date) -> Path:
    key = f"{lat:.4f}_{lon:.4f}_{start.isoformat()}_{end.isoformat()}.json"
    return OM_CACHE / key


def fetch_open_meteo(lat: float, lon: float, start: date, end: date) -> Optional[pd.DataFrame]:
    cache = om_cache_path(lat, lon, start, end)
    if cache.exists():
        payload = json.loads(cache.read_text())
    else:
        params = dict(
            latitude=lat,
            longitude=lon,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            daily=",".join(OPEN_METEO_DAILY_FIELDS),
            timezone="UTC",
            # Match NASA POWER's units so a re-pull yields a consistent dataset.
            # Open-Meteo defaults to kmh/celsius/mm — we force ms/celsius/mm.
            wind_speed_unit="ms",
            temperature_unit="celsius",
            precipitation_unit="mm",
        )
        attempts = 0
        while True:
            try:
                r = requests.get(OPEN_METEO_URL, params=params, timeout=60)
            except Exception as e:
                attempts += 1
                if attempts > 3:
                    print(f"[om-err] {lat},{lon}: {e}")
                    return None
                time.sleep(5)
                continue
            if r.status_code == 429:
                time.sleep(30)
                continue
            if r.status_code != 200:
                print(f"[om-err] {lat},{lon} HTTP {r.status_code}: {r.text[:200]}")
                cache.write_text(json.dumps({"_error": r.text[:500]}))
                return None
            payload = r.json()
            cache.write_text(json.dumps(payload))
            time.sleep(OM_SLEEP)
            break
    daily = payload.get("daily")
    if not daily or "time" not in daily:
        return None
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    df = df.drop(columns=["time"])
    # Cache files written before wind_speed_unit=ms was added are km/h, and the
    # cache key ignores units, so a rerun can read them back. Convert by the file's
    # own daily_units (same rule as 08's load_open_meteo) so every row is m/s.
    units = payload.get("daily_units") or {}
    for col in ("wind_speed_10m_max", "wind_gusts_10m_max"):
        if col in df.columns and "km" in str(units.get(col, "")).lower():
            df[col] = df[col] / 3.6
    df["latitude"] = lat
    df["longitude"] = lon
    return df


# --------------------------------------------------------------------------------------
# Tract / EJ join
# --------------------------------------------------------------------------------------

def load_tract_lookup() -> pd.DataFrame:
    """Return tract_lookup with full EJ + spatial features."""
    p = ROOT / "backend" / "static" / "tract_lookup.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        return df
    # fallback: rebuild from raw
    raise FileNotFoundError(p)


def assign_geoid(df_points: pd.DataFrame, tracts: pd.DataFrame) -> pd.DataFrame:
    """Assign GEOID by nearest-tract-centroid (haversine). Vectorized in chunks."""
    import numpy as np
    pts_lat = df_points["latitude"].to_numpy()
    pts_lon = df_points["longitude"].to_numpy()
    tr_lat = tracts["lat"].to_numpy()
    tr_lon = tracts["lon"].to_numpy()
    tr_geoid = tracts["GEOID"].to_numpy()

    R = 6371.0
    geoids = np.empty(len(pts_lat), dtype=tr_geoid.dtype)
    chunk = 200
    p_lat_r = np.radians(pts_lat)
    p_lon_r = np.radians(pts_lon)
    t_lat_r = np.radians(tr_lat)
    t_lon_r = np.radians(tr_lon)

    for i in range(0, len(pts_lat), chunk):
        a = slice(i, i + chunk)
        dlat = p_lat_r[a, None] - t_lat_r[None, :]
        dlon = p_lon_r[a, None] - t_lon_r[None, :]
        h = (np.sin(dlat / 2) ** 2
             + np.cos(p_lat_r[a, None]) * np.cos(t_lat_r[None, :]) * np.sin(dlon / 2) ** 2)
        d = 2 * R * np.arcsin(np.sqrt(h))
        geoids[a] = tr_geoid[np.argmin(d, axis=1)]
    df_points = df_points.copy()
    df_points["GEOID"] = geoids
    return df_points


# --------------------------------------------------------------------------------------
# Quality control / outlier removal
# --------------------------------------------------------------------------------------

def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that hurt model accuracy:
       - PM2.5 NaN, < PM25_MIN_VALID, or > PM25_MAX_VALID (sensor faults)
       - Per-sensor IQR-based outliers (z-score-ish, beyond 4σ from sensor median)
       - Sensors with < 60 valid days (too noisy / unstable)
    """
    n0 = len(df)
    df = df.dropna(subset=["pm2.5_atm"])
    df = df[(df["pm2.5_atm"] >= PM25_MIN_VALID) & (df["pm2.5_atm"] <= PM25_MAX_VALID)]
    n1 = len(df)
    print(f"[qc] removed {n0-n1} rows for hard PM2.5 limits ({n0} → {n1})")

    # Per-sensor robust outlier removal (median ± PM25_MAD_Z_MAX * MAD).
    grp = df.groupby("sensor_id")["pm2.5_atm"]
    med = grp.transform("median")
    mad = grp.transform(lambda x: (x - x.median()).abs().median())
    mad = mad.replace(0, mad[mad > 0].median() if (mad > 0).any() else 1.0)
    z = (df["pm2.5_atm"] - med).abs() / (1.4826 * mad)
    df = df[z < PM25_MAD_Z_MAX]
    n2 = len(df)
    print(f"[qc] removed {n1-n2} per-sensor outliers (>{PM25_MAD_Z_MAX:g} MAD) ({n1} → {n2})")

    # Drop sensors with too few rows
    counts = df.groupby("sensor_id").size()
    good = counts[counts >= 60].index
    df = df[df["sensor_id"].isin(good)]
    n3 = len(df)
    print(f"[qc] removed {n2-n3} rows from sensors with <60 valid days "
          f"({len(counts)-len(good)} sensors dropped) ({n2} → {n3})")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------

def existing_2025_data() -> pd.DataFrame:
    """Load p2_processed.xls (CSV) — pre-existing 2025 daily data for 240 sensors."""
    p = ROOT / "p2_processed.xls"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.rename(columns={"pm25": "pm2.5_atm"})
    df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)
    keep = ["sensor_id", "date", "pm2.5_atm", "latitude", "longitude", "GEOID"]
    return df[keep]


def main():
    print(f"=== Run start. Budget: {EFFECTIVE_BUDGET:,} points (1M − {SAFETY_RESERVE:,} reserve)")
    print(f"=== Already used: {LEDGER.points_used:,}, remaining: {LEDGER.remaining():,}")

    # Step 1: sensor list
    sensors = fetch_sensor_metadata()
    sensors = filter_sensors(sensors)

    # Step 2: existing 2025 data → mark sensor-date pairs already covered
    existing = existing_2025_data()
    existing_sensor_ids = set(existing["sensor_id"].unique())
    print(f"[existing] {len(existing):,} rows / {len(existing_sensor_ids)} sensors loaded from p2_processed.xls")

    # Step 3: prioritize pull order
    #   1. New sensors (not in existing) → most marginal value, pull first
    #   2. Then existing sensors → only the years not in 2025 data
    new_sensors = [s for s in sensors if s["sensor_index"] not in existing_sensor_ids]
    overlap_sensors = [s for s in sensors if s["sensor_index"] in existing_sensor_ids]
    # Within new, prioritize older sensors (more potential history)
    new_sensors.sort(key=lambda s: s["date_created"])
    overlap_sensors.sort(key=lambda s: s["date_created"])
    pull_order = new_sensors + overlap_sensors
    print(f"[plan] pulling {len(new_sensors)} new + {len(overlap_sensors)} overlap sensors")

    # Step 4: history pulls
    all_pa_frames = []
    for i, sensor in enumerate(pull_order, 1):
        sid = sensor["sensor_index"]
        if LEDGER.remaining() < 5_000:
            print(f"[budget] only {LEDGER.remaining()} pts remaining — stopping pull")
            break
        # Each sensor's window
        sensor_start = max(EARLIEST_START, datetime.utcfromtimestamp(sensor["date_created"]).date() + timedelta(days=1))
        sensor_end = LATEST_END
        if sensor_start >= sensor_end:
            continue
        # If it's an overlap sensor, exclude 2025 (we already have it)
        # We still pull pre-2025 and 2026 separately.
        if sid in existing_sensor_ids:
            existing_dates = set(existing.loc[existing["sensor_id"] == sid, "date"].dt.date.tolist())
        else:
            existing_dates = set()
        df = pull_sensor_history(sensor, sensor_start, sensor_end)
        if df is not None and len(df):
            if existing_dates:
                df = df[~df["date"].dt.date.isin(existing_dates)]
            all_pa_frames.append(df)
        if i % 25 == 0 or i == len(pull_order):
            n_rows = sum(len(f) for f in all_pa_frames)
            print(f"[progress] {i}/{len(pull_order)} sensors | {n_rows:,} new rows | "
                  f"{LEDGER.points_used:,} pts used | {LEDGER.remaining():,} remaining")

    if SKIPPED_WINDOWS:
        print(f"[warn] {len(SKIPPED_WINDOWS)} history windows failed and were skipped "
              f"(not cached); rerun to fill them. First few: {SKIPPED_WINDOWS[:5]}")

    # Step 5: union with existing 2025 data
    if all_pa_frames:
        new_df = pd.concat(all_pa_frames, ignore_index=True)
        new_df = new_df[["sensor_id", "date", "pm2.5_atm", "latitude", "longitude"]]
    else:
        new_df = pd.DataFrame(columns=["sensor_id", "date", "pm2.5_atm", "latitude", "longitude"])
    print(f"[merge] new PurpleAir rows: {len(new_df):,}")
    print(f"[merge] existing 2025 rows: {len(existing):,}")
    existing_min = existing[["sensor_id", "date", "pm2.5_atm", "latitude", "longitude"]]
    pa_all = pd.concat([new_df, existing_min], ignore_index=True)
    pa_all = pa_all.drop_duplicates(subset=["sensor_id", "date"]).reset_index(drop=True)
    print(f"[merge] total rows after dedupe: {len(pa_all):,}")

    # Step 6: tract + EJ join. p2_processed already has GEOID for the 2025 rows.
    # We only need to assign GEOID for sensors that did NOT come from p2_processed.
    tracts = load_tract_lookup()
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
    print(f"[tracts] tract_lookup rows: {len(tracts):,} cols: {list(tracts.columns)}")
    # Existing GEOID (string-zfilled) per sensor from p2_processed
    existing_geoid = (
        existing[["sensor_id", "GEOID"]].drop_duplicates("sensor_id")
    )
    sensor_pts = pa_all[["sensor_id", "latitude", "longitude"]].drop_duplicates("sensor_id").reset_index(drop=True)
    sensor_pts = sensor_pts.merge(existing_geoid, on="sensor_id", how="left")
    # For any sensor missing GEOID, compute via nearest tract centroid
    missing_mask = sensor_pts["GEOID"].isna()
    print(f"[geoid] {missing_mask.sum()} sensors need GEOID assignment via nearest centroid")
    if missing_mask.any():
        assigned = assign_geoid(sensor_pts.loc[missing_mask, ["sensor_id", "latitude", "longitude"]].reset_index(drop=True), tracts)
        sensor_pts.loc[missing_mask, "GEOID"] = assigned["GEOID"].astype(str).str.zfill(11).values
    sensor_pts["GEOID"] = sensor_pts["GEOID"].astype(str).str.zfill(11)
    pa_all = pa_all.drop(columns=["GEOID"], errors="ignore").merge(
        sensor_pts[["sensor_id", "GEOID"]], on="sensor_id", how="left"
    )
    pa_all = pa_all.merge(tracts, on="GEOID", how="left", suffixes=("", "_tract"))
    print(f"[join] after EJ join: {pa_all.shape}")

    # Save raw (pre-weather) checkpoint
    raw_path = ROOT / "pipeline" / "purpleair_pm25_raw.parquet"
    pa_all.to_parquet(raw_path, index=False)
    print(f"[save] raw PA+EJ → {raw_path}")

    # Step 7: Open-Meteo backfill — one call per sensor across full date span
    print("[weather] backfilling weather from Open-Meteo …")
    weather_frames = []
    for i, row in sensor_pts.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        sid = int(row["sensor_id"])
        # Get this sensor's date span
        spans = pa_all.loc[pa_all["sensor_id"] == sid, "date"]
        if spans.empty:
            continue
        s_start = spans.min().date()
        s_end = spans.max().date()
        wdf = fetch_open_meteo(lat, lon, s_start, s_end)
        if wdf is None or wdf.empty:
            continue
        wdf["sensor_id"] = sid
        weather_frames.append(wdf)
        if (i + 1) % 25 == 0:
            print(f"[weather] {i+1}/{len(sensor_pts)} sensors backfilled")
    if weather_frames:
        weather = pd.concat(weather_frames, ignore_index=True)
        weather = weather.drop(columns=["latitude", "longitude"], errors="ignore")
        weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()
        pa_all["date"] = pd.to_datetime(pa_all["date"]).dt.normalize()
        merged = pa_all.merge(weather, on=["sensor_id", "date"], how="left")
    else:
        merged = pa_all
    print(f"[merge] after weather: {merged.shape}")

    # Step 8: feature engineering (cyclical + interactions, mirrors 03_train_enhanced.py)
    import numpy as np
    merged["month"] = merged["date"].dt.month
    merged["dow"] = merged["date"].dt.dayofweek
    merged["doy"] = merged["date"].dt.dayofyear
    merged["hour"] = 12  # daily averages → noon proxy (kept for backward compat with v3 features)
    merged["month_sin"] = np.sin(2 * np.pi * merged["month"] / 12)
    merged["month_cos"] = np.cos(2 * np.pi * merged["month"] / 12)
    merged["dow_sin"] = np.sin(2 * np.pi * merged["dow"] / 7)
    merged["dow_cos"] = np.cos(2 * np.pi * merged["dow"] / 7)
    merged["doy_sin"] = np.sin(2 * np.pi * merged["doy"] / 365)
    merged["doy_cos"] = np.cos(2 * np.pi * merged["doy"] / 365)

    # Rename to backend-canonical column names so the dataset slots into 03_train_enhanced.py
    rename_map = {
        "pm2.5_atm": "pm25",
        "temperature_2m_mean": "temperature",
        "relative_humidity_2m_mean": "humidity",
        "surface_pressure_mean": "pressure",
        "wind_speed_10m_max": "wind_speed",
        "wind_gusts_10m_max": "wind_gusts",
        "precipitation_sum": "precipitation",
    }
    merged = merged.rename(columns=rename_map)
    # Interaction features
    if {"temperature", "humidity"} <= set(merged.columns):
        merged["temp_x_humidity"] = merged["temperature"] * merged["humidity"] / 100
    if {"wind_speed", "temperature"} <= set(merged.columns):
        merged["wind_x_temp"] = merged["wind_speed"] * merged["temperature"] / 100

    # Step 9: QC filter
    merged = quality_filter(merged)

    # Step 10: save final dataset (parquet + csv)
    out_parquet = ROOT / "pipeline" / "purpleair_full_dataset.parquet"
    out_csv = ROOT / "pipeline" / "purpleair_full_dataset.csv"
    merged.to_parquet(out_parquet, index=False)
    merged.to_csv(out_csv, index=False)
    print(f"\n=== DONE ===")
    print(f"Final dataset rows: {len(merged):,}")
    print(f"Sensors retained:    {merged['sensor_id'].nunique()}")
    print(f"Date range:          {merged['date'].min().date()} → {merged['date'].max().date()}")
    print(f"Columns ({len(merged.columns)}): {list(merged.columns)}")
    print(f"PA points used:      {LEDGER.points_used:,} / {EFFECTIVE_BUDGET:,}")
    print(f"PA history calls:    {LEDGER.history_calls:,}")
    print(f"PA rows pulled:      {LEDGER.rows_pulled:,}")
    print(f"Saved → {out_parquet}")
    print(f"Saved → {out_csv}")


if __name__ == "__main__":
    main()
