"""Finisher script (resume after Open-Meteo daily-quota exhaustion).

State at this point:
  - PurpleAir history pulls completed: 487 sensors → ~428k pm2.5 daily rows.
  - PA → EJ join checkpoint saved at pipeline/purpleair_pm25_raw.parquet.
  - Open-Meteo daily quota exhausted at 252 sensors. Their 5y daily weather is
    cached in pipeline/data_pull_cache/openmeteo/.

This script:
  1. Loads the raw PA+EJ checkpoint.
  2. For each sensor:
       - if Open-Meteo cache exists → use it.
       - else → fall back to NASA POWER daily (free, no rate limit).
  3. Computes temporal + cyclical + interaction features.
  4. Applies QC filters (NaN/range hard limits, per-sensor 6-MAD outliers,
     sensors with < 60 valid days dropped).
  5. Writes:
       - pipeline/purpleair_full_dataset.parquet
       - pipeline/purpleair_full_dataset.csv
       - p2_processed_v2.xls  (drop-in replacement for the existing xls)
"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "pipeline" / "purpleair_pm25_raw.parquet"
OM_CACHE = ROOT / "pipeline" / "data_pull_cache" / "openmeteo"
NP_CACHE = ROOT / "pipeline" / "data_pull_cache" / "nasapower"
NP_CACHE.mkdir(parents=True, exist_ok=True)

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_PARAMS = ["T2M", "RH2M", "PS", "WS10M", "WS10M_MAX", "PRECTOTCORR"]
# Mapping into our canonical training column names:
NASA_RENAME = {
    "T2M":          "temperature",       # °C
    "RH2M":         "humidity",          # %
    "PS":           "pressure",          # kPa  (we'll convert → hPa to match Open-Meteo's hPa)
    "WS10M":        "wind_speed_mean",   # m/s — kept for reference
    "WS10M_MAX":    "wind_speed",        # m/s (max-of-day, matches Open-Meteo's wind_speed_10m_max)
    "PRECTOTCORR":  "precipitation",     # mm/day
}

OPEN_METEO_RENAME = {
    "temperature_2m_mean":         "temperature",
    "relative_humidity_2m_mean":   "humidity",
    "surface_pressure_mean":       "pressure",
    "wind_speed_10m_max":          "wind_speed",
    "wind_gusts_10m_max":          "wind_gusts",
    "precipitation_sum":           "precipitation",
}

PM25_MIN, PM25_MAX = 0.0, 200.0
# Per-sensor robust-z outlier threshold. 6 was so tight that EVERY high-PM event
# (NYE fireworks, wildfire smoke, dust storms) got clipped — the dataset's max
# collapsed to 77.8 µg/m³ with zero rows >100. 20 keeps real events while still
# rejecting obvious single-row sensor faults.
MAD_Z_MAX = 20.0


# --------------------------------------------------------------------------------------
# Weather: NASA POWER fallback (free, no rate limit)
# --------------------------------------------------------------------------------------

def np_cache_path(lat: float, lon: float, start: date, end: date) -> Path:
    return NP_CACHE / f"{lat:.4f}_{lon:.4f}_{start.isoformat()}_{end.isoformat()}.json"


def fetch_nasa_power(lat: float, lon: float, start: date, end: date) -> Optional[pd.DataFrame]:
    cache = np_cache_path(lat, lon, start, end)
    if cache.exists():
        payload = json.loads(cache.read_text())
    else:
        params = dict(
            parameters=",".join(NASA_PARAMS),
            community="RE",
            longitude=lon,
            latitude=lat,
            start=start.strftime("%Y%m%d"),
            end=end.strftime("%Y%m%d"),
            format="JSON",
        )
        attempts = 0
        while True:
            try:
                r = requests.get(NASA_POWER_URL, params=params, timeout=120)
            except Exception as e:
                attempts += 1
                if attempts > 3:
                    print(f"[np-err] {lat},{lon}: {e}")
                    return None
                time.sleep(5)
                continue
            if r.status_code == 429:
                time.sleep(30)
                continue
            if r.status_code != 200:
                print(f"[np-err] {lat},{lon} HTTP {r.status_code}: {r.text[:200]}")
                cache.write_text(json.dumps({"_error": r.text[:500]}))
                return None
            payload = r.json()
            cache.write_text(json.dumps(payload))
            time.sleep(0.6)  # be polite
            break
    props = payload.get("properties")
    if not props or "parameter" not in props:
        return None
    params = props["parameter"]
    if "T2M" not in params:
        return None
    # Build wide DataFrame
    dates = sorted(params["T2M"].keys())
    rows = []
    for d in dates:
        rec = {"date": pd.to_datetime(d, format="%Y%m%d")}
        for k in NASA_PARAMS:
            v = params.get(k, {}).get(d)
            # NASA POWER uses -999 as sentinel for missing
            if v is None or v == -999:
                rec[k] = None
            else:
                rec[k] = v
        rows.append(rec)
    df = pd.DataFrame(rows)
    df = df.rename(columns=NASA_RENAME)
    # PS is kPa → convert to hPa to match Open-Meteo
    if "pressure" in df.columns:
        df["pressure"] = df["pressure"] * 10.0
    df["weather_source"] = "nasa_power"
    return df


def load_open_meteo(lat: float, lon: float) -> Optional[pd.DataFrame]:
    """Find any cached OM file for this lat/lon (regardless of date range)."""
    matches = list(OM_CACHE.glob(f"{lat:.4f}_{lon:.4f}_*.json"))
    if not matches:
        return None
    # Pick the largest / most-data one
    matches.sort(key=lambda p: p.stat().st_size, reverse=True)
    payload = json.loads(matches[0].read_text())
    daily = payload.get("daily")
    if not daily or "time" not in daily:
        return None
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    df = df.drop(columns=["time"])
    df = df.rename(columns=OPEN_METEO_RENAME)
    df["weather_source"] = "open_meteo"
    # All rows must share NASA POWER's m/s, or trees learn a "which source?" split
    # instead of weather→PM physics. The existing cache files are km/h (Open-Meteo's
    # default), but 06 now requests wind_speed_unit=ms, so convert only what the
    # file's own daily_units say is km/h; dividing a fresh m/s pull would shrink it.
    units = payload.get("daily_units") or {}
    for raw, col in (("wind_speed_10m_max", "wind_speed"), ("wind_gusts_10m_max", "wind_gusts")):
        if col in df.columns and "km" in str(units.get(raw, "")).lower():
            df[col] = df[col] / 3.6
    return df


# --------------------------------------------------------------------------------------
# QC
# --------------------------------------------------------------------------------------

def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna(subset=["pm25"])
    df = df[(df["pm25"] >= PM25_MIN) & (df["pm25"] <= PM25_MAX)]
    n1 = len(df)
    print(f"[qc] hard PM2.5 limits: removed {n0-n1:,} rows ({n0:,} → {n1:,})")
    grp = df.groupby("sensor_id")["pm25"]
    med = grp.transform("median")
    mad = grp.transform(lambda x: (x - x.median()).abs().median())
    mad = mad.replace(0, mad[mad > 0].median() if (mad > 0).any() else 1.0)
    z = (df["pm25"] - med).abs() / (1.4826 * mad)
    df = df[z < MAD_Z_MAX]
    n2 = len(df)
    print(f"[qc] per-sensor >{MAD_Z_MAX:g} MAD outliers: removed {n1-n2:,} rows ({n1:,} → {n2:,})")
    counts = df.groupby("sensor_id").size()
    good = counts[counts >= 60].index
    n_dropped_sensors = len(counts) - len(good)
    df = df[df["sensor_id"].isin(good)]
    n3 = len(df)
    print(f"[qc] sensors with < 60 valid days: dropped {n_dropped_sensors} sensors / {n2-n3:,} rows ({n2:,} → {n3:,})")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    print(f"[load] {RAW}")
    pa = pd.read_parquet(RAW)
    pa["date"] = pd.to_datetime(pa["date"]).dt.normalize()
    print(f"[load] {len(pa):,} rows / {pa['sensor_id'].nunique()} sensors")
    print(f"[load] cols: {list(pa.columns)}")

    # Per-sensor weather (one source per sensor)
    sensor_pts = (
        pa[["sensor_id", "latitude", "longitude"]].drop_duplicates("sensor_id").reset_index(drop=True)
    )
    print(f"[weather] backfilling for {len(sensor_pts)} sensors "
          f"(Open-Meteo cache → NASA POWER fallback)")

    weather_frames = []
    om_hits = 0
    np_hits = 0
    for i, row in sensor_pts.iterrows():
        sid = int(row["sensor_id"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        sensor_dates = pa.loc[pa["sensor_id"] == sid, "date"]
        if sensor_dates.empty:
            continue
        s_start = sensor_dates.min().date()
        s_end = sensor_dates.max().date()
        wdf = load_open_meteo(lat, lon)
        if wdf is None:
            wdf = fetch_nasa_power(lat, lon, s_start, s_end)
            if wdf is not None:
                np_hits += 1
        else:
            om_hits += 1
        if wdf is None:
            continue
        wdf["sensor_id"] = sid
        weather_frames.append(wdf)
        if (i + 1) % 25 == 0 or (i + 1) == len(sensor_pts):
            print(f"[weather] {i+1}/{len(sensor_pts)}  om={om_hits} nasa_power={np_hits}")

    weather = pd.concat(weather_frames, ignore_index=True) if weather_frames else pd.DataFrame()
    print(f"[weather] total rows: {len(weather):,}  cols: {list(weather.columns)}")

    # Merge weather with PA on (sensor_id, date)
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()
    weather_keep = ["sensor_id", "date", "temperature", "humidity", "pressure",
                    "wind_speed", "precipitation"]
    if "wind_gusts" in weather.columns:
        weather_keep.append("wind_gusts")
    weather_keep.append("weather_source")
    weather = weather[[c for c in weather_keep if c in weather.columns]]
    merged = pa.merge(weather, on=["sensor_id", "date"], how="left")

    # Rename PA pm column to canonical
    merged = merged.rename(columns={"pm2.5_atm": "pm25"})

    # Temporal features (matches pipeline/03_train_enhanced.py FEATURES)
    merged["month"] = merged["date"].dt.month
    merged["dow"] = merged["date"].dt.dayofweek
    merged["day_of_year"] = merged["date"].dt.dayofyear
    merged["hour"] = 12  # daily averages → noon proxy

    merged["month_sin"] = np.sin(2 * np.pi * merged["month"] / 12)
    merged["month_cos"] = np.cos(2 * np.pi * merged["month"] / 12)
    merged["dow_sin"] = np.sin(2 * np.pi * merged["dow"] / 7)
    merged["dow_cos"] = np.cos(2 * np.pi * merged["dow"] / 7)
    merged["doy_sin"] = np.sin(2 * np.pi * merged["day_of_year"] / 365)
    merged["doy_cos"] = np.cos(2 * np.pi * merged["day_of_year"] / 365)

    if {"temperature", "humidity"} <= set(merged.columns):
        merged["temp_x_humidity"] = merged["temperature"] * merged["humidity"] / 100
    if {"wind_speed", "temperature"} <= set(merged.columns):
        merged["wind_x_temp"] = merged["wind_speed"] * merged["temperature"] / 100

    # Season (legacy compat)
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Fall", 10: "Fall", 11: "Fall"}
    merged["season"] = merged["month"].map(season_map)
    if "city" not in merged.columns:
        merged["city"] = ""

    print(f"[merge] post-weather shape: {merged.shape}")
    print(f"[merge] weather coverage: {merged['temperature'].notna().sum():,}/{len(merged):,} rows have temperature")

    merged = quality_filter(merged)

    out_parquet = ROOT / "pipeline" / "purpleair_full_dataset.parquet"
    out_csv = ROOT / "pipeline" / "purpleair_full_dataset.csv"
    out_legacy = ROOT / "p2_processed_v2.xls"
    merged.to_parquet(out_parquet, index=False)
    merged.to_csv(out_csv, index=False)
    merged.to_csv(out_legacy, index=False)

    print()
    print("=" * 70)
    print("FINAL DATASET")
    print("=" * 70)
    print(f"Rows:      {len(merged):,}")
    print(f"Sensors:   {merged['sensor_id'].nunique()}")
    print(f"Date span: {merged['date'].min().date()} → {merged['date'].max().date()}")
    print(f"PM2.5:     min={merged['pm25'].min():.2f}  mean={merged['pm25'].mean():.2f}  "
          f"max={merged['pm25'].max():.2f}  p99={merged['pm25'].quantile(0.99):.2f}")
    if "weather_source" in merged.columns:
        wsrc = merged.groupby("weather_source").size().to_dict()
        print(f"Weather source split: {wsrc}")
    print(f"Columns ({len(merged.columns)}): {list(merged.columns)}")
    print()
    print(f"Saved → {out_parquet}")
    print(f"Saved → {out_csv}")
    print(f"Saved → {out_legacy}  (drop-in replacement for p2_processed.xls)")


if __name__ == "__main__":
    main()
