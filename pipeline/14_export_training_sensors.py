"""Export the sensor locations the deployed model's dist_to_nearest_sensor
feature was built from, so the backend can serve the same feature.

Training (pipeline/03_train_enhanced.py load_data) computes the feature from
every sensor in its source file BEFORE the sensor QC step drops any of them:
load the file, parse `date`, drop rows with no date or no pm25, then take the
first (sensor_id, latitude, longitude) row per sensor. This script repeats
those steps exactly and writes the result to models/training_sensors.json as
[{sensor_id, lat, lon}].

Source resolution matches training: p2_processed_v2.xls when present, else the
tracked pipeline/purpleair_training_ready.parquet. The two hold the same rows
(the .xls is a to_csv dump of the parquet), so a clean clone without the
gitignored .xls exports the same file. The legacy 2025-only p2_processed.xls is
never used: it is a 240-sensor subset and moved the tract median from 7.8 km
to 11.0 km when the backend fell back to it.

The backend reads ONLY this JSON for the feature, and the quantum solver uses
it as the existing-sensor list when live PurpleAir data is unavailable. Re-run
after every retrain that changes the training data:

    python pipeline/14_export_training_sensors.py            # same source as training
    python pipeline/14_export_training_sensors.py --source parquet --out /tmp/x.json

A model trained elsewhere (for example the 8-8-26 retrain, which runs the same
load_data on its own dataset) is exported from that dataset by path:

    python pipeline/14_export_training_sensors.py \
        --source-path "8-8-26 huge retrain/dataset/p2_processed_v3.parquet"
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_PATH = os.path.join(ROOT, "p2_processed_v2.xls")
PARQUET_PATH = os.path.join(ROOT, "pipeline", "purpleair_training_ready.parquet")
OUT_PATH = os.path.join(ROOT, "models", "training_sensors.json")
TARGET = "pm25"


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2)
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_source(source: str, source_path: str | None = None) -> tuple[pd.DataFrame, str]:
    if source_path:
        # Same dispatch as 03_train_enhanced.load_file: parquet by extension,
        # anything else is a to_csv dump (the .xls files are CSV text).
        if source_path.endswith(".parquet"):
            return pd.read_parquet(source_path), source_path
        cols = ["sensor_id", "latitude", "longitude", "date", TARGET]
        return pd.read_csv(source_path, encoding="utf-8-sig", usecols=cols), source_path
    if source == "auto":
        source = "v2" if os.path.exists(V2_PATH) else "parquet"
    if source == "v2":
        # Same reader as 03_train_enhanced.load_file; only three columns are
        # needed, and usecols keeps the 146 MB file from being tokenized whole.
        cols = ["sensor_id", "latitude", "longitude", "date", TARGET]
        return pd.read_csv(V2_PATH, encoding="utf-8-sig", usecols=cols), V2_PATH
    return pd.read_parquet(PARQUET_PATH), PARQUET_PATH


def training_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """The pre-QC sensor table exactly as load_data builds it."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", TARGET])
    return (df[["sensor_id", "latitude", "longitude"]]
            .drop_duplicates("sensor_id")
            .reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", choices=["auto", "v2", "parquet"], default="auto")
    ap.add_argument("--source-path", default=None,
                    help="export from this .parquet/.csv/.xls instead of --source")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    df, path = load_source(args.source, args.source_path)
    sensors = training_sensors(df)
    print(f"source: {os.path.relpath(os.path.abspath(path), ROOT)} ({len(df):,} rows) "
          f"-> {len(sensors)} sensors")

    # Cross-check against the constant the backend clips to: training's
    # sensor -> nearest-OTHER-sensor maximum.
    lats = sensors["latitude"].values
    lons = sensors["longitude"].values
    nearest = np.empty(len(sensors))
    for i in range(len(sensors)):
        d = _haversine_km(lats[i], lons[i], lats, lons)
        d[i] = np.inf
        nearest[i] = d.min()
    print(f"nearest-other-sensor: median={np.median(nearest):.2f} km, "
          f"max={nearest.max():.2f} km")

    # Full float precision (json uses repr), so the backend's distances match
    # the training computation bit for bit.
    rows = [
        {"sensor_id": int(sid) if isinstance(sid, (int, np.integer)) else str(sid),
         "lat": float(lat), "lon": float(lon)}
        for sid, lat, lon in zip(sensors["sensor_id"], lats, lons)
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        # One sensor per line keeps diffs readable after a retrain.
        f.write("[\n" + ",\n".join(json.dumps(r) for r in rows) + "\n]\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
