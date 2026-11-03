"""Serving features must be computed the way training computed them."""
import ast
import asyncio
import importlib.util
import json
import os
from datetime import date, datetime, timedelta, timezone

import httpx
import numpy as np
import pandas as pd
import pytest

from backend import airquality, config, model, predictions, weather_feed
from backend.purpleair import MIN_LIVE_SENSORS, NBR_RADII_KM, compute_neighbor_features

PIPELINE_DIR = os.path.join(config.ROOT, "pipeline")


def _load_pipeline_module(filename):
    """Import a pipeline script by path (some names start with a digit)."""
    path = os.path.join(PIPELINE_DIR, filename)
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py").replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _training_constant(name):
    """A literal module-level constant from 03_train_enhanced.py, read with ast
    so the training script (and its heavy imports) never runs."""
    with open(os.path.join(PIPELINE_DIR, "03_train_enhanced.py")) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


# ── Neighbor features ────────────────────────────────────────────────────────

def _synthetic_network(seed=7):
    """Sensors in two metro clusters plus a sparse rural scatter, and query
    points chosen to hit every branch: dense (all radii), mid-range (no 25 km
    neighbor), and isolated (no neighbor within 100 km)."""
    rng = np.random.default_rng(seed)
    dallas = np.column_stack([rng.normal(32.8, 0.12, 25), rng.normal(-96.8, 0.12, 25)])
    houston = np.column_stack([rng.normal(29.76, 0.15, 20), rng.normal(-95.37, 0.15, 20)])
    rural = np.column_stack([rng.uniform(30.0, 34.0, 12), rng.uniform(-102.0, -98.5, 12)])
    s = np.vstack([dallas, houston, rural])
    sensors = [{"lat": float(la), "lon": float(lo), "pm25": float(pm)}
               for (la, lo), pm in zip(s, rng.gamma(4.0, 2.2, len(s)))]
    queries = np.vstack([
        np.column_stack([rng.normal(32.8, 0.2, 15), rng.normal(-96.8, 0.2, 15)]),
        np.column_stack([rng.normal(29.76, 0.2, 15), rng.normal(-95.37, 0.2, 15)]),
        [[33.25, -96.80], [32.35, -96.80], [30.15, -95.37]],   # ~45-50 km out
        [[26.0, -106.0], [36.4, -93.6], [27.5, -104.5]],      # nothing within 100 km
        rng.uniform([29.0, -104.0], [35.0, -94.0], (20, 2)),
    ])
    return sensors, queries[:, 0], queries[:, 1]


def test_neighbor_radii_match_training():
    nf = _load_pipeline_module("neighbor_features.py")
    assert list(NBR_RADII_KM) == list(nf.RADII_KM)


def test_serving_neighbor_features_match_training():
    nf = _load_pipeline_module("neighbor_features.py")
    sensors, q_lat, q_lon = _synthetic_network()
    served = compute_neighbor_features(q_lat, q_lon, sensors)

    day = pd.Timestamp("2026-10-21")
    pool = pd.DataFrame({
        "latitude": [s["lat"] for s in sensors],
        "longitude": [s["lon"] for s in sensors],
        "pm25": [s["pm25"] for s in sensors],
        "sensor_id": [f"s{i}" for i in range(len(sensors))],
        "date": day,
    })
    # Tracts are query points that are not in the pool. Training's
    # zero-neighbor fallback is the same-day leave-one-out mean
    # (sum - own) / (n - 1); with "own" set to the pool mean that equals the
    # plain pool mean serving uses, so the fallback branch is compared too.
    query = pd.DataFrame({
        "latitude": q_lat, "longitude": q_lon,
        "pm25": float(pool["pm25"].mean()),
        "sensor_id": [f"q{i}" for i in range(len(q_lat))],
        "date": day,
    })
    trained = nf.compute_neighbor_features_df(query, pool, target_col="pm25")

    assert set(served) == set(trained)
    for name in trained:
        if name.startswith("nbr_count"):
            np.testing.assert_array_equal(served[name], trained[name], err_msg=name)
        else:
            np.testing.assert_allclose(served[name], trained[name], rtol=1e-12, atol=1e-12,
                                       err_msg=name)

    # The fixture really exercises each fallback branch.
    c25, c50, c100 = (served[f"nbr_count_{r}km"] for r in (25, 50, 100))
    assert ((c25 > 0) & (c50 > 0)).any()
    assert ((c25 == 0) & (c50 > 0)).any()
    assert (c100 == 0).any()
    isolated = c100 == 0
    np.testing.assert_allclose(served["nbr_pm25_50km"][isolated],
                               np.mean([s["pm25"] for s in sensors]))


def test_neighbor_features_need_enough_sensors():
    few = [{"lat": 30.0 + i * 0.01, "lon": -97.0, "pm25": 5.0}
           for i in range(MIN_LIVE_SENSORS - 1)]
    assert compute_neighbor_features([30.0], [-97.0], few) is None
    assert compute_neighbor_features([30.0], [-97.0], []) is None


# ── Calendar features ────────────────────────────────────────────────────────

def test_calendar_uses_utc_day():
    # 21:30 CDT on Oct 20 is already Oct 21 in UTC, the day training uses.
    from zoneinfo import ZoneInfo
    local = datetime(2026, 10, 20, 21, 30, tzinfo=ZoneInfo("America/Chicago"))
    t = model.get_temporal(local)
    utc_day = date(2026, 10, 21)
    assert t == {"month": 10, "dow": utc_day.weekday(),
                 "day_of_year": utc_day.timetuple().tm_yday}
    assert t["day_of_year"] == 294


def test_calendar_matches_training_conventions():
    # Training: pandas dt.month / dt.dayofweek (Mon=0) / dt.dayofyear.
    for ts in pd.date_range("2026-01-01", "2026-12-31 23:00", freq="37h", tz="UTC"):
        t = model.get_temporal(ts.to_pydatetime())
        assert (t["month"], t["dow"], t["day_of_year"]) == (ts.month, ts.dayofweek, ts.dayofyear)


def test_calendar_default_is_utc_now(monkeypatch):
    fixed = datetime(2026, 12, 31, 23, 30, tzinfo=timezone(timedelta(hours=-6)))

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)

    monkeypatch.setattr(model, "datetime", FrozenDatetime)
    # Dec 31 local evening is Jan 1 in UTC.
    assert model.get_temporal() == {"month": 1, "dow": date(2027, 1, 1).weekday(),
                                   "day_of_year": 1}


def test_compute_path_passes_utc_calendar(client, monkeypatch, no_snapshots):
    """The statewide compute hands the model the UTC calendar."""
    fixed = datetime(2026, 10, 21, 2, 30, tzinfo=timezone.utc)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)

    seen = {}

    class Stop(Exception):
        pass

    def spy(lookup_reset, weather, temporal, header):
        seen["temporal"] = temporal
        seen["columns"] = set(lookup_reset.columns)
        raise Stop

    # The compute path reads the clock in both modules: get_temporal() lives in
    # model, the rest of the compute in predictions.
    monkeypatch.setattr(model, "datetime", FrozenDatetime)
    monkeypatch.setattr(predictions, "datetime", FrozenDatetime)
    monkeypatch.setattr(predictions, "_predict_and_package", spy)
    with pytest.raises(Stop):
        asyncio.run(predictions._compute_texas_predictions())
    assert seen["temporal"] == {"month": 10, "dow": 2, "day_of_year": 294}


# ── Weather grid ─────────────────────────────────────────────────────────────

def _weather_handler(wind_max, wind_mean, unit, captured):
    def handler(request):
        captured.append(request)
        n = len(request.url.params["latitude"].split(","))
        loc = {
            "daily": {
                "temperature_2m_mean": [21.0], "relative_humidity_2m_mean": [55.0],
                "surface_pressure_mean": [990.0], "wind_speed_10m_max": [wind_max],
                "wind_speed_10m_mean": [wind_mean], "precipitation_sum": [0.4],
            },
            "daily_units": {"wind_speed_10m_max": unit},
        }
        return httpx.Response(200, json=[loc] * n)
    return handler


def test_weather_grid_requests_daily_max_wind(net):
    captured = []
    net.route("api.open-meteo.com", _weather_handler(7.5, 3.1, "m/s", captured))
    lats, lons = np.array([30.27, 30.3, 32.78]), np.array([-97.74, -97.7, -96.8])
    snap = asyncio.run(airquality.fetch_weather_grid(lats, lons))

    assert len(captured) == 1
    params = captured[0].url.params
    daily = params["daily"].split(",")
    assert "wind_speed_10m_max" in daily and "wind_speed_10m_mean" not in daily
    assert params["wind_speed_unit"] == "ms"
    assert params["timezone"] == "UTC"
    assert snap["usable"] and snap["schema"] == airquality.WEATHER_GRID_SCHEMA
    assert snap["features"]["wind_speed"] == [7.5, 7.5, 7.5]
    assert snap["features"]["temperature"] == [21.0, 21.0, 21.0]


def test_weather_grid_converts_kmh(net):
    net.route("api.open-meteo.com", _weather_handler(27.0, 10.0, "km/h", []))
    snap = asyncio.run(airquality.fetch_weather_grid(np.array([30.27]), np.array([-97.74])))
    assert snap["features"]["wind_speed"][0] == pytest.approx(7.5)


def test_old_mean_wind_snapshot_is_not_reused():
    feats = {"wind_speed": [3.0, 3.0]}
    current = {"usable": True, "schema": airquality.WEATHER_GRID_SCHEMA, "features": feats}
    legacy = {"usable": True, "features": feats}               # pre-schema, daily mean
    assert weather_feed._grid_matches(current, 2, airquality.WEATHER_GRID_SCHEMA)
    assert not weather_feed._grid_matches(legacy, 2, airquality.WEATHER_GRID_SCHEMA)
    assert not weather_feed._grid_matches({**legacy, "schema": 1}, 2, airquality.WEATHER_GRID_SCHEMA)
    assert not weather_feed._grid_matches(current, 3, airquality.WEATHER_GRID_SCHEMA)


# ── Spatial features ─────────────────────────────────────────────────────────

def test_training_sensors_file():
    sensors = model.load_training_sensors()
    assert len(sensors) == 467
    assert all(np.isfinite([s["lat"], s["lon"]]).all() for s in sensors)
    assert len({s["sensor_id"] for s in sensors}) == len(sensors)


def test_training_sensors_match_parquet_export():
    """models/training_sensors.json is what pipeline/14 exports from the
    tracked training parquet, so a clean clone serves the training feature."""
    export = _load_pipeline_module("14_export_training_sensors.py")
    sensors = export.training_sensors(pd.read_parquet(export.PARQUET_PATH))
    with open(config.TRAINING_SENSORS_PATH) as f:
        committed = json.load(f)
    assert [r["sensor_id"] for r in committed] == [
        int(s) if isinstance(s, (int, np.integer)) else str(s) for s in sensors["sensor_id"]]
    np.testing.assert_array_equal([r["lat"] for r in committed], sensors["latitude"].values)
    np.testing.assert_array_equal([r["lon"] for r in committed], sensors["longitude"].values)


def test_dist_to_nearest_sensor_from_training_sensors(tract_lookup):
    d = tract_lookup["dist_to_nearest_sensor"].values
    assert len(d) == 6896 and np.isfinite(d).all()
    assert np.median(d) == pytest.approx(7.80, abs=0.05)
    assert np.percentile(d, 90) == pytest.approx(54.0, abs=0.5)
    assert d.max() <= config.DIST_TO_SENSOR_MAX_KM
    # The legacy 240-sensor file gave a median of ~11 km; it must not be used.
    assert np.median(d) < 9.0


def test_missing_training_sensors_is_loud_not_silent(tract_lookup, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "TRAINING_SENSORS_PATH", str(tmp_path / "missing.json"))
    df = tract_lookup[["GEOID", "lat", "lon"]].head(20).copy()
    model.enrich_tract_lookup_with_distances(df)
    assert (df["dist_to_nearest_sensor"] == config.DIST_TO_SENSOR_MAX_KM).all()
    assert "WARNING" in capsys.readouterr().out


def test_spatial_reference_points_match_training():
    assert model.TX_COAST_POINTS == [tuple(p) for p in _training_constant("TX_COAST_POINTS")]
    assert model.TX_URBAN_POINTS == [tuple(p) for p in _training_constant("TX_URBAN_POINTS")]


def test_dist_to_coast_is_haversine_km(with_bundle):
    # Houston is ~76 km from the nearest coast point (the removed degree-based
    # formula gave 1.37).
    d = model._min_dist_to_points(np.array([29.76]), np.array([-95.37]), model.TX_COAST_POINTS)
    assert d[0] == pytest.approx(75.9, abs=0.5)
    # A frame without the column gets the same formula inside the batch.
    df = pd.DataFrame({"GEOID": ["48201000100"], "lat": [29.76], "lon": [-95.37]})
    a = model.run_predictions_batch(df.copy(), {}, model.get_temporal())
    b = model.run_predictions_batch(df.assign(dist_to_coast=d), {}, model.get_temporal())
    assert a[0] == b[0]
