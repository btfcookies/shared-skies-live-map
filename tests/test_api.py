"""HTTP contract of the public API, through the real app and lifespan, with
no PurpleAir key and no network (the startup compute uses climatology)."""
import asyncio
import gzip
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import backend.main as main
from backend import limits, placement, visits
from backend.state import EPOCH
from backend.ratelimit import RecentKeys

TOP_LEVEL_KEYS = ["region", "display_name", "generated_at", "expires_at", "weather",
                  "avg_pm25", "avg_epa_aqi", "avg_info", "tract_count", "data_sources",
                  "value_semantics", "tracts"]
TRACT_KEYS = ["geoid", "lat", "lon", "pm25", "epa_aqi", "category", "color", "aqi_range",
              "health_msg", "ejf_score", "pct_people_of_color", "pct_low_income",
              "traffic_proximity", "superfund_proximity", "diesel_pm_proximity",
              "pct_ling_isolated", "county"]
DATA_SOURCE_KEYS = {"live_purpleair_sensors", "using_live_neighbors", "hms_polygons",
                    "hms_data_date", "weather_grid_cells", "purpleair_fetched_at",
                    "purpleair_key_configured", "hms_is_proxy", "refresh_interval_min"}
HEALTH_NEW_KEYS = {"status", "purpleair_key_configured", "live_neighbors",
                   "predictions_age_min", "purpleair_age_min", "refresh_interval_min",
                   "memory_rss_mb"}


# ── /api/health ──────────────────────────────────────────────────────────────

def test_health_without_key_is_degraded(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    h = r.json()
    assert HEALTH_NEW_KEYS | {"model_loaded", "model_features", "lookup_loaded",
                              "live_purpleair_sensors", "data_sources"} <= set(h)
    assert h["status"] == "degraded"
    assert h["purpleair_key_configured"] is False
    assert h["live_neighbors"] is False
    assert h["purpleair_age_min"] is None
    assert h["refresh_interval_min"] == 15
    assert h["model_loaded"] is True and h["model_features"] == 30
    assert isinstance(h["predictions_age_min"], (int, float)) and h["predictions_age_min"] >= 0
    assert h["memory_rss_mb"] is None or h["memory_rss_mb"] > 0


def _fake_texas(age_min, live=True):
    generated = datetime.now(timezone.utc) - timedelta(minutes=age_min)
    return {"data": {"generated_at": generated.isoformat(),
                     "data_sources": {"using_live_neighbors": live}},
            "expires": generated + timedelta(minutes=15)}


@pytest.mark.parametrize("age_min, live, model, status", [
    (2, True, True, "ok"),
    (44, True, True, "ok"),
    (46, True, True, "degraded"),     # older than 3 x refresh_interval_min
    (2, False, True, "degraded"),     # climatology neighbors
    (2, True, False, "degraded"),     # model not loaded
])
def test_health_status_rules(client, monkeypatch, age_min, live, model, status):
    monkeypatch.setitem(main.state, "cache_texas", _fake_texas(age_min, live))
    if not model:
        monkeypatch.setitem(main.state, "bundle", None)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == status


def test_health_is_cheap(client, net):
    # Loose bound for slow CI runners; the real check is no upstream call.
    t0 = time.monotonic()
    client.get("/api/health")
    assert time.monotonic() - t0 < 2.0
    assert net.requests == []


# ── /api/texas/predictions ───────────────────────────────────────────────────

def test_predictions_shape(predictions):
    d = predictions
    assert list(d) == TOP_LEVEL_KEYS
    assert d["region"] == "texas"
    assert d["tract_count"] == len(d["tracts"]) == 6896
    assert list(d["tracts"][0]) == TRACT_KEYS
    assert set(d["data_sources"]) == DATA_SOURCE_KEYS
    assert {"category", "color", "aqi_range", "health_msg"} <= set(d["avg_info"])


def test_predictions_without_key(predictions):
    ds = predictions["data_sources"]
    assert ds["using_live_neighbors"] is False
    assert ds["live_purpleair_sensors"] == 0
    assert ds["purpleair_fetched_at"] is None
    assert ds["purpleair_key_configured"] is False
    assert ds["refresh_interval_min"] == 15
    # NOAA is unreachable offline, so there is no current smoke.
    assert ds["hms_polygons"] == 0 and ds["hms_data_date"] is None
    assert ds["hms_is_proxy"] is False


def test_predictions_values_are_sane(predictions):
    pm = [t["pm25"] for t in predictions["tracts"]]
    assert all(0.0 <= v < 60.0 for v in pm)
    assert len({t["geoid"] for t in predictions["tracts"]}) == 6896


def test_expires_at_is_next_refresh(predictions):
    g = datetime.fromisoformat(predictions["generated_at"])
    e = datetime.fromisoformat(predictions["expires_at"])
    assert (e - g).total_seconds() == pytest.approx(15 * 60, abs=1)


def test_predictions_gzip_etag_and_304(client, predictions):
    r = client.get("/api/texas/predictions", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers["content-encoding"] == "gzip"
    assert r.headers["cache-control"] == "no-cache"
    assert "accept-encoding" in r.headers["vary"].lower()
    etag = r.headers["etag"]
    assert etag.startswith('"') and etag.endswith('"')
    assert r.json() == predictions

    r304 = client.get("/api/texas/predictions",
                      headers={"Accept-Encoding": "gzip", "If-None-Match": etag})
    assert r304.status_code == 304
    assert r304.content == b""
    assert r304.headers["etag"] == etag

    # Weak and cross-encoding tags match too (proxies re-encode or weaken).
    r_weak = client.get("/api/texas/predictions",
                        headers={"Accept-Encoding": "identity", "If-None-Match": "W/" + etag})
    assert r_weak.status_code == 304

    r_other = client.get("/api/texas/predictions", headers={"If-None-Match": '"stale"'})
    assert r_other.status_code == 200


def test_predictions_identity_encoding(client, predictions):
    r = client.get("/api/texas/predictions", headers={"Accept-Encoding": "identity"})
    assert r.status_code == 200
    assert "content-encoding" not in r.headers
    assert json.loads(r.content) == predictions
    # The gzip body is the same bytes, compressed.
    payload = main.state["texas_payload"]
    assert gzip.decompress(payload["gz"]) == payload["raw"] == r.content
    assert r.headers["etag"] == payload["etag"]


def test_city_and_tract_match_the_map(client, predictions):
    hou = [t for t in predictions["tracts"] if t["geoid"].startswith("48201")]
    city = client.get("/api/houston/predictions").json()
    assert city["generated_at"] == predictions["generated_at"]
    assert [t["pm25"] for t in city["tracts"]] == [t["pm25"] for t in hou]

    t0 = hou[0]
    t = client.get(f"/api/tract/{t0['geoid']}").json()
    assert t["pm25"] == t0["pm25"] and t["from_map_batch"] is True
    assert client.get("/api/tract/99999999999").status_code == 404
    assert client.get("/api/nowhere/predictions").status_code == 404


# ── /api/visit ───────────────────────────────────────────────────────────────

@pytest.fixture
def fresh_visits(monkeypatch):
    monkeypatch.setattr(limits, "visit_dedupe", RecentKeys(limits.VISIT_DEDUPE_MIN * 60.0))
    monkeypatch.setattr(visits, "_public_visits_cache",
                        {"count": None, "expires": EPOCH})


def test_visit_counts_once_per_ip(client, fresh_visits):
    start = visits._get_visit_count_sync()

    def visit(xff):
        r = client.post("/api/visit", headers={"X-Forwarded-For": xff})
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"visits"} and isinstance(body["visits"], int)
        return visits._get_visit_count_sync()

    assert visit("203.0.113.10") == start + 1
    assert visit("203.0.113.10") == start + 1
    assert visit("203.0.113.10, 10.0.0.1") == start + 1   # same left-most client
    assert visit("203.0.113.11") == start + 2


def test_visit_window_is_30_min():
    assert limits.VISIT_DEDUPE_MIN == 30
    assert limits.visit_dedupe.ttl_s == 30 * 60


def test_metrics_shape(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200 and set(r.json()) == {"visits"}


# ── /api/quantum/sensor-placement ────────────────────────────────────────────

def test_sensor_placement_contract(client):
    r = client.get("/api/quantum/sensor-placement")
    assert r.status_code == 200
    q = r.json()
    qa = q["methods"]["quantum_annealing"]
    assert set(qa["weights"]) == {"coverage", "error", "ej", "pm25"}
    assert sum(qa["weights"].values()) == pytest.approx(1.0)
    assert qa["feasible"] is True
    assert "simulated annealing" in qa["method_display"].lower()
    assert qa["num_sensors"] == q["num_sensors"] == len(qa["selected_tracts"]) == 25
    # No key, so the network is the training sensor set.
    assert q["num_existing_sensors"] == len(q["existing_sensors"]) == 467
    for s in q["existing_sensors"]:
        assert {"lat", "lon"} <= set(s)
    assert {"coverage", "timing", "num_candidates", "best_energy"} <= set(qa)


def test_sensor_placement_cold_requests_share_one_solve(client, monkeypatch):
    calls = []
    release = threading.Event()

    async def fake_solve():
        calls.append(1)
        while not release.is_set():
            await asyncio.sleep(0.01)
        result = {"methods": {}, "existing_sensors": [], "num_existing_sensors": 0}
        placement.quantum_cache = {"data": result,
                               "expires": datetime.now(timezone.utc) + timedelta(hours=1)}
        return result

    monkeypatch.setattr(placement, "quantum_cache", {"data": None, "expires": EPOCH})
    monkeypatch.setattr(placement, "solve_quantum_and_cache", fake_solve)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(client.get, "/api/quantum/sensor-placement") for _ in range(5)]
        time.sleep(0.3)
        release.set()
        responses = [f.result(timeout=10) for f in futures]
    assert [r.status_code for r in responses] == [200] * 5
    assert len(calls) == 1


def test_solver_inputs_from_main(client, monkeypatch):
    """main.py hands the solver NaN (unknown) for tracts with no LOSO residual,
    the current network as plain {lat, lon}, and the documented parameters."""
    from backend.quantum import qubo_solver

    seen = {}

    def fake_solve(tracts, **kwargs):
        seen["tracts"], seen["kwargs"] = tracts, kwargs
        return {"method_display": "test", "weights": dict(qubo_solver.WEIGHTS), "feasible": True,
                "num_sensors": 0, "num_candidates": 0, "num_reads": 0, "best_energy": 0.0,
                "timing": {}, "selected_tracts": []}

    monkeypatch.setattr(qubo_solver, "solve_quantum", fake_solve)
    monkeypatch.setattr(qubo_solver, "compute_coverage", lambda *a, **k: {})
    result = asyncio.run(placement._run_quantum_placement())

    kw = seen["kwargs"]
    assert (kw["k"], kw["num_reads"], kw["top_candidates"], kw["proximity_threshold_miles"]) == (
        25, 500, 120, 8.0)
    md = kw["model_disagreement"]
    residuals = placement._load_loso_residuals()
    known = np.array([str(t["geoid"]) in residuals for t in seen["tracts"]])
    assert md.dtype == np.float64 and len(md) == len(seen["tracts"]) == 6896
    assert 0 < known.sum() < len(known)
    assert np.isnan(md[~known]).all()
    assert np.isfinite(md[known]).all()
    assert all({"lat", "lon"} <= set(s) for s in kw["existing_sensors"])
    assert result["num_existing_sensors"] == len(kw["existing_sensors"])
    assert result["methods"]["quantum_annealing"]["feasible"] is True
