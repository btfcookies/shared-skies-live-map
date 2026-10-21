"""Sensor-placement solver on a small synthetic instance."""
import numpy as np
import pytest

from backend.quantum.qubo_solver import WEIGHTS, compute_coverage, haversine, solve_quantum

K = 6
SPACING_MI = 8.0


def _instance(seed=3):
    """A 20 x 20 grid of tracts (~0.15 deg apart) across central Texas, a few
    existing sensors, and some tracts that must never be picked: water
    (tract code 99xxxx) and tracts with no EJ score, both given top scores."""
    rng = np.random.default_rng(seed)
    tracts = []
    for i, lat in enumerate(np.linspace(30.0, 32.85, 20)):
        for j, lon in enumerate(np.linspace(-99.0, -96.15, 20)):
            n = len(tracts)
            tracts.append({
                "geoid": f"48{n % 300:03d}{100000 + n:06d}",
                "lat": float(lat), "lon": float(lon),
                "pm25": float(rng.uniform(5, 15)),
                "ejf_score": float(rng.uniform(5, 95)),
            })
    for t in tracts[::37]:
        t["geoid"] = t["geoid"][:5] + "99" + t["geoid"][7:]
        t["ejf_score"], t["pm25"] = 99.0, 30.0
    for t in tracts[5::41]:
        t["ejf_score"], t["pm25"] = 0.0, 30.0
    existing = [{"lat": 31.0, "lon": -97.5}, {"lat": 30.3, "lon": -97.7, "sensor_id": 1},
                {"lat": None, "lon": -97.0}, {"lat": "bad", "lon": -97.0}]
    # LOSO residuals exist only near existing sensors; NaN elsewhere = unknown.
    residuals = np.full(len(tracts), np.nan)
    residuals[rng.choice(len(tracts), 15, replace=False)] = rng.uniform(0.1, 2.0, 15)
    return tracts, existing, residuals


def _solve(tracts, existing, residuals, **kw):
    return solve_quantum(tracts, k=K, num_reads=60, top_candidates=50,
                         proximity_threshold_miles=SPACING_MI, existing_sensors=existing,
                         model_disagreement=residuals, **kw)


def _assert_valid(result, tracts, existing):
    sel = result["selected_tracts"]
    assert result["num_sensors"] == len(sel) == K
    assert result["feasible"] is True
    assert len({t["geoid"] for t in sel}) == K
    for a in range(K):
        for b in range(a + 1, K):
            assert haversine(sel[a]["lat"], sel[a]["lon"], sel[b]["lat"], sel[b]["lon"]) >= SPACING_MI
    for t in sel:
        assert t["geoid"][5:7] != "99"
        assert t["ejf_score"] > 0
        for s in existing[:2]:
            assert haversine(t["lat"], t["lon"], s["lat"], s["lon"]) >= SPACING_MI
        assert np.isfinite(t["composite_score"]) and np.isfinite(t["prediction_error"])
    assert [t["placement_rank"] for t in sel] == list(range(1, K + 1))


def test_exactly_k_feasible_sites_with_nan_residuals():
    tracts, existing, residuals = _instance()
    result = _solve(tracts, existing, residuals)
    _assert_valid(result, tracts, existing)
    assert result["weights"] == WEIGHTS
    assert sum(result["weights"].values()) == pytest.approx(1.0)
    assert "simulated annealing" in result["method_display"].lower()
    assert 0 < result["num_candidates"] <= result["num_eligible_tracts"] < len(tracts)
    assert np.isfinite(result["best_energy"])


def test_all_nan_residuals():
    tracts, existing, residuals = _instance()
    result = _solve(tracts, existing, np.full(len(tracts), np.nan))
    _assert_valid(result, tracts, existing)


def test_deterministic_with_fixed_seed():
    tracts, existing, residuals = _instance()
    a = _solve(tracts, existing, residuals)
    b = _solve(tracts, existing, residuals)
    assert [t["geoid"] for t in a["selected_tracts"]] == [t["geoid"] for t in b["selected_tracts"]]
    assert a["best_energy"] == b["best_energy"]


def test_coverage_report():
    tracts, existing, residuals = _instance()
    result = _solve(tracts, existing, residuals)
    cov = compute_coverage(tracts, result["selected_tracts"], radius_miles=10.0,
                           existing_sensors=existing)
    assert isinstance(cov, dict) and cov


def test_no_new_site_near_an_existing_sensor():
    tracts, existing, residuals = _instance()

    def near(t):
        return min(haversine(t["lat"], t["lon"], s["lat"], s["lon"]) for s in existing[:2]) < SPACING_MI

    # Make the tracts next to existing sensors the most attractive by far.
    for t in tracts:
        if t["ejf_score"] > 0 and t["geoid"][5:7] != "99":
            t["ejf_score"], t["pm25"] = (99.0, 30.0) if near(t) else (10.0, 6.0)
    # Sanity: without the buffer the solver would pick them.
    unbuffered = _solve(tracts, existing, residuals, existing_buffer_miles=0.0)
    assert any(near(t) for t in unbuffered["selected_tracts"])

    result = _solve(tracts, existing, residuals)
    _assert_valid(result, tracts, existing)
    assert not any(near(t) for t in result["selected_tracts"])
