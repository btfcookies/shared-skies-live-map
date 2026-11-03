"""Sensor-placement optimizer: cache, solve, and the /api/quantum/sensor-placement route."""

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from backend import config, snapshots
from backend.config import ROOT
from backend.model import get_temporal, load_training_sensors, run_predictions_batch
from backend.predictions import current_texas
from backend.purpleair_feed import live_purpleair
from backend.state import EPOCH, state
from backend.util import in_flight, parse_ts, retry_after, single_flight
from backend.weather_feed import WEATHER_FALLBACK


router = APIRouter()


def hydrate(now: datetime) -> None:
    """Sensor-placement result from the last snapshot."""
    global quantum_cache
    quantum_snap = snapshots.load(config.QUANTUM_SNAPSHOT_PATH)
    q_ts = parse_ts((quantum_snap or {}).get("generated_at"))
    if quantum_snap and q_ts:
        quantum_cache = {"data": quantum_snap,
                         "expires": q_ts + timedelta(minutes=QUANTUM_CACHE_TTL_MIN)}
        print("Loaded quantum sensor-placement snapshot")


# ── Quantum Sensor Placement ────────────────────────────────────────────────
quantum_cache: dict = {"data": None, "expires": EPOCH}


# Placements change slowly (they depend on the sensor network and the
# tract-level picture, not on hour-to-hour air), and one solve is tens of
# seconds of CPU on the free instance, so refresh a few times a day.
QUANTUM_CACHE_TTL_MIN = int(os.environ.get("QUANTUM_CACHE_TTL_MIN", "360"))


# Sites requested from the solver. The response's top-level num_sensors is
# this request; methods.quantum_annealing.num_sensors is how many it placed,
# which is fewer when too few spaced-out candidates exist.
QUANTUM_NUM_SENSORS = 25


LOSO_RESIDUALS_PATH = os.path.join(ROOT, "models", "loso_residuals.json")


def _existing_sensors_for_solver() -> tuple[list[dict], str]:
    """The current sensor network as [{lat, lon}]: the live PurpleAir list
    when it is usable, else the training sensor set. (The old source was the
    legacy 240-sensor, four-metro p2_processed.xls, which made El Paso and
    the Valley look unmonitored.)"""
    live = live_purpleair()
    if live is not None:
        return [{"lat": s["lat"], "lon": s["lon"]} for s in live["sensors"]], "live PurpleAir"
    return ([{"lat": s["lat"], "lon": s["lon"]} for s in load_training_sensors()],
            "models/training_sensors.json")


def _compute_model_disagreement(df: pd.DataFrame, weather: dict, temporal: dict) -> np.ndarray:
    """
    Compute per-tract prediction uncertainty across weather perturbations.
    Since the ensemble was retrained on full data, individual models may agree
    closely. Instead, we measure sensitivity: how much does the prediction
    change when weather varies? Tracts that are highly sensitive to weather
    inputs are harder to predict accurately without a local sensor.
    """
    bundle = state.get("bundle")
    if bundle is None:
        raise ValueError("Model bundle not loaded")
    def _predict_with_weather(w):
        # Delegate to the one construction the map itself uses. The previous
        # inline builder used `{**w, **temporal}` instead of _compute_v3_shared(),
        # so month_sin/cos, dow_sin/cos, doy_sin/cos, temp_x_humidity and
        # wind_x_temp were absent from `shared` AND from df.columns and stayed at
        # the np.zeros() initialisation; it also skipped the lat/lon ->
        # latitude/longitude rename and filled gaps with 0.0 rather than the
        # training medians. That made the sensitivity probe read the model at a
        # point it was never trained on.
        return run_predictions_batch(df, w, temporal)

    # Predict under several weather perturbations
    base_pred = _predict_with_weather(weather)
    perturbations = [
        {**weather, "temperature": weather["temperature"] + 15},
        {**weather, "temperature": weather["temperature"] - 15},
        {**weather, "humidity": min(100, weather["humidity"] + 25)},
        {**weather, "humidity": max(0, weather["humidity"] - 25)},
        {**weather, "wind_speed": weather["wind_speed"] + 10},
        {**weather, "pressure": weather["pressure"] + 15},
    ]
    all_preds = [base_pred] + [_predict_with_weather(w) for w in perturbations]
    pred_stack = np.array(all_preds)
    disagreement = np.std(pred_stack, axis=0)

    return disagreement


def _load_loso_residuals():
    """Load per-GEOID LOSO-CV residuals for quantum solver."""
    if not os.path.exists(LOSO_RESIDUALS_PATH):
        return None
    try:
        with open(LOSO_RESIDUALS_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load LOSO residuals: {e}")
        return None


async def _run_quantum_placement():
    """Run the sensor-placement solver on the current predictions, the current
    sensor network and the LOSO residuals."""
    from backend.quantum.qubo_solver import solve_quantum, compute_coverage

    texas_data = await current_texas()
    tracts = texas_data.get("tracts", [])

    if not tracts:
        raise HTTPException(503, "No tract predictions available. Wait for predictions to load.")

    existing_sensors, sensors_source = await asyncio.to_thread(_existing_sensors_for_solver)
    print(f"Sensor placement: {len(existing_sensors)} existing sensors from {sensors_source}")

    # Load LOSO-CV residuals (true spatial prediction errors)
    loso_residuals = await asyncio.to_thread(_load_loso_residuals)

    # Per-tract prediction error from LOSO residuals. Residuals exist only for
    # tracts that hosted a training sensor; every other tract is UNKNOWN, so it
    # is NaN (the solver treats NaN as unknown). The old 0.0 fill told the
    # solver that unmonitored tracts were perfectly predicted.
    model_disagreement = None
    if loso_residuals:
        model_disagreement = np.array([
            loso_residuals.get(str(t["geoid"]), np.nan) for t in tracts
        ], dtype=np.float64)
        known = np.isfinite(model_disagreement)
        print(f"LOSO residuals loaded for {int(known.sum())}/{len(tracts)} tracts: "
              f"mean={np.nanmean(model_disagreement):.3f}, "
              f"max={np.nanmax(model_disagreement):.3f}")

    # Fallback to weather sensitivity if no LOSO residuals
    if model_disagreement is None:
        if state.get("bundle") and state.get("tract_lookup") is not None:
            try:
                # Model-unit fallback (the payload's `weather` is the DISPLAY
                # dict in °F/mph — never feed that to the model).
                weather = dict(WEATHER_FALLBACK)
                temporal = get_temporal()
                model_disagreement = await asyncio.to_thread(
                    _compute_model_disagreement, state["tract_lookup"],
                    weather, temporal)
            except Exception as e:
                print(f"Disagreement computation failed: {e}")

    quantum_result = await asyncio.to_thread(
        solve_quantum,
        tracts,
        k=QUANTUM_NUM_SENSORS,
        num_reads=500,
        top_candidates=120,
        proximity_threshold_miles=8.0,
        existing_sensors=existing_sensors,
        model_disagreement=model_disagreement,
    )

    coverage = await asyncio.to_thread(
        compute_coverage,
        tracts, quantum_result["selected_tracts"],
        radius_miles=10.0,
        existing_sensors=existing_sensors,
    )

    avg_ej = float(np.mean([
        t.get("ejf_score", 0.0) or 0.0
        for t in quantum_result["selected_tracts"]
    ])) if quantum_result["selected_tracts"] else 0.0

    avg_composite = float(np.mean([
        t["composite_score"]
        for t in quantum_result["selected_tracts"]
    ])) if quantum_result["selected_tracts"] else 0.0

    now = datetime.now(timezone.utc)
    result = {
        "generated_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=QUANTUM_CACHE_TTL_MIN)).isoformat(),
        "num_total_tracts": len(tracts),
        "num_sensors": QUANTUM_NUM_SENSORS,
        "num_existing_sensors": len(existing_sensors),
        "proximity_threshold_miles": 8.0,
        "coverage_radius_miles": 10.0,
        "existing_sensors": existing_sensors,
        "methods": {
            "quantum_annealing": {
                # The solver describes its own method and reports the weights
                # it actually used, so the UI copy can't drift from the code.
                "method_display": quantum_result["method_display"],
                "weights": quantum_result.get("weights"),
                "feasible": bool(quantum_result.get(
                    "feasible", quantum_result["num_sensors"] == QUANTUM_NUM_SENSORS)),
                "num_sensors": quantum_result["num_sensors"],
                "coverage": coverage,
                "avg_ej_score": round(avg_ej, 1),
                "avg_composite_score": round(avg_composite, 4),
                "timing": quantum_result["timing"],
                "selected_tracts": quantum_result["selected_tracts"],
                "best_energy": quantum_result.get("best_energy"),
                "num_reads": quantum_result.get("num_reads"),
                "num_candidates": quantum_result.get("num_candidates"),
            },
        },
    }

    return result


async def solve_quantum_and_cache() -> dict:
    """One placement solve (run single-flight); updates cache + disk snapshot."""
    global quantum_cache
    try:
        result = await _run_quantum_placement()
    except HTTPException:
        # No predictions yet: nothing expensive ran, so no backoff; the loop
        # solves as soon as predictions exist.
        raise
    except Exception:
        # Back off so a failing solve doesn't rerun every loop tick.
        quantum_cache["expires"] = datetime.now(timezone.utc) + timedelta(minutes=30)
        raise
    quantum_cache = {
        "data": result,
        "expires": datetime.now(timezone.utc) + timedelta(minutes=QUANTUM_CACHE_TTL_MIN),
    }
    await snapshots.save_async(config.QUANTUM_SNAPSHOT_PATH, result)
    print("✓ Sensor placement solved")
    return result


@router.get("/api/quantum/sensor-placement")
async def quantum_sensor_placement():
    """
    Returns optimized sensor placement recommendations. The background loop
    solves after the first predictions and refreshes every
    QUANTUM_CACHE_TTL_MIN; a request only waits when no result exists yet,
    and then it joins the single in-flight solve.
    """
    cached_data = quantum_cache.get("data")
    if cached_data is not None:
        return cached_data
    # After a failed solve, wait out its backoff instead of re-solving per
    # request (seconds of CPU each on the free instance).
    now = datetime.now(timezone.utc)
    if not in_flight("quantum") and now < quantum_cache["expires"]:
        raise HTTPException(503, "Sensor placement is temporarily unavailable.",
                            headers={"Retry-After": retry_after(quantum_cache["expires"], now)})
    try:
        return await single_flight("quantum", solve_quantum_and_cache)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Quantum placement error: {type(e).__name__}: {e}")
        raise HTTPException(500, "Sensor placement is temporarily unavailable.")
