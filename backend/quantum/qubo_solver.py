"""
QUBO (Quadratic Unconstrained Binary Optimization) formulation for
equitable sensor placement across Texas census tracts, solved with
classical simulated annealing.

Formulation:
  - One binary variable per candidate tract (a geographically diverse pool,
    one strong tract per grid cell).
  - Linear reward: the composite placement score (coverage gap, model error,
    EJ burden, PM2.5 level; weights in WEIGHTS).
  - Pairwise terms: a spacing penalty for sites closer than the proximity
    threshold and a coverage-overlap penalty for pairs whose coverage disks
    overlap (the shared area would otherwise be credited twice).
  - Exactly-k budget: lambda * (sum x - k)^2, with lambda chosen above the
    largest marginal gain of any single flip, so every infeasible assignment
    has a strictly better neighbour and the minimum has exactly k sites.

The QUBO is the form a quantum annealer accepts ("quantum-inspired"), but it
is solved here by classical Metropolis simulated annealing on the CPU
(dwave-neal / dwave.samplers), and the best feasible samples are finished by
a swap-move local search. No quantum hardware is involved.

Solvers:
  - solve_quantum: simulated annealing over the QUBO (name kept for API compat)
  - solve_greedy: classical greedy baseline
  - solve_classical_sa: swap-based simulated annealing on the linear objective
"""

import time
import numpy as np

import dimod
from neal import SimulatedAnnealingSampler


# Composite-score weights. The QUBO's linear reward is exactly this composite,
# so these are the weights the solver uses and the ones reported to clients.
WEIGHTS = {"coverage": 0.30, "error": 0.20, "ej": 0.30, "pm25": 0.20}

METHOD_DISPLAY = ("Classical simulated annealing over a quantum-inspired "
                  "QUBO formulation, with swap refinement")

# Coverage disk radius for the overlap penalty; matches the radius main.py
# passes to compute_coverage, so the QUBO optimizes what the API reports.
COVERAGE_RADIUS_MILES = 10.0

# LOSO residuals exist only for sensor-hosting tracts. Unmonitored tracts get
# an inverse-distance-weighted estimate from known residuals within ~100 km
# (model error is spatially correlated at roughly regional scale), else the
# statewide median. Treating them as 0 would mean "no error" exactly where we
# have no evidence, which rewards already-monitored places.
RESIDUAL_IDW_RADIUS_MILES = 62.0
RESIDUAL_IDW_POWER = 2.0

ANNEAL_SEED = 42
# Measured on the live pool (~190 candidates, 500 reads): 100, 300 and 1000
# sweeps all reach the same refined optimum; 200 leaves headroom at ~0.3 s.
ANNEAL_SWEEPS = 200
# Distinct feasible samples passed to swap refinement (lowest energy first)
REFINE_TOP_SAMPLES = 50


# ── Haversine distance (miles) ──────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles between two lat/lon points."""
    R = 3959.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    a = np.clip(a, 0.0, 1.0)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _vectorized_min_dist(t_lats, t_lons, s_lats, s_lons):
    """Vectorized minimum distance from each tract to nearest sensor."""
    n = len(t_lats)
    min_dists = np.full(n, np.inf)
    for sl, sn in zip(s_lats, s_lons):
        dlat = np.radians(t_lats - sl)
        dlon = np.radians(t_lons - sn)
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.radians(t_lats)) * np.cos(np.radians(sl)) *
             np.sin(dlon / 2) ** 2)
        a = np.clip(a, 0.0, 1.0)
        dists = 3959.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        min_dists = np.minimum(min_dists, dists)
    return min_dists


def _cross_dist_matrix(lats1, lons1, lats2, lons2):
    """Distance matrix (miles) between two point sets, vectorized haversine."""
    la1, lo1 = np.radians(lats1)[:, None], np.radians(lons1)[:, None]
    la2, lo2 = np.radians(lats2)[None, :], np.radians(lons2)[None, :]
    a = (np.sin((la1 - la2) / 2) ** 2 +
         np.cos(la1) * np.cos(la2) * np.sin((lo1 - lo2) / 2) ** 2)
    a = np.clip(a, 0.0, 1.0)
    return 3959.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _pairwise_dist_matrix(lats, lons):
    """Compute pairwise distance matrix (miles) using vectorized haversine."""
    return _cross_dist_matrix(lats, lons, lats, lons)


# ── Input cleaning ──────────────────────────────────────────────────────────

def _sensor_coords(sensors):
    """
    (lats, lons) of sensors with finite coordinates. Entries only need lat/lon;
    sensor_id/city are optional and never read.
    """
    lats, lons = [], []
    for s in sensors or []:
        try:
            la, lo = float(s.get("lat")), float(s.get("lon"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(la) and np.isfinite(lo):
            lats.append(la)
            lons.append(lo)
    return np.array(lats, dtype=np.float64), np.array(lons, dtype=np.float64)


def _tract_ej(tracts):
    """
    EJ score per tract, NaN where unknown. The payload coerces a missing score
    to 0.0, and in the tract lookup every 0 belongs to a water/special-use
    (98xxxx/99xxxx) or near-empty tract, so <= 0 is read as missing rather
    than as the lowest burden.
    """
    out = np.full(len(tracts), np.nan)
    for i, t in enumerate(tracts):
        try:
            v = float(t.get("ejf_score"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v > 0:
            out[i] = v
    return out


def _fill_missing_residuals(values, lats, lons):
    """
    Replace NaN entries of a per-tract error array with an inverse-distance
    weighted mean of known entries within RESIDUAL_IDW_RADIUS_MILES, falling
    back to the median of known entries. Returns None if nothing is known.
    """
    vals = np.array(values, dtype=np.float64)
    known = np.isfinite(vals)
    if not known.any():
        return None
    if known.all():
        return vals

    fallback = float(np.median(vals[known]))
    k_lats, k_lons, k_vals = lats[known], lons[known], vals[known]
    missing = np.where(~known)[0]
    # Chunked to keep the distance matrix small on a 512 MB instance
    for start in range(0, len(missing), 1024):
        idx = missing[start:start + 1024]
        d = _cross_dist_matrix(lats[idx], lons[idx], k_lats, k_lons)
        # 1-mile floor so a residual in the same tract doesn't get infinite weight
        w = np.where(d <= RESIDUAL_IDW_RADIUS_MILES,
                     1.0 / np.maximum(d, 1.0) ** RESIDUAL_IDW_POWER, 0.0)
        w_sum = w.sum(axis=1)
        est = (w @ k_vals) / np.where(w_sum > 0, w_sum, 1.0)
        vals[idx] = np.where(w_sum > 0, est, fallback)
    return vals


def _min_dist_to_sensors(lats, lons, existing_sensors):
    """Distance (miles) from each tract to its nearest existing sensor, inf if none."""
    s_lats, s_lons = _sensor_coords(existing_sensors)
    if len(s_lats) == 0:
        return np.full(len(lats), np.inf)
    return _vectorized_min_dist(lats, lons, s_lats, s_lons)


def _eligible_mask(tracts, dist_to_existing, existing_buffer_miles):
    """
    Tracts that may be recommended as new sites. Excludes:
      - water-only tracts (Census tract code 99xxxx) and tracts with
        population 0 when the tract dict carries a population;
      - tracts with no EJ score: EJScreen leaves the index blank where there
        are no residents, so in the served payload this is the best
        available zero-population signal (e.g. 98xxxx special-use tracts);
      - tracts within existing_buffer_miles of an existing sensor, the same
        spacing rule new sites must keep from each other. A new sensor there
        mostly duplicates coverage the network already has.
    """
    ej = _tract_ej(tracts)
    mask = np.isfinite(ej) & (dist_to_existing >= existing_buffer_miles)
    for i, t in enumerate(tracts):
        if not mask[i]:
            continue
        geoid = str(t.get("geoid", ""))
        if len(geoid) >= 11 and geoid[5:7] == "99":
            mask[i] = False
            continue
        pop = t.get("population", t.get("POPULATION"))
        if pop is not None:
            try:
                if float(pop) <= 0:
                    mask[i] = False
            except (TypeError, ValueError):
                pass
    return mask


# ── Score Computation ───────────────────────────────────────────────────────

def compute_tract_scores(tracts, existing_sensors=None, model_disagreement=None):
    """
    Compute per-tract placement priority scores.

    Args:
        tracts: list of tract dicts
        existing_sensors: list of {lat, lon} for the current sensor network
        model_disagreement: optional per-tract error estimate (LOSO residuals,
            or weather-sensitivity spread). NaN means unknown and is filled
            spatially (see _fill_missing_residuals), not treated as zero.
    """
    n = len(tracts)
    lats = np.array([t["lat"] for t in tracts], dtype=np.float64)
    lons = np.array([t["lon"] for t in tracts], dtype=np.float64)
    pm25s = np.array([t["pm25"] for t in tracts], dtype=np.float64)
    ejs = _tract_ej(tracts)

    # ── Coverage need: distance to nearest EXISTING sensor ──
    min_dists = _min_dist_to_sensors(lats, lons, existing_sensors)
    if not np.all(np.isfinite(min_dists)):
        center_lat, center_lon = np.mean(lats), np.mean(lons)
        min_dists = _vectorized_min_dist(
            lats, lons, np.array([center_lat]), np.array([center_lon]))

    # Normalize with soft clipping — don't let a single remote tract dominate
    d_95 = np.percentile(min_dists, 95)
    coverage_need = np.clip(min_dists / max(d_95, 1.0), 0, 1.5) / 1.5

    # ── Prediction error: LOSO residuals / disagreement, or fallback ──
    disagree = None
    if model_disagreement is not None and len(model_disagreement) == n:
        disagree = _fill_missing_residuals(model_disagreement, lats, lons)
    if disagree is not None:
        d_max = np.percentile(disagree, 99)
        prediction_error = np.clip(disagree / max(d_max, 1e-6), 0, 1)
    else:
        # Fallback: deviation from mean (less accurate)
        mean_pm25 = pm25s.mean()
        std_pm25 = pm25s.std() if pm25s.std() > 0 else 1.0
        prediction_error = np.abs(pm25s - mean_pm25) / (3.0 * std_pm25)
        prediction_error = np.clip(prediction_error, 0, 1)

    # ── EJ priority (normalized over known scores only) ──
    known_ej = np.isfinite(ejs)
    if known_ej.any():
        ej_max = np.percentile(ejs[known_ej], 99)
        ej_priority = np.clip(ejs / max(ej_max, 1.0), 0, 1)
        # Unknown EJ tracts are never candidates (_eligible_mask); give them a
        # neutral value so the composite stays finite for reporting.
        ej_priority[~known_ej] = np.median(ej_priority[known_ej])
    else:
        ej_priority = np.zeros(n)

    # ── PM2.5 severity: higher pollution = more need for monitoring ──
    pm25_95 = np.percentile(pm25s, 95)
    pm25_severity = np.clip(pm25s / max(pm25_95, 1.0), 0, 1.2) / 1.2

    composite = (WEIGHTS["coverage"] * coverage_need +
                 WEIGHTS["error"] * prediction_error +
                 WEIGHTS["ej"] * ej_priority +
                 WEIGHTS["pm25"] * pm25_severity)

    scores = {}
    for i, t in enumerate(tracts):
        scores[t["geoid"]] = {
            "coverage_need": round(float(coverage_need[i]), 4),
            "prediction_error": round(float(prediction_error[i]), 4),
            "ej_priority": round(float(ej_priority[i]), 4),
            "pm25_severity": round(float(pm25_severity[i]), 4),
            "composite": round(float(composite[i]), 4),
        }

    return scores, composite, coverage_need


# ── Candidate pool ──────────────────────────────────────────────────────────

def _select_diverse_candidates(tracts, scores_array, coverage_need_array,
                               top_n=100, grid_cells=20, eligible=None):
    """
    Select candidates using geographic diversity + score ranking.
    Divides Texas into a grid and picks top candidates from EACH cell,
    ensuring the candidate pool spans the entire state rather than
    clustering in one high-score region. Only `eligible` tracts are used.
    """
    n = len(tracts)
    if eligible is None:
        eligible = np.ones(n, dtype=bool)
    lats = np.array([tracts[i]["lat"] for i in range(n)], dtype=np.float64)
    lons = np.array([tracts[i]["lon"] for i in range(n)], dtype=np.float64)

    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    # Assign each tract to a grid cell
    lat_bins = np.clip(((lats - lat_min) / max(lat_max - lat_min, 0.01) * grid_cells).astype(int), 0, grid_cells - 1)
    lon_bins = np.clip(((lons - lon_min) / max(lon_max - lon_min, 0.01) * grid_cells).astype(int), 0, grid_cells - 1)
    cell_ids = lat_bins * grid_cells + lon_bins

    # Pool choice only: tilt toward monitoring deserts when picking each
    # cell's representative. The QUBO itself scores sites by `scores_array`.
    combined_score = scores_array + 0.3 * coverage_need_array
    selected = set()

    # First pass: pick the best tract from each occupied cell
    for cell in np.unique(cell_ids[eligible]):
        mask = np.where((cell_ids == cell) & eligible)[0]
        best_in_cell = mask[np.argmax(combined_score[mask])]
        selected.add(int(best_in_cell))

    # Second pass: fill remaining slots from global top scores
    for idx in np.argsort(combined_score)[::-1]:
        if len(selected) >= top_n:
            break
        if eligible[idx]:
            selected.add(int(idx))

    # Third pass: add high-coverage-need tracts (monitoring deserts)
    for idx in np.argsort(coverage_need_array)[::-1]:
        if len(selected) >= top_n + 20:  # slight over-selection for deserts
            break
        if eligible[idx]:
            selected.add(int(idx))

    return np.array(sorted(selected), dtype=int)


# ── QUBO ────────────────────────────────────────────────────────────────────

def _disk_overlap_fraction(d, radius):
    """Fraction of a radius-`radius` disk covered by an equal disk `d` away."""
    d = np.minimum(d, 2.0 * radius)
    lens = (2.0 * radius ** 2 * np.arccos(d / (2.0 * radius))
            - 0.5 * d * np.sqrt(np.maximum(4.0 * radius ** 2 - d ** 2, 0.0)))
    return lens / (np.pi * radius ** 2)


def _build_placement_qubo(rewards, coverage, dist_matrix, k,
                          proximity_threshold_miles,
                          coverage_radius_miles=COVERAGE_RADIUS_MILES):
    """
    Build the exactly-k placement BQM over m candidates.

    Objective part (minimized):
        -sum_i r_i x_i + sum_{i<j} J_ij x_i x_j
    with r_i the composite score and J_ij >= 0:
      - pairs closer than the threshold: spacing penalty of at least max r;
      - pairs whose coverage disks still overlap: the coverage credit the
        two sites would claim twice, WEIGHTS["coverage"] * overlap fraction
        * the smaller coverage need. This replaces an earlier reward for
        pairs 8-35 mi apart, which (once the budget made the anneal matter)
        pulled sites into clusters: measured on the live data it cut new
        tracts covered from 265 to 154 and the picks' mean EJ score from
        88.8 to 75.9.

    Budget part: lam * (sum_i x_i - k)^2, expanded into linear lam*(1-2k),
    quadratic 2*lam and offset lam*k^2, so a k-site assignment's energy is
    exactly its objective value.

    Choice of lam. Turning variable i on changes the objective by
    -r_i + sum_{j in S} J_ij <= gain_on_i = -r_i + sum_j max(J_ij, 0), and
    turning it off by r_i - sum_{j in S} J_ij <= gain_off_i = r_i +
    sum_j max(-J_ij, 0). With |S| = k + t the penalty drops by lam*(2t-1) >=
    lam when any site is turned off; with |S| = k - t it drops by the same
    when any site is turned on. So lam > max(gain_on, gain_off) gives every
    infeasible assignment a strictly lower-energy single-flip neighbour:
    all local minima under single flips, and the global minimum, have
    exactly k sites. We take 10% above that bound; a larger lam would only
    raise the barriers between feasible solutions.
    """
    m = len(rewards)
    J = np.zeros((m, m))
    if m == 0:
        # No eligible candidates: nothing to place, and the bounds below have
        # no maximum. solve_quantum returns an empty, infeasible result.
        return dimod.BinaryQuadraticModel(dimod.BINARY), 0.0, J
    r_max = float(rewards.max())

    off_diag = ~np.eye(m, dtype=bool)
    close = (dist_matrix < proximity_threshold_miles) & off_diag
    # Floor of r_max so keeping a too-close pair never beats dropping one of it
    J[close] = r_max * (2.0 - dist_matrix[close] / proximity_threshold_miles)

    overlap = ((dist_matrix >= proximity_threshold_miles)
               & (dist_matrix < 2.0 * coverage_radius_miles))
    shared_need = np.minimum(coverage[:, None], coverage[None, :])
    J[overlap] = (WEIGHTS["coverage"]
                  * _disk_overlap_fraction(dist_matrix[overlap],
                                           coverage_radius_miles)
                  * shared_need[overlap])

    gain_on = -rewards + np.maximum(J, 0).sum(axis=1)
    gain_off = rewards + np.maximum(-J, 0).sum(axis=1)
    lam = 1.1 * max(float(gain_on.max()), float(gain_off.max()), 1e-6)

    linear = -rewards + lam * (1 - 2 * k)
    quadratic = np.triu(J + 2.0 * lam, k=1)
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, lam * k * k,
                                     dimod.BINARY)
    return bqm, lam, J


def _is_spaced(sel, dist_matrix, threshold):
    """True if no two of the (local) indices in `sel` are closer than threshold."""
    if len(sel) < 2:
        return True
    sub = dist_matrix[np.ix_(sel, sel)]
    return bool(sub[np.triu_indices(len(sel), k=1)].min() >= threshold)


def _swap_refine(sel, rewards, J, close):
    """
    Steepest descent over swap moves (one site out, one candidate in) on the
    objective part of the QUBO. Swaps keep exactly k sites and are only taken
    when the incoming site respects spacing, so feasibility is preserved.
    Single-flip annealing has to cross a lambda-sized barrier to move between
    two k-site solutions; this finishes the local search it cannot.
    """
    x = np.zeros(len(rewards), dtype=bool)
    x[list(sel)] = True
    while True:
        ins, outs = np.where(x)[0], np.where(~x)[0]
        if len(ins) == 0 or len(outs) == 0:
            break
        # field[v]: objective cost of v given the current sites (J has a zero diagonal)
        field = -rewards + J[:, x].sum(axis=1)
        n_close = close[:, x].sum(axis=1)
        delta = (field[outs][None, :] - J[np.ix_(ins, outs)]
                 - field[ins][:, None])
        spaced = (n_close[outs][None, :] - close[np.ix_(ins, outs)]) == 0
        delta[~spaced] = np.inf
        a, b = np.unravel_index(np.argmin(delta), delta.shape)
        if not delta[a, b] < -1e-12:
            break
        x[ins[a]], x[outs[b]] = False, True
    return [int(v) for v in np.where(x)[0]]


def _repair_selection(start, rewards, dist_matrix, k, threshold):
    """
    Fallback when no annealing sample is feasible: starting from `start` (the
    best sample's sites; empty gives the pure greedy spaced set), keep them
    in descending reward order while they respect spacing (dropping the weaker site of each close pair), trim to the k best,
    then greedily add the highest-reward spaced candidates until k.
    May return fewer than k if the pool has fewer mutually spaced sites.
    """
    keep = []
    for i in sorted(start, key=lambda v: -rewards[v]):
        if all(dist_matrix[i, j] >= threshold for j in keep):
            keep.append(i)
    keep = keep[:k]
    for i in np.argsort(-rewards):
        if len(keep) >= k:
            break
        i = int(i)
        if i not in keep and all(dist_matrix[i, j] >= threshold for j in keep):
            keep.append(i)
    return keep


# ── QUBO Solver ─────────────────────────────────────────────────────────────

def solve_quantum(tracts, k=25, num_reads=500, top_candidates=120,
                  proximity_threshold_miles=8.0, existing_sensors=None,
                  model_disagreement=None, existing_buffer_miles=None,
                  num_sweeps=ANNEAL_SWEEPS, seed=ANNEAL_SEED,
                  coverage_radius_miles=COVERAGE_RADIUS_MILES):
    """
    Choose k new sensor sites by simulated annealing over an exactly-k QUBO.

    Annealing samples are read lowest energy first; a sample is feasible when
    it has exactly k sites, no two closer than proximity_threshold_miles. The
    best REFINE_TOP_SAMPLES distinct feasible samples are each refined by
    swap moves that stay feasible (_swap_refine), and the lowest-energy result
    is returned. If no sample is feasible (anneal_feasible=False), the best
    sample and the greedy spaced set are each repaired (_repair_selection)
    and refined, and the better one is returned. feasible reports whether the
    returned selection has exactly k spaced sites; it is False only when the
    pool holds fewer than k mutually spaced candidates (or none at all).
    Candidates within existing_buffer_miles (default: the proximity
    threshold) of an existing sensor are excluded.
    """
    t_start = time.time()
    if existing_buffer_miles is None:
        existing_buffer_miles = proximity_threshold_miles

    scores, scores_array, coverage_need = compute_tract_scores(
        tracts, existing_sensors, model_disagreement)

    lats_all = np.array([t["lat"] for t in tracts], dtype=np.float64)
    lons_all = np.array([t["lon"] for t in tracts], dtype=np.float64)
    dist_existing = _min_dist_to_sensors(lats_all, lons_all, existing_sensors)
    eligible = _eligible_mask(tracts, dist_existing, existing_buffer_miles)

    t_scores = time.time()

    candidate_idx = _select_diverse_candidates(
        tracts, scores_array, coverage_need, top_n=top_candidates,
        eligible=eligible)
    m = len(candidate_idx)

    rewards = scores_array[candidate_idx]
    dist_matrix = _pairwise_dist_matrix(lats_all[candidate_idx],
                                        lons_all[candidate_idx])
    bqm, lam, J = _build_placement_qubo(
        rewards, coverage_need[candidate_idx], dist_matrix, k,
        proximity_threshold_miles, coverage_radius_miles)
    close = (dist_matrix < proximity_threshold_miles) & ~np.eye(m, dtype=bool)

    t_qubo = time.time()

    def energy_of(sel):
        return float(bqm.energy({v: int(v in sel) for v in bqm.variables}))

    local_sel, best_energy, anneal_best, num_feasible = [], 0.0, None, 0
    anneal_feasible = False
    if m > 0:
        # Explicit schedule in units of the largest site reward. neal's
        # automatic range ends too warm for this dense BQM (beta ~5, and some
        # reads came back with k+1 sites); at beta = 50/r_max a flip worth
        # 0.1*r_max is accepted with probability ~e^-5, so reads finish quenched.
        r_scale = max(float(rewards.max()), 1e-6)
        result = SimulatedAnnealingSampler().sample(
            bqm,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            seed=seed,
            beta_range=(0.1 / r_scale, 50.0 / r_scale),
            beta_schedule_type="geometric",
        )
        t_solve = time.time()

        # data(sorted_by='energy') keeps each sample paired with its own energy
        starts, seen, first = [], set(), None
        for sample, energy in result.data(["sample", "energy"],
                                          sorted_by="energy"):
            on = tuple(sorted(v for v, val in sample.items() if val == 1))
            if first is None:
                first = on
            if len(on) == k and _is_spaced(list(on), dist_matrix,
                                           proximity_threshold_miles):
                num_feasible += 1
                if anneal_best is None:
                    anneal_best = float(energy)
                if on not in seen and len(starts) < REFINE_TOP_SAMPLES:
                    seen.add(on)
                    starts.append(list(on))

        anneal_feasible = bool(starts)
        if not anneal_feasible:
            # Also refine from the plain greedy-by-reward spaced set: swaps
            # from a repaired 1-sweep sample alone were measured stuck at
            # -16.98, where the greedy start reaches the full solve's -17.01.
            starts = [_repair_selection(s, rewards, dist_matrix, k,
                                        proximity_threshold_miles)
                      for s in (first, ())]
        for start in starts:
            refined = _swap_refine(start, rewards, J, close)
            e = energy_of(refined)
            if not local_sel or e < best_energy - 1e-12:
                local_sel, best_energy = refined, e
    else:
        t_solve = time.time()
    # Judged on what is returned, not on the raw anneal: a repaired result
    # with k spaced sites is a valid answer. Pool sites are all eligible.
    feasible = (len(local_sel) == k
                and _is_spaced(local_sel, dist_matrix, proximity_threshold_miles))

    selected_tracts = []
    for local_idx in local_sel:
        t_data = tracts[int(candidate_idx[local_idx])]
        s = scores[t_data["geoid"]]
        selected_tracts.append({
            **t_data,
            "coverage_need": s["coverage_need"],
            "prediction_error": s["prediction_error"],
            "ej_priority": s["ej_priority"],
            "pm25_severity": s.get("pm25_severity", 0),
            "composite_score": s["composite"],
        })

    # Sort by composite score descending, assign ranks
    selected_tracts.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, t_item in enumerate(selected_tracts):
        t_item["placement_rank"] = i + 1

    t_end = time.time()

    return {
        "method": "quantum_annealing",
        "method_display": METHOD_DISPLAY,
        "selected_tracts": selected_tracts,
        "num_sensors": len(selected_tracts),
        "num_candidates": int(m),
        "num_eligible_tracts": int(eligible.sum()),
        "num_reads": int(num_reads) if m > 0 else 0,
        "num_sweeps": int(num_sweeps),
        "num_feasible_reads": int(num_feasible),
        # feasible: the returned selection has exactly k spaced, eligible sites.
        # anneal_feasible: at least one raw annealing read already did.
        "feasible": bool(feasible),
        "anneal_feasible": anneal_feasible,
        "penalty_strength": round(float(lam), 6),
        # Best raw annealing sample, before swap refinement (None if infeasible)
        "anneal_best_energy": anneal_best,
        "weights": dict(WEIGHTS),
        "best_energy": float(best_energy),
        "timing": {
            "scoring_ms": round((t_scores - t_start) * 1000, 1),
            "qubo_build_ms": round((t_qubo - t_scores) * 1000, 1),
            "annealing_ms": round((t_solve - t_qubo) * 1000, 1),
            "postprocess_ms": round((t_end - t_solve) * 1000, 1),
            "total_ms": round((t_end - t_start) * 1000, 1),
        },
    }


# ── Classical Greedy Solver (Baseline) ──────────────────────────────────────

def solve_greedy(tracts, k=25, proximity_threshold_miles=8.0,
                 existing_sensors=None, model_disagreement=None):
    """
    Greedy baseline over the same eligible tracts as solve_quantum.
    Selects tracts one at a time, always picking the highest-scoring valid option.
    Sees pairwise overlap only through the hard spacing rule — each decision is independent.
    """
    t_start = time.time()
    scores, scores_array, _ = compute_tract_scores(
        tracts, existing_sensors, model_disagreement)

    lats = np.array([t["lat"] for t in tracts], dtype=np.float64)
    lons = np.array([t["lon"] for t in tracts], dtype=np.float64)
    eligible = _eligible_mask(
        tracts, _min_dist_to_sensors(lats, lons, existing_sensors),
        proximity_threshold_miles)

    selected = []
    for idx in np.argsort(scores_array)[::-1]:
        if len(selected) >= k:
            break
        idx = int(idx)
        if not eligible[idx]:
            continue
        too_close = any(
            haversine(lats[idx], lons[idx], lats[s], lons[s])
            < proximity_threshold_miles for s in selected)
        if not too_close:
            selected.append(idx)

    t_end = time.time()

    selected_tracts = []
    for rank, idx in enumerate(selected):
        t_data = tracts[idx]
        s = scores[t_data["geoid"]]
        selected_tracts.append({
            **t_data,
            "coverage_need": s["coverage_need"],
            "prediction_error": s["prediction_error"],
            "ej_priority": s["ej_priority"],
            "pm25_severity": s.get("pm25_severity", 0),
            "composite_score": s["composite"],
            "placement_rank": rank + 1,
        })

    return {
        "method": "greedy",
        "method_display": "Greedy Submodular Maximization",
        "selected_tracts": selected_tracts,
        "num_sensors": len(selected_tracts),
        "timing": {"total_ms": round((t_end - t_start) * 1000, 1)},
    }


# ── Classical Simulated Annealing Solver (Baseline) ─────────────────────────

def solve_classical_sa(tracts, k=25, proximity_threshold_miles=8.0,
                       iterations=5000, existing_sensors=None,
                       model_disagreement=None):
    """
    Classical simulated annealing. Starts from greedy solution and swaps.
    Uses the SAME objective as greedy (score sum - proximity penalty).
    Its objective has no coverage-overlap term.
    """
    t_start = time.time()
    scores, scores_array, _ = compute_tract_scores(
        tracts, existing_sensors, model_disagreement)

    lats = np.array([t["lat"] for t in tracts], dtype=np.float64)
    lons = np.array([t["lon"] for t in tracts], dtype=np.float64)
    eligible_idx = np.where(_eligible_mask(
        tracts, _min_dist_to_sensors(lats, lons, existing_sensors),
        proximity_threshold_miles))[0]
    rng = np.random.RandomState(42)

    def evaluate(selection):
        reward = sum(scores_array[i] for i in selection)
        penalty = 0.0
        sel_list = list(selection)
        for a in range(len(sel_list)):
            for b in range(a + 1, len(sel_list)):
                dist = haversine(lats[sel_list[a]], lons[sel_list[a]],
                                 lats[sel_list[b]], lons[sel_list[b]])
                if dist < proximity_threshold_miles:
                    penalty += (1.0 - dist / proximity_threshold_miles)
        return reward - 0.5 * penalty

    # Start from top-k eligible tracts by score
    order = eligible_idx[np.argsort(scores_array[eligible_idx])]
    current = set(order[-k:].tolist())
    current_val = evaluate(current)
    best = set(current)
    best_val = current_val

    T = 1.0
    T_min = 0.001
    alpha = (T_min / T) ** (1.0 / max(iterations, 1))

    for step in range(iterations):
        if not current:
            break
        remove = rng.choice(list(current))
        candidates = [int(i) for i in eligible_idx if i not in current]
        if not candidates:
            break
        add = rng.choice(candidates)

        neighbor = set(current)
        neighbor.discard(remove)
        neighbor.add(add)
        neighbor_val = evaluate(neighbor)
        delta = neighbor_val - current_val

        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current = neighbor
            current_val = neighbor_val
            if current_val > best_val:
                best = set(current)
                best_val = current_val

        T *= alpha

    t_end = time.time()

    selected_tracts = []
    sorted_best = sorted(best, key=lambda i: scores_array[i], reverse=True)
    for rank, idx in enumerate(sorted_best):
        t_data = tracts[idx]
        s = scores[t_data["geoid"]]
        selected_tracts.append({
            **t_data,
            "coverage_need": s["coverage_need"],
            "prediction_error": s["prediction_error"],
            "ej_priority": s["ej_priority"],
            "pm25_severity": s.get("pm25_severity", 0),
            "composite_score": s["composite"],
            "placement_rank": rank + 1,
        })

    return {
        "method": "classical_sa",
        "method_display": "Classical Simulated Annealing",
        "selected_tracts": selected_tracts,
        "num_sensors": len(selected_tracts),
        "timing": {"total_ms": round((t_end - t_start) * 1000, 1)},
    }


# ── Coverage Analysis ───────────────────────────────────────────────────────

def compute_coverage(tracts, selected_tracts, radius_miles=10.0,
                     existing_sensors=None):
    """
    Compute coverage statistics — measuring NEW coverage added by the
    proposed sensors, factoring in existing infrastructure.
    """
    n = len(tracts)
    if not selected_tracts:
        return {
            "pct_covered": 0.0, "avg_distance_miles": 0.0,
            "max_distance_miles": 0.0, "coverage_by_ej_quartile": {},
        }

    t_lats = np.array([t["lat"] for t in tracts], dtype=np.float64)
    t_lons = np.array([t["lon"] for t in tracts], dtype=np.float64)

    # Combine existing + new sensors for total coverage
    e_lats, e_lons = _sensor_coords(existing_sensors)
    n_lats, n_lons = _sensor_coords(selected_tracts)
    min_dists = _vectorized_min_dist(t_lats, t_lons,
                                     np.concatenate([e_lats, n_lats]),
                                     np.concatenate([e_lons, n_lons]))

    # Also compute existing-only coverage for delta
    has_existing = len(e_lats) > 0
    if has_existing:
        existing_dists = _vectorized_min_dist(t_lats, t_lons, e_lats, e_lons)
        existing_covered = int(np.sum(existing_dists <= radius_miles))
    else:
        existing_covered = 0

    covered = int(np.sum(min_dists <= radius_miles))
    new_covered = covered - existing_covered

    # EJ quartile breakdown over tracts with a known score only; a missing
    # score is not evidence of low burden, so those tracts sit out.
    ej_scores = _tract_ej(tracts)
    known_ej = np.isfinite(ej_scores)
    valid_ej = ej_scores[known_ej]
    quartiles = np.percentile(valid_ej, [25, 50, 75]) if len(valid_ej) > 0 else [25, 50, 75]

    ej_quartile_coverage = {}
    labels = ["Q1 (Low EJ)", "Q2", "Q3", "Q4 (High EJ)"]
    bounds = [0, quartiles[0], quartiles[1], quartiles[2], 101]

    for q, label in enumerate(labels):
        mask = known_ej & (ej_scores >= bounds[q]) & (ej_scores < bounds[q + 1])
        if not np.any(mask):
            continue
        q_covered = int(np.sum(min_dists[mask] <= radius_miles))
        q_total = int(np.sum(mask))
        # Delta: how many NEW tracts covered vs existing-only
        if has_existing:
            q_existing = int(np.sum(existing_dists[mask] <= radius_miles))
        else:
            q_existing = 0
        ej_quartile_coverage[label] = {
            "covered": q_covered,
            "total": q_total,
            "pct": round(float(q_covered / q_total * 100), 1) if q_total > 0 else 0.0,
            "new_covered": q_covered - q_existing,
            "avg_distance_miles": round(float(np.mean(min_dists[mask])), 1),
        }

    return {
        "pct_covered": round(float(covered / n * 100), 1),
        "covered_count": covered,
        "new_covered": new_covered,
        "existing_covered": existing_covered,
        "total_tracts": n,
        "avg_distance_miles": round(float(np.mean(min_dists)), 1),
        "max_distance_miles": round(float(np.max(min_dists)), 1),
        "median_distance_miles": round(float(np.median(min_dists)), 1),
        "radius_miles": radius_miles,
        "coverage_by_ej_quartile": ej_quartile_coverage,
    }


# ── Comparison ──────────────────────────────────────────────────────────────

def compare_methods(tracts, k=25, num_reads=500, top_candidates=400,
                    proximity_threshold_miles=8.0, coverage_radius_miles=10.0,
                    existing_sensors=None, model_disagreement=None):
    """Run all three solvers and compare results with coverage analysis."""

    quantum_result = solve_quantum(
        tracts, k=k, num_reads=num_reads,
        top_candidates=top_candidates,
        proximity_threshold_miles=proximity_threshold_miles,
        existing_sensors=existing_sensors,
        model_disagreement=model_disagreement,
    )

    greedy_result = solve_greedy(
        tracts, k=k,
        proximity_threshold_miles=proximity_threshold_miles,
        existing_sensors=existing_sensors,
        model_disagreement=model_disagreement,
    )

    classical_sa_result = solve_classical_sa(
        tracts, k=k,
        proximity_threshold_miles=proximity_threshold_miles,
        existing_sensors=existing_sensors,
        model_disagreement=model_disagreement,
    )

    methods_map = {
        "quantum_annealing": quantum_result,
        "greedy": greedy_result,
        "classical_sa": classical_sa_result,
    }

    comparison = {}
    for name, result in methods_map.items():
        coverage = compute_coverage(
            tracts, result["selected_tracts"],
            radius_miles=coverage_radius_miles,
            existing_sensors=existing_sensors,
        )

        sel_ej = _tract_ej(result["selected_tracts"])
        avg_ej = (float(np.mean(sel_ej[np.isfinite(sel_ej)]))
                  if np.isfinite(sel_ej).any() else 0.0)

        avg_composite = float(np.mean([
            t["composite_score"]
            for t in result["selected_tracts"]
        ])) if result["selected_tracts"] else 0.0

        comparison[name] = {
            "method_display": result["method_display"],
            "num_sensors": result["num_sensors"],
            "coverage": coverage,
            "avg_ej_score": round(avg_ej, 1),
            "avg_composite_score": round(avg_composite, 4),
            "timing": result["timing"],
            "selected_tracts": result["selected_tracts"],
        }
        for key in ("best_energy", "num_reads", "feasible", "weights"):
            if key in result:
                comparison[name][key] = result[key]

    return comparison
