# Shared Skies Initiative - Real-Time Map

Real-time PM2.5 air quality predictions for every census tract in Texas.

## What it does

The app displays a live map of model-estimated ground-level PM2.5 (24-hour average) across all 6,896 Texas census tracts. A background loop on the server refreshes the PurpleAir sensor feed every 15 minutes, whether or not anyone is viewing the map, and recomputes every tract's prediction right after each refresh, from those readings plus current weather, CAMS aerosol data and NOAA smoke polygons. The page polls for new predictions every 5 minutes while its tab is visible and shows how old the data is ("Updated N min ago"). A second tab proposes sites for 25 new low-cost sensors, chosen by a quantum-inspired optimizer that runs classical simulated annealing (see Sensor placement).

## How it works

**Model:** A 4-model ML ensemble (Random Forest, LightGBM, XGBoost, CatBoost) trained on 285,798 daily readings from 310 in-Texas PurpleAir sensors (2021-01-01 to 2026-05-01). Blend weights are simplex-optimized via GroupKFold-over-sensors cross-validation; XGBoost ends up with zero weight, so the deployed blend is RF / LightGBM / CatBoost. Honest leave-one-sensor-out (LOSO) CV R² = **0.7134** over all 310 in-Texas sensors (see Known limitations for why accuracy away from sensors is likely lower). (The RMSE/MAE quoted in the ablation table below come from the 10-fold GroupKFold protocol and are not interchangeable with this figure — see Methodology.)

**Features (30 total):** same-day PurpleAir neighbor averages within 25, 50 and 100 km (the dominant inputs), daily weather (temperature, humidity, daily-max wind, surface pressure, precipitation; Open-Meteo or NASA POWER in training, Open-Meteo live), NOAA HMS smoke, CAMS aerosol optical depth and PM2.5, geographic features (latitude/longitude, distance to coast, distance to nearest training sensor), and calendar encodings on the UTC day. No EPA EJScreen variable is a model input — see Methodology.

**Live inference:** Each cycle takes the latest PurpleAir readings (the 24-hour rolling `pm2.5_24hour` value per sensor), rebuilds the neighbor features with the same BallTree code and radii as training, adds the cached weather, CAMS and smoke inputs, and runs the ensemble over all tracts. Serving follows the training pipeline closely but not exactly: weather is sampled on a 1° grid rather than at each point, and the PurpleAir value is a trailing 24 hours rather than a UTC calendar day.

**Refresh schedule** (server side, one background loop, independent of traffic):

| Input / output | Interval | Notes |
|---|---|---|
| PurpleAir sensors | 15 min (`PURPLEAIR_CACHE_TTL_MIN`) | On failure, retries after 5 and 10 min, then every 15 min. Readings older than 180 min (`PURPLEAIR_MAX_AGE_MIN`) stop counting as live and predictions fall back to per-sensor climatology. |
| Tract predictions | right after each PurpleAir refresh, and at least every 15 min | Encoded once per compute and served with an ETag, so unchanged polls get a 304. |
| Open-Meteo weather grid and CAMS air-quality grid | 6 h | |
| NOAA HMS smoke | 60 min | Today's UTC file; yesterday's stands in (flagged `hms_is_proxy`) until today's is published; never older. |
| Sensor placement | 6 h (`QUANTUM_CACHE_TTL_MIN`) | |
| Browser poll | 5 min while the tab is visible | Paused while hidden; refetches on return if the data is older than 5 min. |

**Sensor placement:** The Sensors tab picks 25 new sites. Every candidate tract gets a composite score: coverage gap 30%, model error 20% (per-tract LOSO residuals; tracts without one are filled by inverse-distance weighting of nearby residuals, else the median), EPA EJScreen demographic index 30%, and current predicted PM2.5 20%. The choice is written as a QUBO (quadratic unconstrained binary optimization, the input format of quantum annealers) with an exactly-25 budget term and a penalty on pairs whose 10-mile coverage areas overlap, over about 190 geographically spread candidates (one per grid cell). It is solved by **classical** simulated annealing on the server's CPU (`dwave-neal`), then refined with feasibility-preserving swaps; no quantum hardware is involved, and "quantum-inspired" describes only the problem formulation. Sites must be at least 8 miles apart and at least 8 miles from an existing sensor (the live PurpleAir network, or the training sensor set when live data is unavailable); water tracts and tracts without an EJScreen score are excluded. On the October 2026 data the result matched a greedy ranking over the same ~190-candidate pool on 24 of 25 sites, so the annealer adds little on this instance.

## Methodology

**No EJScreen-derived variable is used as a prediction input.** Earlier versions of the model took eight EPA EJScreen features — four demographic (EJF score, % people of color, % low income, % linguistically isolated) and four physical source-proximity (traffic, Superfund, RMP, diesel PM) — for 11.1% of total blended importance. All eight were removed.

The reason is that this model is used to study PM2.5 disparities across demographic groups. If a demographic variable is a model input, any disparity the model reproduces is partly an artifact of its own inputs rather than a property of the atmosphere, and feature-attribution (SHAP) results on it are not interpretable as physical findings. The four physical proximity features were dropped as well, even though they are not circular, so that no EJScreen product enters the prediction path at all.

The accuracy cost was measured, not assumed. Three arms were run on frozen folds — identical data, folds, seed, models and blend weights, varying only the feature list — under 10-fold GroupKFold over sensors on 285,798 rows / 310 sensors, with 95% CIs from a 2000-repetition per-sensor cluster bootstrap on the paired delta:

| Arm | Features | 10-fold GroupKFold R² | RMSE | MAE | ΔR² vs production | 95% CI |
|---|---|---|---|---|---|---|
| A — production | 38 | 0.71359 | 4.2146 | 2.6204 | — | — |
| B — no demographic | 34 | 0.71141 | 4.2307 | 2.6295 | +0.00218 | [−0.00315, +0.00833] |
| C — no EJScreen | 30 | 0.71407 | 4.2111 | 2.6149 | −0.00048 | [−0.00704, +0.00662] |

Both confidence intervals include zero, so neither removal is distinguishable from noise. (The absolute scores in this table are optimistic and only the deltas between arms should be compared; see Known limitations.) Arm C — the deployed configuration — has a slightly *better* point estimate than the 38-feature model and lower RMSE and MAE, and beats it in 7 of 10 folds. Removing all eight EJScreen features carries no measurable accuracy cost.

These three arms hold the blend weights **frozen** at the previous deployment's values (rf 0.6827 / lgbm 0.0079 / xgb 0 / cat 0.3094) so that the only thing varying is the feature list. That is why arm C's 0.71407 is not the same number as the deployed model's 0.7134: the deployed retrain re-optimizes its own simplex weights (rf 0.79352 / lgbm 0.070187 / xgb 0 / cat 0.136293) on the new out-of-fold predictions, and is evaluated under the 310-site LOSO protocol rather than 10-fold GroupKFold. Both protocols agree on the conclusion: the deployed 30-feature model scores **0.7134** against the previous 38-feature model's **0.7136**, a difference of −0.0002.

**Loss function.** The same retrain also switched LightGBM to `objective="huber"` and XGBoost to `objective="reg:pseudohubererror"`. PM2.5 is heavy-tailed and a squared-error objective lets a handful of smoke and dust days dominate the fit; the Huber losses bound that influence. This is disclosed here because it means the deployed model differs from its predecessor in **two** ways — feature set and loss — not one. The ablation above isolates the feature effect correctly (all three arms use the same models), but it should not be read as a full accounting of the change from the 38-feature deployment.

**Post-hoc environmental-justice analysis.** Because the model never receives a demographic variable, any disparity in its output is inferred from atmospheric measurements and geography alone. Predicting all 6,896 tracts across 365 days (2025-05-02 → 2026-05-01, mean field 8.92 µg/m³) and correlating against tract demographics afterwards:

| Demographic (EJScreen percentile) | Pearson r | Q5 − Q1 (µg/m³) | 95% CI |
|---|---|---|---|
| % low income | +0.127 | +0.719 (+8.4%) | [+0.576, +0.849] |
| EJ index | +0.088 | +0.561 (+6.4%) | [+0.416, +0.710] |
| % people of color | +0.029 | +0.370 (+4.1%) | [+0.216, +0.526] |
| % linguistically isolated | +0.003 | +0.122 (+1.4%) | [−0.043, +0.282] |

Low income shows the clearest monotonic gradient. The % people-of-color relationship is **not monotonic** — the least-POC quintile (9.01) is also elevated relative to the middle quintiles (8.67–8.75) before the most-POC quintile rises to 9.38 — so Spearman (+0.124) exceeds Pearson (+0.029) by four-fold. The linguistic-isolation interval includes zero and should not be reported as a disparity. Full results: `xai/outputs/posthoc_ej/`.

EJScreen data is still used in two places that are not model inputs: the post-hoc environmental-justice analysis (which correlates predictions against demographics *after* the fact), and the quantum sensor-placement optimizer, where prioritizing under-monitored overburdened communities is a deliberate policy choice applied downstream of the model.

Full results: `models/ablation_ejscreen.json`, per-fold log in `models/ablation_ejscreen_foldlog.txt`.

## Stack

| Layer | Tech |
|---|---|
| Frontend | React + Vite, Leaflet, i18n (EN/ES) |
| Backend | FastAPI (Python), served on Render |
| Hosting | Vercel (frontend) + Render (backend) |
| Data | PurpleAir API, Open-Meteo (weather, CC BY 4.0), Copernicus Atmosphere Monitoring Service via Open-Meteo Air Quality, NASA POWER (training weather), NOAA HMS, U.S. Census TIGER (EPA EJScreen: post-hoc analysis and sensor placement only, never a model input) |
| Maps and search | Stadia Maps basemap (Esri fallback), OpenStreetMap data, Photon and Nominatim geocoding |
| ML | scikit-learn, LightGBM, XGBoost, CatBoost, scipy; dimod + dwave-neal for sensor placement |

## Data notes

- PM2.5 values are raw PurpleAir ATM-channel concentrations, with no EPA correction applied. Live inference uses the same channel as most training targets; see Known limitations for the exception.
- Color scale: Good 0–9 µg/m³ / Moderate 9–13 / Elevated 13–17 / High 17+. The 9 µg/m³ break matches the U.S. EPA annual NAAQS (2024).
- Training sensors: 310 in-Texas PurpleAir sensors. 117 additional border-state sensors are used as same-day neighbors but excluded from training targets.

## Known limitations

- **Mixed PM2.5 basis in the training targets.** 61,224 of the 412,507 rows in the training dataset (14.8%; 59,640 of the 285,798 in-Texas rows used as training targets, about 21%) are the 2025 legacy rows copied from `p2_processed.xls`. Their PM2.5 is not raw `pm2.5_atm`: against the cached ATM history for the same sensor-days they fit about 0.54 × ATM + 1.6. Fixing this means replacing those rows with the cached ATM values and retraining, which has not been done yet. The 8-8-26 retrain still contains 39,105 of these rows.
- **The EJScreen ablation's absolute scores are optimistic.** `pipeline/13_ablation_ejscreen.py` uses neighbor features computed once from the full sensor pool rather than recomputed per fold, so a held-out sensor's features still include other held-out sensors' same-day readings. The R², RMSE and MAE in the ablation table are therefore higher than an honest held-out score. The deltas between arms remain comparable, because all three arms share the same features and folds.
- **LOSO has no spatial buffer.** Leave-one-sensor-out keeps every other sensor, however close, in training and in the neighbor features. A training sensor's nearest other sensor is a median 3.6 km away, while a tract's nearest training sensor is a median 7.8 km away (90th percentile 54 km). Accuracy at unmonitored tracts, especially rural ones, is therefore likely lower than the 0.7134 LOSO score.
- **No EPA correction.** Raw PurpleAir ATM values usually read higher than regulatory reference monitors, while the color bands are anchored to the EPA annual standard, which is defined for reference-monitor annual means. A tract shown above 9 µg/m³ on a given day is not an exceedance of that standard.

## Deployment

The backend runs on Render's free tier (one uvicorn process, 512 MB RAM, ephemeral disk); the frontend runs on Vercel, where `frontend/vercel.json` rewrites `/api/*` to the Render service. `render.yaml` describes the service, but a service created by hand in the Render dashboard ignores it, so check the dashboard's Environment tab. Set the health check path to `/api/health`.

| Variable | Default | Purpose |
|---|---|---|
| `PURPLEAIR_API_KEY` | none (required) | PurpleAir READ key. Without it, live neighbor features are off, predictions use climatology, and `/api/health` reports `degraded`. |
| `PURPLEAIR_CACHE_TTL_MIN` | 15 | PurpleAir refresh and prediction recompute interval, in minutes. |
| `PURPLEAIR_MAX_AGE_MIN` | 180 | How old PurpleAir data may get before predictions stop treating it as live. |
| `QUANTUM_CACHE_TTL_MIN` | 360 | Sensor-placement re-solve interval. |
| `OPENMETEO_GRID_TTL_MIN` | 360 | Weather and CAMS grid refresh interval. |
| `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` | unset | Persistent visit counter. Without them the count lives in SQLite and resets on every Render restart. |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS allowlist. |
| `KEEPALIVE_URL`, `KEEPALIVE_INTERVAL` | `RENDER_EXTERNAL_URL`, 840 s | The server pings its own public `/api/health` every 14 min so the free tier does not spin it down. |
| `SSI_SNAPSHOTS_DIR` | `backend/static/snapshots` | Where the latest inputs and results are written. On Render this disk is wiped on every restart or deploy. |
| `PYTHON_VERSION`, `MALLOC_ARENA_MAX`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | see `render.yaml` | Python version and memory settings for the 512 MB instance. |

**PurpleAir cost.** Measured on 2026-10-20, one statewide pull returned 497 sensors and cost 3,484 points. At a 15-minute refresh that is about 335,000 points a day, or about 10M points a month; the organization balance was about 400M at the time. Raising `PURPLEAIR_CACHE_TTL_MIN` lowers the cost proportionally. Each Render restart also makes one extra pull (the snapshot disk is wiped), and a local server run with the key spends points the same way.

**Health.** `GET /api/health` reads only in-memory state, so it is safe as the health check, and always returns HTTP 200. `status` is `degraded` when the model is not loaded, live neighbor features are not in use, or predictions are older than three refresh intervals. It also reports `purpleair_key_configured`, `live_neighbors`, `predictions_age_min`, `purpleair_age_min`, `refresh_interval_min`, `memory_rss_mb` and the per-source `data_sources` block.

## Running locally

From the repo root (the folder `git clone` creates), in two terminals:

```
# Terminal 1: backend
python3 -m venv .venv && . .venv/bin/activate      # Python 3.11 or 3.12
pip install -r backend/requirements.txt
# Optional: put PURPLEAIR_API_KEY=<your READ key> in backend/.env (gitignored).
[ -f backend/.env ] && { set -a; . backend/.env; set +a; }
uvicorn backend.main:app --port 8000
```

```
# Terminal 2: frontend
cd frontend && npm ci && npm run dev               # http://localhost:5173, proxies /api to :8000
```

Without the key the backend still runs, on climatological neighbor features. With it, a local server spends PurpleAir points exactly like production, so consider a larger `PURPLEAIR_CACHE_TTL_MIN` for development.

**After a retrain** that changes the training data, re-run `pipeline/14_export_training_sensors.py` and commit `models/training_sensors.json` together with the new model: the backend computes `dist_to_nearest_sensor` from that file alone. In the same commit, update the test constants that pin the shipped model and sensor set (see Running tests).

## Running tests

From the repo root:

```
pip install -r backend/requirements.txt pytest==9.1.1 pytest-asyncio==1.4.0
python -m pytest
```

The suite is offline: it needs no PurpleAir key and mocks every outbound call. CI (`.github/workflows/ci.yml`) runs it on Python 3.11 and 3.12 and builds the frontend.

Some tests pin values that are expected to change when the model or its inputs change on purpose. A retrain, or a new `models/sensor_recent_pm.json` or `models/training_sensors.json`, moves them; a dependency bump should not. Regenerate them from the new files rather than loosening the tolerances:

- `GOLDEN_MEAN` and `GOLDEN_TRACTS` in `tests/test_model.py` (climatology-fallback predictions, checked to 1e-6; the comment there says how to recompute them);
- the 467-sensor count and the 7.80 km median and 54.0 km 90th-percentile tract-to-sensor distances in `tests/test_features.py`;
- `num_existing_sensors == 467` in `tests/test_api.py`.

If the median or 90th percentile changes, update the figures in Known limitations too.

## Structure

```
backend/      FastAPI app + prediction pipeline
frontend/     React/Vite app
pipeline/     Data pull, feature engineering, model training scripts
models/       Trained ensemble bundle + metrics
```

## Author

Saketh Chebrolu — [Shared Skies Initiative](https://sharedskiesinitiative.org/real-time-map)
