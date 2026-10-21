# 8-8-26 Huge Retrain

Full retrain of the Shared Skies live-map PM2.5 ensemble on GT PACE Phoenix,
using the AQNet v4 PurpleAir archival pull (started August 8, 2026) merged with
the existing production dataset. Data loading, features, models and
hyperparameters come from `pipeline/03_train_enhanced.py`, imported
unmodified. The steps of its `__main__` and its leave-one-site-out loop were
re-implemented in `train_full.py`, `loso_shard.py` and `loso_merge.py` so the
loop could run as a 24-task SLURM array. They follow 03's logic but are not
03 itself, so this is not a byte-identical rerun of production.

**The production files in `models/` at the repo root are untouched.** Everything
from this run lives in this folder. Promotion to the live site is a deliberate
step, documented at the bottom.

## What happened with the download

The plan was to retrain on the complete v4 pull: 10,874 sensors across seven
western states, of which 505 are in Texas (the live map's domain). The pull
died at 43% when the PurpleAir API key ran out of points (HTTP 402 Payment
Required on every call). The fetcher's internal budget guard counted rows, but
PurpleAir bills rows times fields (seven fields per row on this pull), so the
account hit its real ceiling long before the row guard tripped. Two of four
shards (even sensor indexes) completed; two died mid-pull.

What landed for Texas: 200 of 505 sensors (86 of the 6-hourly near-monitor
tier, 114 daily), 228,164 of 564,379 sensor-days. The decision was made to
retrain on what landed rather than wait for the account to be topped up. The
pull is fully resumable: finished sensors are never re-fetched, and the
remaining 305 Texas sensors can be added with a rerun of this same chain once
the key has points again.

## The dataset (`dataset/p2_processed_v3.parquet`)

Built by `scripts/build_dataset.py` in the exact 41-column schema of
`p2_processed_v2.xls`:

- **New part**: the 200 fetched Texas sensors reduced to daily UTC rows.
  Target pm25 reconstructed as the row-wise mean of the A and B ATM channels
  (production pulls PurpleAir's own blended `pm2.5_atm`; on 139,856
  overlapping sensor-days the two agree at r = 0.983, MAE 0.48, with a
  +0.44 ug/m3 mean offset). Production QC applied: 0-200 range, per-sensor
  MAD z < 20, minimum 60 days. 173,026 raw rows -> 162,335 kept across 174
  sensors.
- **Old part**: the shipped production dataset for the 325 sensors not in the
  new pull (sensors that have died since, plus out-of-state border neighbors),
  and gap-fill days for overlapping sensors (new rows win on conflicts). It
  still carries 39,105 of the legacy 2025 rows whose PM2.5 is not raw
  `pm2.5_atm` (see Known limitations in the top-level README).
- **Covariates, all rebuilt fresh through Aug 7, 2026**: per-sensor Open-Meteo
  daily weather (all 505 sensors, 99.99% row coverage, zero NASA POWER
  fallbacks), NOAA HMS smoke via the shared train/serve point-in-polygon code
  (2,045 days), CAMS aerosol and ERA5 extras on the 0.5 degree cell grid, GEOID
  by nearest tract centroid, Texas membership audited by point-in-polygon
  (89 new sensors audited, 25 outside Texas kept as neighbors only).

Final table: **434,986 rows, 499 sensors, 2021-01-01 to 2026-08-07**
(production was 412,507 rows, 467 sensors, 2021-01-01 to 2026-05-01).

## Results

Same validation protocol as production: pooled leave-one-site-out (LOSO)
cross-validation over in-Texas sensors with per-fold neighbor recompute, then
simplex-convex ensemble weights fit on the out-of-fold predictions and scored
with GroupKFold-over-sensors cross-fit (the honest, no-optimism number).

| Metric | Production (v6) | This retrain (v7) |
|---|---|---|
| LOSO R2, optimized weights (headline) | 0.7134 | **0.7076** |
| LOSO R2, inverse-MSE baseline | 0.7090 | 0.7020 |
| LOSO RMSE (ug/m3), inverse-MSE baseline | 4.2486 | 4.2828 |
| LOSO sites | 310 | 319 |
| Random-split ensemble R2 | 0.8033 | 0.8058 |
| Chosen weights | RF .794 / LGBM .070 / CAT .136 | RF .818 / LGBM .182 |
| Sensors / rows | 467 / 412,507 | 499 / 434,986 |
| Data ends | 2026-05-01 | 2026-08-07 |

Read: the new model is statistically indistinguishable from production on
spatial generalization (0.708 vs 0.713 pooled LOSO R2, on a slightly larger
and partly different fold population), while being trained on fresher data
with 32 more sensors and about 14 more weeks of coverage, including the
smoke periods of summer 2026. Single-model LOSO R2: RF 0.7069, CAT 0.6941, LGBM
0.6937, XGB 0.6816. The convex optimizer again kept RF dominant and zeroed
XGB; CatBoost dropped out this round in favor of LightGBM.

The honest caveat: this was meant to be a much bigger jump. With the pull
stopped at 43%, the effective new signal is 174 refreshed sensors rather than
505. The remaining 305 Texas sensors are the upside still on the table.

## What is in this folder

- `models/` - the complete new model set, drop-in compatible with the backend:
  `ensemble.joblib` (v7 bundle, LOSO-optimized weights, 30 features),
  `metrics.json`, `feature_names.json`, `loso_residuals.json` (per-GEOID, for
  the quantum sensor-placement QUBO), `loso_oof.npz` (per-model out-of-fold
  matrix), `sensor_recent_pm.json` (climatology fallback), `site_metrics.json`,
  `random_split_ctx.json`.
- `dataset/` - `p2_processed_v3.parquet` (the training table),
  `build_report.json` (QC and agreement numbers), updated
  `sensor_tx_membership.csv`.
- `scripts/` - everything that ran on Phoenix: dataset builder, covariate
  pullers (weather, HMS, cells, membership), the three wrappers that
  re-implement the production script's main flow and LOSO loop around its
  imported functions (`train_full.py`, `loso_shard.py`, `loso_merge.py`), and
  the SLURM job generator and dependency-chained submitter.
- `logs/` - the build, train, and merge job logs from Phoenix.

## Promoting to production

The live backend loads `models/ensemble.joblib` and friends from the repo
root. To ship this model:

    cp "8-8-26 huge retrain/models/ensemble.joblib" models/
    cp "8-8-26 huge retrain/models/feature_names.json" models/
    cp "8-8-26 huge retrain/models/metrics.json" models/
    cp "8-8-26 huge retrain/models/loso_residuals.json" models/
    cp "8-8-26 huge retrain/models/sensor_recent_pm.json" models/
    cp "8-8-26 huge retrain/dataset/sensor_tx_membership.csv" pipeline/

The backend also reads `dist_to_nearest_sensor` from `models/training_sensors.json`,
which must be regenerated from this run's 499-sensor dataset before shipping:

    python pipeline/14_export_training_sensors.py \
        --source-path "8-8-26 huge retrain/dataset/p2_processed_v3.parquet"

The test suite pins the shipped model and sensor set, so CI goes red until
these are regenerated in the same commit (see Running tests in the top-level
README): `GOLDEN_MEAN` / `GOLDEN_TRACTS` in `tests/test_model.py`, the
467-sensor count and 7.80 km / 54.0 km tract-to-sensor distances in
`tests/test_features.py`, and `num_existing_sensors == 467` in
`tests/test_api.py`. Note that `tests/test_features.py` also checks
`models/training_sensors.json` against an export of the production parquet,
so that test must be pointed at this dataset too.

Then commit and push (Render redeploys from main). Given the flat headline
number, shipping is optional until the full pull completes; the strongest
argument for shipping now is the fresher climatology fallback and residual map.

## Finishing the job when the key has points

1. Resume the fetch on Phoenix (it skips all finished sensors):
   the other session's `pa-v4c-*` jobs are already queued, or rerun
   `fetch_pa_v4.py` shards from the aqnet repo.
2. Rerun this chain: `bash ~/scratch/livemap_retrain/retrain/submit_chain.sh`
   after removing `~/scratch/livemap_retrain/repo/models/loso_shard_*.done.joblib`
   and the outputs of the build (it is idempotent otherwise).
3. Collect as here.

Run provenance: PACE Phoenix, account gts-ar70, embers QOS, jobs
11759862 (build), 11759863 (HMS), 11759864 (train), 11759865[0-23] (LOSO
array), 11759866 (merge), August 9, 2026. Total chain wall time roughly 30
minutes for build and covariates plus ~25 minutes of 24-way LOSO.
