"""Reproduce the three-arm EJScreen feature ablation reported in README.md.

The deployed model drops all eight EPA EJScreen-derived features. This script is
the evidence for the claim that doing so costs nothing measurable — it was
previously run ad hoc and only its JSON output was committed, which made the
central "cost measured, not assumed" statement unauditable. Running this file
regenerates models/ablation_ejscreen.json and the per-fold log from scratch.

Design (matches the committed artifact):
  * Frame            load_data() -> PM25_TRAIN_CAP -> in_tx filter, i.e. exactly
                     03_train_enhanced.__main__'s deployed-model frame.
  * Arms             A = 38 features (the pre-removal production set)
                     B = 34 (minus the 4 demographic)
                     C = 30 (minus all 8 EJScreen)  <- the deployed set
  * Folds            10-fold GroupKFold over sensor_id, deterministic.
  * Models           pipeline/03_train_enhanced.train_ensemble(), imported rather
                     than re-specified, so hyperparameters cannot drift.
  * Blend            FROZEN at the pre-removal deployment weights so that the
                     only thing varying across arms is the feature list. This is
                     why arm C's R2 differs from the deployed model's LOSO R2 —
                     the deployed retrain re-optimizes its own simplex weights.
  * Uncertainty      2000-repetition per-sensor cluster bootstrap on the PAIRED
                     delta vs arm A (resample sensors, not rows: residuals are
                     spatially clustered and a row bootstrap understates CIs).

Known optimism: the nbr_* neighbor features come from load_data(), which builds
them once from the FULL sensor pool (leave-self-out only). They are not
recomputed per fold the way loso_cv() does, so a held-out sensor's features
still average same-day readings from other held-out sensors. The absolute R2,
RMSE and MAE of every arm are therefore optimistic and are not comparable to
the deployed LOSO score. The arm-vs-arm deltas remain comparable, because all
three arms share the same neighbor features and folds.

Run from the repo root:
    python pipeline/13_ablation_ejscreen.py
    python pipeline/13_ablation_ejscreen.py --folds 10 --boot 2000
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEMOGRAPHIC = ["ejf_score", "pct_people_of_color", "pct_low_income",
               "pct_ling_isolated"]
PHYSICAL_EJ = ["traffic_proximity", "superfund_proximity", "rmp_proximity",
               "diesel_pm_proximity"]

# The blend the pre-removal model shipped with. Frozen on purpose (see header).
FROZEN_WEIGHTS = {"rf": 0.6827, "lgbm": 0.0079, "xgb": 0.0, "cat": 0.3094}


def _load_trainer():
    path = os.path.join(ROOT, "pipeline", "03_train_enhanced.py")
    spec = importlib.util.spec_from_file_location("train_enhanced", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def blend(models, X):
    """Weighted ensemble prediction using the frozen deployment weights."""
    out = None
    for name, w in FROZEN_WEIGHTS.items():
        if w == 0.0 or name not in models:
            continue
        p = models[name].predict(X) * w
        out = p if out is None else out + p
    return out


def run_arm(df, feats, groups, y, n_folds, label, log):
    """Pooled out-of-fold predictions for one feature set."""
    X = df[feats].to_numpy(dtype=np.float64)
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_folds)
    trainer = _load_trainer()
    t0 = time.time()
    for k, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        # train_ensemble carves its early-stopping holdout off the TAIL of the
        # array, so hand it date-sorted rows to keep that holdout temporal.
        order = np.argsort(df["date"].to_numpy()[tr], kind="stable")
        tr = tr[order]
        models = trainer.train_ensemble(X[tr], y[tr], verbose=False)
        oof[te] = blend(models, X[te])
        msg = (f"  [{label}] fold {k}/{n_folds}  "
               f"R2={r2_score(y[te], oof[te]):.4f}  ({time.time()-t0:.0f}s)")
        print(msg, flush=True)
        log.append(msg)
    return oof


def metrics(y, p):
    return {
        "r2": float(r2_score(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    trainer = _load_trainer()
    df = trainer.load_data()

    cap = trainer.PM25_TRAIN_CAP
    target = trainer.TARGET
    pool_rows = len(df)
    if cap is not None:
        df = df[df[target] <= cap].reset_index(drop=True)
    if "in_tx" in df.columns:
        df = df[df["in_tx"]].reset_index(drop=True)
    df = df.dropna(subset=[target]).reset_index(drop=True)

    deployed = list(trainer.FEATURES)                       # 30
    arm_c = deployed
    arm_b = deployed + PHYSICAL_EJ                          # 34
    arm_a = deployed + PHYSICAL_EJ + DEMOGRAPHIC            # 38

    for name, feats in (("A", arm_a), ("B", arm_b), ("C", arm_c)):
        missing = [f for f in feats if f not in df.columns]
        if missing:
            sys.exit(f"arm {name}: columns absent from the frame: {missing}")

    y = df[target].to_numpy(dtype=np.float64)
    groups = df["sensor_id"].to_numpy()
    print(f"\n[ablation] {len(df):,} rows, {pd.Series(groups).nunique()} sensors, "
          f"{args.folds}-fold GroupKFold, frozen weights {FROZEN_WEIGHTS}\n")

    log = []
    oof = {}
    for name, feats in (("A_38_production", arm_a),
                        ("B_34_no_demographic", arm_b),
                        ("C_30_no_ejscreen", arm_c)):
        oof[name] = run_arm(df, feats, groups, y, args.folds, name, log)

    arms = {}
    for name, feats in (("A_38_production", arm_a),
                        ("B_34_no_demographic", arm_b),
                        ("C_30_no_ejscreen", arm_c)):
        arms[name] = {"n_features": len(feats), **metrics(y, oof[name])}

    # Paired per-sensor cluster bootstrap of each arm's delta vs arm A.
    sensors = np.unique(groups)
    idx_by = {s: np.where(groups == s)[0] for s in sensors}
    rng = np.random.default_rng(args.seed)
    draws = [np.concatenate([idx_by[s] for s in
                             rng.choice(sensors, size=len(sensors), replace=True)])
             for _ in range(args.boot)]
    base = oof["A_38_production"]
    for name in ("B_34_no_demographic", "C_30_no_ejscreen"):
        d = np.array([r2_score(y[i], oof[name][i]) - r2_score(y[i], base[i])
                      for i in draws])
        lo, hi = np.percentile(d, [2.5, 97.5])
        arms[name]["delta_r2_vs_A"] = float(arms[name]["r2"] - arms["A_38_production"]["r2"])
        arms[name]["delta_r2_ci95"] = [float(lo), float(hi)]
        arms[name]["ci_includes_zero"] = bool(lo <= 0.0 <= hi)

    out = {
        "design": {
            "folds": f"{args.folds}-fold GroupKFold over sensors (deterministic)",
            "seed": args.seed,
            "bootstrap_reps": args.boot,
            "bootstrap": "per-sensor cluster bootstrap on the paired delta",
            "blend_weights": FROZEN_WEIGHTS,
            "blend_weights_note": (
                "frozen at the pre-removal deployment so the only variable "
                "across arms is the feature list; the deployed model "
                "re-optimizes its own simplex weights and is scored under "
                "310-site LOSO, so its R2 is not this table's arm C"),
            "target": "raw PurpleAir ATM",
            "cap": cap,
            "n_rows": int(len(df)),
            "n_sensors": int(len(sensors)),
            "neighbour_pool_rows": int(pool_rows),
            "models": "pipeline/03_train_enhanced.train_ensemble (imported)",
        },
        "arms": arms,
    }
    dest = os.path.join(ROOT, "models", "ablation_ejscreen.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(ROOT, "models", "ablation_ejscreen_foldlog.txt"), "w") as f:
        f.write("\n".join(log) + "\n")

    print("\n" + "=" * 66)
    for k, v in arms.items():
        extra = ""
        if "delta_r2_vs_A" in v:
            lo, hi = v["delta_r2_ci95"]
            extra = f"  dR2={v['delta_r2_vs_A']:+.5f}  CI[{lo:+.5f}, {hi:+.5f}]"
        print(f"  {k:22s} n={v['n_features']:2d}  R2={v['r2']:.5f}  "
              f"RMSE={v['rmse']:.4f}  MAE={v['mae']:.4f}{extra}")
    print("=" * 66)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
