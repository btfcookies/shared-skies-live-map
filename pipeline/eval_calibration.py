"""Read-only evaluator: does monotone (isotonic) calibration of the deployed
ensemble blend honestly improve the LOSO-CV R²?

This NEVER touches the feature pipeline or the backend — it only reads the
already-computed per-model out-of-fold predictions in models/loso_oof.npz, builds
the production blend (using the weights stored in models/ensemble.joblib), and asks:

  Does a MONOTONE post-hoc calibrator g(blend) -> y, fit ONLY on held-out
  sensors (GroupKFold-over-sensors), raise the pooled LOSO R²?

Isotonic regression (PAVA) finds the monotone g minimizing held-out MSE; it can
ONLY help if the blend has genuine monotone miscalibration (tree ensembles shrink
toward the mean and under-predict extremes). Fitting per grouped fold and scoring
on the held-out fold makes the reported gain optimism-free and paper-defensible
(van der Laan Super-Learner / Platt calibration).

Run AFTER training completes:
    python pipeline/eval_calibration.py

loso_oof.npz is gitignored and only a local training run rewrites it, so a stale
copy can sit next to a newer bundle pulled from git. The script therefore checks
that the OOF file, metrics.json and the bundle come from the same run (model set,
feature count, OOF row and sensor counts, blend weights) and exits with an error
if they do not, rather than scoring one model's OOF with another model's weights.
It prints the uncalibrated vs grouped-CV-calibrated LOSO R². It does NOT modify
any artifact — wiring the calibrator into the bundle is a separate, deliberate
step taken only if v5 needs the margin.
"""
import os
import json

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")


def load_oof():
    p = os.path.join(MODELS_DIR, "loso_oof.npz")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} — run training first (loso_cv saves it).")
    z = np.load(p, allow_pickle=True)
    model_names = [str(m) for m in z["model_names"]]
    oof = {n: z[f"oof_{n}"] for n in model_names}
    # The trainer does not store the feature list in the npz today; a future one
    # that does gets compared directly.
    feature_names = [str(f) for f in z["feature_names"]] if "feature_names" in z.files else None
    return z["y"], z["valid"], z["sensor_id"], model_names, oof, feature_names


def load_deployed():
    bp = os.path.join(MODELS_DIR, "ensemble.joblib")
    mp = os.path.join(MODELS_DIR, "metrics.json")
    for p in (bp, mp):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run training first.")
    with open(mp) as f:
        metrics = json.load(f)
    return joblib.load(bp), metrics


def check_same_run(bundle, metrics, model_names, oof_features, valid, sid_all):
    """Exit loudly unless the OOF file belongs to the deployed bundle. The npz
    has no feature list, so metrics.json (written by the same training run as
    the npz) links the two: its OOF row/sensor counts must match the npz and
    its feature count and weights must match the bundle."""
    problems = []
    bundle_models = sorted(bundle["models"])
    if sorted(model_names) != bundle_models:
        problems.append(f"model set: npz {sorted(model_names)} vs bundle {bundle_models}")
    bundle_feats = list(bundle["feature_names"])
    if oof_features is not None and oof_features != bundle_feats:
        problems.append(f"feature list: npz has {len(oof_features)} features, bundle {len(bundle_feats)}")
    if metrics.get("n_features") != len(bundle_feats):
        problems.append(f"feature count: metrics.json {metrics.get('n_features')} vs bundle {len(bundle_feats)}")
    ewo = metrics.get("ensemble_weight_optimization", {})
    n_rows = int(np.asarray(valid).sum())
    n_sensors = len(np.unique(np.asarray(sid_all)[np.asarray(valid)]))
    if ewo.get("n_oof_rows") != n_rows or ewo.get("n_sensors") != n_sensors:
        problems.append(f"OOF size: npz {n_rows:,} rows / {n_sensors} sensors vs metrics.json "
                        f"{ewo.get('n_oof_rows')} rows / {ewo.get('n_sensors')} sensors")
    if ewo.get("models") is not None and sorted(ewo["models"]) != sorted(model_names):
        problems.append(f"model set: npz {sorted(model_names)} vs metrics.json {sorted(ewo['models'])}")
    mw = metrics.get("loso_cv_optimized", {}).get("weights", {})
    bw = bundle.get("weights", {})
    # metrics.json rounds weights to 4 decimals.
    if any(abs(float(mw.get(n, 0.0)) - float(bw.get(n, 0.0))) > 1e-3 for n in set(mw) | set(bw)):
        problems.append(f"blend weights: metrics.json {mw} vs bundle {bw}")
    if problems:
        raise SystemExit(
            "models/loso_oof.npz does not match the deployed models/ensemble.joblib:\n  - "
            + "\n  - ".join(problems)
            + "\nRe-run pipeline/03_train_enhanced.py, or use the npz, metrics.json and bundle "
              "from the same training run."
        )


def chosen_weights(bundle, model_names):
    """The blend weights the deployed bundle actually serves with."""
    w = bundle["weights"]
    return {n: float(w.get(n, 0.0)) for n in model_names}


def grouped_isotonic_r2(blend, y, groups, n_splits=5):
    """Pooled LOSO R² after monotone calibration fit on held-out sensors."""
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    cal = np.full(len(y), np.nan)
    for tr, te in gkf.split(blend, y, groups):
        ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
        ir.fit(blend[tr], y[tr])
        cal[te] = ir.predict(blend[te])
    return cal


def main():
    y_all, valid, sid_all, model_names, oof, oof_features = load_oof()
    bundle, metrics = load_deployed()
    check_same_run(bundle, metrics, model_names, oof_features, valid, sid_all)
    stack = np.column_stack([oof[n] for n in model_names])
    row_valid = np.asarray(valid) & np.all(np.isfinite(stack), axis=1)
    P = stack[row_valid]
    y = np.asarray(y_all)[row_valid]
    groups = np.asarray(sid_all)[row_valid]

    w = chosen_weights(bundle, model_names)
    wv = np.array([w[n] for n in model_names])
    wv = wv / wv.sum()
    blend = P @ wv

    print("=" * 66)
    print("ISOTONIC CALIBRATION EVALUATION (read-only, on LOSO OOF)")
    print("=" * 66)
    print(f"  rows={len(y):,}  sensors={len(np.unique(groups))}  models={model_names}")
    print(f"  chosen weights: " + "  ".join(f"{n.upper()}:{w[n]:.3f}" for n in model_names))

    r2_un = r2_score(y, blend)
    rmse_un = np.sqrt(mean_squared_error(y, blend))
    mae_un = mean_absolute_error(y, blend)
    print(f"\n  UNCALIBRATED blend   R²={r2_un:.4f}  RMSE={rmse_un:.4f}  MAE={mae_un:.4f}")

    cal = grouped_isotonic_r2(blend, y, groups)
    r2_cal = r2_score(y, cal)
    rmse_cal = np.sqrt(mean_squared_error(y, cal))
    mae_cal = mean_absolute_error(y, cal)
    print(f"  CALIBRATED (grpCV)   R²={r2_cal:.4f}  RMSE={rmse_cal:.4f}  MAE={mae_cal:.4f}")
    print(f"\n  Δ from calibration:  R² {r2_cal - r2_un:+.4f}   (honest, grouped-CV, no optimism)")

    if r2_cal > r2_un:
        print(f"\n  → calibration HELPS by {r2_cal - r2_un:+.4f}. If v5 needs the margin,")
        print(f"    fit a FINAL isotonic on ALL OOF and store its (x,y) knots in the")
        print(f"    bundle, applied right after the weighted blend at inference.")
    else:
        print(f"\n  → calibration does NOT help (blend already well-calibrated). Skip it.")
    print(f"\n  HEADLINE: uncal={r2_un:.4f}  cal={r2_cal:.4f}")


if __name__ == "__main__":
    main()
