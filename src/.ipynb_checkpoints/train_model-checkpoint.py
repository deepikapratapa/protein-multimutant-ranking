#!/usr/bin/env python3
"""
Train ΔΔG regression models on engineered features and evaluate.

Usage:
python -m src.train_model \
  --features data/processed/features_basic.csv \
  --train data/processed/train_split.csv \
  --test  data/processed/test_split.csv \
  --outdir results/models

Outputs (in --outdir):
  - metrics_test.json
  - best_model.pkl            (sklearn Pipeline)
  - feature_list.txt
  - predictions_test.csv
  - plots/ (optional scatter/violin)
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

RANDOM_STATE = 42

ID_COLS = ["clid","pdb_id","chain","wt","mut","res_index","pos","pH","temp_C"]
TARGET = "ddg_exp_kcal"

CANDIDATES = {
    "rf": RandomForestRegressor(
        n_estimators=600, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=RANDOM_STATE
    ),
    "gbr": GradientBoostingRegressor(
        n_estimators=1000, learning_rate=0.02, max_depth=3,
        random_state=RANDOM_STATE
    ),
}
if HAS_XGB:
    CANDIDATES["xgb"] = XGBRegressor(
        n_estimators=1200, learning_rate=0.03, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1,
        tree_method="hist"
    )

def build_feature_list(df: pd.DataFrame) -> list[str]:
    """All numeric feature columns (exclude identifiers + target)."""
    drop = set(["ddg_exp_kcal"] + [c for c in ID_COLS if c in df.columns])
    feats = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    return feats

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def eval_model(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    r2 = cross_val_score(model, X, y, cv=kf, scoring="r2").mean()
    neg_rmse = cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error").mean()
    return {"cv_r2": float(r2), "cv_rmse": float(-neg_rmse)}

def fit_best(features_csv: Path, train_csv: Path, test_csv: Path, outdir: Path):
    outdir = Path(outdir); (outdir / "plots").mkdir(parents=True, exist_ok=True)

    feats_df = pd.read_csv(features_csv)
    train_idx = pd.read_csv(train_csv)
    test_idx  = pd.read_csv(test_csv)

    # Join to get engineered features for the same rows
    # Merge keys:
    keys = ["pdb_id","chain","wt","res_index","mut"]
    train = train_idx.merge(feats_df, on=keys, how="left")
    test  = test_idx.merge(feats_df,  on=keys, how="left")

    # Build feature list
    feat_cols = build_feature_list(train)
    with open(outdir / "feature_list.txt", "w") as f:
        f.write("\n".join(feat_cols))

    X_train, y_train = train[feat_cols].values, train[TARGET].values
    X_test,  y_test  = test[feat_cols].values,  test[TARGET].values

    # Common preprocessor
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), list(range(len(feat_cols))))],
        remainder="drop"
    )

    best_name, best_cv, best_pipe = None, {"cv_r2": -9e9, "cv_rmse": 9e9}, None
    for name, reg in CANDIDATES.items():
        pipe = Pipeline([("pre", pre), ("reg", reg)])
        cvm = eval_model(pipe, X_train, y_train)
        print(f"[CV] {name}: R2={cvm['cv_r2']:.3f}, RMSE={cvm['cv_rmse']:.3f}")
        if cvm["cv_r2"] > best_cv["cv_r2"]:
            best_name, best_cv, best_pipe = name, cvm, pipe

    # Fit best on full train
    best_pipe.fit(X_train, y_train)

    # Test metrics
    y_pred = best_pipe.predict(X_test)
    metrics = {
        "model": best_name,
        **best_cv,
        "test_r2": r2_score(y_test, y_pred),
        "test_rmse": rmse(y_test, y_pred),
        "test_mae": mean_absolute_error(y_test, y_pred),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "features": feat_cols,
    }
    with open(outdir / "metrics_test.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[TEST]", json.dumps(metrics, indent=2))

    # Save predictions
    pred_df = test[keys + [TARGET]].copy()
    pred_df["y_pred"] = y_pred
    pred_df.to_csv(outdir / "predictions_test.csv", index=False)

    # Save model
    import joblib
    joblib.dump(best_pipe, outdir / "best_model.pkl")

    # Optional quick scatter
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(figsize=(4,4))
        plt.scatter(y_test, y_pred, s=12, alpha=0.6)
        lims = [min(y_test.min(), y_pred.min())-0.5, max(y_test.max(), y_pred.max())+0.5]
        plt.plot(lims, lims, 'r--'); plt.xlim(lims); plt.ylim(lims)
        plt.xlabel("Experimental ΔΔG"); plt.ylabel("Predicted ΔΔG")
        plt.title(f"Test scatter — {best_name}")
        plt.tight_layout()
        plt.savefig(outdir / "plots" / "scatter_test.png", dpi=200)
        plt.close()
    except Exception as e:
        print(f"[WARN] Plotting skipped: {e}")

    return metrics

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--test",  required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    return ap.parse_args()

def main():
    args = parse_args()
    fit_best(args.features, args.train, args.test, args.outdir)

if __name__ == "__main__":
    main()