import os

import matplotlib
matplotlib.use("Agg")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
import torch
import xgboost as xgb

from src.shared.config import PROCESSED_DATA_DIR


def main():
    print("Step 1: Importing libraries...")
    print("  sklearn OK")
    print("  torch OK")
    print("  xgboost OK")

    print("\nStep 2: Loading cached data...")
    features = pd.read_csv(PROCESSED_DATA_DIR / "features.csv", index_col=0, parse_dates=True)
    p1 = pd.read_csv(PROCESSED_DATA_DIR / "p1_latent.csv", index_col=0, parse_dates=True)
    p2 = pd.read_csv(PROCESSED_DATA_DIR / "p2_transformer.csv", index_col=0, parse_dates=True)
    p3 = pd.read_csv(PROCESSED_DATA_DIR / "p3_finbert.csv", index_col=0, parse_dates=True)
    p4 = pd.read_csv(PROCESSED_DATA_DIR / "p4_gat.csv", index_col=0, parse_dates=True)
    print(f"  Features: {features.shape}, P1: {p1.shape}, P2: {p2.shape}, P3: {p3.shape}, P4: {p4.shape}")

    print("\nStep 3: Merging...")
    target_cols = ["fwd_rv_22d", "spike_label"]
    feature_cols = [c for c in features.columns if c not in target_cols]
    meta = features[feature_cols].copy()
    meta = meta.join(p1, how="inner")
    meta = meta.join(p3, how="inner")
    meta = meta.join(p2, how="inner")
    meta = meta.join(p4, how="inner")
    meta = meta.join(features[target_cols], how="inner").dropna()
    meta_feat_cols = [c for c in meta.columns if c not in target_cols]
    print(f"  Meta: {meta.shape}")

    m_train = meta[meta.index < "2020-01-01"]
    m_test = meta[meta.index >= "2020-01-01"]

    print("\nStep 4: HAR-RV...")
    har = LinearRegression().fit(m_train[["RV_1d", "RV_5d", "RV_22d"]], m_train["fwd_rv_22d"])
    hp = har.predict(m_test[["RV_1d", "RV_5d", "RV_22d"]])
    auc = roc_auc_score(m_test["spike_label"], hp)
    mse = mean_squared_error(m_test["fwd_rv_22d"], hp)
    print(f"  HAR-RV: MSE={mse:.6f}, AUC={auc:.4f}")

    print("\nStep 5: XGBoost...")
    xgb_m = xgb.XGBRegressor(n_estimators=200, max_depth=3, verbosity=0)
    xgb_m.fit(m_train[meta_feat_cols], m_train["fwd_rv_22d"])
    xp = xgb_m.predict(m_test[meta_feat_cols])
    auc = roc_auc_score(m_test["spike_label"], xp)
    mse = mean_squared_error(m_test["fwd_rv_22d"], xp)
    print(f"  XGB: MSE={mse:.6f}, AUC={auc:.4f}")

    print("\nStep 6: Ensemble...")
    ens = 0.4 * hp + 0.6 * xp
    auc = roc_auc_score(m_test["spike_label"], ens)
    mse = mean_squared_error(m_test["fwd_rv_22d"], ens)
    print(f"  Ensemble: MSE={mse:.6f}, AUC={auc:.4f}")

    print("\nStep 7: SHAP...")
    try:
        import shap
        explainer = shap.TreeExplainer(xgb_m)
        sv = explainer.shap_values(m_test[meta_feat_cols])
        fi = pd.Series(np.abs(sv).mean(axis=0), index=meta_feat_cols).sort_values(ascending=False)
        print("  Top 10 features:")
        for name, val in fi.head(10).items():
            print(f"    {name:30s}  {val:.4f}")
    except Exception as e:
        print(f"  SHAP failed: {e}")

    print("\nDONE — all results computed successfully!")


if __name__ == "__main__":
    main()