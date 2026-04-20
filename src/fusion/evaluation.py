"""Evaluation: Diebold-Mariano, calibration, conformal prediction, SHAP, walk-forward."""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, mean_squared_error, roc_auc_score

from src.shared.config import (
    DEVICE,
    ENSEMBLE_HAR_WEIGHT_CALM,
    ENSEMBLE_HAR_WEIGHT_CRISIS,
    ENSEMBLE_HAR_WEIGHT_NORMAL,
    ENSEMBLE_HAR_WEIGHT_STRESS,
    PROC_DIR,
    XGB_PARAMS,
)


def diebold_mariano(actual, p1, p2, h=22):
    e1 = (actual - p1)**2; e2 = (actual - p2)**2; d = e1 - e2
    n = len(d); d_mean = d.mean()
    gamma = [np.cov(d[:-k] if k > 0 else d, d[k:] if k > 0 else d)[0,1] for k in range(h)]
    var_d = (gamma[0] + 2*sum(gamma[1:])) / n
    if var_d <= 0: return 0, 1.0
    dm = d_mean / np.sqrt(var_d)
    return dm, 2*stats.t.sf(abs(dm), df=n-1)


def get_regime_weight(vix_level):
    if vix_level > 35: return ENSEMBLE_HAR_WEIGHT_CRISIS
    elif vix_level > 25: return ENSEMBLE_HAR_WEIGHT_STRESS
    elif vix_level > 18: return ENSEMBLE_HAR_WEIGHT_NORMAL
    else: return ENSEMBLE_HAR_WEIGHT_CALM


def run_evaluation(results):
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    m_test = results["m_test"]
    m_train = results["m_train"]
    meta_feat_cols = results["meta_feat_cols"]
    preds = results["predictions"]
    actual = m_test["fwd_rv_22d"].values
    spike = m_test["spike_label"].values

    # Diebold-Mariano
    print("\nDIEBOLD-MARIANO TESTS")
    tests = [
        ("HAR-RV", preds["har"], "Cross-Attn", preds["ca_reg"]),
        ("HAR-RV", preds["har"], "Ensemble", preds["ensemble"]),
        ("XGB", preds["xgb"], "Cross-Attn", preds["ca_reg"]),
        ("Cross-Attn", preds["ca_reg"], "Ensemble", preds["ensemble"]),
    ]
    for n1, p1, n2, p2 in tests:
        dm, pval = diebold_mariano(actual, p1, p2)
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        print(f"  {n1} vs {n2}: DM={dm:.3f}, p={pval:.4f} {sig}")

    # Platt Calibration
    print("\nPLATT CALIBRATION")
    cal_mask = m_train.index >= "2017-01-01"
    cal_data = m_train[cal_mask]
    if cal_data["spike_label"].nunique() < 2:
        cal_data = m_train
    print(f"  Calibration set: {len(cal_data)} days, spikes: {cal_data['spike_label'].sum()}")

    scaler = results["scaler"]
    groups = results["groups"]

    cal_feat = scaler.transform(cal_data[meta_feat_cols])
    cal_pipes = [p.to(DEVICE) for p in split_pipelines(cal_feat, meta_feat_cols, groups)]

    ca = results["ca_model"]
    ca.eval()
    import torch
    with torch.no_grad():
        _, cal_cls = ca(cal_pipes)
    cal_prob = torch.sigmoid(cal_cls.squeeze()).cpu().numpy()

    platt = LogisticRegression(C=1.0)
    platt.fit(cal_prob.reshape(-1, 1), cal_data["spike_label"].values)
    calibrated = platt.predict_proba(preds["ca_cls"].reshape(-1, 1))[:, 1]

    print(f"  Before: Brier={brier_score_loss(spike, preds['ca_cls']):.4f}")
    print(f"  After:  Brier={brier_score_loss(spike, calibrated):.4f}")
    print(f"  Calibrated AUC: {roc_auc_score(spike, calibrated):.4f}")

    # Conformal Prediction
    print("\nCONFORMAL PREDICTION")
    with torch.no_grad():
        cal_reg, _ = ca(cal_pipes)
    cal_pred = cal_reg.squeeze().cpu().numpy()
    cal_actual = cal_data["fwd_rv_22d"].values
    scores = np.abs(cal_actual - cal_pred)
    alpha = 0.10
    n_cal = len(scores)
    q = np.quantile(scores, np.ceil((1-alpha)*(n_cal+1))/n_cal)

    lower = preds["ca_reg"] - q
    upper = preds["ca_reg"] + q
    coverage = np.mean((actual >= lower) & (actual <= upper))
    print(f"  90% target → actual coverage: {100*coverage:.1f}%")
    print(f"  Interval width: ±{q:.4f}")

    # Plots
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax = axes[0, 0]
        for name, pred in [("HAR-RV", preds["har"]), ("XGB", preds["xgb"]),
                            ("Cross-Attn", preds["ca_reg"]), ("Ensemble", preds["ensemble"])]:
            bins = np.quantile(pred, np.linspace(0, 1, 11))
            bm, am = [], []
            for j in range(len(bins)-1):
                mask = (pred >= bins[j]) & (pred < bins[j+1])
                if mask.sum() > 0: bm.append(pred[mask].mean()); am.append(actual[mask].mean())
            ax.plot(bm, am, 'o-', label=name, markersize=4)
        ax.plot([0, 0.6], [0, 0.6], 'k--', alpha=0.3)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Regression Calibration"); ax.legend()

        ax = axes[0, 1]
        for name, probs in [("Raw", preds["ca_cls"]), ("Platt", calibrated)]:
            fp, mp = calibration_curve(spike, probs, n_bins=8, strategy='quantile')
            ax.plot(mp, fp, 'o-', label=name, markersize=5)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_title("Spike Calibration"); ax.legend()

        ax = axes[1, 0]
        ax.fill_between(m_test.index, lower, upper, alpha=0.2, color="blue")
        ax.plot(m_test.index, actual, linewidth=0.6, color="black", alpha=0.6, label="Actual")
        ax.plot(m_test.index, preds["ca_reg"], linewidth=0.6, color="red", alpha=0.7, label="Pred")
        ax.set_title(f"Conformal Intervals ({100*coverage:.0f}% coverage)"); ax.legend()

        ax = axes[1, 1]
        ax.plot(m_test.index, actual, linewidth=0.6, color="black", alpha=0.6, label="Actual")
        ax.plot(m_test.index, preds["ensemble"], linewidth=0.6, color="red", alpha=0.7, label="Ensemble")
        ax.set_title("Ensemble Predictions vs Actual"); ax.legend()

        plt.suptitle("V5 Evaluation (Regime-Conditional Ensemble)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(PROC_DIR, "evaluation_plots.png"), dpi=150)
        plt.close()
        print(f"  Plots saved to {os.path.join(PROC_DIR, 'evaluation_plots.png')}")
    except Exception as e:
        print(f"  Plotting failed (non-critical): {e}")

    # SHAP
    print("\nSHAP ANALYSIS")
    try:
        xgb_m = results["xgb_model"]
        explainer = shap.TreeExplainer(xgb_m)
        sv = explainer.shap_values(m_test[meta_feat_cols])

        plt.figure(figsize=(10, 14))
        shap.summary_plot(sv, m_test[meta_feat_cols], plot_type="bar", show=False, max_display=30)
        plt.title("SHAP — V5")
        plt.tight_layout()
        plt.savefig(os.path.join(PROC_DIR, "shap_overall.png"), dpi=150)
        plt.close()
        print(f"  SHAP plot saved to {os.path.join(PROC_DIR, 'shap_overall.png')}")

        fi = pd.Series(np.abs(sv).mean(axis=0), index=meta_feat_cols).sort_values(ascending=False)
        p5 = ["RV_1d", "RV_5d", "RV_22d"]
        p1 = [c for c in meta_feat_cols if c.startswith("latent") or c in ["recon_error", "vae_anomaly", "crisis_distance"]]
        p2t = [c for c in meta_feat_cols if c.startswith("trans_")]
        p3 = [c for c in meta_feat_cols if c.startswith("finbert") or c in ["hawkish_dovish", "risk_appetite", "vol_expectation", "systemic_risk"]
              or c.endswith("_5d") or c.endswith("_22d")]
        p3 = [c for c in p3 if c in fi.index and c not in p5 + p1 + p2t]
        p4 = [c for c in meta_feat_cols if c.startswith("gat") or c == "mean_abs_corr"]
        p2m = [c for c in meta_feat_cols if c not in p5+p1+p2t+p3+p4]

        total = fi.sum()
        print("\nPIPELINE CONTRIBUTIONS")
        for name, cols in [("P5 HAR-RV", p5), ("P2 market", p2m), ("P2 Transformer", p2t),
                            ("P1 VAE", p1), ("P3 Text/Claude", p3), ("P4 GAT", p4)]:
            val = fi[[c for c in cols if c in fi.index]].sum()
            print(f"  {name:20s}  SHAP={val:.4f}  ({100*val/total:.1f}%)")
    except Exception as e:
        print(f"  SHAP failed (non-critical): {e}")

    return {"calibrated_prob": calibrated, "conformal_q": q}


def split_pipelines(X, feat_cols, groups):
    """Import helper for evaluation."""
    tensors = []
    import torch
    for cols in groups:
        idx = [feat_cols.index(c) for c in cols if c in feat_cols]
        if len(idx) > 0:
            tensors.append(torch.FloatTensor(X[:, idx]))
        else:
            tensors.append(torch.zeros(X.shape[0], 1))
    return tensors


def run_walk_forward(meta, meta_feat_cols):
    print("\n" + "=" * 60)
    print("WALK-FORWARD CV (Regime-Conditional Ensemble)")
    print("=" * 60)

    test_years = [y for y in range(2016, 2027) if meta[meta.index.year == y].shape[0] > 50]
    wf = []

    for year in test_years:
        tr = meta[meta.index.year < year]; te = meta[meta.index.year == year]
        if len(tr) < 500 or len(te) < 30: continue

        h = LinearRegression().fit(tr[["RV_1d","RV_5d","RV_22d"]], tr["fwd_rv_22d"])
        hp = h.predict(te[["RV_1d","RV_5d","RV_22d"]])

        xg = xgb.XGBRegressor(**XGB_PARAMS)
        xg.fit(tr[meta_feat_cols], tr["fwd_rv_22d"])
        xp = xg.predict(te[meta_feat_cols])

        # regime-conditional ensemble
        vix_vals = te["vix_level"].values if "vix_level" in te.columns else np.full(len(te), 18)
        har_w = np.array([get_regime_weight(v) for v in vix_vals])
        ens = har_w * hp + (1 - har_w) * xp

        h_auc = roc_auc_score(te["spike_label"], hp) if te["spike_label"].nunique() > 1 else np.nan
        x_auc = roc_auc_score(te["spike_label"], xp) if te["spike_label"].nunique() > 1 else np.nan
        e_auc = roc_auc_score(te["spike_label"], ens) if te["spike_label"].nunique() > 1 else np.nan

        wf.append({"Year": year, "HAR_MSE": mean_squared_error(te["fwd_rv_22d"], hp), "HAR_AUC": h_auc,
                     "Full_MSE": mean_squared_error(te["fwd_rv_22d"], xp), "Full_AUC": x_auc,
                     "Ens_MSE": mean_squared_error(te["fwd_rv_22d"], ens), "Ens_AUC": e_auc})

        if not np.isnan(h_auc):
            avg_w = np.mean(har_w)
            print(f"  {year}: HAR={h_auc:.3f} | Full={x_auc:.3f} | Ens={e_auc:.3f} (avg HAR weight: {avg_w:.0%})")
        else:
            print(f"  {year}: no spikes")

    wf_df = pd.DataFrame(wf).set_index("Year")
    valid = wf_df.dropna(subset=["HAR_AUC"])
    print(f"\nAverages:")
    print(f"  HAR-RV:   MSE={valid['HAR_MSE'].mean():.6f}  AUC={valid['HAR_AUC'].mean():.3f}")
    print(f"  Full:     MSE={valid['Full_MSE'].mean():.6f}  AUC={valid['Full_AUC'].mean():.3f}")
    print(f"  Ensemble: MSE={valid['Ens_MSE'].mean():.6f}  AUC={valid['Ens_AUC'].mean():.3f}")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for col, label in [("HAR_MSE", "HAR-RV"), ("Full_MSE", "Full"), ("Ens_MSE", "Ensemble")]:
            axes[0].plot(wf_df.index, wf_df[col], 'o-', label=label, markersize=4)
        axes[0].set_title("MSE by Year"); axes[0].legend()
        for col, label in [("HAR_AUC", "HAR-RV"), ("Full_AUC", "Full"), ("Ens_AUC", "Ensemble")]:
            d = wf_df[col].dropna()
            axes[1].plot(d.index, d.values, 'o-', label=label, markersize=4)
        axes[1].set_title("AUC by Year"); axes[1].legend()
        plt.suptitle("Walk-Forward CV — V5 (Regime-Conditional)"); plt.tight_layout()
        plt.savefig(os.path.join(PROC_DIR, "walk_forward.png"), dpi=150)
        plt.close()
        print(f"  Walk-forward plot saved to {os.path.join(PROC_DIR, 'walk_forward.png')}")
    except Exception as e:
        print(f"  Walk-forward plotting failed: {e}")

    return wf_df
