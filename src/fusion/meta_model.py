"""Meta-model: cross-attention fusion + regime-conditional ensemble."""
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
import xgboost as xgb
from config import *


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        return (bce * (1 - pt) ** self.gamma).mean()


class CrossAttentionFusion(nn.Module):
    def __init__(self, pipe_dims, hidden=META_HIDDEN, heads=META_HEADS, dropout=META_DROPOUT):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU())
            for d in pipe_dims
        ])
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.reg_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.cls_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, pipe_tensors):
        projected = [proj(p) for proj, p in zip(self.projections, pipe_tensors)]
        seq = torch.stack(projected, dim=1)
        attn_out, _ = self.attn(seq, seq, seq)
        fused = self.norm(attn_out.mean(dim=1))
        return self.reg_head(fused), self.cls_head(fused)


def split_pipelines(X, feat_cols, groups):
    tensors = []
    for cols in groups:
        idx = [feat_cols.index(c) for c in cols if c in feat_cols]
        if len(idx) > 0:
            tensors.append(torch.FloatTensor(X[:, idx]))
        else:
            tensors.append(torch.zeros(X.shape[0], 1))
    return tensors


def get_regime_weight(vix_level):
    """Return HAR-RV weight based on VIX regime."""
    if vix_level > 35:
        return ENSEMBLE_HAR_WEIGHT_CRISIS
    elif vix_level > 25:
        return ENSEMBLE_HAR_WEIGHT_STRESS
    elif vix_level > 18:
        return ENSEMBLE_HAR_WEIGHT_NORMAL
    else:
        return ENSEMBLE_HAR_WEIGHT_CALM


def train_meta(feat, feature_cols, latent_df, trans_df, text_df, gat_df, har_pred):
    print("\n" + "=" * 60)
    print("META-MODEL: CROSS-ATTENTION FUSION")
    print("=" * 60)

    target_cols = ["fwd_rv_22d", "spike_label"]
    base_cols = [c for c in feature_cols if c not in target_cols]

    meta = feat[base_cols].copy()
    meta = meta.join(latent_df, how="inner")
    meta = meta.join(text_df, how="inner")
    meta = meta.join(trans_df, how="inner")
    meta = meta.join(gat_df, how="inner")
    meta = meta.join(feat[target_cols], how="inner").dropna()

    meta_feat_cols = [c for c in meta.columns if c not in target_cols]
    print(f"  Meta dataset: {meta.shape[0]} rows, {len(meta_feat_cols)} features")

    # define pipeline groups
    p5_cols = ["RV_1d", "RV_5d", "RV_22d"]
    p1_cols = [c for c in meta_feat_cols if c.startswith("latent") or c in ["recon_error", "vae_anomaly", "crisis_distance", "latent_velocity", "latent_acceleration"]]
    p2t_cols = [c for c in meta_feat_cols if c.startswith("trans_")]
    p3_cols = [c for c in meta_feat_cols if c.startswith("finbert") or c in ["hawkish_dovish", "risk_appetite", "vol_expectation", "systemic_risk"] or c.endswith("_5d") or c.endswith("_22d")]
    p3_cols = [c for c in p3_cols if c in meta_feat_cols]
    p4_cols = [c for c in meta_feat_cols if c.startswith("gat") or c == "mean_abs_corr"]
    p2m_cols = [c for c in meta_feat_cols if c not in p5_cols + p1_cols + p2t_cols + p3_cols + p4_cols]

    groups = [p2m_cols + p5_cols, p1_cols, p2t_cols, p3_cols, p4_cols]
    groups = [g for g in groups if len(g) > 0]
    dims = [len(g) for g in groups]
    print(f"  Pipeline groups: {len(groups)}, dims: {dims}")

    # split
    m_train = meta[meta.index < SPLIT_DATE]
    m_test = meta[meta.index >= SPLIT_DATE]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(m_train[meta_feat_cols])
    X_test = scaler.transform(m_test[meta_feat_cols])

    # cross-attention model
    train_pipes = [p.to(DEVICE) for p in split_pipelines(X_train, meta_feat_cols, groups)]
    test_pipes = [p.to(DEVICE) for p in split_pipelines(X_test, meta_feat_cols, groups)]

    y_reg = torch.FloatTensor(m_train["fwd_rv_22d"].values).to(DEVICE)
    y_cls = torch.FloatTensor(m_train["spike_label"].values).to(DEVICE)

    ca = CrossAttentionFusion(dims).to(DEVICE)
    opt = torch.optim.AdamW(ca.parameters(), lr=META_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, META_EPOCHS)
    focal = FocalLoss(META_FOCAL_GAMMA)

    ca.train()
    for epoch in range(1, META_EPOCHS + 1):
        opt.zero_grad()
        reg_out, cls_out = ca(train_pipes)
        loss_r = nn.functional.mse_loss(reg_out.squeeze(), y_reg)
        loss_c = focal(cls_out.squeeze(), y_cls)
        loss = loss_r + 0.5 * loss_c
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ca.parameters(), 1.0)
        opt.step()
        scheduler.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}/{META_EPOCHS}  Loss: {loss.item():.6f}")

    torch.save(ca.state_dict(), os.path.join(MODEL_DIR, "cross_attention.pt"))

    # free training data
    import gc
    del train_pipes, y_reg, y_cls
    gc.collect()

    # predictions
    ca.eval()
    with torch.no_grad():
        test_reg, test_cls = ca(test_pipes)
    ca_reg = test_reg.squeeze().cpu().numpy()
    ca_cls = torch.sigmoid(test_cls.squeeze()).cpu().numpy()

    # HAR-RV
    har = LinearRegression().fit(m_train[["RV_1d", "RV_5d", "RV_22d"]], m_train["fwd_rv_22d"])
    hp = har.predict(m_test[["RV_1d", "RV_5d", "RV_22d"]])

    # XGBoost
    xgb_m = xgb.XGBRegressor(**XGB_PARAMS)
    xgb_m.fit(m_train[meta_feat_cols], m_train["fwd_rv_22d"])
    xp = xgb_m.predict(m_test[meta_feat_cols])

    # Regime-conditional ensemble
    vix_test = m_test["vix_level"].values if "vix_level" in m_test.columns else np.full(len(m_test), 18)
    har_weights = np.array([get_regime_weight(v) for v in vix_test])
    ensemble = har_weights * hp + (1 - har_weights) * xp

    actual = m_test["fwd_rv_22d"].values
    spike = m_test["spike_label"].values

    print(f"\n{'=' * 60}")
    print("RESULTS (test: 2020+)")
    print(f"{'=' * 60}")
    print(f"  HAR-RV                     MSE={mean_squared_error(actual, hp):.6f}  AUC={roc_auc_score(spike, hp):.4f}")
    print(f"  XGB full                   MSE={mean_squared_error(actual, xp):.6f}  AUC={roc_auc_score(spike, xp):.4f}")
    print(f"  Cross-Attn (reg)           MSE={mean_squared_error(actual, ca_reg):.6f}  AUC={roc_auc_score(spike, ca_reg):.4f}")
    print(f"  Ensemble (regime-cond)     MSE={mean_squared_error(actual, ensemble):.6f}  AUC={roc_auc_score(spike, ensemble):.4f}")
    print(f"  Cross-Attn (cls)           AUC={roc_auc_score(spike, ca_cls):.4f}")

    # show regime weights used
    print(f"\n  Regime weights used:")
    print(f"    Crisis (VIX>35): HAR={ENSEMBLE_HAR_WEIGHT_CRISIS:.0%}")
    print(f"    Stress (VIX>25): HAR={ENSEMBLE_HAR_WEIGHT_STRESS:.0%}")
    print(f"    Normal (VIX>18): HAR={ENSEMBLE_HAR_WEIGHT_NORMAL:.0%}")
    print(f"    Calm   (VIX≤18): HAR={ENSEMBLE_HAR_WEIGHT_CALM:.0%}")

    return {
        "m_train": m_train, "m_test": m_test,
        "meta_feat_cols": meta_feat_cols, "meta": meta,
        "predictions": {"har": hp, "xgb": xp, "ca_reg": ca_reg, "ca_cls": ca_cls, "ensemble": ensemble},
        "xgb_model": xgb_m, "ca_model": ca, "scaler": scaler, "groups": groups,
    }
