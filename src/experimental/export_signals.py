"""
Export daily model signals as CSV for QuantConnect backtesting.
Loads cached pipeline outputs, runs HAR-RV + XGBoost + regime-conditional ensemble,
and produces a CSV with daily predictions that QuantConnect can consume.

Usage:
    cd ~/Desktop/vol_project
    python export_signals.py

Output:
    data/processed/daily_signals.csv
"""
import numpy as np, pandas as pd, os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# ── Load config values ──
SPLIT_DATE = "2020-01-01"
SEED = 42
np.random.seed(SEED)

XGB_PARAMS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.5,
    reg_alpha=0.3, reg_lambda=3.0,
    random_state=SEED, verbosity=0,
)

# Regime-conditional weights
def get_regime(vix):
    if vix > 35: return "CRISIS"
    elif vix > 25: return "STRESS"
    elif vix > 18: return "NORMAL"
    else: return "CALM"

def get_har_weight(vix):
    if vix > 35: return 0.70
    elif vix > 25: return 0.60
    elif vix > 18: return 0.40
    else: return 0.30


def main():
    proc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")

    print("=" * 60)
    print("EXPORTING MODEL SIGNALS FOR QUANTCONNECT")
    print("=" * 60)

    # ── Load cached pipeline outputs ──
    print("\nLoading cached data...")
    features = pd.read_csv(os.path.join(proc_dir, "features.csv"), index_col=0, parse_dates=True)
    p1 = pd.read_csv(os.path.join(proc_dir, "p1_latent.csv"), index_col=0, parse_dates=True)
    p2 = pd.read_csv(os.path.join(proc_dir, "p2_transformer.csv"), index_col=0, parse_dates=True)
    p3 = pd.read_csv(os.path.join(proc_dir, "p3_finbert.csv"), index_col=0, parse_dates=True)
    p4 = pd.read_csv(os.path.join(proc_dir, "p4_gat.csv"), index_col=0, parse_dates=True)
    print(f"  Features: {features.shape}")
    print(f"  P1 VAE: {p1.shape}")
    print(f"  P2 Transformer: {p2.shape}")
    print(f"  P3 Claude Text: {p3.shape}")
    print(f"  P4 GAT: {p4.shape}")

    # ── Merge into meta dataset ──
    target_cols = ["fwd_rv_22d", "spike_label"]
    feature_cols = [c for c in features.columns if c not in target_cols]

    meta = features[feature_cols].copy()
    meta = meta.join(p1, how="inner")
    meta = meta.join(p3, how="inner")
    meta = meta.join(p2, how="inner")
    meta = meta.join(p4, how="inner")
    meta = meta.join(features[target_cols], how="inner").dropna()
    meta_feat_cols = [c for c in meta.columns if c not in target_cols]
    print(f"  Meta dataset: {meta.shape[0]} rows, {len(meta_feat_cols)} features")

    # ── Walk-forward signal generation ──
    # For each day, use only data available up to that point
    # Start generating signals from 2016 onward (need enough training data)
    print("\nGenerating walk-forward signals...")

    all_signals = []
    years = sorted(meta.index.year.unique())
    signal_start_year = 2016

    for year in years:
        if year < signal_start_year:
            continue

        # train on all data before this year
        train = meta[meta.index.year < year]
        test = meta[meta.index.year == year]

        if len(train) < 500 or len(test) < 10:
            continue

        # HAR-RV
        har = LinearRegression()
        har.fit(train[["RV_1d", "RV_5d", "RV_22d"]], train["fwd_rv_22d"])
        har_pred = har.predict(test[["RV_1d", "RV_5d", "RV_22d"]])

        # XGBoost on all features
        xgb_m = xgb.XGBRegressor(**XGB_PARAMS)
        xgb_m.fit(train[meta_feat_cols], train["fwd_rv_22d"])
        xgb_pred = xgb_m.predict(test[meta_feat_cols])

        # Spike probability (using XGBoost classification)
        xgb_cls = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.5,
            reg_alpha=0.3, reg_lambda=3.0,
            random_state=SEED, verbosity=0, eval_metric='logloss',
        )
        xgb_cls.fit(train[meta_feat_cols], train["spike_label"])
        spike_prob_raw = xgb_cls.predict_proba(test[meta_feat_cols])[:, 1]

        # Platt calibration for spike probability
        # Use last 2 years of training data for calibration
        cal_start = str(year - 2)
        cal_data = train[train.index >= cal_start]
        if cal_data["spike_label"].nunique() >= 2 and len(cal_data) > 100:
            cal_prob = xgb_cls.predict_proba(cal_data[meta_feat_cols])[:, 1]
            platt = LogisticRegression(C=1.0)
            platt.fit(cal_prob.reshape(-1, 1), cal_data["spike_label"].values)
            spike_prob_calibrated = platt.predict_proba(spike_prob_raw.reshape(-1, 1))[:, 1]
        else:
            spike_prob_calibrated = spike_prob_raw

        # Regime-conditional ensemble for each day
        vix_vals = test["vix_level"].values if "vix_level" in test.columns else np.full(len(test), 18)

        for j in range(len(test)):
            d = test.index[j]
            vix = vix_vals[j]
            regime = get_regime(vix)
            har_w = get_har_weight(vix)

            ensemble_pred = har_w * har_pred[j] + (1 - har_w) * xgb_pred[j]
            spike_p = spike_prob_calibrated[j]

            # Generate trade signal
            # We don't have real IV here, but we can use VIX as a proxy for 30-day IV
            iv_proxy = vix / 100  # VIX is in percentage points, convert to decimal
            rv_pred = ensemble_pred
            gap = rv_pred - iv_proxy

            if spike_p > 0.30:
                signal = "BUY_PUTS"
                signal_strength = min(spike_p, 1.0)
            elif gap > 0.04:
                signal = "BUY_VOL"
                signal_strength = min(gap / 0.08, 1.0)
            elif gap > 0.02:
                signal = "LEAN_LONG_VOL"
                signal_strength = min(gap / 0.06, 1.0)
            elif gap < -0.04:
                signal = "SELL_VOL"
                signal_strength = min(abs(gap) / 0.08, 1.0)
            elif gap < -0.02:
                signal = "LEAN_SHORT_VOL"
                signal_strength = min(abs(gap) / 0.06, 1.0)
            else:
                signal = "NEUTRAL"
                signal_strength = 0.0

            # Get key text features if available
            hawkish = test.iloc[j].get("hawkish_dovish", 0) if "hawkish_dovish" in test.columns else 0
            uncertainty = test.iloc[j].get("finbert_uncertainty", 0) if "finbert_uncertainty" in test.columns else 0
            systemic = test.iloc[j].get("systemic_risk", 0) if "systemic_risk" in test.columns else 0

            all_signals.append({
                "Date": d.strftime("%Y-%m-%d"),
                "predicted_rv": round(float(rv_pred), 6),
                "har_pred": round(float(har_pred[j]), 6),
                "xgb_pred": round(float(xgb_pred[j]), 6),
                "spike_prob": round(float(spike_p), 4),
                "vix_level": round(float(vix), 2),
                "iv_proxy": round(float(iv_proxy), 6),
                "rv_iv_gap": round(float(gap), 6),
                "regime": regime,
                "har_weight": round(float(har_w), 2),
                "signal": signal,
                "signal_strength": round(float(signal_strength), 4),
                "hawkish_dovish": round(float(hawkish), 4),
                "uncertainty": round(float(uncertainty), 4),
                "systemic_risk": round(float(systemic), 4),
                "actual_rv": round(float(test.iloc[j]["fwd_rv_22d"]), 6),
                "actual_spike": int(test.iloc[j]["spike_label"]),
            })

        # year summary
        year_signals = [s for s in all_signals if s["Date"].startswith(str(year))]
        buy_count = sum(1 for s in year_signals if "BUY" in s["signal"])
        sell_count = sum(1 for s in year_signals if "SELL" in s["signal"])
        neutral_count = sum(1 for s in year_signals if s["signal"] == "NEUTRAL")
        print(f"  {year}: {len(year_signals)} days | BUY_VOL/PUTS: {buy_count} | SELL_VOL: {sell_count} | NEUTRAL: {neutral_count}")

    # ── Save CSV ──
    signals_df = pd.DataFrame(all_signals)
    output_path = os.path.join(proc_dir, "daily_signals.csv")
    signals_df.to_csv(output_path, index=False)
    print(f"\n  Saved {len(signals_df)} daily signals to {output_path}")

    # ── Summary statistics ──
    print(f"\n{'=' * 60}")
    print("SIGNAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total trading days: {len(signals_df)}")
    print(f"  Date range: {signals_df['Date'].iloc[0]} to {signals_df['Date'].iloc[-1]}")
    print(f"\n  Signal distribution:")
    for sig in ["BUY_PUTS", "BUY_VOL", "LEAN_LONG_VOL", "NEUTRAL", "LEAN_SHORT_VOL", "SELL_VOL"]:
        count = (signals_df["signal"] == sig).sum()
        pct = 100 * count / len(signals_df)
        print(f"    {sig:20s}  {count:4d} days  ({pct:.1f}%)")

    print(f"\n  Regime distribution:")
    for reg in ["CALM", "NORMAL", "STRESS", "CRISIS"]:
        count = (signals_df["regime"] == reg).sum()
        pct = 100 * count / len(signals_df)
        print(f"    {reg:10s}  {count:4d} days  ({pct:.1f}%)")

    # ── Signal accuracy analysis ──
    print(f"\n  Signal accuracy (did RV exceed IV proxy?):")
    for sig in ["BUY_VOL", "BUY_PUTS", "LEAN_LONG_VOL", "SELL_VOL", "LEAN_SHORT_VOL"]:
        subset = signals_df[signals_df["signal"] == sig]
        if len(subset) > 10:
            if "BUY" in sig or "LONG" in sig:
                correct = (subset["actual_rv"] > subset["iv_proxy"]).mean()
                print(f"    {sig:20s}  {len(subset):3d} days  RV > IV: {100*correct:.1f}%")
            else:
                correct = (subset["actual_rv"] < subset["iv_proxy"]).mean()
                print(f"    {sig:20s}  {len(subset):3d} days  RV < IV: {100*correct:.1f}%")

    print(f"\n  Average RV-IV gap by signal:")
    for sig in ["BUY_PUTS", "BUY_VOL", "LEAN_LONG_VOL", "NEUTRAL", "LEAN_SHORT_VOL", "SELL_VOL"]:
        subset = signals_df[signals_df["signal"] == sig]
        if len(subset) > 0:
            avg_gap = subset["rv_iv_gap"].mean()
            avg_actual = subset["actual_rv"].mean()
            print(f"    {sig:20s}  avg gap: {avg_gap:+.4f}  avg actual RV: {avg_actual:.4f}")

    # ── Preview ──
    print(f"\n  First 5 rows:")
    print(signals_df.head().to_string(index=False))
    print(f"\n  Last 5 rows:")
    print(signals_df.tail().to_string(index=False))

    return signals_df


if __name__ == "__main__":
    main()
