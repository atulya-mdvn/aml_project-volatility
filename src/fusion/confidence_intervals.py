"""
confidence_intervals.py
=======================
Standalone script that plugs into your existing vol_project.

Reads the cached pipeline outputs from data/processed/, trains
quantile XGBoost models for the 10th, 50th, and 90th percentiles,
calibrates the intervals using conformal prediction, and exports
everything to a CSV.

Usage:
    cd ~/Desktop/vol_project
    conda activate pydata
    python confidence_intervals.py

Output:
    data/processed/predictions_with_ci.csv

Columns in the output:
    date              — trading date
    predicted_rv      — point estimate (median, 50th percentile)
    ci_lower          — 10th percentile (lower bound of 80% CI)
    ci_upper          — 90th percentile (upper bound of 80% CI)
    ci_width          — ci_upper - ci_lower (uncertainty measure)
    ci_width_pct      — ci_width / predicted_rv (relative uncertainty)
    spike_prob        — spike probability from classification head
    actual_rv         — actual forward 22d RV (NaN for dates without future data)
    coverage_flag     — 1 if actual_rv fell inside [ci_lower, ci_upper], else 0
    confidence_level  — categorical: HIGH / MEDIUM / LOW based on ci_width percentile
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from src.shared.config import PROC_DIR


# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
TRAIN_CUTOFF = "2020-01-01"      # everything before this = train
CALIB_CUTOFF = "2022-01-01"      # train < 2020, calibration 2020-2022, test > 2022
QUANTILES = [0.10, 0.50, 0.90]   # 80% prediction interval
TARGET_COVERAGE = 0.80            # conformal target


def load_pipeline_outputs():
    """
    Load all cached pipeline outputs and merge on date index.
    Same logic as your existing meta_model.py / test_eval.py.
    """
    print("Loading pipeline outputs...")

    features = pd.read_csv(PROC_DIR / "features.csv", index_col=0, parse_dates=True)

    # Try to load each pipeline — skip gracefully if missing
    pipeline_dfs = {}
    for name, fname in [("p1", "p1_latent.csv"), ("p2", "p2_transformer.csv"),
                        ("p3", "p3_finbert.csv"), ("p4", "p4_gat.csv")]:
        path = PROC_DIR / fname
        if os.path.exists(path):
            pipeline_dfs[name] = pd.read_csv(path, index_col=0, parse_dates=True)
            print(f"  {name}: {pipeline_dfs[name].shape}")
        else:
            print(f"  {name}: NOT FOUND — skipping")

    # Separate targets from features
    target_cols = ["fwd_rv_22d", "spike_label"]
    feature_cols = [c for c in features.columns if c not in target_cols]

    # Start with base features
    meta = features[feature_cols].copy()

    # Join each pipeline
    for name, df in pipeline_dfs.items():
        # Drop any columns that overlap with existing (avoid _x, _y suffixes)
        overlap = set(meta.columns) & set(df.columns)
        if overlap:
            df = df.drop(columns=list(overlap))
        meta = meta.join(df, how="inner")

    # Add targets back
    meta = meta.join(features[target_cols], how="inner").dropna()

    meta_feat_cols = [c for c in meta.columns if c not in target_cols]
    print(f"  Final merged shape: {meta.shape}")
    print(f"  Feature count: {len(meta_feat_cols)}")
    print(f"  Date range: {meta.index.min()} to {meta.index.max()}")

    return meta, meta_feat_cols, target_cols


def train_quantile_models(meta, feat_cols, target="fwd_rv_22d"):
    """
    Train three XGBoost models: one for each quantile (10th, 50th, 90th).

    Uses the same hyperparameters as your existing meta-model but with
    quantile loss instead of squared error.

    The split is:
      - Train: before 2020 (no COVID, no 2022 rate shock)
      - Calibration: 2020–2022 (used ONLY for conformal adjustment)
      - Test: 2022+ (true out-of-sample)
    """
    print("\n" + "=" * 60)
    print("TRAINING QUANTILE MODELS")
    print("=" * 60)

    train = meta[meta.index < TRAIN_CUTOFF]
    calib = meta[(meta.index >= TRAIN_CUTOFF) & (meta.index < CALIB_CUTOFF)]
    test = meta[meta.index >= CALIB_CUTOFF]

    print(f"  Train:       {len(train)} days ({train.index.min()} to {train.index.max()})")
    print(f"  Calibration: {len(calib)} days ({calib.index.min()} to {calib.index.max()})")
    print(f"  Test:        {len(test)}  days ({test.index.min()} to {test.index.max()})")

    X_train = train[feat_cols]
    y_train = train[target]
    X_calib = calib[feat_cols]
    y_calib = calib[target]
    X_test = test[feat_cols]
    y_test = test[target]

    models = {}
    predictions = {}

    for q in QUANTILES:
        label = f"q{int(q*100):02d}"
        print(f"\n  Training {label} (quantile={q})...")

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_lambda=3.0,
            objective="reg:quantileerror",
            quantile_alpha=q,
            verbosity=0,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Predict on ALL data (train + calib + test)
        pred_all = model.predict(meta[feat_cols])
        predictions[label] = pred_all
        models[label] = model

        # Report calibration-set coverage
        pred_calib = model.predict(X_calib)
        print(f"    Calib predictions — min: {pred_calib.min():.4f}, "
              f"median: {np.median(pred_calib):.4f}, max: {pred_calib.max():.4f}")

    # ── Also train a standard point-estimate model for spike probability ──
    print("\n  Training spike classifier...")
    spike_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_lambda=3.0,
        scale_pos_weight=len(y_train[train["spike_label"] == 0]) /
                         max(1, len(y_train[train["spike_label"] == 1])),
        verbosity=0,
        random_state=42,
    )
    spike_model.fit(X_train, train["spike_label"])
    spike_prob_all = spike_model.predict_proba(meta[feat_cols])[:, 1]

    return models, predictions, spike_prob_all, {
        "train": train, "calib": calib, "test": test,
        "X_calib": X_calib, "y_calib": y_calib,
        "X_test": X_test, "y_test": y_test,
    }


def conformal_calibration(models, splits):
    """
    Conformal prediction calibration.

    The idea: raw quantile predictions may not have honest coverage.
    The 10th percentile prediction might actually contain the true value
    only 15% of the time below it, not 10%.

    Conformal prediction fixes this by computing the empirical residuals
    on the calibration set and adjusting the intervals to guarantee
    the target coverage.

    This is the part that makes the confidence intervals HONEST —
    when we say "80% CI", it actually contains the true value ~80% of the time.
    """
    print("\n" + "=" * 60)
    print("CONFORMAL CALIBRATION")
    print("=" * 60)

    X_calib = splits["X_calib"]
    y_calib = splits["y_calib"]

    # Get raw quantile predictions on calibration set
    q10_calib = models["q10"].predict(X_calib)
    q90_calib = models["q90"].predict(X_calib)

    # Nonconformity scores: how far outside the interval is the true value?
    # If true value is inside [q10, q90], score = 0
    # If below q10, score = q10 - true
    # If above q90, score = true - q90
    scores = np.maximum(q10_calib - y_calib.values, y_calib.values - q90_calib)
    scores = np.sort(scores)

    # Raw coverage on calibration set
    raw_coverage = np.mean((y_calib.values >= q10_calib) & (y_calib.values <= q90_calib))
    print(f"  Raw interval coverage on calibration set: {raw_coverage:.1%}")

    # Find the conformal quantile that achieves target coverage
    # This is the (1 - alpha)(1 + 1/n) quantile of nonconformity scores
    n_calib = len(scores)
    conformal_level = TARGET_COVERAGE
    q_index = int(np.ceil((conformal_level) * (n_calib + 1))) - 1
    q_index = min(q_index, n_calib - 1)
    adjustment = scores[q_index]

    print(f"  Conformal adjustment: ±{adjustment:.4f}")
    print(f"  (This widens the interval to guarantee {TARGET_COVERAGE:.0%} coverage)")

    # Verify on calibration set
    adjusted_lower = q10_calib - adjustment
    adjusted_upper = q90_calib + adjustment
    adjusted_coverage = np.mean((y_calib.values >= adjusted_lower) &
                                (y_calib.values <= adjusted_upper))
    print(f"  Adjusted coverage on calibration set: {adjusted_coverage:.1%}")

    return adjustment


def build_output_csv(meta, feat_cols, predictions, spike_prob_all,
                     conformal_adj, splits):
    """
    Assemble the final CSV with predictions, CIs, and diagnostics.
    """
    print("\n" + "=" * 60)
    print("BUILDING OUTPUT CSV")
    print("=" * 60)

    df = pd.DataFrame(index=meta.index)
    df.index.name = "date"

    # Point estimate = median (q50)
    df["predicted_rv"] = predictions["q50"]

    # Confidence interval = q10/q90 adjusted by conformal correction
    df["ci_lower"] = predictions["q10"] - conformal_adj
    df["ci_upper"] = predictions["q90"] + conformal_adj

    # Floor ci_lower at 0 (volatility can't be negative)
    df["ci_lower"] = df["ci_lower"].clip(lower=0.0)

    # Uncertainty measures
    df["ci_width"] = df["ci_upper"] - df["ci_lower"]
    df["ci_width_pct"] = (df["ci_width"] / df["predicted_rv"].clip(lower=0.01)) * 100

    # Spike probability
    df["spike_prob"] = spike_prob_all

    # Actual RV (for evaluation — will be NaN for most recent dates)
    df["actual_rv"] = meta["fwd_rv_22d"]

    # Coverage flag: did the actual fall inside the CI?
    df["coverage_flag"] = ((df["actual_rv"] >= df["ci_lower"]) &
                           (df["actual_rv"] <= df["ci_upper"])).astype(int)
    # NaN where actual_rv is NaN
    df.loc[df["actual_rv"].isna(), "coverage_flag"] = np.nan

    # Confidence level based on relative CI width
    # Use expanding percentile (same logic as V6 — no hindsight)
    width_pcts = df["ci_width_pct"].expanding().rank(pct=True)
    df["confidence_level"] = "MEDIUM"
    df.loc[width_pcts <= 0.33, "confidence_level"] = "HIGH"
    df.loc[width_pcts >= 0.67, "confidence_level"] = "LOW"

    # ── Save ──
    out_path = PROC_DIR / "predictions_with_ci.csv"
    df.to_csv(out_path, float_format="%.6f")
    print(f"  Saved to: {out_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    return df


def print_evaluation(df, splits):
    """
    Print comprehensive evaluation metrics.
    """
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    for period_name, mask in [("Calibration (2020-2022)",
                                (df.index >= TRAIN_CUTOFF) & (df.index < CALIB_CUTOFF)),
                               ("Test (2022+)",
                                df.index >= CALIB_CUTOFF),
                               ("Full out-of-sample (2020+)",
                                df.index >= TRAIN_CUTOFF)]:
        subset = df[mask].dropna(subset=["actual_rv"])
        if len(subset) == 0:
            continue

        coverage = subset["coverage_flag"].mean()
        avg_width = subset["ci_width"].mean()
        avg_width_pct = subset["ci_width_pct"].mean()
        mse = mean_squared_error(subset["actual_rv"], subset["predicted_rv"])
        mae = mean_absolute_error(subset["actual_rv"], subset["predicted_rv"])

        # Interval score (proper scoring rule for quantile forecasts)
        # Penalizes both miscoverage AND unnecessarily wide intervals
        alpha = 1.0 - TARGET_COVERAGE
        lower = subset["ci_lower"].values
        upper = subset["ci_upper"].values
        actual = subset["actual_rv"].values
        interval_score = (
            (upper - lower)
            + (2.0 / alpha) * (lower - actual) * (actual < lower)
            + (2.0 / alpha) * (actual - upper) * (actual > upper)
        ).mean()

        print(f"\n  {period_name} ({len(subset)} days):")
        print(f"    Coverage:        {coverage:.1%}  (target: {TARGET_COVERAGE:.0%})")
        print(f"    Avg CI width:    {avg_width:.4f}  ({avg_width_pct:.1f}% of prediction)")
        print(f"    MSE:             {mse:.6f}")
        print(f"    MAE:             {mae:.4f}")
        print(f"    Interval score:  {interval_score:.4f}  (lower is better)")

        # Coverage by confidence level
        for level in ["HIGH", "MEDIUM", "LOW"]:
            level_subset = subset[subset["confidence_level"] == level]
            if len(level_subset) > 0:
                level_cov = level_subset["coverage_flag"].mean()
                level_width = level_subset["ci_width_pct"].mean()
                print(f"    {level:6s} confidence: {len(level_subset):4d} days, "
                      f"coverage={level_cov:.1%}, avg width={level_width:.1f}%")

    # Spike detection AUC
    oos = df[df.index >= TRAIN_CUTOFF].dropna(subset=["actual_rv"])
    if len(oos) > 0:
        # Derive spike labels from actual_rv
        # Use the expanding 90th percentile (same as your pipeline)
        threshold = oos["actual_rv"].expanding().quantile(0.90)
        spike_actual = (oos["actual_rv"] > threshold).astype(int)
        if spike_actual.sum() > 0 and spike_actual.sum() < len(spike_actual):
            auc = roc_auc_score(spike_actual, oos["spike_prob"])
            print(f"\n  Spike detection AUC (out-of-sample): {auc:.4f}")


def print_sample_rows(df):
    """Show some example rows so you can see what the output looks like."""
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT")
    print("=" * 60)

    # Show a calm period, a stress period, and a transition
    for label, date in [("Calm (2017-06-15)", "2017-06-15"),
                        ("Pre-COVID (2020-02-14)", "2020-02-14"),
                        ("COVID peak (2020-03-16)", "2020-03-16"),
                        ("Rate shock (2022-06-13)", "2022-06-13"),
                        ("Calm again (2024-01-15)", "2024-01-15")]:
        # Find nearest date
        idx = df.index.get_indexer([pd.Timestamp(date)], method="nearest")[0]
        row = df.iloc[idx]
        actual_str = f"{row['actual_rv']:.4f}" if pd.notna(row['actual_rv']) else "N/A"
        cov_str = ("YES" if row['coverage_flag'] == 1 else "NO") if pd.notna(row['coverage_flag']) else "N/A"

        print(f"\n  {label} ({df.index[idx].strftime('%Y-%m-%d')}):")
        print(f"    Predicted RV:   {row['predicted_rv']:.4f}")
        print(f"    80% CI:         [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
        print(f"    CI width:       {row['ci_width']:.4f} ({row['ci_width_pct']:.1f}%)")
        print(f"    Spike prob:     {row['spike_prob']:.3f}")
        print(f"    Confidence:     {row['confidence_level']}")
        print(f"    Actual RV:      {actual_str}")
        print(f"    Inside CI?      {cov_str}")


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Volatility Confidence Intervals — Quantile + Conformal    ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Step 1: Load data
    meta, feat_cols, target_cols = load_pipeline_outputs()

    # Step 2: Train quantile models
    models, predictions, spike_prob_all, splits = train_quantile_models(
        meta, feat_cols
    )

    # Step 3: Conformal calibration
    conformal_adj = conformal_calibration(models, splits)

    # Step 4: Build output CSV
    df = build_output_csv(meta, feat_cols, predictions, spike_prob_all,
                          conformal_adj, splits)

    # Step 5: Evaluate
    print_evaluation(df, splits)

    # Step 6: Show examples
    print_sample_rows(df)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Output: {PROC_DIR}/predictions_with_ci.csv")
    print(f"Use this with V7 of the trading algorithm for")
    print(f"uncertainty-aware position sizing.")


if __name__ == "__main__":
    main()
