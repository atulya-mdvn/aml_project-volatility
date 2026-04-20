import sys
import gc
import warnings
import json
import os

import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

from src.shared.config import MODEL_DIR, PROC_DIR, RAW_DIR


def load_spx_csv(path):
    """Load SPX CSV handling both yfinance formats."""
    import pandas as pd
    try:
        df = pd.read_csv(path, header=[0, 1, 2], index_col=0, parse_dates=True).droplevel([1, 2], axis=1)
        return df
    except Exception:
        return pd.read_csv(path, index_col=0, parse_dates=True)


def main():
    skip_download = "--skip-download" in sys.argv
    import pandas as pd

    # ── Step 1: Download or load data ──
    if not skip_download:
        from src.shared.data_download import download_all

        data = download_all()

        # Fix yfinance multi-header if present
        if isinstance(data["spx"].columns, pd.MultiIndex):
            data["spx"].columns = data["spx"].columns.get_level_values(0)
    else:
        print("Loading cached data...")
        data = {
            "spx": load_spx_csv(RAW_DIR / "spx_daily.csv"),
            "vix": pd.read_csv(RAW_DIR / "vix_term_structure.csv", index_col=0, parse_dates=True),
            "cross_assets": pd.read_csv(RAW_DIR / "cross_assets.csv", index_col=0, parse_dates=True),
            "spx_intraday": None,
            "options": None,
        }

        intra_path = RAW_DIR / "spx_intraday.csv"
        if intra_path.exists():
            data["spx_intraday"] = pd.read_csv(intra_path, index_col=0, parse_dates=True)

        opts_path = RAW_DIR / "spx_options.csv"
        if opts_path.exists():
            data["options"] = pd.read_csv(opts_path, index_col=0)

        news_path = RAW_DIR / "news_headlines.json"
        if news_path.exists():
            with open(news_path) as f:
                data["news"] = json.load(f)
        else:
            data["news"] = {}

        print("Cached data loaded.")

    # ── Step 2: Build features ──
    from src.shared.features import build_features

    feat, feature_cols, target_cols, threshold = build_features(
        data["spx"],
        data["vix"],
        data.get("spx_intraday"),
        data.get("options"),
    )

    # ── Step 3: P5 HAR-RV baseline ──
    from src.p5_har_rv.p5_har_rv import train_har_rv

    har_model, har_pred = train_har_rv(feat)
    gc.collect()

    # ── Step 4: P1 VAE ──
    from src.p1_regime.p1_vae import train_vae

    latent_df, vae_model = train_vae(feat, feature_cols)
    del vae_model
    gc.collect()
    print("  [memory freed: P1 model released]")

    # ── Step 5: P2 Transformer ──
    from src.p2_transformer.p2_transformer import train_transformer

    trans_df, trans_model = train_transformer(feat, feature_cols)
    del trans_model
    gc.collect()
    print("  [memory freed: P2 model released]")

    # ── Step 6: P3 Claude text analysis ──
    from src.p3_text.p3_finbert import train_finbert

    text_df = train_finbert(feat, data.get("news", {}))
    gc.collect()
    print("  [memory freed: P3 released]")

    # ── Step 7: P4 GAT ──
    from src.p4_graph.p4_gat import train_gat

    gat_df, gat_model = train_gat(feat, data["cross_assets"])
    del gat_model
    del data
    gc.collect()
    print("  [memory freed: P4 model + raw data released]")

    # ── Step 8: Meta-model ──
    from src.fusion.meta_model import train_meta

    results = train_meta(feat, feature_cols, latent_df, trans_df, text_df, gat_df, har_pred)

    # ── Step 9: Evaluation ──
    from src.fusion.evaluation import run_evaluation, run_walk_forward

    eval_results = run_evaluation(results)
    wf_df = run_walk_forward(results["meta"], results["meta_feat_cols"])

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {PROC_DIR}")
    print(f"Models saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()