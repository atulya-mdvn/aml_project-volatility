from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.shared.schemas import (
    P4_NODE_FEATURE_COLUMNS,
    P4_EDGE_FEATURE_COLUMNS,
    P4_FEATURE_COLUMNS,
)


NODE_INPUT_PATH = Path("data/interim/p4_node_features.parquet")
EDGE_INPUT_PATH = Path("data/interim/p4_edge_features.parquet")
OUTPUT_PATH = Path("data/processed/p4_features.parquet")


def create_empty_p4_features() -> pd.DataFrame:
    """Create an empty P4 feature table with the official schema."""
    return pd.DataFrame(columns=P4_FEATURE_COLUMNS)


def validate_p4_schema(df: pd.DataFrame) -> None:
    """Raise an error if the DataFrame columns do not match the expected schema."""
    if list(df.columns) != P4_FEATURE_COLUMNS:
        raise ValueError("P4 schema does not match expected columns.")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found in P4 feature table.")


def validate_p4_node_schema(df: pd.DataFrame) -> None:
    if list(df.columns) != P4_NODE_FEATURE_COLUMNS:
        raise ValueError("Node input schema does not match P4_NODE_FEATURE_COLUMNS.")

    required_cols = ["date", "asset", "asset_class", "close", "volume"]
    for col in required_cols:
        if df[col].isna().any():
            raise ValueError(f"Missing values found in required node column: {col}")


def validate_p4_edge_schema(df: pd.DataFrame) -> None:
    if list(df.columns) != P4_EDGE_FEATURE_COLUMNS:
        raise ValueError("Edge input schema does not match P4_EDGE_FEATURE_COLUMNS.")

    required_cols = ["date", "asset_i", "asset_j", "p4_corr_60d", "p4_abs_corr_60d"]
    for col in required_cols:
        if df[col].isna().any():
            raise ValueError(f"Missing values found in required edge column: {col}")


def load_node_features(path: Path = NODE_INPUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Node input file not found: {path}")

    df = pd.read_parquet(path)
    validate_p4_node_schema(df)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    return df


def load_edge_features(path: Path = EDGE_INPUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Edge input file not found: {path}")

    df = pd.read_parquet(path)
    validate_p4_edge_schema(df)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    return df


def safe_mean(series: pd.Series) -> float:
    if series.dropna().empty:
        return 0.0
    return float(series.mean())


def safe_std(series: pd.Series) -> float:
    vals = series.dropna()
    if len(vals) <= 1:
        return 0.0
    value = float(vals.std(ddof=0))
    return 0.0 if np.isnan(value) else value


def safe_max(series: pd.Series) -> float:
    vals = series.dropna()
    if vals.empty:
        return 0.0
    return float(vals.max())


def safe_min(series: pd.Series) -> float:
    vals = series.dropna()
    if vals.empty:
        return 0.0
    return float(vals.min())


def compute_daily_node_summary(node_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a daily summary matrix from node-level features.
    This will later be reduced into p4_graph_pc* columns using PCA.
    """
    grouped = node_df.groupby("date", sort=True)

    rows = []
    for date_value, group in grouped:
        row = {
            "date": date_value,
            "node_close_mean": safe_mean(group["close"]),
            "node_close_std": safe_std(group["close"]),
            "node_volume_mean": safe_mean(group["volume"]),
            "node_volume_std": safe_std(group["volume"]),
            "node_ret_1d_mean": safe_mean(group["p4_ret_1d"]),
            "node_ret_1d_std": safe_std(group["p4_ret_1d"]),
            "node_ret_5d_mean": safe_mean(group["p4_ret_5d"]),
            "node_ret_5d_std": safe_std(group["p4_ret_5d"]),
            "node_vol_20d_mean": safe_mean(group["p4_vol_20d"]),
            "node_vol_20d_std": safe_std(group["p4_vol_20d"]),
            "node_momentum_20d_mean": safe_mean(group["p4_momentum_20d"]),
            "node_momentum_20d_std": safe_std(group["p4_momentum_20d"]),
            "node_volume_z_mean": safe_mean(group["p4_volume_z"]),
            "node_volume_z_std": safe_std(group["p4_volume_z"]),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def compute_daily_graph_summary(
    node_df: pd.DataFrame, edge_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build graph-level daily summary fields from node and edge tables.
    """
    node_counts = (
        node_df.groupby("date")["asset"]
        .nunique()
        .reset_index(name="p4_graph_num_nodes")
    )

    edge_grouped = edge_df.groupby("date", sort=True)

    edge_rows = []
    for date_value, group in edge_grouped:
        num_edges = int(len(group))
        num_nodes = int(
            node_counts.loc[node_counts["date"] == date_value, "p4_graph_num_nodes"].iloc[0]
        )

        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0.0

        edge_rows.append(
            {
                "date": date_value,
                "p4_graph_num_edges": num_edges,
                "p4_graph_density": float(density),
                "p4_graph_avg_abs_corr": safe_mean(group["p4_abs_corr_60d"]),
                "p4_graph_max_abs_corr": safe_max(group["p4_abs_corr_60d"]),
                "p4_graph_min_corr": safe_min(group["p4_corr_60d"]),
                "p4_graph_corr_dispersion": safe_std(group["p4_corr_60d"]),
            }
        )

    edge_summary = pd.DataFrame(edge_rows)

    result = node_counts.merge(edge_summary, on="date", how="inner")
    result = result.sort_values("date").reset_index(drop=True)
    return result


def compute_graph_pca_features(node_summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run PCA across dates using daily node-summary vectors.

    If too few dates exist to support 20 components, compute as many as
    possible and pad the rest with zeros.
    """
    feature_cols = [col for col in node_summary_df.columns if col != "date"]
    X = node_summary_df[feature_cols].to_numpy(dtype=float)

    n_rows, n_cols = X.shape
    max_components = min(20, n_rows, n_cols)

    pc_df = pd.DataFrame({"date": node_summary_df["date"].values})

    if max_components >= 1:
        pca = PCA(n_components=max_components)
        transformed = pca.fit_transform(X)

        for i in range(max_components):
            pc_df[f"p4_graph_pc{i + 1}"] = transformed[:, i]

    for i in range(max_components + 1, 21):
        pc_df[f"p4_graph_pc{i}"] = 0.0

    return pc_df


def build_p4_features(node_df: pd.DataFrame, edge_df: pd.DataFrame) -> pd.DataFrame:
    graph_summary_df = compute_daily_graph_summary(node_df, edge_df)
    node_summary_df = compute_daily_node_summary(node_df)
    graph_pc_df = compute_graph_pca_features(node_summary_df)

    result = graph_summary_df.merge(graph_pc_df, on="date", how="inner")
    result = result.sort_values("date").reset_index(drop=True)

    fill_zero_cols = [col for col in result.columns if col != "date"]
    result[fill_zero_cols] = result[fill_zero_cols].fillna(0.0)

    result = result[P4_FEATURE_COLUMNS].copy()
    validate_p4_schema(result)
    return result


def main() -> None:
    node_df = load_node_features(NODE_INPUT_PATH)
    edge_df = load_edge_features(EDGE_INPUT_PATH)

    p4_features_df = build_p4_features(node_df, edge_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    p4_features_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Loaded {len(node_df)} node rows from {NODE_INPUT_PATH}")
    print(f"Loaded {len(edge_df)} edge rows from {EDGE_INPUT_PATH}")
    print(f"Saved {len(p4_features_df)} daily rows to {OUTPUT_PATH}")
    print("P4 processed feature table established.")
    print(f"Columns: {list(p4_features_df.columns)}")


if __name__ == "__main__":
    main()