from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.shared.schemas import (
    P3_DOC_FEATURE_COLUMNS,
    P3_FEATURE_COLUMNS,
)


INPUT_PATH = Path("data/interim/p3_doc_features.parquet")
OUTPUT_PATH = Path("data/processed/p3_features.parquet")


def create_empty_p3_features() -> pd.DataFrame:
    """Create an empty P3 feature table with the official schema."""
    return pd.DataFrame(columns=P3_FEATURE_COLUMNS)


def validate_p3_schema(df: pd.DataFrame) -> None:
    """Raise an error if the DataFrame columns do not match the expected schema."""
    if list(df.columns) != P3_FEATURE_COLUMNS:
        raise ValueError("P3 schema does not match expected columns.")

    if "date" in df.columns and df["date"].duplicated().any():
        raise ValueError("Duplicate dates found in P3 feature table.")


def load_p3_doc_features(path: Path = INPUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_parquet(path)
    validate_p3_doc_features_schema(df)
    return df


def validate_p3_doc_features_schema(df: pd.DataFrame) -> None:
    if list(df.columns) != P3_DOC_FEATURE_COLUMNS:
        raise ValueError("Input schema does not match P3_DOC_FEATURE_COLUMNS.")

    if df["doc_id"].duplicated().any():
        raise ValueError("Duplicate doc_id values found in P3 doc features.")

    required_cols = [
        "doc_id",
        "date",
        "source_type",
        "p3_sent_pos",
        "p3_sent_neg",
        "p3_sent_neu",
        "p3_uncertainty_score",
    ]
    for col in required_cols:
        if df[col].isna().any():
            raise ValueError(f"Missing values found in required input column: {col}")


def get_embedding_columns() -> list[str]:
    return [f"p3_emb_{i}" for i in range(1, 769)]


def compute_daily_counts(group: pd.DataFrame) -> dict:
    source_types = group["source_type"].fillna("")

    fed_count = int(source_types.str.startswith("fed").sum())
    news_count = int(source_types.str.startswith("news").sum())
    earnings_count = int(source_types.str.startswith("earnings").sum())

    return {
        "p3_doc_count": int(len(group)),
        "p3_fed_count": fed_count,
        "p3_news_count": news_count,
        "p3_earnings_count": earnings_count,
    }


def safe_std(series: pd.Series) -> float:
    """
    Return population std for consistency.
    If only one observation exists, return 0.0 instead of NaN.
    """
    if len(series) <= 1:
        return 0.0
    value = float(series.std(ddof=0))
    return 0.0 if np.isnan(value) else value


def compute_daily_summary_stats(group: pd.DataFrame) -> dict:
    return {
        "p3_sent_pos_mean": float(group["p3_sent_pos"].mean()),
        "p3_sent_neg_mean": float(group["p3_sent_neg"].mean()),
        "p3_sent_neu_mean": float(group["p3_sent_neu"].mean()),
        "p3_sent_pos_std": safe_std(group["p3_sent_pos"]),
        "p3_sent_neg_std": safe_std(group["p3_sent_neg"]),
        "p3_sent_neu_std": safe_std(group["p3_sent_neu"]),
        "p3_uncertainty_mean": float(group["p3_uncertainty_score"].mean()),
        "p3_uncertainty_std": safe_std(group["p3_uncertainty_score"]),
    }


def compute_document_pca_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-document PCA features from the 768 embedding columns.

    If there are too few documents to support 20 principal components,
    compute as many as possible and pad the remaining columns with zeros.
    """
    emb_cols = get_embedding_columns()
    emb_matrix = df[emb_cols].to_numpy(dtype=float)

    n_docs, n_emb = emb_matrix.shape
    max_components = min(20, n_docs, n_emb)

    pc_df = pd.DataFrame(index=df.index)

    if max_components >= 1:
        pca = PCA(n_components=max_components)
        transformed = pca.fit_transform(emb_matrix)

        for i in range(max_components):
            pc_df[f"p3_text_pc{i + 1}"] = transformed[:, i]

    for i in range(max_components + 1, 21):
        pc_df[f"p3_text_pc{i}"] = 0.0

    return pc_df


def build_p3_features(doc_features_df: pd.DataFrame) -> pd.DataFrame:
    df = doc_features_df.copy()

    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

    pc_df = compute_document_pca_features(df)
    df = pd.concat([df.reset_index(drop=True), pc_df.reset_index(drop=True)], axis=1)

    rows = []
    pc_columns = [f"p3_text_pc{i}" for i in range(1, 21)]

    for date_value, group in df.groupby("date", sort=True):
        row = {"date": date_value}
        row.update(compute_daily_counts(group))
        row.update(compute_daily_summary_stats(group))

        for pc_col in pc_columns:
            row[pc_col] = float(group[pc_col].mean())

        rows.append(row)

    result = pd.DataFrame(rows, columns=P3_FEATURE_COLUMNS)
    result = result.sort_values("date").reset_index(drop=True)

    fill_zero_cols = [col for col in result.columns if col != "date"]
    result[fill_zero_cols] = result[fill_zero_cols].fillna(0.0)

    validate_p3_schema(result)
    return result


def main() -> None:
    doc_features_df = load_p3_doc_features(INPUT_PATH)
    p3_features_df = build_p3_features(doc_features_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    p3_features_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Loaded {len(doc_features_df)} document-level rows from {INPUT_PATH}")
    print(f"Saved {len(p3_features_df)} daily rows to {OUTPUT_PATH}")
    print("P3 processed feature table established.")
    print(f"Columns: {list(p3_features_df.columns)}")


if __name__ == "__main__":
    main()