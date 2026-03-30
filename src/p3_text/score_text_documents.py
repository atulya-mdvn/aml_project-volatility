from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd

from src.shared.schemas import (
    P3_DOCUMENT_COLUMNS,
    P3_DOC_FEATURE_COLUMNS,
)


INPUT_PATH = Path("data/interim/p3_documents.parquet")
OUTPUT_PATH = Path("data/interim/p3_doc_features.parquet")


POSITIVE_WORDS = {
    "strong",
    "solid",
    "improve",
    "improved",
    "improvement",
    "growth",
    "growing",
    "expanded",
    "expanding",
    "resilient",
    "favorable",
    "progress",
    "healthy",
    "gain",
    "gains",
    "stability",
    "stable",
}

NEGATIVE_WORDS = {
    "weak",
    "weakened",
    "decline",
    "declining",
    "slowdown",
    "deterioration",
    "deteriorate",
    "risk",
    "risks",
    "stress",
    "stressed",
    "inflationary",
    "volatile",
    "volatility",
    "uncertain",
    "uncertainty",
    "loss",
    "losses",
}

UNCERTAINTY_WORDS = {
    "uncertain",
    "uncertainty",
    "may",
    "might",
    "could",
    "monitor",
    "monitored",
    "monitoring",
    "appears",
    "appear",
    "possible",
    "possibly",
    "suggest",
    "suggests",
    "anticipated",
    "anticipate",
    "expect",
    "expected",
    "expects",
    "depending",
    "outlook",
    "roughly",
    "approximately",
    "potential",
    "potentially",
    "risk",
    "risks",
}


def load_documents(path: Path = INPUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_parquet(path)
    validate_documents_schema(df)
    return df


def validate_documents_schema(df: pd.DataFrame) -> None:
    if list(df.columns) != P3_DOCUMENT_COLUMNS:
        raise ValueError("Input document schema does not match P3_DOCUMENT_COLUMNS.")

    if df["doc_id"].duplicated().any():
        raise ValueError("Duplicate doc_id values found in input documents.")

    required_cols = ["doc_id", "date", "source_type", "text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")


def tokenize_text(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def compute_sentiment_scores(text: str) -> tuple[float, float, float]:
    """
    Very simple placeholder sentiment scoring.

    Returns:
        (positive_score, negative_score, neutral_score)

    Scores sum to 1.0.
    """
    tokens = tokenize_text(text)

    if not tokens:
        return 0.0, 0.0, 1.0

    pos_count = sum(token in POSITIVE_WORDS for token in tokens)
    neg_count = sum(token in NEGATIVE_WORDS for token in tokens)
    total = len(tokens)

    pos_score = pos_count / total
    neg_score = neg_count / total

    neutral_score = max(0.0, 1.0 - pos_score - neg_score)

    score_sum = pos_score + neg_score + neutral_score
    if score_sum == 0:
        return 0.0, 0.0, 1.0

    pos_score /= score_sum
    neg_score /= score_sum
    neutral_score /= score_sum

    return pos_score, neg_score, neutral_score


def compute_uncertainty_score(text: str) -> float:
    """
    Lexicon-based uncertainty score:
    fraction of tokens that match uncertainty-related words.
    """
    tokens = tokenize_text(text)

    if not tokens:
        return 0.0

    uncertainty_count = sum(token in UNCERTAINTY_WORDS for token in tokens)
    return uncertainty_count / len(tokens)


def make_placeholder_embedding(text: str, dim: int = 768) -> list[float]:
    """
    Deterministic placeholder embedding.

    Uses a hash of the text to seed a random number generator so the same
    text always produces the same embedding vector. This is only a scaffold
    until a real FinBERT embedding step is added.
    """
    if not isinstance(text, str):
        text = ""

    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)

    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(dim)

    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    return emb.astype(float).tolist()


def score_document(row: pd.Series) -> dict:
    text = row["text"] if pd.notna(row["text"]) else ""

    sent_pos, sent_neg, sent_neu = compute_sentiment_scores(text)
    uncertainty_score = compute_uncertainty_score(text)
    embedding = make_placeholder_embedding(text, dim=768)

    output = {
        "doc_id": row["doc_id"],
        "date": row["date"],
        "source_type": row["source_type"],
        "p3_sent_pos": sent_pos,
        "p3_sent_neg": sent_neg,
        "p3_sent_neu": sent_neu,
        "p3_uncertainty_score": uncertainty_score,
    }

    for i, value in enumerate(embedding, start=1):
        output[f"p3_emb_{i}"] = value

    return output


def build_doc_features(documents_df: pd.DataFrame) -> pd.DataFrame:
    rows = [score_document(row) for _, row in documents_df.iterrows()]
    df = pd.DataFrame(rows, columns=P3_DOC_FEATURE_COLUMNS)
    validate_doc_features_schema(df)
    return df


def validate_doc_features_schema(df: pd.DataFrame) -> None:
    if list(df.columns) != P3_DOC_FEATURE_COLUMNS:
        raise ValueError("Output schema does not match P3_DOC_FEATURE_COLUMNS.")

    if df["doc_id"].duplicated().any():
        raise ValueError("Duplicate doc_id values found in output doc features.")

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
            raise ValueError(f"Missing values found in required output column: {col}")


def main() -> None:
    documents_df = load_documents(INPUT_PATH)
    doc_features_df = build_doc_features(documents_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc_features_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Loaded {len(documents_df)} documents from {INPUT_PATH}")
    print(f"Saved {len(doc_features_df)} rows to {OUTPUT_PATH}")
    print("P3 document feature schema established.")
    print(f"Columns: {list(doc_features_df.columns[:10])} ... {list(doc_features_df.columns[-5:])}")


if __name__ == "__main__":
    main()