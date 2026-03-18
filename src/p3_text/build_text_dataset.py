from pathlib import Path


def load_raw_text_sources():
    """Load raw text data from Fed, news, and earnings sources."""
    pass


def clean_and_standardize_text(df):
    """Clean text fields, standardize schema, and assign date columns."""
    pass


def embed_text(df):
    """Generate document-level FinBERT embeddings and sentiment features."""
    pass


def aggregate_to_daily(df):
    """Aggregate document-level features into one row per day."""
    pass


def save_outputs(df, output_path: Path):
    """Save processed daily text features."""
    pass


def main():
    output_path = Path("data/processed/p3_features.parquet")
    print("Building P3 text dataset...")
    # pipeline will go here
    print(f"Planned output: {output_path}")


if __name__ == "__main__":
    main()