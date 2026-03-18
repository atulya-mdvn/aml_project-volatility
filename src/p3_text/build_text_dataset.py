from pathlib import Path
import pandas as pd

from src.shared.schemas import P3_FEATURE_COLUMNS


OUTPUT_PATH = Path("data/processed/p3_features.parquet")


def create_empty_p3_features() -> pd.DataFrame:
    """Create an empty P3 feature table with the official schema."""
    return pd.DataFrame(columns=P3_FEATURE_COLUMNS)


def validate_p3_schema(df: pd.DataFrame) -> None:
    """Raise an error if the DataFrame columns do not match the expected schema."""
    if list(df.columns) != P3_FEATURE_COLUMNS:
        raise ValueError("P3 schema does not match expected columns.")


def main() -> None:
    df = create_empty_p3_features()
    validate_p3_schema(df)

    print("P3 schema established.")
    print(f"Columns: {list(df.columns)}")
    print(f"Planned output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()