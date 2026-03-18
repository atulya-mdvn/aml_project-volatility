from pathlib import Path
import pandas as pd

from src.shared.schemas import P4_FEATURE_COLUMNS


OUTPUT_PATH = Path("data/processed/p4_features.parquet")


def create_empty_p4_features() -> pd.DataFrame:
    """Create an empty P4 feature table with the official schema."""
    return pd.DataFrame(columns=P4_FEATURE_COLUMNS)


def validate_p4_schema(df: pd.DataFrame) -> None:
    """Raise an error if the DataFrame columns do not match the expected schema."""
    if list(df.columns) != P4_FEATURE_COLUMNS:
        raise ValueError("P4 schema does not match expected columns.")


def main() -> None:
    df = create_empty_p4_features()
    validate_p4_schema(df)

    print("P4 schema established.")
    print(f"Columns: {list(df.columns)}")
    print(f"Planned output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()