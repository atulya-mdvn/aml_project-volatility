from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


P4_FEATURES_PATH = Path("data/processed/p4_features.parquet")
TARGETS_PATH = Path("data/processed/p5_features.parquet")

PREDICTIONS_OUTPUT_PATH = Path("outputs/p4/predictions.parquet")
METRICS_OUTPUT_PATH = Path("outputs/p4/metrics.json")


# Change this once your P5 target column name is finalized.
DEFAULT_REGRESSION_TARGET = "target_rv_22d"


def load_p4_features(path: Path = P4_FEATURES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"P4 feature file not found: {path}")

    df = pd.read_parquet(path)
    if "date" not in df.columns:
        raise ValueError("P4 feature table must contain a 'date' column.")

    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    return df


def load_targets(path: Path = TARGETS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Target file not found: {path}. "
            "You need data/processed/p5_features.parquet before training P4 predictions."
        )

    df = pd.read_parquet(path)
    if "date" not in df.columns:
        raise ValueError("Target table must contain a 'date' column.")

    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    return df


def get_p4_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col != "date"]


def prepare_modeling_table(
    p4_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_col: str = DEFAULT_REGRESSION_TARGET,
) -> tuple[pd.DataFrame, list[str]]:
    if target_col not in target_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in target table. "
            f"Available columns: {list(target_df.columns)}"
        )

    merged = p4_df.merge(
        target_df[["date", target_col]],
        on="date",
        how="inner",
    )

    merged = merged.dropna(subset=[target_col]).sort_values("date").reset_index(drop=True)

    feature_cols = get_p4_feature_columns(p4_df)
    if not feature_cols:
        raise ValueError("No P4 feature columns found for modeling.")

    if merged.empty:
        raise ValueError("Merged modeling table is empty after joining P4 features to targets.")

    return merged, feature_cols


def time_split_train_test(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    n = len(df)
    split_idx = max(1, int(n * (1 - test_fraction)))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Train/test split produced an empty partition. "
            "You likely need more rows."
        )

    return train_df, test_df


def train_regression_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[RandomForestRegressor, pd.DataFrame, dict]:
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    predictions_df = test_df[["date"]].copy()
    predictions_df["actual"] = y_test.values
    predictions_df["prediction"] = y_pred
    predictions_df["residual"] = predictions_df["actual"] - predictions_df["prediction"]
    predictions_df["model"] = "RandomForestRegressor"
    predictions_df["pillar"] = "p4"
    predictions_df["target_col"] = target_col

    metrics = {
        "pillar": "p4",
        "model": "RandomForestRegressor",
        "target_col": target_col,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }

    return model, predictions_df, metrics


def save_outputs(predictions_df: pd.DataFrame, metrics: dict) -> None:
    PREDICTIONS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_parquet(PREDICTIONS_OUTPUT_PATH, index=False)

    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved predictions to {PREDICTIONS_OUTPUT_PATH}")
    print(f"Saved metrics to {METRICS_OUTPUT_PATH}")


def main() -> None:
    p4_df = load_p4_features(P4_FEATURES_PATH)
    target_df = load_targets(TARGETS_PATH)

    target_col = DEFAULT_REGRESSION_TARGET
    modeling_df, feature_cols = prepare_modeling_table(
        p4_df,
        target_df,
        target_col=target_col,
    )

    train_df, test_df = time_split_train_test(modeling_df, test_fraction=0.2)

    _, predictions_df, metrics = train_regression_model(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    save_outputs(predictions_df, metrics)

    print("Standalone P4 prediction run complete.")
    print(f"Used {len(feature_cols)} P4 features.")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()