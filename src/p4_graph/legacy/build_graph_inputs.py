from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import yfinance as yf

from src.shared.schemas import (
    P4_NODE_FEATURE_COLUMNS,
    P4_EDGE_FEATURE_COLUMNS,
)


NODE_OUTPUT_PATH = Path("data/interim/p4_node_features.parquet")
EDGE_OUTPUT_PATH = Path("data/interim/p4_edge_features.parquet")


def get_asset_map() -> dict[str, str]:
    """
    Starter cross-asset universe using liquid ETF proxies from Yahoo Finance.
    Keys are tickers and values are asset classes.
    """
    return {
        "SPY": "equity",
        "QQQ": "equity",
        "IWM": "equity",
        "EFA": "equity",
        "EEM": "equity",
        "SHY": "treasury",
        "IEF": "treasury",
        "TLT": "treasury",
        "LQD": "credit",
        "HYG": "credit",
        "UUP": "fx",
        "FXE": "fx",
        "FXY": "fx",
        "GLD": "commodity",
        "SLV": "commodity",
        "USO": "commodity",
        "DBC": "commodity",
    }


def download_market_data(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download daily market data for all tickers from Yahoo Finance.

    Returns a DataFrame with a DatetimeIndex and MultiIndex columns such as:
    ('Close', 'SPY'), ('Volume', 'SPY'), ...
    """
    print(f"Downloading Yahoo Finance data for {len(tickers)} tickers...")
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if df.empty:
        raise ValueError("Downloaded market data is empty.")

    required_fields = {"Close", "Volume"}
    available_fields = set(df.columns.get_level_values(0))
    missing = required_fields - available_fields
    if missing:
        raise ValueError(f"Missing required Yahoo fields: {missing}")

    return df


def reshape_market_data(raw_df: pd.DataFrame, asset_map: dict[str, str]) -> pd.DataFrame:
    """
    Convert Yahoo wide data into a long table with:
    date, asset, asset_class, close, volume
    """
    close_df = raw_df["Close"].copy()
    volume_df = raw_df["Volume"].copy()

    close_long = close_df.stack().reset_index()
    close_long.columns = ["date", "asset", "close"]

    volume_long = volume_df.stack().reset_index()
    volume_long.columns = ["date", "asset", "volume"]

    df = close_long.merge(volume_long, on=["date", "asset"], how="inner")
    df["asset_class"] = df["asset"].map(asset_map)

    df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    return df


def safe_volume_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std(ddof=0)

    z = (series - rolling_mean) / rolling_std
    z = z.replace([np.inf, -np.inf], np.nan)
    return z


def compute_node_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute node-level daily features for each asset.
    """
    df = df.copy()
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)

    grouped = df.groupby("asset", group_keys=False)

    df["p4_ret_1d"] = grouped["close"].pct_change(1)
    df["p4_ret_5d"] = grouped["close"].pct_change(5)
    df["p4_vol_20d"] = grouped["p4_ret_1d"].transform(
        lambda s: s.rolling(window=20, min_periods=20).std(ddof=0)
    )
    df["p4_momentum_20d"] = grouped["close"].pct_change(20)
    df["p4_volume_z"] = grouped["volume"].transform(
        lambda s: safe_volume_zscore(s, window=20)
    )

    node_df = df[
        [
            "date",
            "asset",
            "asset_class",
            "close",
            "volume",
            "p4_ret_1d",
            "p4_ret_5d",
            "p4_vol_20d",
            "p4_momentum_20d",
            "p4_volume_z",
        ]
    ].copy()

    node_df["date"] = pd.to_datetime(node_df["date"]).dt.date.astype(str)

    return node_df


def compute_edge_features(node_df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Compute rolling pairwise correlations using 1-day returns.

    Stores each undirected edge once as (asset_i, asset_j) with asset_i < asset_j.
    """
    returns_wide = node_df.pivot(index="date", columns="asset", values="p4_ret_1d")
    returns_wide = returns_wide.sort_index()

    edge_rows = []
    assets = sorted(returns_wide.columns.tolist())

    print(f"Computing rolling {window}-day correlations for {len(assets)} assets...")

    for asset_i, asset_j in combinations(assets, 2):
        pair_df = returns_wide[[asset_i, asset_j]].copy()

        rolling_corr = pair_df[asset_i].rolling(window=window, min_periods=window).corr(
            pair_df[asset_j]
        )

        pair_out = pd.DataFrame(
            {
                "date": rolling_corr.index,
                "asset_i": asset_i,
                "asset_j": asset_j,
                "p4_corr_60d": rolling_corr.values,
            }
        )

        pair_out["p4_abs_corr_60d"] = pair_out["p4_corr_60d"].abs()
        edge_rows.append(pair_out)

    if not edge_rows:
        return pd.DataFrame(columns=P4_EDGE_FEATURE_COLUMNS)

    edge_df = pd.concat(edge_rows, ignore_index=True)
    edge_df["date"] = pd.to_datetime(edge_df["date"]).dt.date.astype(str)

    edge_df = edge_df.dropna(subset=["p4_corr_60d", "p4_abs_corr_60d"]).reset_index(drop=True)

    edge_df = edge_df[
        [
            "date",
            "asset_i",
            "asset_j",
            "p4_corr_60d",
            "p4_abs_corr_60d",
        ]
    ].copy()

    return edge_df


def validate_node_schema(df: pd.DataFrame) -> None:
    if list(df.columns) != P4_NODE_FEATURE_COLUMNS:
        raise ValueError("Node feature schema does not match P4_NODE_FEATURE_COLUMNS.")

    if df.empty:
        raise ValueError("Node feature DataFrame is empty.")

    required_cols = ["date", "asset", "asset_class", "close", "volume"]
    for col in required_cols:
        if df[col].isna().any():
            raise ValueError(f"Missing values found in required node column: {col}")


def validate_edge_schema(df: pd.DataFrame) -> None:
    if list(df.columns) != P4_EDGE_FEATURE_COLUMNS:
        raise ValueError("Edge feature schema does not match P4_EDGE_FEATURE_COLUMNS.")

    if df.empty:
        raise ValueError("Edge feature DataFrame is empty.")

    required_cols = ["date", "asset_i", "asset_j", "p4_corr_60d", "p4_abs_corr_60d"]
    for col in required_cols:
        if df[col].isna().any():
            raise ValueError(f"Missing values found in required edge column: {col}")


def main() -> None:
    asset_map = get_asset_map()
    tickers = list(asset_map.keys())

    raw_df = download_market_data(tickers=tickers, start="2015-01-01", end=None)
    base_df = reshape_market_data(raw_df, asset_map)
    node_df = compute_node_features(base_df)
    edge_df = compute_edge_features(node_df, window=60)

    node_df = node_df[P4_NODE_FEATURE_COLUMNS].copy()
    edge_df = edge_df[P4_EDGE_FEATURE_COLUMNS].copy()

    validate_node_schema(node_df)
    validate_edge_schema(edge_df)

    NODE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    node_df.to_parquet(NODE_OUTPUT_PATH, index=False)
    edge_df.to_parquet(EDGE_OUTPUT_PATH, index=False)

    print(f"Saved {len(node_df)} node rows to {NODE_OUTPUT_PATH}")
    print(f"Saved {len(edge_df)} edge rows to {EDGE_OUTPUT_PATH}")
    print(f"Assets used: {tickers}")


if __name__ == "__main__":
    main()