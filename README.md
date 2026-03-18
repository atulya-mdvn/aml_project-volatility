# aml_project-volatility

Applied Machine Learning final project on multi-modal volatility intelligence.

## Project Overview

This project studies whether market volatility can be better predicted by combining multiple data modalities:

- **P1**: LSTM autoencoder for regime detection
- **P2**: Transformer for daily market time-series attention
- **P3**: FinBERT-based text pipeline using Fed/news/earnings text
- **P4**: Graph attention pipeline over cross-asset relationships
- **P5**: HAR-RV baseline and target construction

The final stage will merge outputs from all five pillars and train an **XGBoost fusion model**.

## Repository Structure

- `data/raw/` — original downloaded data
- `data/interim/` — cleaned intermediate data
- `data/processed/` — final modeling-ready feature tables

- `src/p1_regime/` — regime detection code
- `src/p2_transformer/` — transformer pipeline
- `src/p3_text/` — text pipeline
- `src/p4_graph/` — graph pipeline
- `src/p5_har_rv/` — baseline and target construction
- `src/fusion/` — feature merge + XGBoost
- `src/shared/` — shared utilities/configs

- `notebooks/` — exploratory and modeling notebooks by pillar
- `outputs/` — figures, tables, and saved results

## Collaboration Plan

- **Atulya**: P3 and P4
- **Karim**: P1, P2, and P5

## Expected Pillar Outputs

Each pillar should eventually export a file like:

- `p1_features.parquet`
- `p2_features.parquet`
- `p3_features.parquet`
- `p4_features.parquet`
- `p5_features.parquet`

Each file should contain:
- a `date` column
- pillar-specific features
- no duplicate dates
- consistent formatting for downstream merge

## Fusion Plan

The final fusion step will:
1. merge all pillar outputs by date
2. align with the regression/classification targets
3. train an XGBoost meta-model
4. evaluate performance and run ablations