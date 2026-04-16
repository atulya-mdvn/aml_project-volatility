# aml_project-volatility

Applied Machine Learning final project on multi-modal volatility forecasting.

**Karim Nabilsi & Atulya Madhavan**  
Columbia University — Spring 2026

---

## Project Overview

This project studies whether future market volatility can be better predicted by combining multiple data modalities rather than relying on price history alone.

The target variable is primarily **22-day forward S&P 500 realized volatility**.

The system is organized into five modeling pillars:

- **P1: Regime Detection**  
  Variational LSTM Autoencoder used to learn latent volatility regimes, anomalies, and state transitions.

- **P2: Sequential Time-Series Modeling**  
  Transformer encoder used to capture medium-term temporal structure in market features.

- **P3: Financial Text Signals**  
  Text pipeline using financial headlines / macro text to estimate sentiment, uncertainty, and risk-related features.

- **P4: Cross-Asset Graph Modeling**  
  Graph Attention Network (GAT) using cross-asset relationships and rolling correlations.

- **P5: HAR-RV Baseline**  
  Classical Heterogeneous Autoregressive Realized Volatility benchmark model.

These pillar outputs are merged into downstream fusion models for final forecasting and evaluation.

---

## Final Modeling Stack

The repository contains multiple final-stage approaches, including:

- HAR-RV linear benchmark
- XGBoost tabular model
- Cross-attention fusion model
- Regime-conditional ensemble models
- Forecast calibration and confidence interval tools

---

## Repository Structure

```text
aml_project-volatility/
├── README.md
├── requirements.txt
├── .gitignore
├── run.py
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── src/
│   ├── p1_regime/
│   ├── p2_transformer/
│   ├── p3_text/
│   ├── p4_graph/
│   ├── p5_har_rv/
│   ├── fusion/
│   ├── shared/
│   └── experimental/
├── notebooks/
└── outputs/