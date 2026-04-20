"""
Central configuration for the Volatility Intelligence project.
"""
from pathlib import Path

# ─── Basic Runtime Settings ───────────────────────────────
SEED = 42
DEVICE = "cpu"   # training scripts can override with torch.device(...)

# ─── Repo Root / Paths ────────────────────────────────────
# config.py lives in: src/shared/config.py
# so repo root is two levels up from this file
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = REPO_ROOT / "outputs"
MODELS_DIR = REPO_ROOT / "models"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

for path in [
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUTS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# Optional backwards-compatible aliases while we repair the repo
RAW_DIR = RAW_DATA_DIR
PROC_DIR = PROCESSED_DATA_DIR
MODEL_DIR = MODELS_DIR
BASE_DIR = REPO_ROOT

# ─── IBKR Connection ──────────────────────────────────────
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497
IBKR_CLIENT_ID = 1
IBKR_TIMEOUT = 30

# ─── Data Settings ─────────────────────────────────────────
START_DATE = "2000-01-01"
END_DATE = "2025-12-31"
SPLIT_DATE = "2020-01-01"
TARGET_WINDOW = 22
SPIKE_PERCENTILE = 90

IBKR_BAR_SIZE = "5 mins"
IBKR_DURATION = "1 Y"

CROSS_ASSETS = {
    "TLT": "20Y Treasury", "IEF": "7-10Y Treasury", "SHY": "1-3Y Treasury",
    "HYG": "High Yield", "LQD": "Inv Grade", "JNK": "Junk Bonds",
    "GLD": "Gold", "SLV": "Silver", "USO": "Oil", "UNG": "Nat Gas",
    "UUP": "USD Index", "FXE": "EUR/USD", "FXB": "GBP/USD", "FXY": "JPY/USD",
    "EEM": "EM Equity", "EWJ": "Japan", "FXI": "China", "EWZ": "Brazil",
    "VNQ": "Real Estate", "XLF": "Financials", "XLK": "Tech", "XLE": "Energy",
    "XLV": "Healthcare", "XLU": "Utilities", "DBA": "Agriculture", "COPX": "Copper",
}

# ─── Model Hyperparameters ─────────────────────────────────
# P1: VAE
P1_SEQ_LEN = 30
P1_LATENT_DIM = 8
P1_HIDDEN_DIM = 32
P1_DROPOUT = 0.1
P1_EPOCHS = 60
P1_LR = 1e-3
P1_KL_WEIGHT = 0.01
P1_VOL_WEIGHT = 0.1

# P2: Transformer
P2_SEQ_LEN = 60
P2_D_MODEL = 64
P2_HEADS = 4
P2_LAYERS = 2
P2_DROPOUT = 0.15
P2_OUTPUT_DIM = 32
P2_EPOCHS = 40
P2_LR = 5e-4

# P4: GAT
P4_HIDDEN = 16
P4_OUTPUT = 8
P4_HEADS = 4
P4_DROPOUT = 0.15
P4_EPOCHS = 50
P4_LR = 1e-3
P4_CORR_WINDOW = 60
P4_PATIENCE = 10

# Meta: cross-attention fusion
META_HIDDEN = 48
META_HEADS = 4
META_DROPOUT = 0.2
META_EPOCHS = 60
META_LR = 1e-3
META_FOCAL_GAMMA = 2.0

# Ensemble — regime-conditional weights
ENSEMBLE_HAR_WEIGHT_CALM = 0.30
ENSEMBLE_HAR_WEIGHT_NORMAL = 0.40
ENSEMBLE_HAR_WEIGHT_STRESS = 0.60
ENSEMBLE_HAR_WEIGHT_CRISIS = 0.70

# XGBoost
XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.5,
    reg_alpha=0.3,
    reg_lambda=3.0,
    random_state=SEED,
    verbosity=0,
)