from pathlib import Path

# ---------------------------------------------------------
# Project root (directory containing config.py)
# ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------
# Data directories
# ---------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Modeling settings
# ---------------------------------------------------------
DEFAULT_MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
DEFAULT_RESIDUAL_STD_PATH = MODEL_DIR / "residual_std.pkl"