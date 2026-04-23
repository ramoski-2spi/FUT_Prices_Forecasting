'''from pathlib import Path

# -----------------------------
# FUTBIN scraping settings
# -----------------------------

FUTBIN_BASE_URL = "https://www.futbin.com/26/player"
USER_AGENT = "Mozilla/5.0"

# -----------------------------
# Data directories
# -----------------------------

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

# Create folders if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Modeling settings
# -----------------------------

DEFAULT_MODEL_PATH = MODEL_DIR / "xgb_model.pkl"

# -----------------------------
# Forecasting settings
# -----------------------------

DEFAULT_FORECAST_DAYS = 30'''

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
# FUTBIN scraping settings
# ---------------------------------------------------------
FUTBIN_BASE_URL = "https://www.futbin.com/26/player"
USER_AGENT = "Mozilla/5.0"

# ---------------------------------------------------------
# Modeling settings
# ---------------------------------------------------------
DEFAULT_MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
DEFAULT_RESIDUAL_STD_PATH = MODEL_DIR / "residual_std.pkl"

# ---------------------------------------------------------
# Forecasting settings
# ---------------------------------------------------------
DEFAULT_FORECAST_DAYS = 30