from pathlib import Path

# Base
BASE_DIR = Path(__file__).resolve().parent.parent

# Data
DATA_DIR       = BASE_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"

# Raw files
TRAIN_TX = RAW_DIR / "train_transaction.csv"
TRAIN_ID = RAW_DIR / "train_identity.csv"
TEST_TX  = RAW_DIR / "test_transaction.csv"
TEST_ID  = RAW_DIR / "test_identity.csv"

# Artifacts
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR    = ARTIFACTS_DIR / "models"
LOGS_DIR      = ARTIFACTS_DIR / "logs"

# Models
XGB_HEAVY  = MODELS_DIR / "xgb_heavy.pkl"
LGBM_HEAVY = MODELS_DIR / "lgbm_heavy.pkl"
XGB_LIGHT  = MODELS_DIR / "xgb_light.pkl"
LGBM_LIGHT = MODELS_DIR / "lgbm_light.pkl"
ISO_FOREST = MODELS_DIR / "iso_forest.pkl"

# Metadata
FEATURE_MEDIANS   = MODELS_DIR / "feature_medians.csv"
TOP_35_FEATURES   = MODELS_DIR / "top35_features.pkl"
ALL_FEATURES_FILE = MODELS_DIR / "all_features.pkl"

# Auto create dirs
for path in [
    DATA_DIR, RAW_DIR, PROCESSED_DIR,
    ARTIFACTS_DIR, MODELS_DIR, LOGS_DIR
]:
    path.mkdir(parents=True, exist_ok=True)