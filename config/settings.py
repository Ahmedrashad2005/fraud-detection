# General
APP_NAME     = "Fraud Detection System"
VERSION      = "1.0.0"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# Model Weights
MODEL_WEIGHTS = {
    "xgb": 0.45,
    "lgbm": 0.35,
    "iso": 0.20
}

# Thresholds
PREDICTION_THRESHOLD = 0.3

RISK_THRESHOLDS = {
    "high": 0.7,
    "medium": 0.4
}

RISK_LABELS = {
    "high": "HIGH RISK",
    "medium": "MEDIUM RISK",
    "low": "LOW RISK"
}

# Database
DB_NAME      = "fraud_detection.db"
DB_PATH      = f"artifacts/{DB_NAME}"
DB_TABLE_TX  = "transactions"
DB_TABLE_LOG = "prediction_logs"

# Dashboard
REFRESH_RATE     = 30
MAX_ROWS_DISPLAY = 100

# Logging
LOG_LEVEL = "INFO"

# MLOps
DRIFT_THRESHOLD = 0.85
MLFLOW_EXP_NAME = "fraud-detection"