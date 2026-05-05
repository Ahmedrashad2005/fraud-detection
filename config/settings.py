# ============================================================
# GENERAL CONFIG
# ============================================================

APP_NAME     = "Fraud Detection System"
VERSION      = "1.0.0"
RANDOM_STATE = 42
TEST_SIZE    = 0.2


# ============================================================
# MODEL WEIGHTS (for ensemble)
# ============================================================

MODEL_WEIGHTS = {
    "xgb": 0.5,
    "lgbm": 0.4,
    "iso": 0.1
}


# ============================================================
# THRESHOLDS
# ============================================================

# Main prediction threshold (binary decision)
PREDICTION_THRESHOLD = 0.5

# Optional: review layer (bank-style decision)
REVIEW_THRESHOLD = 0.6

# Risk levels (for UI / business logic)
RISK_THRESHOLDS = {
    "high": 0.75,
    "medium": 0.5
}

RISK_LABELS = {
    "high": "HIGH RISK",
    "medium": "MEDIUM RISK",
    "low": "LOW RISK"
}


# ============================================================
# DATABASE
# ============================================================

DB_NAME      = "fraud_detection.db"
DB_PATH      = f"artifacts/{DB_NAME}"

DB_TABLE_TX  = "transactions"
DB_TABLE_LOG = "prediction_logs"


# ============================================================
# DASHBOARD
# ============================================================

REFRESH_RATE     = 30   # seconds
MAX_ROWS_DISPLAY = 100


# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = "INFO"


# ============================================================
# MLOPS
# ============================================================

DRIFT_THRESHOLD = 0.85
MLFLOW_EXP_NAME = "fraud-detection"