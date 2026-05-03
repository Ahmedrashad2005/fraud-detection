# models/train.py

import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score,
    classification_report, confusion_matrix
)

from config.paths import ARTIFACTS_DIR
from config.params import (
    XGB_HEAVY_PARAMS,
    XGB_LIGHT_PARAMS,
    LGBM_HEAVY_PARAMS,
    LGBM_LIGHT_PARAMS,
    ISO_PARAMS
)
from config.settings import TEST_SIZE, RANDOM_STATE

from data.load_data import load_raw_data
from features.build_features import build_features


TARGET = "isFraud"
THRESHOLD = 0.3


# ================================================================
# 1. Split
# ================================================================
def split_data(df):
    print("\n" + "=" * 50)
    print("SPLIT DATA")
    print("=" * 50)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Train : {X_train.shape} | Fraud: {y_train.sum():,}")
    print(f"Test  : {X_test.shape}  | Fraud: {y_test.sum():,}")

    return X_train, X_test, y_train, y_test


# ================================================================
# 2. Preprocess + Features
# ================================================================
def process_data(X_train, X_test):
    print("\n" + "=" * 50)
    print("PREPROCESS + FEATURES")
    print("=" * 50)

    # fit on train
    X_train, encoders, dropped_cols = preprocess(X_train, fit=True)

    # transform test
    X_test, _, _ = preprocess(
        X_test,
        fit=False,
        encoders=encoders,
        cols_to_drop=dropped_cols
    )

    # feature engineering
    X_train = build_features(X_train)
    X_test  = build_features(X_test)

    # align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"✅ Train shape: {X_train.shape}")
    print(f"✅ Test  shape: {X_test.shape}")

    return X_train, X_test, encoders, dropped_cols


# ================================================================
# 3. Evaluation
# ================================================================
def evaluate(model, X, y, name="Model"):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > THRESHOLD).astype(int)

    print("\n" + "=" * 50)
    print(f"📊 {name}")
    print("=" * 50)

    print(f"AUC : {roc_auc_score(y, probs):.4f}")
    print(f"F1  : {f1_score(y, preds):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, preds))

    print("\nClassification Report:")
    print(classification_report(y, preds))


# ================================================================
# 4. Isolation Forest
# ================================================================
def train_iso(X_train):
    print("\nTraining Isolation Forest...")
    iso = IsolationForest(**ISO_PARAMS)
    iso.fit(X_train)
    return iso


# ================================================================
# 5. Heavy Models
# ================================================================
def train_heavy(X_train, y_train, X_test, y_test):
    print("\n🚀 Training HEAVY models...")

    xgb = XGBClassifier(**XGB_HEAVY_PARAMS)
    xgb.fit(X_train, y_train)

    lgbm = LGBMClassifier(**LGBM_HEAVY_PARAMS)
    lgbm.fit(X_train, y_train)

    evaluate(xgb, X_test, y_test, "XGBoost Heavy")
    evaluate(lgbm, X_test, y_test, "LightGBM Heavy")

    return xgb, lgbm


# ================================================================
# 6. Light Models (Top 35 dynamic)
# ================================================================
def train_light(X_train, y_train, X_test, y_test, xgb_heavy):
    print("\n⚡ Training LIGHT models...")

    importances = xgb_heavy.feature_importances_
    top_idx = np.argsort(importances)[-35:]
    features = X_train.columns[top_idx].tolist()

    print(f"Top features: {len(features)}")

    X_tr = X_train[features]
    X_te = X_test[features]

    xgb_l = XGBClassifier(**XGB_LIGHT_PARAMS)
    xgb_l.fit(X_tr, y_train)

    lgbm_l = LGBMClassifier(**LGBM_LIGHT_PARAMS)
    lgbm_l.fit(X_tr, y_train)

    evaluate(xgb_l, X_te, y_test, "XGBoost Light")
    evaluate(lgbm_l, X_te, y_test, "LightGBM Light")

    return xgb_l, lgbm_l, features


# ================================================================
# 7. Save Artifacts
# ================================================================
def save_all(models,
             encoders,
             dropped_cols,
             all_features,
             light_features,
             X_train):

    print("\n💾 Saving artifacts...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # models
    for name, model in models.items():
        joblib.dump(model, ARTIFACTS_DIR / f"{name}.pkl")

    # preprocessing artifacts
    joblib.dump(encoders, ARTIFACTS_DIR / "encoders.pkl")
    joblib.dump(dropped_cols, ARTIFACTS_DIR / "dropped_cols.pkl")

    # feature lists
    joblib.dump(all_features, ARTIFACTS_DIR / "all_features.pkl")
    joblib.dump(light_features, ARTIFACTS_DIR / "top35_features.pkl")

    # medians
    medians = X_train.median().to_dict()
    joblib.dump(medians, ARTIFACTS_DIR / "feature_medians.pkl")

    print("✅ All artifacts saved!")


# ================================================================
# MAIN
# ================================================================
def main():
    print("\n" + "=" * 50)
    print("TRAINING PIPELINE")
    print("=" * 50)

    # 1. load
    df = load_raw_data()

    # 2. split
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. preprocess
    X_train, X_test, encoders, dropped_cols = process_data(X_train, X_test)

    # 4. iso
    iso = train_iso(X_train)

    # 5. heavy
    xgb, lgbm = train_heavy(X_train, y_train, X_test, y_test)

    # 6. light
    xgb_l, lgbm_l, light_features = train_light(
        X_train, y_train, X_test, y_test, xgb
    )

    # 7. save
    save_all(
        {
            "xgb_heavy": xgb,
            "lgbm_heavy": lgbm,
            "iso_forest": iso,
            "xgb_light": xgb_l,
            "lgbm_light": lgbm_l
        },
        encoders,
        dropped_cols,
        X_train.columns.tolist(),
        light_features,
        X_train
    )

    print("\n🎉 TRAINING COMPLETE!")


if __name__ == "__main__":
    main()