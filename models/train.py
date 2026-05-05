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
    classification_report, confusion_matrix,
    precision_recall_curve
)

from config.paths import ARTIFACTS_DIR
from config.params import (
    XGB_HEAVY_PARAMS, XGB_LIGHT_PARAMS,
    LGBM_HEAVY_PARAMS, LGBM_LIGHT_PARAMS,
    ISO_PARAMS
)
from config.settings import TEST_SIZE, RANDOM_STATE
from data.load_data import load_raw_data
from features.preprocess import preprocess_train, preprocess_inference
from features.build_features import build_features

TARGET = "isFraud"


# ================================================================
# 1. Split → Train / Val / Test
# ================================================================
def split_data(df):
    print("\n" + "="*50)
    print("SPLIT DATA (70 / 15 / 15)")
    print("="*50)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 70% Train | 30% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 15% Val | 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    print(f"Train : {X_train.shape} | Fraud: {y_train.sum():,}")
    print(f"Val   : {X_val.shape}   | Fraud: {y_val.sum():,}")
    print(f"Test  : {X_test.shape}  | Fraud: {y_test.sum():,}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ================================================================
# 2. Preprocess + Feature Engineering
# ================================================================
def process_data(X_train, X_val, X_test):
    print("\n" + "="*50)
    print("PREPROCESS + FEATURES")
    print("="*50)

    # Fit on train only
    X_train = preprocess_train(X_train)
    X_val   = preprocess_inference(X_val)
    X_test  = preprocess_inference(X_test)

    # Feature Engineering
    X_train = build_features(X_train)
    X_val   = build_features(X_val)
    X_test  = build_features(X_test)

    # Align columns
    X_val  = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"✅ Train : {X_train.shape}")
    print(f"✅ Val   : {X_val.shape}")
    print(f"✅ Test  : {X_test.shape}")

    return X_train, X_val, X_test


# ================================================================
# 3. Aggregation Features (fit on train only)
# ================================================================
def add_aggregation_features(X_train, X_val, X_test):
    print("\n" + "="*50)
    print("AGGREGATION FEATURES")
    print("="*50)

    agg_configs = [
        ('card1',         'TransactionAmt', ['mean', 'std', 'count']),
        ('card2',         'TransactionAmt', ['mean', 'count']),
        ('P_emaildomain', 'TransactionAmt', ['mean', 'count']),
        ('DeviceType',    'TransactionAmt', ['mean', 'count']),
    ]

    for group_col, value_col, aggs in agg_configs:
        if group_col not in X_train.columns:
            continue
        if value_col not in X_train.columns:
            continue

        for agg in aggs:
            col_name = f"{group_col}_{value_col}_{agg}"

            # Fit on train only
            agg_map = X_train.groupby(group_col)[value_col].agg(agg)

            # Transform all three
            X_train[col_name] = X_train[group_col]\
                                 .map(agg_map).fillna(0)
            X_val[col_name]   = X_val[group_col]\
                                 .map(agg_map).fillna(0)
            X_test[col_name]  = X_test[group_col]\
                                 .map(agg_map).fillna(0)

            print(f"  ✅ {col_name}")

    # Amount vs card mean ratio
    if 'card1_TransactionAmt_mean' in X_train.columns:
        for df_ in [X_train, X_val, X_test]:
            df_['amt_vs_card1_mean'] = (
                df_['TransactionAmt'] /
                (df_['card1_TransactionAmt_mean'] + 1)
            )
        print("  ✅ amt_vs_card1_mean")

    print(f"\n✅ Train : {X_train.shape}")
    print(f"✅ Val   : {X_val.shape}")
    print(f"✅ Test  : {X_test.shape}")

    return X_train, X_val, X_test


# ================================================================
# 4. Isolation Forest
# ================================================================
def train_iso(X_train):
    print("\n" + "="*50)
    print("ISOLATION FOREST")
    print("="*50)

    iso = IsolationForest(**ISO_PARAMS)
    iso.fit(X_train)
    print("✅ Isolation Forest trained")
    return iso


# ================================================================
# 5. Find Best Threshold (on Val set)
# ================================================================
def find_best_threshold(y_true, probs, beta=0.5):
    """
    beta < 1 → Precision أهم
    beta = 1 → توازن
    beta > 1 → Recall أهم
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true, probs
    )

    f_beta = ((1 + beta**2) * precision * recall) / \
             (beta**2 * precision + recall + 1e-6)

    best_idx       = np.argmax(f_beta)
    best_threshold = float(thresholds[best_idx])

    print(f"Best Threshold : {best_threshold:.3f}")
    print(f"Precision      : {precision[best_idx]:.3f}")
    print(f"Recall         : {recall[best_idx]:.3f}")
    print(f"F{beta} Score  : {f_beta[best_idx]:.3f}")

    return best_threshold


# ================================================================
# 6. Evaluation
# ================================================================
def evaluate(probs, y, threshold, name="Model"):
    preds = (probs > threshold).astype(int)

    print(f"\n{'='*50}")
    print(f"📊 {name}")
    print(f"{'='*50}")
    print(f"AUC       : {roc_auc_score(y, probs):.4f}")
    print(f"F1        : {f1_score(y, preds):.4f}")
    print(f"Threshold : {threshold:.3f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y, preds))
    print(f"\nClassification Report:")
    print(classification_report(y, preds,
                                target_names=['Normal', 'Fraud']))


# ================================================================
# 7. Train Heavy Models
# ================================================================
def train_heavy(X_train, y_train, X_val, y_val):
    print("\n" + "="*50)
    print("MODEL 1 — HEAVY (All Features)")
    print("="*50)
    print(f"Features: {X_train.shape[1]}")

    xgb = XGBClassifier(**XGB_HEAVY_PARAMS)
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    lgbm = LGBMClassifier(**LGBM_HEAVY_PARAMS)
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)]
    )

    print(f"\nXGBoost  VAL AUC: "
          f"{roc_auc_score(y_val, xgb.predict_proba(X_val)[:,1]):.4f}")
    print(f"LightGBM VAL AUC: "
          f"{roc_auc_score(y_val, lgbm.predict_proba(X_val)[:,1]):.4f}")

    return xgb, lgbm


# ================================================================
# 8. Train Light Models
# ================================================================
def train_light(X_train, y_train, X_val, y_val, xgb_heavy):
    print("\n" + "="*50)
    print("MODEL 2 — LIGHT (Top 35 Features)")
    print("="*50)

    # Dynamic feature selection from Heavy
    importances = xgb_heavy.feature_importances_
    top_idx     = np.argsort(importances)[-35:]
    features    = X_train.columns[top_idx].tolist()
    print(f"Features selected: {len(features)}")

    X_tr = X_train[features]
    X_va = X_val[features]

    xgb_l = XGBClassifier(**XGB_LIGHT_PARAMS)
    xgb_l.fit(
        X_tr, y_train,
        eval_set=[(X_va, y_val)],
        verbose=100
    )

    lgbm_l = LGBMClassifier(**LGBM_LIGHT_PARAMS)
    lgbm_l.fit(X_tr, y_train)

    print(f"\nXGBoost Light  VAL AUC: "
          f"{roc_auc_score(y_val, xgb_l.predict_proba(X_va)[:,1]):.4f}")
    print(f"LightGBM Light VAL AUC: "
          f"{roc_auc_score(y_val, lgbm_l.predict_proba(X_va)[:,1]):.4f}")

    return xgb_l, lgbm_l, features


# ================================================================
# 9. Ensemble Predict
# ================================================================
def ensemble_predict(xgb, lgbm, xgb_l, lgbm_l,
                     X, light_features):
    p1 = xgb.predict_proba(X)[:, 1]
    p2 = lgbm.predict_proba(X)[:, 1]
    p3 = xgb_l.predict_proba(X[light_features])[:, 1]
    p4 = lgbm_l.predict_proba(X[light_features])[:, 1]

    final = (
        0.40 * p1 +
        0.30 * p2 +
        0.20 * p3 +
        0.10 * p4
    )
    return final


# ================================================================
# 10. Save Artifacts
# ================================================================
def save_all(models, all_features, light_features,
             X_train, threshold):
    print("\n" + "="*50)
    print("SAVING ARTIFACTS")
    print("="*50)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Models
    for name, model in models.items():
        joblib.dump(model, ARTIFACTS_DIR / f"{name}.pkl")
        print(f"✅ {name}.pkl")

    # Features
    joblib.dump(all_features,
                ARTIFACTS_DIR / "all_features.pkl")
    joblib.dump(light_features,
                ARTIFACTS_DIR / "top35_features.pkl")

    # Medians
    medians = X_train.median().to_dict()
    joblib.dump(medians,
                ARTIFACTS_DIR / "feature_medians.pkl")

    # Threshold
    joblib.dump(threshold,
                ARTIFACTS_DIR / "threshold.pkl")

    print("✅ all_features.pkl")
    print("✅ top35_features.pkl")
    print("✅ feature_medians.pkl")
    print(f"✅ threshold.pkl → {threshold:.3f}")


# ================================================================
# MAIN
# ================================================================
def main():
    print("\n" + "="*50)
    print("TRAINING PIPELINE")
    print("="*50)

    # 1. Load
    df = load_raw_data()

    # 2. Split → Train / Val / Test
    X_train, X_val, X_test, \
    y_train, y_val, y_test = split_data(df)

    # 3. Preprocess + Feature Engineering
    X_train, X_val, X_test = process_data(
        X_train, X_val, X_test
    )

    # 4. Aggregation Features (no leakage)
    X_train, X_val, X_test = add_aggregation_features(
        X_train, X_val, X_test
    )

    # 5. Isolation Forest + iso_score feature
    iso = train_iso(X_train)
    X_train['iso_score'] = iso.decision_function(X_train)
    X_val['iso_score']   = iso.decision_function(X_val)
    X_test['iso_score']  = iso.decision_function(X_test)
    print("✅ iso_score added as feature")

    # 6. Heavy Models (eval on Val)
    xgb, lgbm = train_heavy(
        X_train, y_train,
        X_val,   y_val
    )

    # 7. Light Models (eval on Val)
    xgb_l, lgbm_l, light_features = train_light(
        X_train, y_train,
        X_val,   y_val,
        xgb
    )

    # 8. Threshold tuning on Val
    print("\n" + "="*50)
    print("THRESHOLD TUNING ON VAL SET")
    print("="*50)
    val_probs = ensemble_predict(
        xgb, lgbm, xgb_l, lgbm_l,
        X_val, light_features
    )
    threshold = find_best_threshold(y_val, val_probs, beta=0.5)

    # 9. Final Evaluation on Test (clean)
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50)
    test_probs = ensemble_predict(
        xgb, lgbm, xgb_l, lgbm_l,
        X_test, light_features
    )
    evaluate(test_probs, y_test, threshold, "FINAL ENSEMBLE")

    # 10. Save
    save_all(
        {"xgb_heavy":  xgb,
         "lgbm_heavy": lgbm,
         "iso_forest": iso,
         "xgb_light":  xgb_l,
         "lgbm_light": lgbm_l},
        X_train.columns.tolist(),
        light_features,
        X_train,
        threshold
    )

    print("\n🎉 TRAINING COMPLETE!")


if __name__ == "__main__":
    main()