# features/preprocess.py

import pandas as pd
import numpy as np
import joblib
from config.paths import ARTIFACTS_DIR

DROP_COLS     = ['TransactionID', 'TransactionDT']
ENCODERS_PATH = ARTIFACTS_DIR / "encoders.pkl"
MEDIANS_PATH  = ARTIFACTS_DIR / "medians.pkl"
COLUMNS_PATH  = ARTIFACTS_DIR / "feature_columns.pkl"


# ================================================================
# Basic Cleaning
# ================================================================

def drop_useless_cols(df):
    cols = [c for c in DROP_COLS if c in df.columns]
    df   = df.drop(columns=cols)
    print(f"✅ Dropped {len(cols)} useless columns")
    return df


def drop_high_missing(df, threshold=0.9):
    missing_rate = df.isnull().mean()
    high_missing = missing_rate[missing_rate > threshold].index.tolist()
    df = df.drop(columns=high_missing)
    print(f"✅ Dropped {len(high_missing)} high-missing columns")
    return df, high_missing


# ================================================================
# Missing Values
# ================================================================

def fill_missing_train(df):
    medians = {}
    for col in df.select_dtypes(include=np.number).columns:
        med          = df[col].median()
        df[col]      = df[col].fillna(med)
        medians[col] = med
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna("Unknown")
    print("✅ Missing filled (train)")
    return df, medians


def fill_missing_inference(df, medians):
    for col, med in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(med)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna("Unknown")
    print("✅ Missing filled (inference)")
    return df


# ================================================================
# Encoding
# ================================================================

def encode_train(df):
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        uniques       = df[col].astype(str).unique().tolist()
        mapping       = {val: idx for idx, val in enumerate(uniques)}
        df[col]       = df[col].astype(str).map(mapping)
        encoders[col] = mapping
    print(f"✅ Encoded {len(encoders)} columns (train)")
    return df, encoders


def encode_inference(df, encoders):
    for col, mapping in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping).fillna(-1)
    print("✅ Encoded (inference)")
    return df


# ================================================================
# Memory Optimization
# ================================================================

def reduce_memory(df):
    before = df.memory_usage().sum() / 1e6
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    after = df.memory_usage().sum() / 1e6
    print(f"✅ Memory: {before:.1f} → {after:.1f} MB")
    return df


# ================================================================
# preprocess_train
# ================================================================

def preprocess_train(df):
    print("\n" + "="*50)
    print("TRAIN PREPROCESSING")
    print("="*50)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Drop useless
    df = drop_useless_cols(df)

    # 2. Drop high missing + save dropped cols
    df, dropped_cols = drop_high_missing(df)

    # 3. Fill missing + save medians
    df, medians = fill_missing_train(df)

    # 4. Encode + save encoders
    df, encoders = encode_train(df)

    # 5. Memory optimization
    df = reduce_memory(df)

    # 6. Save all artifacts
    joblib.dump(encoders,            ENCODERS_PATH)
    joblib.dump(medians,             MEDIANS_PATH)
    joblib.dump(df.columns.tolist(), COLUMNS_PATH)
    joblib.dump(dropped_cols,
                ARTIFACTS_DIR / "dropped_cols.pkl")

    print(f"✅ Artifacts saved")
    print(f"Final shape: {df.shape}")
    print("="*50)
    return df


# ================================================================
# preprocess_inference
# ================================================================

def preprocess_inference(df):
    """
    ⚠️ لا نعمل drop_high_missing هنا عشان:
    - الأعمدة اللي اتشالت في train محفوظة في COLUMNS_PATH
    - بنستخدم df[columns] في الآخر عشان يشيل الزيادة
    - ده بيضمن إن الـ inference دايماً بنفس شكل الـ train
    """
    print("\n" + "="*50)
    print("INFERENCE PREPROCESSING")
    print("="*50)

    # Load saved artifacts
    encoders = joblib.load(ENCODERS_PATH)
    medians  = joblib.load(MEDIANS_PATH)
    columns  = joblib.load(COLUMNS_PATH)

    # 1. Drop useless cols فقط (مش high missing)
    df = drop_useless_cols(df)

    # 2. Fill missing باستخدام medians من train
    df = fill_missing_inference(df, medians)

    # 3. Encode باستخدام encoders من train
    df = encode_inference(df, encoders)

    # 4. Align columns
    # - أضف الناقص بـ 0
    # - شيل الزيادة
    # - ده بيعوض drop_high_missing تلقائياً ✅
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]

    # 5. Memory optimization
    df = reduce_memory(df)

    print(f"Final shape: {df.shape}")
    print("="*50)
    return df


# ================================================================
# Test Run
# ================================================================

if __name__ == "__main__":
    from data.load_data import load_raw_data

    df = load_raw_data()
    df = preprocess_train(df)
    print("\npreprocess_train ✅")

    df2 = load_raw_data()
    df2 = preprocess_inference(df2)
    print("\npreprocess_inference ✅")