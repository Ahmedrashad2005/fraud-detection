# features/preprocess.py

import pandas as pd
import numpy as np
import joblib

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.paths import ARTIFACTS_DIR


# ==============================
# CONFIG
# ==============================

DROP_COLS = ['TransactionID', 'TransactionDT']

ENCODERS_PATH = ARTIFACTS_DIR / "encoders.pkl"
MEDIANS_PATH  = ARTIFACTS_DIR / "medians.pkl"
COLUMNS_PATH  = ARTIFACTS_DIR / "feature_columns.pkl"


# ==============================
# BASIC CLEANING
# ==============================

def drop_useless_cols(df):
    cols = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols)
    print(f"✅ Dropped {len(cols)} useless columns")
    return df


def drop_high_missing(df, threshold=0.9):
    missing_rate = df.isnull().mean()
    high_missing = missing_rate[missing_rate > threshold].index.tolist()
    df = df.drop(columns=high_missing)
    print(f"✅ Dropped {len(high_missing)} high-missing columns")
    return df


# ==============================
# MISSING VALUES
# ==============================

def fill_missing_train(df):
    medians = {}

    # Numeric
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        med = df[col].median()
        df[col] = df[col].fillna(med)
        medians[col] = med

    # Categorical
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    print("✅ Missing values filled (train)")
    return df, medians


def fill_missing_inference(df, medians):
    for col, med in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(med)

    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    print("✅ Missing values filled (inference)")
    return df


# ==============================
# ENCODING
# ==============================

def encode_train(df):
    encoders = {}

    cat_cols = df.select_dtypes(include='object').columns

    for col in cat_cols:
        uniques = df[col].astype(str).unique().tolist()
        mapping = {val: idx for idx, val in enumerate(uniques)}

        df[col] = df[col].astype(str).map(mapping)
        encoders[col] = mapping

    print(f"✅ Encoded {len(encoders)} categorical columns")
    return df, encoders


def encode_inference(df, encoders):
    for col, mapping in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping)

            # unseen values
            df[col] = df[col].fillna(-1)

    print("✅ Encoding applied (inference)")
    return df


# ==============================
# MEMORY OPTIMIZATION
# ==============================

def reduce_memory(df):
    before = df.memory_usage().sum() / 1e6

    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    after = df.memory_usage().sum() / 1e6

    print(f"✅ Memory: {before:.1f} → {after:.1f} MB")
    return df


# ==============================
# TRAIN PIPELINE
# ==============================

def preprocess_train(df):
    print("\n" + "="*50)
    print("TRAIN PREPROCESSING")
    print("="*50)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = drop_useless_cols(df)
    df = drop_high_missing(df)

    df, medians  = fill_missing_train(df)
    df, encoders = encode_train(df)

    df = reduce_memory(df)

    # save artifacts
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(medians, MEDIANS_PATH)
    joblib.dump(df.columns.tolist(), COLUMNS_PATH)

    print("💾 Artifacts saved")

    print(f"Final shape: {df.shape}")
    print("="*50)

    return df


# ==============================
# INFERENCE PIPELINE
# ==============================

def preprocess_inference(df):
    print("\n" + "="*50)
    print("INFERENCE PREPROCESSING")
    print("="*50)

    encoders = joblib.load(ENCODERS_PATH)
    medians  = joblib.load(MEDIANS_PATH)
    columns  = joblib.load(COLUMNS_PATH)

    df = drop_useless_cols(df)

    df = fill_missing_inference(df, medians)
    df = encode_inference(df, encoders)

    # align columns
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]

    df = reduce_memory(df)

    print(f"Final shape: {df.shape}")
    print("="*50)

    return df


# ==============================
# TEST RUN
# ==============================

if __name__ == "__main__":
    from data.load_data import load_raw_data, save_processed

    df = load_raw_data()
    df = preprocess_train(df)

    save_processed(df, "preprocessed_train.csv")