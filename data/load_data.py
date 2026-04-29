# data/load_data.py

import pandas as pd
from pathlib import Path
from config.paths import (RAW_DIR, PROCESSED_DIR,
                           TRAIN_TX, TRAIN_ID,
                           TEST_TX,  TEST_ID)


def load_raw_data(verbose=True):
    if not TRAIN_TX.exists():
        raise FileNotFoundError(f"Not found: {TRAIN_TX}")
    if not TRAIN_ID.exists():
        raise FileNotFoundError(f"Not found: {TRAIN_ID}")

    if verbose:
        print("Loading raw data...")

    train_tx = pd.read_csv(TRAIN_TX, low_memory=False)
    train_id = pd.read_csv(TRAIN_ID, low_memory=False)

    df = train_tx.merge(train_id, on='TransactionID', how='left')

    if verbose:
        print(f"train_transaction : {train_tx.shape}")
        print(f"train_identity    : {train_id.shape}")
        print(f"After merge       : {df.shape}")
        print(f"Fraud rate        : {df['isFraud'].mean()*100:.2f}%")

    return df


def load_test_data(verbose=True):
    if not TEST_TX.exists():
        raise FileNotFoundError(f"Not found: {TEST_TX}")
    if not TEST_ID.exists():
        raise FileNotFoundError(f"Not found: {TEST_ID}")

    if verbose:
        print("Loading test data...")

    test_tx = pd.read_csv(TEST_TX, low_memory=False)
    test_id = pd.read_csv(TEST_ID, low_memory=False)

    df = test_tx.merge(test_id, on='TransactionID', how='left')

    if verbose:
        print(f"test_transaction : {test_tx.shape}")
        print(f"test_identity    : {test_id.shape}")
        print(f"After merge      : {df.shape}")

    return df


def save_processed(df, filename):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    print(f"✅ Saved processed data → {path}")


def load_processed(filename):
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"✅ Loaded processed data → {path} — shape: {df.shape}")
    return df


def get_data_info(df):
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    print(f"Shape           : {df.shape}")
    print(f"Total rows      : {len(df):,}")
    print(f"Total columns   : {df.shape[1]}")
    print(f"Numeric cols    : {df.select_dtypes('number').shape[1]}")
    print(f"Categorical cols: {df.select_dtypes('object').shape[1]}")
    print(f"Missing values  : {df.isnull().sum().sum():,}")
    print(f"Duplicate rows  : {df.duplicated().sum():,}")

    if 'isFraud' in df.columns:
        fraud_count  = df['isFraud'].sum()
        normal_count = (df['isFraud'] == 0).sum()
        print(f"Fraud count     : {fraud_count:,}")
        print(f"Fraud rate      : {df['isFraud'].mean()*100:.2f}%")

        if fraud_count > 0:
            print(f"Imbalance ratio : {int(normal_count / fraud_count)}:1")
        else:
            print("Imbalance ratio : No fraud found in dataset")

    print("="*50)


if __name__ == "__main__":
    df = load_raw_data()
    get_data_info(df)
    save_processed(df, "merged_train.csv")