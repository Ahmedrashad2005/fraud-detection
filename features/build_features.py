# features/build_features.py

import pandas as pd
import numpy as np

from datetime import datetime, timedelta


# ==============================
# TIME FEATURES
# ==============================

def build_time_features(df):
    if 'TransactionDT' not in df.columns:
        return df

    ref_date = datetime(2017, 11, 30)

    dt = df['TransactionDT'].apply(
        lambda x: ref_date + timedelta(seconds=x)
    )

    df['hour']        = dt.dt.hour
    df['day_of_week'] = dt.dt.dayofweek

    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['is_night']    = (df['hour'] < 6).astype(int)
    df['is_morning']  = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_afternoon']= ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)

    df['is_rush_hour'] = df['hour'].isin([8, 9, 17, 18]).astype(int)

    # cyclic encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['dow_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 7)

    print("✅ Time features built")
    return df


# ==============================
# AMOUNT FEATURES
# ==============================

def build_amount_features(df):
    if 'TransactionAmt' not in df.columns:
        return df

    amt = df['TransactionAmt']

    df['amount_log']  = np.log1p(amt)
    df['amount_sqrt'] = np.sqrt(amt)

    df['is_high_amount']  = (amt > 1000).astype(int)
    df['is_round_number'] = (amt % 100 == 0).astype(int)

    df['amount_cents'] = amt % 1
    df['has_no_cents'] = (df['amount_cents'] == 0).astype(int)

    df['amount_category'] = pd.cut(
        amt,
        bins=[0, 50, 200, 1000, float('inf')],
        labels=[0, 1, 2, 3]
    ).astype(float)

    print("✅ Amount features built")
    return df


# ==============================
# EMAIL FEATURES
# ==============================

def build_email_features(df):
    if 'P_emaildomain' not in df.columns:
        return df

    free_emails  = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
    risky_emails = ['anonymous.com', 'guerrillamail.com', 'temp-mail.org']

    df['is_free_email']  = df['P_emaildomain'].isin(free_emails).astype(int)
    df['is_risky_email'] = df['P_emaildomain'].isin(risky_emails).astype(int)
    df['email_missing']  = df['P_emaildomain'].isnull().astype(int)

    print("✅ Email features built")
    return df


# ==============================
# CARD FEATURES
# ==============================

def build_card_features(df):
    if 'card4' in df.columns:
        risk_map = {
            'visa': 0.018,
            'mastercard': 0.031,
            'american express': 0.052,
            'discover': 0.078
        }

        df['card_risk_score'] = df['card4'].map(risk_map).fillna(0.04)
        df['is_discover'] = (df['card4'] == 'discover').astype(int)
        df['is_amex']     = (df['card4'] == 'american express').astype(int)

    if 'card6' in df.columns:
        df['is_credit'] = (df['card6'] == 'credit').astype(int)

    print("✅ Card features built")
    return df


# ==============================
# DEVICE FEATURES
# ==============================

def build_device_features(df):
    if 'DeviceType' not in df.columns:
        return df

    device_risk = {
        'mobile': 0.062,
        'desktop': 0.031
    }

    df['is_mobile'] = (df['DeviceType'] == 'mobile').astype(int)
    df['device_risk_score'] = df['DeviceType'].map(device_risk).fillna(0.04)
    df['device_missing'] = df['DeviceType'].isnull().astype(int)

    print("✅ Device features built")
    return df


# ==============================
# DISTANCE FEATURES
# ==============================

def build_distance_features(df):
    if 'dist1' not in df.columns:
        return df

    df['dist1_log'] = np.log1p(df['dist1'].fillna(0))
    df['dist1_missing'] = df['dist1'].isnull().astype(int)
    df['is_far_distance'] = (df['dist1'] > 500).astype(int)

    print("✅ Distance features built")
    return df


# ==============================
# CROSS FEATURES
# ==============================

def build_cross_features(df):
    if 'is_night' in df.columns and 'is_high_amount' in df.columns:
        df['night_x_high_amount'] = df['is_night'] * df['is_high_amount']

    if 'is_mobile' in df.columns:
        df['mobile_x_high_amount'] = df['is_mobile'] * df.get('is_high_amount', 0)
        df['mobile_x_night']       = df['is_mobile'] * df.get('is_night', 0)

    if 'is_discover' in df.columns:
        df['discover_x_night'] = df['is_discover'] * df.get('is_night', 0)

    print("✅ Cross features built")
    return df


# ==============================
# COMPOSITE RISK
# ==============================

def build_composite_risk(df):
    df['composite_risk'] = (
        df.get('device_risk_score', 0) * 0.25 +
        df.get('card_risk_score', 0)   * 0.25 +
        df.get('is_night', 0)          * 0.20 +
        df.get('is_high_amount', 0)    * 0.15 +
        df.get('is_risky_email', 0)    * 0.10 +
        df.get('is_far_distance', 0)   * 0.05
    )

    print("✅ Composite risk built")
    return df


# ==============================
# MAIN PIPELINE
# ==============================

def build_features(df):
    print("\n" + "="*50)
    print("FEATURE ENGINEERING")
    print("="*50)

    before = df.shape[1]

    df = build_time_features(df)
    df = build_amount_features(df)
    df = build_email_features(df)
    df = build_card_features(df)
    df = build_device_features(df)
    df = build_distance_features(df)
    df = build_cross_features(df)
    df = build_composite_risk(df)

    after = df.shape[1]

    print(f"📈 Features added: {after - before}")
    print(f"Total columns: {after}")
    print("="*50)

    return df


# ==============================
# TEST
# ==============================

if __name__ == "__main__":
    from data.load_data import load_raw_data
    from features.preprocess import preprocess_train

    df = load_raw_data()
    df = build_features(df)
    df = preprocess_train(df)