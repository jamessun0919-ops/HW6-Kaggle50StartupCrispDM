import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from src.config import SEED, TEST_SIZE


def feature_engineering(df):
    df = df.copy()
    df["TotalSpend"] = df["R&D Spend"] + df["Marketing Spend"] + df["Administration"]
    df["ROI"] = df["Profit"] / df["TotalSpend"].replace(0, np.nan)
    df["RDRatio"] = df["R&D Spend"] / df["TotalSpend"].replace(0, np.nan)
    df["MarketingRatio"] = df["Marketing Spend"] / df["TotalSpend"].replace(0, np.nan)
    return df


def encode_state(df):
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    state_encoded = encoder.fit_transform(df[["State"]])
    state_cols = encoder.get_feature_names_out(["State"])
    df_encoded = pd.concat(
        [df.drop(columns=["State"]),
         pd.DataFrame(state_encoded, columns=state_cols, index=df.index)], axis=1
    )
    return df_encoded, encoder


def prepare_features(df, use_engineered=True):
    df_fe = feature_engineering(df) if use_engineered else df.copy()
    df_enc, encoder = encode_state(df_fe)

    exclude = ["Profit"]
    X = df_enc.drop(columns=exclude)
    y = df_enc["Profit"]
    return X, y, encoder


def split_data(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)


def scale_features(X_train, X_test, columns):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[columns] = scaler.fit_transform(X_train[columns])
    X_test_scaled[columns] = scaler.transform(X_test[columns])
    return X_train_scaled, X_test_scaled, scaler


def run_data_preparation(df, use_engineered=True):
    X, y, encoder = prepare_features(df, use_engineered)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_orig, X_test_orig = X_train.copy(), X_test.copy()

    scale_cols = ["R&D Spend", "Administration", "Marketing Spend"]
    if use_engineered:
        scale_cols = [c for c in X_train.columns if c in scale_cols]

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, scale_cols)

    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print(f"Features: {list(X_train_scaled.columns)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoder, X_train_orig, X_test_orig
