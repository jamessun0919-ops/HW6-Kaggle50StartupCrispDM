import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from src.config import SEED


RANDOM_STATE = SEED


def build_models():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
    }


def build_advanced_models():
    models = {}
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=200, random_state=RANDOM_STATE)
    except ImportError:
        print("XGBoost not installed, skipping")
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(n_estimators=200, random_state=RANDOM_STATE)
    except ImportError:
        print("LightGBM not installed, skipping")
    return models


def train_all_models(X_train, y_train, include_advanced=False):
    models = build_models()
    if include_advanced:
        models.update(build_advanced_models())

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"  ✓ {name} 訓練完成")

    return trained
