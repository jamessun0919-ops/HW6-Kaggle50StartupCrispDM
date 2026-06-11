import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.config import VIF_THRESHOLD_GOOD, VIF_THRESHOLD_MODERATE


def compute_vif(df, features):
    vif_data = pd.DataFrame()
    vif_data["feature"] = features
    vif_data["VIF"] = [
        _calc_vif(df, f, features) for f in features
    ]
    return vif_data


def _calc_vif(df, target, features):
    predictors = [f for f in features if f != target]
    if len(predictors) == 0:
        return 1.0
    X = df[predictors].values
    y = df[target].values
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    return 1.0 / (1.0 - r2) if r2 < 1 else float("inf")


def interpret_vif(vif_value):
    if vif_value < VIF_THRESHOLD_GOOD:
        return "良好 (無共線性)"
    elif vif_value < VIF_THRESHOLD_MODERATE:
        return "中等 (可能有共線性)"
    else:
        return "嚴重 (存在高度共線性)"


def run_vif_analysis(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    vif_df = compute_vif(X, numeric_cols)
    vif_df["interpretation"] = vif_df["VIF"].apply(interpret_vif)

    print("\n[VIF 分析結果]")
    print(vif_df.to_string(index=False))

    severe = vif_df[vif_df["VIF"] >= VIF_THRESHOLD_MODERATE]
    if not severe.empty:
        print(f"\n⚠ 以下特徵存在共線性問題:")
        for _, row in severe.iterrows():
            print(f"  - {row['feature']}: VIF={row['VIF']:.2f}")

    return vif_df
