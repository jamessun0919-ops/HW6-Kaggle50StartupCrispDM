import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from src.config import SEED


def lasso_feature_selection(X_train, y_train):
    lasso = LassoCV(cv=5, random_state=SEED, max_iter=10000)
    lasso.fit(X_train, y_train)

    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": lasso.coef_
    })
    coef_df["selected"] = np.abs(coef_df["coefficient"]) > 1e-6
    coef_df = coef_df.sort_values("coefficient", key=abs, ascending=False)

    selected = coef_df[coef_df["selected"]]["feature"].tolist()
    print(f"\n[LassoCV 特徵選擇]")
    print(f"選擇的特徵 ({len(selected)}): {selected}")
    print(f"最佳 alpha: {lasso.alpha_:.6f}")

    return lasso, coef_df, selected


def rfe_feature_selection(X_train, y_train, n_features=None):
    if n_features is None:
        n_features = max(1, X_train.shape[1] // 2)

    estimator = LinearRegression()
    rfe = RFE(estimator, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)

    ranking_df = pd.DataFrame({
        "feature": X_train.columns,
        "selected": rfe.support_,
        "rank": rfe.ranking_
    }).sort_values("rank")

    selected = ranking_df[ranking_df["selected"]]["feature"].tolist()
    print(f"\n[RFE 特徵選擇]")
    print(f"選擇的特徵 ({len(selected)}): {selected}")

    return rfe, ranking_df, selected


def run_feature_selection(X_train, y_train):
    lasso_result = lasso_feature_selection(X_train, y_train)
    rfe_result = rfe_feature_selection(X_train, y_train)
    return lasso_result, rfe_result
