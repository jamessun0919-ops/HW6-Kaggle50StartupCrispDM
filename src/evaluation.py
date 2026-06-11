import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from src.config import CV_FOLDS, R2_MINIMUM


METRICS = {"R2": r2_score, "MAE": mean_absolute_error, "RMSE": lambda y, p: np.sqrt(mean_squared_error(y, p))}


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    }


def cross_validate_model(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="r2")
    return {
        "mean_r2": scores.mean(),
        "std_r2": scores.std(),
        "scores": scores.tolist(),
    }


def evaluate_all_models(trained_models, X_train, y_train, X_test, y_test):
    results = []
    cv_results = []

    for name, model in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test)
        cv = cross_validate_model(model, X_train, y_train)
        metrics["Model"] = name
        metrics["CV_R2_Mean"] = cv["mean_r2"]
        metrics["CV_R2_Std"] = cv["std_r2"]
        results.append(metrics)
        cv_results.append(cv)

        status = "✓ PASS" if metrics["R2"] >= R2_MINIMUM else "✗ FAIL"
        print(f"  {status} | {name:20s} | R²={metrics['R2']:.4f} | MAE={metrics['MAE']:.2f} | RMSE={metrics['RMSE']:.2f} | CV_R²={cv['mean_r2']:.4f}±{cv['std_r2']:.4f}")

    result_df = pd.DataFrame(results)
    result_df = result_df[["Model", "R2", "MAE", "RMSE", "CV_R2_Mean", "CV_R2_Std"]]
    return result_df


def find_best_model(result_df, metric="R2"):
    best_idx = result_df[metric].idxmax()
    best = result_df.iloc[best_idx]
    print(f"\n🏆 最佳模型: {best['Model']} ({metric}={best[metric]:.4f})")
    return best["Model"]
