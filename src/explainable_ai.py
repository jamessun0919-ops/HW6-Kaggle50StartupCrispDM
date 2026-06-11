import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from src.config import REPORTS_DIR, SEED


def shap_analysis(model, X_train, X_test, model_name="model"):
    try:
        import shap
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f"SHAP Summary Plot - {model_name}")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"shap_summary_{model_name}.png", dpi=150, bbox_inches="tight")
        plt.show()

        shap.plots.waterfall(shap_values[0], show=False)
        plt.title(f"SHAP Waterfall - Sample 0 - {model_name}")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"shap_waterfall_{model_name}.png", dpi=150, bbox_inches="tight")
        plt.show()

        mean_shap = np.abs(shap_values.values).mean(axis=0)
        feature_importance = pd.DataFrame({
            "feature": X_test.columns,
            "mean_abs_shap": mean_shap
        }).sort_values("mean_abs_shap", ascending=False)

        print(f"\n[SHAP 特徵重要度 - {model_name}]")
        print(feature_importance.to_string(index=False))

        return {"shap_values": shap_values, "feature_importance": feature_importance}

    except Exception as e:
        print(f"SHAP 分析失敗: {e}")
        return None


def permutation_importance_analysis(model, X_test, y_test, model_name="model"):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=SEED)

    imp_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(imp_df["feature"], imp_df["importance_mean"], xerr=imp_df["importance_std"])
    plt.xlabel("Permutation Importance")
    plt.title(f"Permutation Importance - {model_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"permutation_importance_{model_name}.png", dpi=150)
    plt.show()

    print(f"\n[Permutation Importance - {model_name}]")
    print(imp_df.to_string(index=False))

    return imp_df


def run_explainability(best_model, X_train, X_test, y_test, model_name="BestModel"):
    print("\n" + "=" * 50)
    print("SHAP 分析")
    print("=" * 50)
    shap_result = shap_analysis(best_model, X_train, X_test, model_name)

    print("\n" + "=" * 50)
    print("Permutation Importance")
    print("=" * 50)
    perm_result = permutation_importance_analysis(best_model, X_test, y_test, model_name)

    return {"shap": shap_result, "permutation_importance": perm_result}
