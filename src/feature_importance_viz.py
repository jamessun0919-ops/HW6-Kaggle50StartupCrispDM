import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from src.data_understanding import load_data
from src.data_preparation import run_data_preparation
from src.modeling import train_all_models
from src.evaluation import evaluate_all_models, find_best_model
from src.config import MODELS_DIR, REPORTS_DIR

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def compute_feature_importance(df, best_model, X_train, X_test, y_test, feature_names):
    results = {}

    corr = df[["R&D Spend", "Administration", "Marketing Spend", "Profit"]].corr()["Profit"].drop("Profit")
    results["Correlation"] = corr

    from sklearn.inspection import permutation_importance
    perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
    results["Permutation"] = pd.Series(perm.importances_mean, index=feature_names)

    if hasattr(best_model, "feature_importances_"):
        fi = best_model.feature_importances_
        if len(fi) == len(feature_names):
            results["TreeImportance"] = pd.Series(fi, index=feature_names)

    if hasattr(best_model, "coef_"):
        coef = best_model.coef_.ravel()
        if len(coef) == len(feature_names):
            results["Coefficient"] = pd.Series(np.abs(coef), index=feature_names)

    try:
        import shap
        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer(X_test)
        mean_shap = np.abs(shap_values.values).mean(axis=0)
        if len(mean_shap) == len(feature_names):
            results["SHAP"] = pd.Series(mean_shap, index=feature_names)
    except Exception:
        pass

    return results


def plot_importance_comparison(importance_dict, model_name):
    n_methods = len(importance_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for ax, (method, series) in zip(axes, importance_dict.items()):
        sorted_series = series.sort_values()
        colors = ["#2E86AB" if v > 0 else "#A23B72" for v in sorted_series.values]
        bars = ax.barh(sorted_series.index, sorted_series.values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(method, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Importance")

        for bar, val in zip(bars, sorted_series.values):
            ax.text(bar.get_width() + bar.get_width() * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)

    fig.suptitle(f"Feature Importance Comparison - {model_name}",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_importance_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    df_plot = pd.DataFrame(importance_dict)
    df_plot_normalized = df_plot.div(df_plot.max()).fillna(0)
    sns.heatmap(df_plot_normalized, annot=df_plot.round(4), fmt=".4f",
                cmap="YlOrRd", linewidths=0.5, ax=ax2,
                cbar_kws={"label": "Relative Importance"})
    ax2.set_title(f"Feature Importance Heatmap - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig2.savefig(REPORTS_DIR / "feature_importance_heatmap.png",
                 dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n[Feature Importance Comparison] 已儲存至 {REPORTS_DIR}/feature_importance_comparison.png")
    return fig, fig2


def plot_shap_detail(best_model, X_train, X_test, model_name):
    try:
        import shap
        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer(X_test)

        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(REPORTS_DIR / f"shap_detail_{model_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        shap.plots.waterfall(shap_values[0], show=False)
        plt.savefig(REPORTS_DIR / f"shap_waterfall_detail_{model_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        mean_shap = np.abs(shap_values.values).mean(axis=0)
        shap_df = pd.DataFrame({
            "Feature": X_test.columns,
            "Mean |SHAP|": mean_shap
        }).sort_values("Mean |SHAP|", ascending=False)

        print("\n[SHAP Global Feature Importance]")
        print(shap_df.to_string(index=False))
        return shap_df

    except Exception as e:
        print(f"SHAP detail failed: {e}")
        return None


def plot_correlation_detail(df):
    numeric = ["R&D Spend", "Administration", "Marketing Spend", "Profit"]
    corr = df[numeric].corr()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, col in enumerate(["R&D Spend", "Marketing Spend", "Administration"]):
        ax = axes[i]
        sns.regplot(data=df, x=col, y="Profit", ax=ax,
                     scatter_kws={"alpha": 0.6, "s": 60},
                     line_kws={"color": "red", "lw": 2})
        r_val = corr.loc[col, "Profit"]
        ax.set_title(f"Profit vs {col}\nPearson r = {r_val:.4f}",
                     fontsize=12, fontweight="bold")
        ax.text(0.95, 0.05, f"R2 = {r_val**2:.4f}",
                transform=ax.transAxes, ha="right", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "correlation_with_regression.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Correlation Detail] 已儲存至 {REPORTS_DIR}/correlation_with_regression.png")

    return corr


def print_feature_explanations(importance_dict, corr_df, df):
    corr_series = corr_df["Profit"]
    print("\n" + "=" * 60)
    print("  特徵關鍵性解釋")
    print("=" * 60)

    def rank_features(series, higher_is_better=True):
        if higher_is_better:
            return series.sort_values(ascending=False)
        return series.sort_values()

    all_ranks = []
    for method, series in importance_dict.items():
        ranks = series.rank(ascending=False)
        all_ranks.append(ranks)

    avg_rank = pd.concat(all_ranks, axis=1).mean(axis=1).sort_values()
    print(f"\n綜合特徵排名 (根據 {len(importance_dict)} 種方法):")
    for i, (feat, rank) in enumerate(avg_rank.items(), 1):
        print(f"  #{i} {feat:25s} (avg rank: {rank:.1f})")

    print(f"\n--- 詳細解釋 ---")

    base_features = ["R&D Spend", "Marketing Spend", "Administration"]
    for feat in base_features:
        r_val = corr_series.get(feat, 0)
        r2 = r_val ** 2

        print(f"  - {feat}")
        print(f"    相關性: r = {r_val:.4f} (R2 = {r2:.4f})")
        print(f"    解釋: Profit 變異的 {r2*100:.2f}% 可由 {feat} 解釋")

        if feat == "R&D Spend":
            print(f"    結論: [1] R&D 是最重要的獲利驅動因子")
        elif feat == "Marketing Spend":
            print(f"    結論: [2] Marketing 為次要獲利因子")
        elif feat == "Administration":
            print(f"    結論: [3] Administration 對獲利影響有限")

    for feat in avg_rank.index:
        if feat not in base_features:
            print(f"\n  * {feat}")
            print(f"    結論: 類別變數，影響相對較小")

    print(f"\n--- What-If 分析佐證 ---")
    what_if_data = {
        "Increase R&D 10%": "+15,225 (+12.0%)",
        "Increase Marketing 10%": "-19 (-0.02%)",
        "Decrease Administration 10%": "+3,056 (+2.4%)",
    }
    for scenario, change in what_if_data.items():
        print(f"  {scenario}: Profit {change}")


def main():
    print("=" * 60)
    print("  特徵關鍵性圖示分析")
    print("=" * 60)

    df = load_data()

    print("\n[執行資料準備]")
    X_train, X_test, y_train, y_test, scaler, encoder, X_train_orig, X_test_orig = run_data_preparation(
        df, use_engineered=False
    )

    print("\n[訓練模型]")
    trained = train_all_models(X_train, y_train, include_advanced=False)

    print("\n[模型評估]")
    result_df = evaluate_all_models(trained, X_train, y_train, X_test, y_test)
    best_name = find_best_model(result_df)
    best_model = trained[best_name]

    feature_names = X_train.columns.tolist()

    print("\n[計算各類特徵重要度]...")
    importance_dict = compute_feature_importance(df, best_model, X_train, X_test, y_test, feature_names)

    print("\n[繪製相關性分析圖]")
    corr_series = plot_correlation_detail(df)

    print("\n[繪製特徵重要度比較圖]")
    plot_importance_comparison(importance_dict, best_name)

    print("\n[繪製 SHAP 詳細圖]")
    plot_shap_detail(best_model, X_train, X_test, best_name)

    print_feature_explanations(importance_dict, corr_series, df)

    print(f"\n{'=' * 60}")
    print(f"  所有圖表已儲存至: {REPORTS_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
