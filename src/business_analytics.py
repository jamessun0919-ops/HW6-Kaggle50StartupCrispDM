import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from src.config import REPORTS_DIR, SEED


def what_if_analysis(model, X_test_scaled, scaler, feature_columns, df_original_row=None):
    if df_original_row is not None:
        base_row_orig = df_original_row.iloc[0:1].copy()
    else:
        rd_col = feature_columns.index("R&D Spend") if "R&D Spend" in feature_columns else 0
        base_row_orig = pd.DataFrame([{
            "R&D Spend": 100000, "Administration": 120000, "Marketing Spend": 200000
        }])

    scenarios = {
        "Increase_R&D_10%": ("R&D Spend", 1.10),
        "Increase_Marketing_10%": ("Marketing Spend", 1.10),
        "Decrease_Administration_10%": ("Administration", 0.90),
    }

    def predict_from_original(orig_row):
        row = orig_row.copy()
        for col in feature_columns:
            if col not in row.columns:
                row[col] = 0
        row = row[feature_columns]
        scale_cols = [c for c in ["R&D Spend", "Administration", "Marketing Spend"] if c in row.columns]
        if scaler is not None and scale_cols:
            row[scale_cols] = scaler.transform(row[scale_cols])
        return model.predict(row)[0]

    base_pred = predict_from_original(base_row_orig)
    print(f"\n[情境分析] 基準預測 Profit: {base_pred:.2f}")
    results = {"Scenario": ["Baseline"], "Profit": [base_pred], "Change": [0], "Change_%": [0]}

    for scenario_name, (col, factor) in scenarios.items():
        row = base_row_orig.copy()
        if col in row.columns:
            row[col] = row[col] * factor
        pred = predict_from_original(row)
        change = pred - base_pred
        results["Scenario"].append(scenario_name)
        results["Profit"].append(pred)
        results["Change"].append(change)
        results["Change_%"].append(change / base_pred * 100)

    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))

    return result_df


def budget_analysis(df):
    df = df.copy()
    df["TotalSpend"] = df["R&D Spend"] + df["Marketing Spend"] + df["Administration"]
    median_profit = df["Profit"].median()
    high_profit = df[df["Profit"] >= median_profit]
    low_profit = df[df["Profit"] < median_profit]

    categories = ["R&D Spend", "Marketing Spend", "Administration"]
    comparison = pd.DataFrame({
        "Category": categories,
        "HighProfit_Mean": [high_profit[c].mean() for c in categories],
        "HighProfit_Pct": [high_profit[c].mean() / high_profit["TotalSpend"].mean() * 100 for c in categories],
        "LowProfit_Mean": [low_profit[c].mean() for c in categories],
        "LowProfit_Pct": [low_profit[c].mean() / low_profit["TotalSpend"].mean() * 100 for c in categories],
    })

    print("\n[預算配置分析 - 高獲利 vs 低獲利企業]")
    print(comparison.round(2).to_string(index=False))

    comparison_melted = comparison.melt(
        id_vars=["Category"],
        value_vars=["HighProfit_Pct", "LowProfit_Pct"],
        var_name="Group", value_name="Percentage"
    )

    plt.figure(figsize=(10, 5))
    bar_width = 0.35
    x = np.arange(len(categories))
    plt.bar(x - bar_width / 2, comparison["HighProfit_Pct"], bar_width, label="High Profit", alpha=0.8)
    plt.bar(x + bar_width / 2, comparison["LowProfit_Pct"], bar_width, label="Low Profit", alpha=0.8)
    plt.xticks(x, categories)
    plt.ylabel("Spending Percentage (%)")
    plt.title("Budget Allocation: High Profit vs Low Profit Companies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "budget_allocation.png", dpi=150)
    plt.show()

    return comparison


def roi_analysis(df):
    df = df.copy()
    df["TotalSpend"] = df["R&D Spend"] + df["Marketing Spend"] + df["Administration"]
    df["ROI"] = df["Profit"] / df["TotalSpend"].replace(0, np.nan)

    top_roi = df.nlargest(5, "ROI")[["R&D Spend", "Marketing Spend", "Administration", "Profit", "ROI"]]
    print("\n[ROI 分析 - 最高 ROI 企業]")
    print(top_roi.round(2).to_string(index=False))

    return top_roi


def cluster_analysis(df):
    cluster_features = ["R&D Spend", "Marketing Spend", "Administration"]
    X_cluster = df[cluster_features].copy()

    kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=10)
    df_clustered = df.copy()
    df_clustered["Cluster"] = kmeans.fit_predict(X_cluster)

    cluster_labels = {0: "InnovationDriven", 1: "MarketingDriven", 2: "Balanced"}
    df_clustered["ClusterLabel"] = df_clustered["Cluster"].map(cluster_labels)

    cluster_summary = df_clustered.groupby("ClusterLabel")[cluster_features + ["Profit"]].mean()
    print("\n[集群分析 - 各集群平均]")
    print(cluster_summary.round(2).to_string())

    colors = {"InnovationDriven": "red", "MarketingDriven": "blue", "Balanced": "green"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label in cluster_labels.values():
        subset = df_clustered[df_clustered["ClusterLabel"] == label]
        axes[0].scatter(subset["R&D Spend"], subset["Profit"],
                        c=colors[label], label=label, alpha=0.7, s=80)
    axes[0].set_xlabel("R&D Spend")
    axes[0].set_ylabel("Profit")
    axes[0].set_title("Clusters: R&D Spend vs Profit")
    axes[0].legend()

    for label in cluster_labels.values():
        subset = df_clustered[df_clustered["ClusterLabel"] == label]
        axes[1].scatter(subset["Marketing Spend"], subset["Profit"],
                        c=colors[label], label=label, alpha=0.7, s=80)
    axes[1].set_xlabel("Marketing Spend")
    axes[1].set_ylabel("Profit")
    axes[1].set_title("Clusters: Marketing Spend vs Profit")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "cluster_analysis.png", dpi=150)
    plt.show()

    return df_clustered, kmeans


def run_business_analytics(best_model, X_test_scaled, feature_columns, scaler, df_original, X_test_original=None):
    print("\n" + "=" * 50)
    print("What-If 分析")
    print("=" * 50)
    what_if_analysis(best_model, X_test_scaled, scaler, feature_columns, df_original_row=X_test_original)

    print("\n" + "=" * 50)
    print("預算配置分析")
    print("=" * 50)
    budget_analysis(df_original)

    print("\n" + "=" * 50)
    print("ROI 分析")
    print("=" * 50)
    roi_analysis(df_original)

    print("\n" + "=" * 50)
    print("集群分析 (K-Means)")
    print("=" * 50)
    cluster_analysis(df_original)
