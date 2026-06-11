import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from src.config import RAW_DATA_PATH, REPORTS_DIR, ANOVA_ALPHA


def load_data(path=RAW_DATA_PATH):
    df = pd.read_csv(path)
    print(f"資料集形狀: {df.shape}")
    return df


def quality_checks(df):
    print("=" * 50)
    print("資料品質檢查")
    print("=" * 50)

    print("\n[資料型態]")
    df.info()

    print(f"\n[缺失值]\n{df.isnull().sum()}")
    print(f"\n[重複記錄]: {df.duplicated().sum()}")

    print(f"\n[基本統計量]")
    print(df.describe().round(2))

    return {"missing": df.isnull().sum().to_dict(), "duplicates": int(df.duplicated().sum())}


def univariate_analysis(df):
    numeric_cols = ["R&D Spend", "Administration", "Marketing Spend", "Profit"]
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i, 0])
        axes[i, 0].set_title(f"{col} - Histogram + KDE")

        sns.boxplot(y=df[col], ax=axes[i, 1])
        axes[i, 1].set_title(f"{col} - Box Plot")

        axes[i, 2].text(0.5, 0.5,
                        f"{col}\n\nMean: {df[col].mean():.2f}\nStd: {df[col].std():.2f}\n"
                        f"Min: {df[col].min():.2f}\n25%: {df[col].quantile(0.25):.2f}\n"
                        f"50%: {df[col].median():.2f}\n75%: {df[col].quantile(0.75):.2f}\n"
                        f"Max: {df[col].max():.2f}",
                        ha="center", va="center", fontsize=11,
                        transform=axes[i, 2].transAxes)
        axes[i, 2].axis("off")

    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "univariate_analysis.png", dpi=150)
    plt.show()


def bivariate_analysis(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [("R&D Spend", "Profit"), ("Marketing Spend", "Profit"), ("Administration", "Profit")]
    titles = ["Profit vs R&D Spend", "Profit vs Marketing Spend", "Profit vs Administration"]

    for i, ((x, y), title) in enumerate(zip(pairs, titles)):
        sns.scatterplot(data=df, x=x, y=y, ax=axes[i])
        axes[i].set_title(title)

        slope = np.corrcoef(df[x], df[y])[0, 1]
        axes[i].text(0.05, 0.9, f"Corr: {slope:.3f}", transform=axes[i].transAxes,
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "bivariate_analysis.png", dpi=150)
    plt.show()


def correlation_analysis(df):
    corr = df[["R&D Spend", "Administration", "Marketing Spend", "Profit"]].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".3f",
                linewidths=0.5, square=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.show()

    print("\n[與 Profit 的相關性]")
    print(corr["Profit"].drop("Profit").sort_values(ascending=False).round(4))
    return corr


def state_analysis(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="State", y="Profit")
    plt.title("Profit Distribution by State")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "state_boxplot.png", dpi=150)
    plt.show()

    groups = [group["Profit"].values for _, group in df.groupby("State")]
    f_stat, p_value = f_oneway(*groups)
    print(f"\n[ANOVA - Profit by State]")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"顯著性 (α={ANOVA_ALPHA}): {'顯著' if p_value < ANOVA_ALPHA else '不顯著'}")

    return {"f_statistic": float(f_stat), "p_value": float(p_value),
            "significant": bool(p_value < ANOVA_ALPHA)}


def outlier_analysis(df):
    numeric_cols = ["R&D Spend", "Administration", "Marketing Spend", "Profit"]
    results = {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        results[col] = {"count": len(outliers), "indices": outliers.index.tolist(),
                        "lower": lower, "upper": upper}

        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(f"{col} (離群值: {len(outliers)})")

    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "outlier_analysis.png", dpi=150)
    plt.show()

    for col, res in results.items():
        print(f"{col}: {res['count']} 個離群值")
    return results


def run_data_understanding():
    df = load_data()
    print(df.head(), "\n")
    quality_checks(df)
    univariate_analysis(df)
    bivariate_analysis(df)
    correlation_analysis(df)
    state_analysis(df)
    outlier_analysis(df)
    return df
