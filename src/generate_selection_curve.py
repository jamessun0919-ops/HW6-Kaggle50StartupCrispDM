import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_understanding import load_data
from src.data_preparation import run_data_preparation
from src.config import REPORTS_DIR

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DARK_BG = "#1a1a2e"
CARD_BG = "#16213e"
ACCENT_BLUE = "#0f3460"
GOLD = "#e94560"
GREEN = "#2ecc71"
CYAN = "#00d2ff"
ORANGE = "#f39c12"
PURPLE = "#9b59b6"
WHITE = "#ffffff"
LIGHT_GRAY = "#cccccc"
RED = "#e74c3c"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": "none",
    "axes.labelcolor": WHITE,
    "text.color": WHITE,
    "xtick.color": LIGHT_GRAY,
    "ytick.color": LIGHT_GRAY,
    "grid.color": (1, 1, 1, 0.06),
    "font.family": "sans-serif",
    "font.size": 11,
    "legend.facecolor": CARD_BG,
    "legend.edgecolor": (1, 1, 1, 0.15),
    "legend.labelcolor": LIGHT_GRAY,
})


def main():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, encoder, _, _ = run_data_preparation(
        df, use_engineered=False
    )

    feature_sets = [
        ["R&D Spend"],
        ["R&D Spend", "Marketing Spend"],
        ["R&D Spend", "Marketing Spend", "State_New York"],
        ["R&D Spend", "Marketing Spend", "State_New York", "Administration"],
        ["R&D Spend", "Marketing Spend", "State_New York", "Administration", "State_Florida"],
    ]

    labels = [
        "1: R&D Only",
        "2: + Marketing",
        "3: + State_NY",
        "4: + Admin",
        "5: All Features",
    ]

    short_labels = [
        "R&D",
        "R&D\n+ Mkt",
        "R&D+Mkt\n+StateNY",
        "R&D+Mkt\n+StateNY\n+Admin",
        "All 5\nFeatures",
    ]

    r2_list = []
    mae_list = []
    rmse_list = []
    cv_mean_list = []
    cv_std_list = []

    for cols in feature_sets:
        model = LinearRegression()
        model.fit(X_train[cols], y_train)
        y_pred = model.predict(X_test[cols])

        r2_list.append(r2_score(y_test, y_pred))
        mae_list.append(mean_absolute_error(y_test, y_pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))

        cv = cross_val_score(model, X_train[cols], y_train, cv=5, scoring="r2")
        cv_mean_list.append(cv.mean())
        cv_std_list.append(cv.std())

    x = np.arange(len(feature_sets))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    line_r2 = ax.plot(x, r2_list, "-o", color=CYAN, linewidth=3,
                      markersize=12, markerfacecolor=CYAN,
                      markeredgecolor=WHITE, markeredgewidth=2,
                      label="Test R2", zorder=5)

    line_cv = ax.plot(x, cv_mean_list, "--s", color=ORANGE, linewidth=2.5,
                      markersize=11, markerfacecolor=ORANGE,
                      markeredgecolor=WHITE, markeredgewidth=1.5,
                      label="CV R2 (5-Fold)", zorder=4)

    ax.fill_between(x,
                     [m - s for m, s in zip(cv_mean_list, cv_std_list)],
                     [m + s for m, s in zip(cv_mean_list, cv_std_list)],
                     alpha=0.15, color=ORANGE, label="CV Std Dev")

    for i, (r2, cv) in enumerate(zip(r2_list, cv_mean_list)):
        ax.annotate(f"R2={r2:.4f}", (x[i], r2),
                    textcoords="offset points", xytext=(0, -22),
                    ha="center", fontsize=10, color=CYAN, fontweight="bold")
        ax.annotate(f"CV={cv:.4f}", (x[i], cv),
                    textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=9, color=ORANGE)

    ax.axhline(y=0.90, color=GREEN, linestyle="--", linewidth=2, alpha=0.7)
    ax.text(4.3, 0.905, "Target R2 = 0.90", fontsize=10, color=GREEN,
            fontweight="bold", ha="left")

    threshold_y = 0.90
    for i, r2 in enumerate(r2_list):
        if r2 >= threshold_y:
            ax.axvspan(i - 0.3, len(x) - 0.5, alpha=0.06, color=GREEN)
            ax.text(i, 0.04, f"✓ PASS", fontsize=11, color=GREEN,
                    fontweight="bold", ha="center", va="bottom")
            break

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.5, 4.8)
    ax.set_xlabel("Feature Set (cumulative)", fontsize=13, color=WHITE, labelpad=10)
    ax.set_ylabel("R2 Score", fontsize=13, color=WHITE, labelpad=10)
    ax.set_title("Forward Feature Selection — R2 Progression",
                 fontsize=15, fontweight="bold", color=GOLD, pad=15)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.12)

    ax2 = axes[1]
    color_mae = PURPLE
    color_rmse = GOLD

    ax2_twin = ax2.twinx()
    l1 = ax2.plot(x, mae_list, "-^", color=color_mae, linewidth=2.5,
                  markersize=11, markerfacecolor=color_mae,
                  markeredgecolor=WHITE, markeredgewidth=1.5,
                  label="MAE", zorder=5)
    l2 = ax2_twin.plot(x, rmse_list, "-D", color=color_rmse, linewidth=2.5,
                       markersize=11, markerfacecolor=color_rmse,
                       markeredgecolor=WHITE, markeredgewidth=1.5,
                       label="RMSE", zorder=5)

    for i, (mae, rmse) in enumerate(zip(mae_list, rmse_list)):
        ax2.annotate(f"{mae:,.0f}", (x[i], mae),
                     textcoords="offset points", xytext=(0, -20),
                     ha="center", fontsize=9, color=color_mae, fontweight="bold")
        ax2_twin.annotate(f"{rmse:,.0f}", (x[i], rmse),
                          textcoords="offset points", xytext=(0, 14),
                          ha="center", fontsize=9, color=color_rmse, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, fontsize=9)
    ax2.set_xlabel("Feature Set (cumulative)", fontsize=13, color=WHITE, labelpad=10)
    ax2.set_ylabel("MAE", fontsize=13, color=color_mae, labelpad=10)
    ax2_twin.set_ylabel("RMSE", fontsize=13, color=color_rmse, labelpad=10)
    ax2.set_title("Error Metrics — MAE & RMSE Progression",
                  fontsize=15, fontweight="bold", color=GOLD, pad=15)
    ax2.grid(True, alpha=0.12)

    ax2.tick_params(axis="y", colors=color_mae)
    ax2_twin.tick_params(axis="y", colors=color_rmse)

    lines = l1 + l2
    legend_labels = [l.get_label() for l in lines]
    ax2.legend(lines, legend_labels, loc="upper right", fontsize=10)

    fig.suptitle("Feature Selection Analysis — Sequential Addition of Features",
                 fontsize=16, fontweight="bold", color=GOLD, y=0.98)

    fig.text(0.5, 0.01,
             "Model: LinearRegression | Target: R2 >= 0.90 | R&D alone explains 94.7% of Profit variance",
             ha="center", fontsize=9, color=LIGHT_GRAY, style="italic")

    output_path = REPORTS_DIR / "selection_curve.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()

    root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "selection_curve.png")
    fig.savefig(root_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()

    print(f"[OK] 已儲存至 {output_path}")

    print("\n=== 各階段結果 ===")
    print(f"{'Features':45s} {'R2':>8s} {'MAE':>10s} {'RMSE':>10s} {'CV_R2':>8s}")
    print("-" * 85)
    for i, cols in enumerate(feature_sets):
        print(f"{labels[i]:45s} {r2_list[i]:8.4f} {mae_list[i]:>10,.0f} {rmse_list[i]:>10,.0f} {cv_mean_list[i]:8.4f}")


if __name__ == "__main__":
    main()
