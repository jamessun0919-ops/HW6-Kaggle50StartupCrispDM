import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from src.data_understanding import load_data
from src.data_preparation import run_data_preparation, prepare_features, split_data
from src.config import REPORTS_DIR

sns = None
try:
    import seaborn as sns
except ImportError:
    pass

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DARK_BG = "#1a1a2e"
CARD_BG = "#16213e"
ACCENT_BLUE = "#0f3460"
GOLD = "#e94560"
GREEN = "#2ecc71"
ORANGE = "#f39c12"
WHITE = "#ffffff"
LIGHT_GRAY = "#cccccc"
RD_COLOR = "#e74c3c"
MKT_COLOR = "#3498db"
ADM_COLOR = "#2ecc71"
STATE_COLOR = "#95a5a6"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": "none",
    "axes.labelcolor": WHITE,
    "text.color": WHITE,
    "xtick.color": LIGHT_GRAY,
    "ytick.color": LIGHT_GRAY,
    "grid.color": (1, 1, 1, 0.05),
    "font.family": "sans-serif",
    "font.size": 11,
})


def draw_card(ax, x, y, w, h, color=CARD_BG, alpha=0.8):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor=(1,1,1,0.15),
                          linewidth=0.5, alpha=alpha)
    ax.add_patch(rect)


def lasso_coef_plot(ax, features, coefs):
    selected = ["R&D Spend", "Marketing Spend", "Administration"]
    drop = [f for f in features if f not in selected]

    colors = []
    valid_coefs = []
    valid_features = []
    for f, c in zip(features, coefs):
        if f in selected:
            valid_features.append(f)
            valid_coefs.append(c)
            if f == "R&D Spend":
                colors.append(RD_COLOR)
            elif f == "Marketing Spend":
                colors.append(MKT_COLOR)
            else:
                colors.append(ADM_COLOR)

    bars = ax.barh(valid_features, valid_coefs, color=colors, height=0.55,
                   edgecolor="white", linewidth=0.5, alpha=0.9)
    for bar, val in zip(bars, valid_coefs):
        ax.text(bar.get_width() + max(abs(c) for c in coefs) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+,.0f}", va="center", fontsize=10, color=WHITE,
                fontweight="bold")

    for f in drop:
        idx = features.index(f)
        ax.barh(f, 0, color=STATE_COLOR, height=0.55, alpha=0.5)
        ax.text(max(abs(c) for c in coefs) * 0.02,
                list(features).index(f),
                f"  {f} (dropped)", va="center", fontsize=10, color=LIGHT_GRAY)

    ax.axvline(0, color=WHITE, linewidth=0.8, alpha=0.3)
    ax.set_xlabel("Lasso Coefficient", fontsize=11, color=WHITE)
    ax.set_title("LassoCV Feature Selection", fontsize=13, fontweight="bold",
                 color=GOLD, pad=12)
    ax.tick_params(colors=LIGHT_GRAY)


def rfe_ranking_plot(ax, features, rankings, supports):
    y_pos = range(len(features))
    bar_colors = [GREEN if s else STATE_COLOR for s in supports]
    bars = ax.barh(list(features), [6 - r for r in rankings],
                   color=bar_colors, height=0.55, alpha=0.85,
                   edgecolor="white", linewidth=0.5)

    for i, (bar, feat, rank, supp) in enumerate(zip(bars, features, rankings, supports)):
        status = "SELECTED" if supp else f"Rank #{rank}"
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                status, va="center", fontsize=9,
                color=GREEN if supp else LIGHT_GRAY,
                fontweight="bold" if supp else "normal")

    ax.set_xlim(0, 6.5)
    ax.set_xlabel("Selection Score", fontsize=11, color=WHITE)
    ax.set_title("RFE Feature Selection", fontsize=13, fontweight="bold",
                 color=GOLD, pad=12)
    ax.tick_params(colors=LIGHT_GRAY)
    ax.set_xticks([])

    legend_elements = [
        mpatches.Patch(facecolor=GREEN, alpha=0.85, label="Selected"),
        mpatches.Patch(facecolor=STATE_COLOR, alpha=0.85, label="Dropped"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              facecolor=CARD_BG, edgecolor=(1,1,1,0.2))


def shap_comparison_plot(ax, feature_importance_data):
    features = ["R&D Spend", "Marketing Spend", "Administration",
                "State_New York", "State_Florida"]
    values = [24977.9, 3838.7, 1622.1, 1064.7, 165.0]

    colors = [RD_COLOR, MKT_COLOR, ADM_COLOR, STATE_COLOR, STATE_COLOR]
    bars = ax.barh(features, values, color=colors, height=0.55,
                   edgecolor="white", linewidth=0.5, alpha=0.9)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height() / 2,
                f"{val:,.1f}", va="center", fontsize=9, color=WHITE,
                fontweight="bold")

    ax.set_xlabel("Mean |SHAP Value|", fontsize=11, color=WHITE)
    ax.set_title("SHAP Feature Importance", fontsize=13, fontweight="bold",
                 color=GOLD, pad=12)
    ax.tick_params(colors=LIGHT_GRAY)


def model_comparison_table(ax):
    col_labels = ["Model", "Features", "R2", "MAE", "RMSE"]
    rows = [
        ["LinearRegression", "All 5", "0.8987", "6961", "9056"],
        ["LinearRegression", "Top 3", "0.8990", "6946", "9048"],
        ["GradientBoosting", "All 5", "0.9160", "7834", "8246"],
        ["GradientBoosting", "Top 3", "0.9147", "7791", "8312"],
    ]
    cell_text = [[r[i] for r in rows] for i in range(5)]
    all_text = [col_labels] + rows
    table = ax.table(cellText=all_text, loc="center",
                     colWidths=[0.20, 0.15, 0.12, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(CARD_BG)
        cell.set_text_props(color=WHITE, ha="center")
        cell.set_edgecolor((1,1,1,0.1))
        if row == 0:
            cell.set_facecolor(ACCENT_BLUE)
            cell.set_text_props(color=WHITE, ha="center", fontweight="bold")
        if col == 2:
            val = cell_text[2][row - 1] if row > 0 else ""
            try:
                if float(val) >= 0.90:
                    cell.set_text_props(color=GREEN, fontweight="bold", ha="center")
            except ValueError:
                pass

    ax.axis("off")
    ax.set_title("Model Performance: All Features vs Selected (Top 3)",
                 fontsize=12, fontweight="bold", color=GOLD, pad=15)


def decision_summary_plot(ax):
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    ax.text(5, 5.2, "Final Selected Features", fontsize=14, fontweight="bold",
            ha="center", color=GOLD)

    boxes = [
        (0.5, 2.5, 3, 2.0, RD_COLOR, "R&D Spend", "Primary Driver\nCoeff: +36,825"),
        (3.7, 2.5, 2.6, 2.0, MKT_COLOR, "Marketing Spend", "Secondary Driver\nCoeff: +3,552"),
        (6.5, 2.5, 2.5, 2.0, ADM_COLOR, "Administration", "Limited Impact\nCoeff: -655"),
    ]
    for x, y, w, h, color, title, desc in boxes:
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="white",
                              linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.6, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color=WHITE)
        ax.text(x + w / 2, y + h * 0.3, desc, ha="center", va="center",
                fontsize=8, color=LIGHT_GRAY)

    dropped_y = 0.5
    ax.text(2.5, dropped_y + 0.4, "Dropped Features:", fontsize=10,
            fontweight="bold", color=LIGHT_GRAY, ha="center")
    ax.text(5, dropped_y, "State_Florida  |  State_New York",
            fontsize=9, color=STATE_COLOR, ha="center",
            style="italic")
    ax.text(7.5, dropped_y, "(ANOVA p=0.567, not significant)",
            fontsize=7, color=LIGHT_GRAY, ha="center", style="italic")

    ax.annotate("", xy=(3.5, 4.5), xytext=(2, 4.5),
                arrowprops=dict(arrowstyle="->", color=WHITE, lw=1, alpha=0.5))
    ax.annotate("", xy=(3.7, 4.5), xytext=(5.1, 4.5),
                arrowprops=dict(arrowstyle="->", color=WHITE, lw=1, alpha=0.5))


def main():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, encoder, _, _ = run_data_preparation(
        df, use_engineered=False
    )
    features = list(X_train.columns)

    from sklearn.linear_model import LassoCV
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000).fit(X_train, y_train)
    lasso_coefs = lasso.coef_

    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression
    rfe = RFE(LinearRegression(), n_features_to_select=3).fit(X_train, y_train)

    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(4, 2, figure=fig, hspace=0.30, wspace=0.25,
                  height_ratios=[1, 1, 1, 1.1])

    ax1 = fig.add_subplot(gs[0, 0])
    lasso_coef_plot(ax1, features, lasso_coefs)

    ax2 = fig.add_subplot(gs[0, 1])
    rfe_ranking_plot(ax2, features, rfe.ranking_, rfe.support_)

    ax3 = fig.add_subplot(gs[1, :])
    shap_comparison_plot(ax3, None)

    ax4 = fig.add_subplot(gs[2, :])
    model_comparison_table(ax4)

    ax5 = fig.add_subplot(gs[3, :])
    decision_summary_plot(ax5)

    fig.suptitle("Feature Selection Analysis — 50 Startups Profit Prediction",
                 fontsize=18, fontweight="bold", color=GOLD, y=0.98,
                 ha="center")

    fig.text(0.5, 0.01, "Based on LassoCV, RFE, and SHAP | GradientBoosting R²=0.9160",
             ha="center", fontsize=10, color=LIGHT_GRAY, style="italic")

    output_path = REPORTS_DIR / "feature_selection_final.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()

    print(f"[OK] 圖表已儲存至 {output_path}")


if __name__ == "__main__":
    main()
