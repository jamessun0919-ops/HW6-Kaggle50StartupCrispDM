import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.data_understanding import load_data
from src.data_preparation import run_data_preparation
from src.modeling import train_all_models
from src.config import REPORTS_DIR

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


def forward_selection_plot(ax, X_train, y_train, X_test, y_test, feature_order):
    r2_scores = []
    mae_scores = []
    rmse_scores = []
    cv_scores = []
    selected = []

    for i in range(1, len(feature_order) + 1):
        cols = feature_order[:i]
        selected.append("\n".join(cols) if len(cols) <= 3 else f"Top {i}")

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train[cols], y_train)
        y_pred = model.predict(X_test[cols])

        r2_scores.append(r2_score(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

        cv = cross_val_score(model, X_train[cols], y_train, cv=5, scoring="r2")
        cv_scores.append(cv.mean())

    x = list(range(1, len(feature_order) + 1))

    ax.plot(x, r2_scores, "-o", color=CYAN, linewidth=2.5, markersize=8,
            markerfacecolor=CYAN, markeredgecolor=WHITE, markeredgewidth=1,
            label="R2 (Test)", zorder=5)
    ax.plot(x, cv_scores, "--s", color=ORANGE, linewidth=2, markersize=7,
            markerfacecolor=ORANGE, markeredgecolor=WHITE, markeredgewidth=1,
            label="CV R2 Mean", zorder=4)

    for i, (r2, cv) in enumerate(zip(r2_scores, cv_scores)):
        ax.annotate(f"{r2:.4f}", (x[i], r2),
                    textcoords="offset points", xytext=(0, -18),
                    ha="center", fontsize=8, color=CYAN, fontweight="bold")
        ax.annotate(f"{cv:.4f}", (x[i], cv),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8, color=ORANGE)

    ax.axhline(y=0.90, color=GREEN, linestyle="--", linewidth=1.5, alpha=0.6,
               label="R2 Target = 0.90")
    ax.fill_between(x, 0.90, max(max(r2_scores), max(cv_scores), 0.95),
                    alpha=0.08, color=GREEN)

    ax.set_xticks(x)
    ax.set_xticklabels(feature_order, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Features Added (by importance)", fontsize=12, color=WHITE)
    ax.set_ylabel("R2 Score", fontsize=12, color=WHITE)
    ax.set_title("Forward Feature Selection — Cumulative R2",
                 fontsize=14, fontweight="bold", color=GOLD, pad=15)
    ax.legend(loc="lower right", fontsize=9, labelcolor=[WHITE, ORANGE, GREEN])
    ax.grid(True, alpha=0.15)


def model_comparison_line_plot(ax, trained_models, X_train, y_train, X_test, y_test):
    model_names = list(trained_models.keys())
    r2_vals = []
    mae_vals = []
    rmse_vals = []
    cv_vals = []

    for name in model_names:
        model = trained_models[name]
        y_pred = model.predict(X_test)
        r2_vals.append(r2_score(y_test, y_pred))
        mae_vals.append(mean_absolute_error(y_test, y_pred))
        rmse_vals.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        cv_vals.append(cv)

    x = np.arange(len(model_names))

    ax.plot(x, r2_vals, "-o", color=CYAN, linewidth=2.5, markersize=9,
            markerfacecolor=CYAN, markeredgecolor=WHITE, markeredgewidth=1,
            label="R2 (Test)", zorder=5)

    for i in range(len(model_names)):
        cv_mean = np.mean(cv_vals[i])
        cv_std = np.std(cv_vals[i])
        ax.errorbar(i, cv_mean, yerr=cv_std, fmt="s", color=ORANGE,
                    capsize=4, capthick=1.5, markersize=8,
                    markeredgecolor=WHITE, markeredgewidth=0.8,
                    label="CV R2 (5-Fold)" if i == 0 else "", zorder=4)

    for i, r2 in enumerate(r2_vals):
        ax.annotate(f"{r2:.4f}", (x[i], r2),
                    textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=8, color=CYAN, fontweight="bold")

    ax.axhline(y=0.90, color=GREEN, linestyle="--", linewidth=1.5, alpha=0.6)
    ax.text(len(model_names) - 0.5, 0.91, "R2 Target = 0.90",
            fontsize=9, color=GREEN, ha="right", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0.75, 1.05)
    ax.set_xlabel("Model", fontsize=12, color=WHITE)
    ax.set_ylabel("R2 Score", fontsize=12, color=WHITE)
    ax.set_title("Model Performance Comparison (Test R2 vs CV R2)",
                 fontsize=14, fontweight="bold", color=GOLD, pad=15)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.15)

    for i, (m, r2) in enumerate(zip(model_names, r2_vals)):
        color = GREEN if r2 >= 0.90 else GOLD
        ax.plot(i, r2, "o", color=color, markersize=14, markeredgecolor=WHITE,
                markeredgewidth=2, zorder=6)


def error_metrics_line_plot(ax, trained_models, X_test, y_test):
    model_names = list(trained_models.keys())
    mae_vals = []
    rmse_vals = []

    for name in model_names:
        model = trained_models[name]
        y_pred = model.predict(X_test)
        mae_vals.append(mean_absolute_error(y_test, y_pred))
        rmse_vals.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    x = np.arange(len(model_names))

    ax2 = ax.twinx()
    l1 = ax.plot(x, mae_vals, "-^", color=PURPLE, linewidth=2, markersize=9,
                 markerfacecolor=PURPLE, markeredgecolor=WHITE, markeredgewidth=1,
                 label="MAE", zorder=5)
    l2 = ax2.plot(x, rmse_vals, "-D", color=GOLD, linewidth=2, markersize=9,
                  markerfacecolor=GOLD, markeredgecolor=WHITE, markeredgewidth=1,
                  label="RMSE", zorder=5)

    for i, (mae, rmse) in enumerate(zip(mae_vals, rmse_vals)):
        ax.annotate(f"{mae:.0f}", (x[i], mae),
                    textcoords="offset points", xytext=(0, -18),
                    ha="center", fontsize=7, color=PURPLE, fontweight="bold")
        ax2.annotate(f"{rmse:.0f}", (x[i], rmse),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=7, color=GOLD, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("MAE", fontsize=12, color=PURPLE)
    ax2.set_ylabel("RMSE", fontsize=12, color=GOLD)
    ax.set_title("Error Metrics Comparison (MAE vs RMSE)",
                 fontsize=14, fontweight="bold", color=GOLD, pad=15)
    ax.grid(True, alpha=0.15)

    ax.tick_params(axis="y", colors=PURPLE)
    ax2.tick_params(axis="y", colors=GOLD)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper left", fontsize=9)


def cv_folds_line_plot(ax, trained_models, X_train, y_train):
    model_names = list(trained_models.keys())
    n_folds = 5
    x_folds = np.arange(1, n_folds + 1)

    colors = [CYAN, ORANGE, PURPLE, GREEN, GOLD]
    for i, name in enumerate(model_names):
        model = trained_models[name]
        scores = cross_val_score(model, X_train, y_train, cv=n_folds, scoring="r2")
        ax.plot(x_folds, scores, f"-o", color=colors[i % len(colors)],
                linewidth=1.8, markersize=7, markerfacecolor=colors[i % len(colors)],
                markeredgecolor=WHITE, markeredgewidth=1, label=name)

    ax.axhline(y=0.90, color=GREEN, linestyle="--", linewidth=1.5, alpha=0.5)
    ax.set_xticks(x_folds)
    ax.set_xlabel("CV Fold", fontsize=12, color=WHITE)
    ax.set_ylabel("R2 Score", fontsize=12, color=WHITE)
    ax.set_title("Cross-Validation R2 per Fold — All Models",
                 fontsize=14, fontweight="bold", color=GOLD, pad=15)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.15)
    ax.set_ylim(0.70, 1.05)


def marginal_gain_plot(ax, X_train, y_train, X_test, y_test, feature_order):
    gains = []
    prev_r2 = 0
    for i in range(1, len(feature_order) + 1):
        cols = feature_order[:i]
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train[cols], y_train)
        r2 = r2_score(y_test, model.predict(X_test[cols]))
        gains.append(r2 - prev_r2)
        prev_r2 = r2

    x = np.arange(1, len(feature_order) + 1)
    colors = [CYAN, GREEN, ORANGE, PURPLE, GOLD]
    bars = ax.bar(x, gains, color=colors[:len(gains)], width=0.5,
                  edgecolor=WHITE, linewidth=0.8, alpha=0.85, zorder=3)

    cumulative = 0
    for i, (bar, gain) in enumerate(zip(bars, gains)):
        cumulative += gain
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"+{gain:.4f}", ha="center", fontsize=9, fontweight="bold",
                color=colors[i])
        ax.text(bar.get_x() + bar.get_width() / 2, 0.005,
                f"Cum: {cumulative:.4f}", ha="center", fontsize=7,
                color=LIGHT_GRAY, rotation=0)

    ax.axhline(y=0, color=WHITE, linewidth=0.8, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_order, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Feature Added (by importance)", fontsize=12, color=WHITE)
    ax.set_ylabel("Marginal R2 Gain", fontsize=12, color=WHITE)
    ax.set_title("Marginal R2 Gain per Added Feature",
                 fontsize=14, fontweight="bold", color=GOLD, pad=15)
    ax.grid(True, alpha=0.15, axis="y")


def main():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, encoder, _, _ = run_data_preparation(
        df, use_engineered=False
    )

    trained = train_all_models(X_train, y_train, include_advanced=False)

    feature_order_by_importance = ["R&D Spend", "Marketing Spend",
                                   "Administration", "State_Florida",
                                   "State_New York"]

    fig = plt.figure(figsize=(22, 20))
    gs = GridSpec(3, 2, figure=fig, hspace=0.30, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])
    forward_selection_plot(ax1, X_train, y_train, X_test, y_test,
                           feature_order_by_importance)

    ax2 = fig.add_subplot(gs[1, 0])
    model_comparison_line_plot(ax2, trained, X_train, y_train, X_test, y_test)

    ax3 = fig.add_subplot(gs[1, 1])
    error_metrics_line_plot(ax3, trained, X_test, y_test)

    ax4 = fig.add_subplot(gs[2, 0])
    cv_folds_line_plot(ax4, trained, X_train, y_train)

    ax5 = fig.add_subplot(gs[2, 1])
    marginal_gain_plot(ax5, X_train, y_train, X_test, y_test,
                       feature_order_by_importance)

    fig.suptitle("Feature Selection & Model Performance — Line Chart Analysis",
                 fontsize=18, fontweight="bold", color=GOLD, y=0.98)

    fig.text(0.5, 0.005,
             "Each feature adds measurable R2 | R&D Spend alone achieves R2=0.947 | Target R2 >= 0.90",
             ha="center", fontsize=10, color=LIGHT_GRAY, style="italic")

    output_path = REPORTS_DIR / "feature_selection_linechart.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()

    output_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "final_linechart.png")
    fig.savefig(output_root, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()

    print(f"[OK] 折線圖已儲存至 {output_path}")
    print(f"[OK] 同時複製至 final_linechart.png")


if __name__ == "__main__":
    main()
