import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "font.family": "sans-serif",
    "font.size": 11,
})

fig, ax = plt.subplots(figsize=(20, 26))
ax.set_xlim(0, 20)
ax.set_ylim(0, 26)
ax.axis("off")

def box(x, y, w, h, color, text, sub_text="", fc="white"):
    main_color = color
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.15",
                          facecolor=fc, edgecolor=main_color,
                          linewidth=2.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.58, text, ha="center", va="center",
            fontsize=10, fontweight="bold", color=main_color)
    if sub_text:
        ax.text(x + w / 2, y + h * 0.28, sub_text, ha="center", va="center",
                fontsize=8, color="#666666")

def arrow(x1, y1, x2, y2, color="#888888"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=2.5, connectionstyle="arc3,rad=0"),
                zorder=2)

def phase_box(x, y, w, h, color, number, title, items, sub_items=None):
    fc = "#ffffff"
    border_color = color
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.12",
                          facecolor=fc, edgecolor=border_color,
                          linewidth=2.5, zorder=3)
    ax.add_patch(rect)

    header = FancyBboxPatch((x, y + h - 0.7), w, 0.7,
                            boxstyle="round,pad=0.05",
                            facecolor=color, edgecolor=color,
                            linewidth=0, zorder=4)
    ax.add_patch(header)
    ax.text(x + w / 2, y + h - 0.35, f"Phase {number}: {title}",
            ha="center", va="center", fontsize=11, fontweight="bold", color="white", zorder=5)

    y_offset = y + h - 1.0
    for item in items:
        ax.text(x + 0.3, y_offset, f"  {item}", ha="left", va="center",
                fontsize=9, color="#333333")
        y_offset -= 0.38

    if sub_items:
        y_offset -= 0.1
        for s in sub_items:
            ax.text(x + 0.5, y_offset, s, ha="left", va="center",
                    fontsize=8, color="#888888", style="italic")
            y_offset -= 0.32

# ============================================================
# Phase 1: Business Understanding
# ============================================================
phase_box(0.5, 22.5, 8.5, 3.2, "#e74c3c", "1",
          "Business Understanding",
          ["Objectives: Find key Profit drivers",
           "Hypotheses: H1~H4",
           "Business Questions: 6 questions",
           "Success Criteria: R2 >= 0.90"],
          ["- R&D is primary driver (H1)",
           "- Marketing is secondary (H2)",
           "- Admin has limited impact (H3)",
           "- State not significant (H4)"])

phase_box(11, 22.5, 8.5, 3.2, "#e67e22", "2",
          "Data Understanding (EDA)",
          ["Load data (50 rows, 5 columns)",
           "Quality checks: missing, duplicates",
           "Univariate: histograms + boxplots",
           "Bivariate: scatter + correlation",
           "State analysis: ANOVA test",
           "Outlier detection: IQR method"],
          ["- R&D r=0.973 (very high)",
           "- Marketing r=0.748 (moderate)",
           "- Admin r=0.201 (low)",
           "- State ANOVA p=0.567 (not sig)"])

arrow(4.75, 22.5, 4.75, 21.2, "#e74c3c")
arrow(15.25, 22.5, 15.25, 21.2, "#e67e22")
ax.text(4.75, 21.8, "Business Context", ha="center", fontsize=7, color="#e74c3c")
ax.text(15.25, 21.8, "Data Insights", ha="center", fontsize=7, color="#e67e22")

y_base = 18.3

# Box: Data Preparation
phase_box(0.5, y_base, 5.8, 2.8, "#27ae60", "3",
          "Data Preparation",
          ["Feature Engineering",
           "  TotalSpend, ROI, RDRatio",
           "One-Hot Encoding (State)",
           "Train/Test Split (80/20)",
           "StandardScaler"],
          None)

# Box: Multicollinearity
phase_box(7, y_base, 5.5, 2.8, "#2980b9", "4",
          "Multicollinearity",
          ["VIF Analysis"],
          ["- All features VIF < 5",
           "- No multicollinearity"])

# Box: Feature Selection
phase_box(13.2, y_base, 6.3, 2.8, "#8e44ad", "5",
          "Feature Selection",
          ["LassoCV: automatic selection",
           "RFE: recursive elimination"],
          ["- Lasso selects 3 features",
           "- RFE selects 2 features",
           "- R&D + Marketing key"])

arrow(3.4, y_base + 2.8, 3.4, y_base + 3.8, "#27ae60")
arrow(9.75, y_base + 2.8, 9.75, y_base + 3.8, "#2980b9")
arrow(16.35, y_base + 2.8, 16.35, y_base + 3.8, "#8e44ad")
ax.text(3.4, y_base + 3.3, "Prepare", ha="center", fontsize=7, color="#27ae60")
ax.text(9.75, y_base + 3.3, "Check", ha="center", fontsize=7, color="#2980b9")
ax.text(16.35, y_base + 3.3, "Select", ha="center", fontsize=7, color="#8e44ad")

y_mid = 13.8

# Arrow from Phase 3,4,5 to Phase 6
arrow(3.4, y_base, 3.4, y_mid + 2.8)
arrow(9.75, y_base, 9.75, y_mid + 2.8)
arrow(16.35, y_base, 16.35, y_mid + 2.8)

# Phase 6: Modeling
phase_box(0.5, y_mid, 9.5, 3.8, "#d35400", "6",
          "Modeling",
          ["Baseline: LinearRegression",
           "Regularized: Ridge, Lasso",
           "Ensemble: RandomForest (200 trees)",
           "Boosting: GradientBoosting (200 trees)",
           "Advanced: XGBoost, LightGBM"],
          ["- 5 models default, 7 with advanced",
           "- All trained on scaled features"])

# Phase 7: Evaluation
phase_box(10.5, y_mid, 9, 3.8, "#c0392b", "7",
          "Evaluation",
          ["Metrics: R2, MAE, RMSE",
           "Cross-Validation: 5-fold",
           "Model Comparison Table",
           "Best Model: GradientBoosting"],
          ["- R2=0.9160 (passes 0.90 target)",
           "- RandomForest R2=0.9061",
           "- LinearRegression R2=0.8987"])

arrow(5.25, y_mid + 3.8, 5.25, y_mid + 4.8)
arrow(15, y_mid + 3.8, 15, y_mid + 4.8)
ax.text(5.25, y_mid + 4.3, "Train", ha="center", fontsize=7, color="#d35400")
ax.text(15, y_mid + 4.3, "Evaluate", ha="center", fontsize=7, color="#c0392b")

arrow(10, y_mid + 1.9, 10.5, y_mid + 1.9, "#d35400")

y_bot = 9.3

# Arrow from 6,7 to 8
arrow(5.25, y_mid, 5.25, y_bot + 2.8)
arrow(15, y_mid, 15, y_bot + 2.8)

# Phase 8: Explainable AI
phase_box(0.5, y_bot, 9.5, 3.0, "#16a085", "8",
          "Explainable AI",
          ["SHAP: Global + Local explanation",
           "  Summary plot, Waterfall plot",
           "Permutation Feature Importance",
           "R&D is #1 in ALL methods"],
          ["- SHAP R&D: 24,978",
           "- Permutation R&D: 1.89"])

# Phase 9: Business Analytics
phase_box(10.5, y_bot, 9, 3.0, "#2c3e50", "9",
          "Business Analytics",
          ["What-If Analysis: 3 scenarios",
           "Budget: High vs Low Profit",
           "ROI Analysis: top companies",
           "K-Means Clustering: 3 types"],
          ["- R&D+10% -> Profit+12%",
           "- Admin-10% -> Profit+2.4%"])

arrow(5.25, y_bot + 3.0, 5.25, 7.8)
arrow(15, y_bot + 3.0, 15, 7.8)

# Phase 10: Insights
phase_box(0.5, 4.8, 19, 3.2, "#7f8c8d", "10",
          "Business Insights & Recommendations",
          ["Insight 1: R&D is the most important profit driver (R=0.973, explains 94.6% variance)",
           "Insight 2: Marketing is secondary (R=0.748), but ROI increase is marginal",
           "Insight 3: Administration has limited impact (R=0.201), cost center",
           "Insight 4: State is not statistically significant (ANOVA p=0.567)",
           "Insight 5: High-profit companies invest differently (MarketingDriven cluster)"],
          None)

arrow(10, 7.8, 10, 8.0, "#7f8c8d")

# Phase 11: Deployment
phase_box(0.5, 0.5, 19, 4.0, "#e84393", "11",
          "Deployment",
          ["Streamlit Dashboard: interactive prediction & analysis",
           "  - Profit Prediction (input R&D, Mkt, Admin, State)",
           "  - What-If Simulation (R&D+10%, Mkt+10%, Admin-10%)",
           "  - SHAP Explanation (feature contribution breakdown)",
           "  - Feature Importance (interactive charts)",
           "  - ROI Dashboard (budget allocation analysis)"],
          None)

arrow(10, 4.8, 10, 4.5, "#e84393")
arrow(10, 0.5, 10, 0.0)

ax.text(10, -0.3, "Deliverables: CRISP-DM Report | EDA Report | ML Models | SHAP Report | Business Insights | Streamlit Dashboard",
        ha="center", fontsize=9, color="#888888", style="italic")

ax.text(0.2, 25.5, "Kaggle 50 Startups — CRISP-DM + Machine Learning Pipeline",
        fontsize=16, fontweight="bold", color="#2c3e50")

ax.text(0.2, 25.0, "Target: Predict Profit | Problem: Regression | Dataset: 50 Startups | Best Model: GradientBoosting (R2=0.9160)",
        fontsize=10, color="#666666")

fig.savefig(REPORTS_DIR / "workflow.png", dpi=200, bbox_inches="tight", pad_inches=0.3)
fig.savefig("workflow.png", dpi=200, bbox_inches="tight", pad_inches=0.3)
plt.close()

print("[OK] workflow.png")
