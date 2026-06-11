import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectKBest, f_regression, SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.config import REPORTS_DIR, SEED

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = SEED

plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#f8f9fa",
    "axes.edgecolor": "#dee2e6",
    "axes.labelcolor": "#212529",
    "text.color": "#212529",
    "xtick.color": "#495057",
    "ytick.color": "#495057",
    "grid.color": "#e9ecef",
    "font.family": "sans-serif",
    "font.size": 12,
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#dee2e6",
})

df = pd.read_csv("50_Startups.csv")
df_enc = pd.get_dummies(df, columns=["State"], drop_first=True)
y = df_enc["Profit"]
X = df_enc.drop(columns=["Profit"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

feature_names = X.columns.tolist()
n_features = len(feature_names)

def r2_for_features(cols):
    m = LinearRegression().fit(X_train[cols], y_train)
    return float(r2_score(y_test, m.predict(X_test[cols])))

results = {}
n_range = list(range(1, n_features + 1))

def rank_by_abs_coef(coef):
    return np.argsort(np.abs(coef))[::-1]

print("=== RFE ===")
rfe_r2 = []
for k in n_range:
    rfe = RFE(LinearRegression(), n_features_to_select=k).fit(X_train, y_train)
    cols = [f for f, s in zip(feature_names, rfe.support_) if s]
    rfe_r2.append(r2_for_features(cols))
    print(f"  k={k}: {cols} -> R2={rfe_r2[-1]:.4f}")
results["RFE"] = rfe_r2

print("\n=== SelectKBest (f_regression) ===")
skb_r2 = []
for k in n_range:
    skb = SelectKBest(f_regression, k=k).fit(X_train, y_train)
    cols = [feature_names[i] for i in skb.get_support(indices=True)]
    skb_r2.append(r2_for_features(cols))
    print(f"  k={k}: {cols} -> R2={skb_r2[-1]:.4f}")
results["SelectKBest"] = skb_r2

print("\n=== LassoCV ===")
lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000).fit(X_train, y_train)
lasso_rank = rank_by_abs_coef(lasso.coef_)
lasso_r2 = []
for k in n_range:
    cols = [feature_names[i] for i in lasso_rank[:k]]
    lasso_r2.append(r2_for_features(cols))
    print(f"  k={k}: {cols} -> R2={lasso_r2[-1]:.4f}")
results["Lasso"] = lasso_r2

print("\n=== Random Forest ===")
rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE).fit(X_train, y_train)
rf_rank = np.argsort(rf.feature_importances_)[::-1]
rf_r2 = []
for k in n_range:
    cols = [feature_names[i] for i in rf_rank[:k]]
    rf_r2.append(r2_for_features(cols))
    print(f"  k={k}: {cols} -> R2={rf_r2[-1]:.4f}")
results["RandomForest"] = rf_r2

print("\n=== SFS (SequentialFeatureSelector) ===")
sfs = SequentialFeatureSelector(
    LinearRegression(), n_features_to_select=n_features - 1,
    direction="forward", cv=5, n_jobs=-1
).fit(X_train, y_train)
sfs_rank = np.argsort(sfs.support_.astype(int))[::-1]

sfs_r2 = []
for k in n_range:
    if k <= n_features - 1:
        cols = [feature_names[i] for i in sfs_rank[:k]]
    else:
        cols = feature_names
    sfs_r2.append(r2_for_features(cols))
    print(f"  k={k}: {cols} -> R2={sfs_r2[-1]:.4f}")
results["SFS"] = sfs_r2

colors = {
    "SFS": "#2ecc71",
    "RFE": "#e74c3c",
    "SelectKBest": "#3498db",
    "Lasso": "#f39c12",
    "RandomForest": "#9b59b6",
}
markers = {
    "SFS": "o",
    "RFE": "s",
    "SelectKBest": "^",
    "Lasso": "D",
    "RandomForest": "v",
}

fig, ax = plt.subplots(figsize=(12, 7))

for name in ["SFS", "RFE", "SelectKBest", "Lasso", "RandomForest"]:
    ax.plot(n_range, results[name], f"-{markers[name]}",
            color=colors[name], linewidth=2.5, markersize=9,
            markerfacecolor=colors[name], markeredgecolor="white",
            markeredgewidth=1.5, label=name, zorder=5)

for name in ["SFS", "RFE", "SelectKBest", "Lasso", "RandomForest"]:
    vals = results[name]
    ax.annotate(f"{vals[0]:.4f}", (1, vals[0]),
                textcoords="offset points", xytext=(8, -10),
                fontsize=8, color=colors[name], fontweight="bold")
    ax.annotate(f"{vals[-1]:.4f}", (n_features, vals[-1]),
                textcoords="offset points", xytext=(8, -10),
                fontsize=8, color=colors[name], fontweight="bold")

ax.axhline(y=0.90, color="#2ecc71", linestyle="--", linewidth=2, alpha=0.6)
ax.text(n_features + 0.1, 0.902, "Target R² = 0.90", fontsize=10,
        color="#2ecc71", fontweight="bold", va="bottom")

ax.set_xticks(n_range)
ax.set_xlabel("Number of Features Selected", fontsize=13, labelpad=10)
ax.set_ylabel("R² Score", fontsize=13, labelpad=10)
ax.set_ylim(0, 1.0)
ax.set_title("R² by Number of Features — Comparison of Selection Methods",
             fontsize=15, fontweight="bold", pad=15)
ax.legend(loc="lower right", fontsize=11, title="Selection Method",
          title_fontsize=11)
ax.grid(True, alpha=0.4)

fig.tight_layout()
fig.savefig(REPORTS_DIR / "r2_by_method.png", dpi=200, bbox_inches="tight")
fig.savefig("r2_by_method.png", dpi=200, bbox_inches="tight")
plt.close()

print("\n" + "=" * 60)
print("R2 by Number of Features - All Methods")
print("=" * 60)
header = f"{'k':>3s}"
for name in ["SFS", "RFE", "SelectKBest", "Lasso", "RandomForest"]:
    header += f" {name:>14s}"
print(header)
print("-" * (3 + 15 * 5))
for i, k in enumerate(n_range):
    row = f"{k:3d}"
    for name in ["SFS", "RFE", "SelectKBest", "Lasso", "RandomForest"]:
        row += f" {results[name][i]:>14.4f}"
    print(row)

print(f"\n[OK] r2_by_method.png")
