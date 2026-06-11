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
from sklearn.metrics import mean_squared_error
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

def rmse_for_features(cols):
    m = LinearRegression().fit(X_train[cols], y_train)
    y_pred = m.predict(X_test[cols])
    return float(np.sqrt(mean_squared_error(y_test, y_pred)))

results = {}
n_range = list(range(1, n_features + 1))

def rank_by_abs_coef(coef):
    return np.argsort(np.abs(coef))[::-1]

print("=== RFE ===")
rfe_rmse = []
for k in n_range:
    rfe = RFE(LinearRegression(), n_features_to_select=k).fit(X_train, y_train)
    cols = [f for f, s in zip(feature_names, rfe.support_) if s]
    rfe_rmse.append(rmse_for_features(cols))
    print(f"  k={k}: {cols} -> RMSE={rfe_rmse[-1]:.0f}")
results["RFE"] = rfe_rmse

print("\n=== SelectKBest (f_regression) ===")
skb_rmse = []
for k in n_range:
    skb = SelectKBest(f_regression, k=k).fit(X_train, y_train)
    cols = [feature_names[i] for i in skb.get_support(indices=True)]
    skb_rmse.append(rmse_for_features(cols))
    print(f"  k={k}: {cols} -> RMSE={skb_rmse[-1]:.0f}")
results["SelectKBest"] = skb_rmse

print("\n=== LassoCV ===")
lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000).fit(X_train, y_train)
lasso_rank = rank_by_abs_coef(lasso.coef_)
lasso_rmse = []
for k in n_range:
    cols = [feature_names[i] for i in lasso_rank[:k]]
    # ensure at least some features have non-zero coef; if all remaining are 0, just use them anyway
    lasso_rmse.append(rmse_for_features(cols))
    print(f"  k={k}: {cols} -> RMSE={lasso_rmse[-1]:.0f}")
results["Lasso"] = lasso_rmse

print("\n=== Random Forest ===")
rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE).fit(X_train, y_train)
rf_rank = np.argsort(rf.feature_importances_)[::-1]
rf_rmse = []
for k in n_range:
    cols = [feature_names[i] for i in rf_rank[:k]]
    rf_rmse.append(rmse_for_features(cols))
    print(f"  k={k}: {cols} -> RMSE={rf_rmse[-1]:.0f}")
results["RandomForest"] = rf_rmse

print("\n=== SFS (SequentialFeatureSelector) ===")
sfs = SequentialFeatureSelector(
    LinearRegression(), n_features_to_select=n_features - 1,
    direction="forward", cv=5, n_jobs=-1
).fit(X_train, y_train)
sfs_rank = np.argsort(sfs.support_.astype(int))[::-1]

sfs_rmse = []
for k in n_range:
    if k <= n_features - 1:
        cols = [feature_names[i] for i in sfs_rank[:k]]
    else:
        cols = feature_names
    sfs_rmse.append(rmse_for_features(cols))
    print(f"  k={k}: {cols} -> RMSE={sfs_rmse[-1]:.0f}")
results["SFS"] = sfs_rmse

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
    for k, v in zip(n_range, results[name]):
        if k == 1:
            ax.annotate(f"{v:,.0f}", (k, v),
                        textcoords="offset points", xytext=(0, -20),
                        ha="center", fontsize=8, color=colors[name], fontweight="bold")
        elif k == n_features:
            ax.annotate(f"{v:,.0f}", (k, v),
                        textcoords="offset points", xytext=(12, 0),
                        ha="left", fontsize=8, color=colors[name], fontweight="bold")

ax.set_xticks(n_range)
ax.set_xlabel("Number of Features Selected", fontsize=13, labelpad=10)
ax.set_ylabel("RMSE", fontsize=13, labelpad=10)
ax.set_title("RMSE by Number of Features — Comparison of Selection Methods",
             fontsize=15, fontweight="bold", pad=15)
ax.legend(loc="lower right", fontsize=11, title="Selection Method",
          title_fontsize=11)
ax.grid(True, alpha=0.4)
ax.ticklabel_format(axis="y", style="plain")

fig.tight_layout()
fig.savefig(REPORTS_DIR / "rmse_by_method.png", dpi=200, bbox_inches="tight")
fig.savefig("rmse_by_method.png", dpi=200, bbox_inches="tight")
plt.close()

print("\n" + "=" * 60)
print("RMSE by Number of Features — All Methods")
print("=" * 60)
header = f"{'k':>3s}"
for name in ["SFS", "RFE", "SelectKBest", "Lasso", "RandomForest"]:
    header += f" {name:>12s}"
print(header)
print("-" * (3 + 13 * 5))
for i, k in enumerate(n_range):
    row = f"{k:3d}"
    for name in ["SFS", "RFE", "SelectKBest", "Lasso", "RandomForest"]:
        row += f" {results[name][i]:>12,.0f}"
    print(row)

print(f"\n[OK] rmse_by_method.png")
