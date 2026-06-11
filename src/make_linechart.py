import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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

features = ["R&D", "+ Marketing", "+ State_NY", "+ Admin", "+ All 5"]
x = np.arange(len(features))
r2_test = [0.9265, 0.9168, 0.9161, 0.8996, 0.8987]
r2_cv   = [0.9395, 0.9404, 0.9324, 0.9316, 0.9289]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, r2_test, "-o", color="#e74c3c", linewidth=2.8,
        markersize=10, markerfacecolor="#e74c3c",
        markeredgecolor="white", markeredgewidth=2, label="Test R²", zorder=5)

ax.plot(x, r2_cv, "--s", color="#3498db", linewidth=2.5,
        markersize=10, markerfacecolor="#3498db",
        markeredgecolor="white", markeredgewidth=2, label="CV R² (5-fold)", zorder=4)

for i, (t, c) in enumerate(zip(r2_test, r2_cv)):
    ax.annotate(f"{t:.4f}", (x[i], t),
                textcoords="offset points", xytext=(0, -20),
                ha="center", fontsize=9, color="#e74c3c", fontweight="bold")
    ax.annotate(f"{c:.4f}", (x[i], c),
                textcoords="offset points", xytext=(0, 14),
                ha="center", fontsize=9, color="#3498db")

ax.axhline(y=0.90, color="#2ecc71", linestyle="--", linewidth=2, alpha=0.8)
ax.text(4, 0.902, "Target R² = 0.90", fontsize=11, color="#2ecc71",
        fontweight="bold", ha="right", va="bottom")

ax.fill_between(x, 0.90, max(max(r2_test), max(r2_cv)) + 0.02,
                alpha=0.08, color="#2ecc71")

ax.set_xticks(x)
ax.set_xticklabels(features, fontsize=11)
ax.set_ylim(0.86, 0.98)
ax.set_xlim(-0.3, 4.3)
ax.set_xlabel("Feature Set (cumulative addition)", fontsize=13, labelpad=10)
ax.set_ylabel("R² Score", fontsize=13, labelpad=10)
ax.set_title("Forward Feature Selection — Effect on Model Performance",
             fontsize=15, fontweight="bold", pad=15)
ax.legend(loc="lower left", fontsize=11)
ax.grid(True, alpha=0.4)

fig.tight_layout()
fig.savefig(REPORTS_DIR / "linechart.png", dpi=200, bbox_inches="tight")
fig.savefig("linechart.png", dpi=200, bbox_inches="tight")
plt.close()
print("[OK] linechart.png")
