import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
rmse = [7714, 8206, 8243, 9015, 9056]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, rmse, "-D", color="#9b59b6", linewidth=2.8,
        markersize=10, markerfacecolor="#9b59b6",
        markeredgecolor="white", markeredgewidth=2, zorder=5)

ax.fill_between(x, min(rmse) - 200, rmse, alpha=0.1, color="#9b59b6")

for i, v in enumerate(rmse):
    ax.annotate(f"{v:,}", (x[i], v),
                textcoords="offset points", xytext=(0, -18),
                ha="center", fontsize=10, color="#9b59b6", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(features, fontsize=11)
ax.set_ylim(7000, 9500)
ax.set_xlim(-0.3, 4.3)
ax.set_xlabel("Feature Set (cumulative addition)", fontsize=13, labelpad=10)
ax.set_ylabel("RMSE", fontsize=13, labelpad=10)
ax.set_title("Forward Feature Selection — RMSE Progression",
             fontsize=15, fontweight="bold", pad=15)
ax.grid(True, alpha=0.4)
ax.ticklabel_format(axis="y", style="plain")

fig.tight_layout()
fig.savefig(REPORTS_DIR / "rmse_linechart.png", dpi=200, bbox_inches="tight")
fig.savefig("rmse_linechart.png", dpi=200, bbox_inches="tight")
plt.close()
print("[OK] rmse_linechart.png")
