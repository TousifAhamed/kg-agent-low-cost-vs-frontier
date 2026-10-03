"""Generate the two paper figures from the real result tables (no fabricated numbers).
Palette: validated dataviz reference (blue #2a78d6 = agent/KG systems, gray #898781 = text baselines).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

BLUE, GRAY, GREEN = "#2a78d6", "#898781", "#008300"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": ["DejaVu Sans", "sans-serif"], "svg.fonttype": "none"})

# ---- Figure 1: main normalized-EM comparison ----
systems = ["S1\nLLM-only", "S2\nText-RAG", "S3\nKG-retrieval", "S4\n(full agent)",
           "S4 no\ncompute_metric", "S4 no\nguardrail"]
norm_em = [0.0278, 0.5833, 0.2778, 0.8981, 0.7500, 0.8611]
colors = [GRAY, GRAY, GRAY, BLUE, BLUE, BLUE]

fig, ax = plt.subplots(figsize=(6.75, 2.6), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
bars = ax.bar(systems, norm_em, color=colors, width=0.6, zorder=3)
for b, v in zip(bars, norm_em):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
            fontsize=8.5, color=INK)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Normalized Exact-Match", fontsize=9, color=INK2)
ax.set_title("Baselines and ablations, small backbone (36 EM items)",
             fontsize=9.5, color=INK, loc="left")
ax.tick_params(axis="x", labelsize=8, colors=INK)
ax.tick_params(axis="y", labelsize=8, colors=MUTED)
ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.spines["left"].set_color(BASE); ax.spines["bottom"].set_color(BASE)
legend_handles = [
    Patch(facecolor=GRAY, label="Non-agentic baseline (S1, S2, S3)"),
    Patch(facecolor=BLUE, label="Full KG agent and ablations (S4, S4 variants)"),
]
ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=8, labelcolor=INK2)
fig.tight_layout()
fig.savefig(OUT / "fig_main_results.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: RQ2 robustness under KG perturbation ----
dropout_x = [0, 30, 60]
dropout_y = [0.8981, 0.6944, 0.7778]
noise_x = [0, 30, 60]
noise_y = [0.8981, 0.8333, 0.8889]
s2_baseline = 0.5833

fig, ax = plt.subplots(figsize=(6.75, 2.6), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.plot(dropout_x, dropout_y, marker="o", color=BLUE, linewidth=2, label="Edge dropout", zorder=3)
# weight_noise() perturbs CRM connectivity weights only (src/agent/perturb.py), not sensor readings
ax.plot(noise_x, noise_y, marker="s", color=GREEN, linewidth=2,
        label="Connectivity-weight noise (relative σ)", zorder=3)
ax.axhline(s2_baseline, color=GRAY, linewidth=1.5, linestyle="--", zorder=2,
           label="S2 Text-RAG (undegraded)")
ax.set_xticks([0, 30, 60])
ax.set_xlabel("Perturbation level (dropout rate or relative σ, %)", fontsize=9, color=INK2)
ax.set_ylabel("Normalized Exact-Match", fontsize=9, color=INK2)
ax.set_ylim(0, 1.0)
ax.set_title("RQ2: edge dropout and connectivity-weight noise (small backbone, single draws)",
             fontsize=9.5, color=INK, loc="left")
ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK2)
ax.tick_params(axis="both", labelsize=8, colors=MUTED)
ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.spines["left"].set_color(BASE); ax.spines["bottom"].set_color(BASE)
fig.tight_layout()
fig.savefig(OUT / "fig_robustness.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

print("wrote fig_main_results.png, fig_robustness.png")
