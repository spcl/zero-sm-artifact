#!/usr/bin/env python3
"""End-to-end pipeline-parallel figure: speedup of one-sided puts over two-sided Send/Recv per recipe (columns, ordered
by pipeline-bubble share b) and platform (rows), mean and SD over paired blocks, with the estimated saving over a
15T-token pretraining run.

Usage: figures/e2e_pp.py SPEEDUP_CSV [-o OUT.pdf]
SPEEDUP_CSV is the output of e2e/pp/analyze.py.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "grid.linewidth": 0.4, "grid.color": "#d9d9d9", "pdf.fonttype": 42, "ps.fonttype": 42,
    "hatch.linewidth": 0.6,
})

# Columns left to right (increasing b, then label): recipe -> (label, GPUs, tokens per step = GBS x sequence length).
RECIPES = {
    "llama70b": ("Llama-70B", 64, 64 * 8192), "qwen3-32b": ("Qwen3-32B", 64, 64 * 4096),
    "deepseek-v3": ("DeepSeek-V3", 32, 64 * 4096), "llama405b": ("Llama-405B", 64, 64 * 8192),
    "nemotron15b": ("Nemotron-15B", 32, 32 * 4096), "qwen3-235b": ("Qwen3-235B", 64, 64 * 4096),
    "qwen3-next": ("Qwen3-Next", 64, 64 * 4096),
}
PLATFORMS = [("gb200", "GB200 NVL72"), ("b200", "B200 NVL8")]

# Cost of one token = price per GPU-hour x GPUs x two-sided step time / tokens per step; one-sided saves that cost
# times its step-time reduction over a full run. CoreWeave on-demand prices; run size = Llama 3 pretraining budget.
PRICE_PER_GPU_HOUR = {"gb200": 10.50, "b200": 8.60}
RUN_TOKENS = 15e12
LIM = 10.0


def run_saving_usd(platform, recipe, base_ms, speedup_pct):
    _, gpus, tokens = RECIPES[recipe]
    usd_per_token = PRICE_PER_GPU_HOUR[platform] * gpus * base_ms / 1000 / 3600 / tokens
    return usd_per_token * RUN_TOKENS * (1 - 1 / (1 + speedup_pct / 100))


def fmt_usd(x):
    sign = "−" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x / 1e6:.1f}M" if x >= 0.95e6 else f"{sign}${x / 1e3:.0f}k"


def slow_fast_cmap():
    return LinearSegmentedColormap.from_list("slow_fast", [(0.0, "#a50f15"), (0.25, "#fc8d59"), (0.5, "#f7f7f7"),
                                                           (0.75, "#74c476"), (1.0, "#00441b")])


def heat_cell(ax, x, y, cell, cmap, norm, usd):
    if cell is None:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc="#f0f0f0", ec="white", lw=0.8))
        ax.text(x, y, "–", ha="center", va="center", fontsize=6, color="#999999")
        return
    m, sd = cell
    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc=cmap(norm(max(-LIM, min(LIM, m)))), ec="white", lw=0.8))
    dark = abs(m) > 0.62 * LIM
    ink, sub = ("white", "#eeeeee") if dark else ("#111111", "#555555")
    mean_txt = "0.0" if abs(m) < 0.05 else f"{m:+.1f}"
    group = HPacker(children=[
        TextArea(mean_txt, textprops=dict(fontsize=5.2, color=ink)),
        TextArea(f"±{sd:.1f}", textprops=dict(fontsize=4.0, color=sub))], align="baseline", pad=0, sep=1.2)
    ax.add_artist(AnnotationBbox(group, (x, y - 0.15), frameon=False, box_alignment=(0.5, 0.5), pad=0, zorder=4))
    ax.text(x, y + 0.28, fmt_usd(usd), ha="center", va="center", fontsize=4.6, color=sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("speedup_csv")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "figures/e2e_pp.pdf"))
    a = ap.parse_args()
    cells, base, bubble = {}, {}, {}
    for r in csv.DictReader(open(a.speedup_csv)):
        key = (r["platform"], r["recipe"])
        cells[key] = (float(r["speedup_pct"]), float(r["sd_pct"]))
        base[key] = float(r["two_sided_step_ms"])
        bubble[r["recipe"]] = float(r["bubble_pct"])

    cmap, norm = slow_fast_cmap(), TwoSlopeNorm(vmin=-LIM, vcenter=0.0, vmax=LIM)
    fig, ax = plt.subplots(figsize=(3.45, 1.02))
    for j, rec in enumerate(RECIPES):
        if rec in bubble:
            ax.text(j, -0.74, f"{bubble[rec]:.1f}", ha="center", va="center", fontsize=5.8, color="#333333")
        for i, (plat, _) in enumerate(PLATFORMS):
            cell = cells.get((plat, rec))
            usd = run_saving_usd(plat, rec, base[(plat, rec)], cell[0]) if cell else None
            heat_cell(ax, j, i, cell, cmap, norm, usd)
    ax.text(-0.56, -0.74, "Bubble share $b$ (%)", ha="right", va="center", fontsize=5.8, color="#333333")
    ax.set_xlim(-0.5, len(RECIPES) - 0.5)
    ax.set_ylim(len(PLATFORMS) - 0.5, -1.05)
    ax.set_yticks(range(len(PLATFORMS)))
    ax.set_yticklabels([p[1] for p in PLATFORMS], fontsize=6.0)
    ax.set_xticks(range(len(RECIPES)))
    ax.set_xticklabels([v[0] for v in RECIPES.values()], fontsize=5.8, rotation=25, ha="right",
                       rotation_mode="anchor")
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(a.out, dpi=300)
    print("wrote", os.path.relpath(a.out))


if __name__ == "__main__":
    main()
