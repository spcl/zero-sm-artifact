#!/usr/bin/env python3
"""End-to-end FSDP figure: recipes x (platform, arm) heatmap of the speedup over legacy, mean and SD over paired
blocks, with the estimated saving over a 15T-token pretraining run.

Usage: figures/e2e_fsdp.py SPEEDUP_CSV [-o OUT.pdf]
SPEEDUP_CSV is the output of e2e/fsdp/analyze.py.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import Polygon, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "grid.linewidth": 0.4, "grid.color": "#d9d9d9", "pdf.fonttype": 42, "ps.fonttype": 42,
    "hatch.linewidth": 0.6,
})
CE_COLOR = "#006d2c"

# Rows top to bottom, grouped by pacing on GB200: recipe -> (label, GPUs, tokens per step = GBS x sequence length).
GROUPS = [
    ("Host-bound", ["nemotron15b-h", "llama70b-h", "qwen3-32b-h", "qwen3-next-h"]),
    ("GPU-bound", ["deepseek-v3-g", "nemotron15b-g1", "nemotron15b-g2", "qwen3-32b-g2",
                   "llama70b-g", "llama405b-g", "qwen3-235b-g", "qwen3-32b-g1"]),
]
RECIPES = {
    "llama405b-g": ("Llama-405B-G", 64, 64 * 8192), "llama70b-g": ("Llama-70B-G", 64, 128 * 8192),
    "llama70b-h": ("Llama-70B-H", 64, 64 * 8192), "qwen3-32b-g1": ("Qwen3-32B-G1", 64, 256 * 4096),
    "qwen3-32b-g2": ("Qwen3-32B-G2", 32, 64 * 4096), "qwen3-32b-h": ("Qwen3-32B-H", 64, 64 * 4096),
    "nemotron15b-g1": ("Nemotron-15B-G1", 32, 256 * 4096), "nemotron15b-g2": ("Nemotron-15B-G2", 32, 128 * 4096),
    "nemotron15b-h": ("Nemotron-15B-H", 32, 32 * 4096), "qwen3-235b-g": ("Qwen3-235B-G", 64, 64 * 16384),
    "qwen3-next-h": ("Qwen3-Next-H", 64, 64 * 4096), "deepseek-v3-g": ("DeepSeek-V3-G", 32, 64 * 4096),
}
COLS = [("gb200", "symmetric_st", "Sym.\nST"), ("gb200", "symmetric_stmc", "Sym.\nSTMC"),
        ("gb200", "ce_unicast", "CE\nunicast"), ("gb200", "ce_multicast", "CE\nmulticast"),
        ("b200", "symmetric_stmc", "Sym.\nSTMC"), ("b200", "ce_multicast", "CE\nmulticast")]

# Cost of one token = price per GPU-hour x GPUs x legacy step time / tokens per step; an arm saves that cost times its
# step-time reduction over a full run. CoreWeave on-demand prices; run size = Llama 3 pretraining budget.
PRICE_PER_GPU_HOUR = {"gb200": 10.50, "b200": 8.60}
RUN_TOKENS = 15e12
LIM = 4.0
BEST_MARK_ON_LIGHT, BEST_MARK_ON_DARK = "#00441b", "#c7e9c0"


def run_saving_usd(platform, recipe, legacy_ms, speedup_pct):
    _, gpus, tokens = RECIPES[recipe]
    usd_per_token = PRICE_PER_GPU_HOUR[platform] * gpus * legacy_ms / 1000 / 3600 / tokens
    return usd_per_token * RUN_TOKENS * (1 - 1 / (1 + speedup_pct / 100))


def fmt_usd(x):
    sign = "−" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x / 1e6:.1f}M" if x >= 0.95e6 else f"{sign}${x / 1e3:.0f}k"


def slow_fast_cmap():
    return LinearSegmentedColormap.from_list("slow_fast", [(0.0, "#a50f15"), (0.25, "#fc8d59"), (0.5, "#f7f7f7"),
                                                           (0.75, "#74c476"), (1.0, "#00441b")])


def heat_cell(ax, x, y, cell, cmap, norm, best, usd):
    if cell is None:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc="#f0f0f0", ec="white", lw=0.8))
        ax.text(x, y, "–", ha="center", va="center", fontsize=6, color="#999999")
        return
    m, sd = cell
    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc=cmap(norm(max(-LIM, min(LIM, m)))), ec="white", lw=0.8))
    dark = abs(m) > 0.62 * LIM
    ink, sub = ("white", "#eeeeee") if dark else ("#111111", "#555555")
    if best:
        ax.add_patch(Polygon([(x + 0.5, y - 0.5), (x + 0.33, y - 0.5), (x + 0.5, y - 0.27)], closed=True,
                             fc=BEST_MARK_ON_DARK if dark else BEST_MARK_ON_LIGHT, ec="none", zorder=3))
    mean_txt = "0.0" if abs(m) < 0.05 else f"{m:+.1f}"
    group = HPacker(children=[
        TextArea(mean_txt, textprops=dict(fontsize=5.7, color=ink, fontweight="bold" if best else "normal")),
        TextArea(f"±{sd:.1f}", textprops=dict(fontsize=4.0, color=sub))], align="baseline", pad=0, sep=1.2)
    ax.add_artist(AnnotationBbox(group, (x, y - 0.15), frameon=False, box_alignment=(0.5, 0.5), pad=0, zorder=4))
    ax.text(x, y + 0.28, fmt_usd(usd), ha="center", va="center", fontsize=4.6, color=sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("speedup_csv")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "figures/e2e_fsdp.pdf"))
    a = ap.parse_args()
    cells, legacy = {}, {}
    for r in csv.DictReader(open(a.speedup_csv)):
        cells[(r["platform"], r["recipe"], r["arm"])] = (float(r["speedup_pct"]), float(r["sd_pct"]))
        legacy[(r["platform"], r["recipe"])] = float(r["legacy_step_ms"])

    rows, headers, y = [], [], 0.0
    for title, members in GROUPS:
        headers.append((y, title))
        y += 0.62
        for rec in members:
            rows.append((y, rec))
            y += 1.0
    cmap, norm = slow_fast_cmap(), TwoSlopeNorm(vmin=-LIM, vcenter=0.0, vmax=LIM)
    fig, ax = plt.subplots(figsize=(3.5, 0.17 * y + 0.6))
    n_gb = sum(c[0] == "gb200" for c in COLS)
    for ry, rec in rows:
        # Bold value and corner mark: best arm per platform at printed precision, if faster than legacy.
        best = set()
        for plat in ("gb200", "b200"):
            means = [(round(cells[(p, rec, arm)][0], 1), j) for j, (p, arm, _) in enumerate(COLS)
                     if p == plat and (p, rec, arm) in cells]
            top = max((m for m, _ in means), default=0.0)
            best |= {j for m, j in means if top > 0 and m == top}
        for j, (plat, arm, _) in enumerate(COLS):
            cell = cells.get((plat, rec, arm))
            usd = run_saving_usd(plat, rec, legacy[(plat, rec)], cell[0]) if cell else None
            heat_cell(ax, j, ry, cell, cmap, norm, j in best, usd)
    for hy, text in headers:
        ax.text(-0.5, hy, text, ha="left", va="center", fontsize=6.3, style="italic", color="#333333")
    ax.axvline(n_gb - 0.5, color="black", lw=1.0)
    ax.set_xlim(-0.5, len(COLS) - 0.5)
    ax.set_ylim(y - 0.5, headers[0][0] - 0.3)
    ax.set_yticks([ry for ry, _ in rows])
    ax.set_yticklabels([RECIPES[rec][0] for _, rec in rows], fontsize=6.0)
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels([c[2] for c in COLS], fontsize=5.6, linespacing=1.0)
    for tick, (_, arm, _) in zip(ax.xaxis.get_major_ticks(), COLS):
        if arm.startswith("ce_"):
            for lab in (tick.label1, tick.label2):
                lab.set_color(CE_COLOR)
                lab.set_fontweight("bold")
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for name, x0, x1 in (("GB200 NVL72", -0.5, n_gb - 0.5), ("B200 NVL8", n_gb - 0.5, len(COLS) - 0.5)):
        ax.annotate(name, ((x0 + x1) / 2, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 13.5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7)
    fig.tight_layout(pad=0.25)
    fig.savefig(a.out, dpi=300)
    print("wrote", os.path.relpath(a.out))


if __name__ == "__main__":
    main()
