#!/usr/bin/env python3
"""Overlap benchmark figure: median slowdown of every GEMM along the compute chain.

Usage: figures/overlap_timeline.py TSV ... [-o OUT.pdf]

Inputs are chain-position tables (chain_position.tsv written by overlap/analyze.py). One row per
AllGather configuration, one column per model. Each panel is labelled with the number
of GEMMs slowed by more than 1% and the median slowdown of those GEMMs.
"""
import argparse
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COL = "process_all_slowdown_pct_median_median"

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "grid.linewidth": 0.4, "grid.color": "#d9d9d9", "pdf.fonttype": 42, "ps.fonttype": 42,
})
CE_C, SYM_C, LEG_C, INK, GREY = "#00441b", "#238b45", "#74c476", "#333333", "#777777"
# (model, column title, GEMMs shown)
MODELS = [
    ("qwen3_30b_a3b", "Qwen3-30B MoE, MXFP8, 144 MiB AG", 100),
    ("llama3_8b", "Llama-3-8B, FP8, 26 MiB AG", 96),
]
# (config, label, color)
ROWS = [
    ("zero", "CE (0 CTAs)", CE_C),
    ("legacy", "Ring, 32 CTAs (default)", LEG_C),
    ("default", "Symmetric, 4 CTAs (default)", SYM_C),
    ("cta_32", "Symmetric, 32 CTAs", SYM_C),
]


def load(paths):
    table = {}
    for path in paths:
        for r in csv.DictReader(open(path), delimiter="\t"):
            table.setdefault((r["model"], r["config"]), []).append(r)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "figures", "overlap_timeline.pdf"))
    a = ap.parse_args()
    table = load(a.inputs)

    def rows_for(model, cfg, n_show):
        sel = [r for r in table.get((model, cfg), []) if int(r["global_chain_position"]) <= n_show]
        return sorted(sel, key=lambda r: int(r["global_chain_position"]))

    fig, axes = plt.subplots(len(ROWS), len(MODELS), figsize=(3.45, 0.36 * len(ROWS)), sharex="col",
                             sharey=True, gridspec_kw=dict(hspace=0.1, wspace=0.05))
    # One y-scale for every panel; headroom so the in-panel labels clear the tallest bars.
    peak = max([float(r[COL]) for model, _, n in MODELS for cfg, _, _ in ROWS
                for r in rows_for(model, cfg, n)] + [0.0])
    ymax = 1.5 * peak
    for g, (model, title, n_show) in enumerate(MODELS):
        for row, (cfg, label, color) in enumerate(ROWS):
            ax = axes[row, g]
            sel = rows_for(model, cfg, n_show)
            ax.set_xlim(0.5, n_show + 0.5)
            ax.set_ylim(0, ymax)
            ax.set_yticks([0, 40])
            ax.tick_params(labelsize=5, length=1.5, pad=1)
            ax.grid(True, axis="y")
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(title, loc="center", fontsize=5.6, pad=2)
            if not sel:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center",
                        fontsize=7, color=GREY)
                continue
            xs = [int(r["global_chain_position"]) for r in sel]
            # Display floor of 0%; the statistics below use the signed values.
            ax.bar(xs, [max(float(r[COL]), 0.0) for r in sel], width=0.8, color=color, zorder=2)
            hit = sorted(v for v in (float(r[COL]) for r in sel) if v > 1.0)
            stats = f"{len(hit)}/{len(sel)} slowed >1%" + (f", median {statistics.median(hit):.1f}%" if hit else "")
            box = dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.75)
            ax.text(0.99, 0.94, label, transform=ax.transAxes, ha="right", va="top", fontsize=4.6,
                    fontweight="bold", color="black", zorder=4, bbox=box)
            ax.annotate(stats, (0.99, 0.94), xycoords="axes fraction", xytext=(0, -5.4),
                        textcoords="offset points", ha="right", va="top", fontsize=4.2, color=INK,
                        linespacing=1.0, zorder=4, bbox=box)
        axes[-1, g].set_xlabel("GEMM index", fontsize=5.8, labelpad=1)
    fig.supylabel("Median per-GEMM slowdown (%)", fontsize=5.8, x=0.005)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.92, bottom=0.12)
    fig.savefig(a.out, dpi=300)
    print("wrote", os.path.relpath(a.out))


if __name__ == "__main__":
    main()
