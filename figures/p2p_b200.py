#!/usr/bin/env python3
"""Fig. p2p_b200: send/recv vs. put (and VCCL) on B200, NVLink (solid) and one 400 Gb/s rail (dashed).
Outer axis: bandwidth vs. message size (mean of 10 trials, band = +-1 std); inset: small-message latency.

Usage: figures/p2p_b200.py P2P_CSV [-o OUT.pdf]   (P2P_CSV from microbench/analysis/build_csv.py)
"""
import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from microbench_style import (NIC_BW, NVLINK_BW, ROOT, STYLE, bound, draw, load, mfc, save, series,
                              size_axis)

ARMS = ["NCCL put", "NCCL send/recv", "VCCL SM-free", "VCCL"]
LABEL = {"NCCL put": "NCCL put (zero-SM)"}
DOMAIN_LS = {"scale-up": "-", "scale-out": "--"}


def main(csv_path, out):
    groups = defaultdict(list)
    for r in load(csv_path):
        groups[(r["domain"], r["arm"])].append(r)

    fig, ax = plt.subplots(figsize=(3.45, 1.3))
    ins = ax.inset_axes([0.60, 0.17, 0.38, 0.39])
    for dom, ls in DOMAIN_LS.items():
        for arm in ARMS:
            if not groups[(dom, arm)]:
                continue
            st = STYLE[arm]
            x, y, sd, t, _ = series(groups[(dom, arm)], factor=1.0)
            draw(ax, x, y, sd, st, ls=ls)
            keep = [i for i, s in enumerate(x) if s <= 64 * 1024]
            ins.plot([x[i] for i in keep], [t[i] for i in keep], color=st["color"], ls=ls, lw=0.9,
                     marker=st["marker"], ms=2.2, mfc=mfc(st), mec=st["color"], mew=0.6, zorder=st["z"])

    ax.set_yscale("log")
    ax.set_ylim(3e-4, 5000)
    size_axis(ax, 8, 4 * 2**30, step=3)
    ax.set_xlim(6, 5.8 * 2**30)
    ax.set_xlabel("Message size (bytes)")
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=6.8, labelpad=1)
    ax.grid(True, which="major")
    bound(ax, NVLINK_BW, "NVLink5: 900 GB/s")
    bound(ax, NIC_BW, "IB NDR: 50 GB/s")

    size_axis(ins, 8, 64 * 1024, step=4)
    ins.set_ylabel("Lat. (µs)", fontsize=5, labelpad=1)
    ins.tick_params(labelsize=5.5, pad=1)
    ins.set_ylim(0, 75)
    ins.grid(True, lw=0.3)
    ins.patch.set_alpha(0.92)

    handles = [Line2D([], [], color=STYLE[a]["color"], marker=STYLE[a]["marker"], ms=3.5, mfc=mfc(STYLE[a]),
                      mec=STYLE[a]["color"], lw=1.1, label=LABEL.get(a, a)) for a in ARMS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.75), ncol=1, frameon=False,
              handlelength=1.5, fontsize=5.5, labelspacing=0.15, columnspacing=0.8, handletextpad=0.4,
              borderaxespad=0.3, markerscale=0.8)
    fig.tight_layout(pad=0.3)
    save(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "figures", "p2p_b200.pdf"))
    a = ap.parse_args()
    main(a.csv, a.out)
