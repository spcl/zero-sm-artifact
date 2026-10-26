#!/usr/bin/env python3
"""Fig. allgather: AllGather bus bandwidth on GB200 (row 1) and B200 (row 2); inset: small-message latency.
Lines are means over 10 trials; bands are +-1 standard deviation.

Usage: figures/allgather.py AG_CSV [-o OUT.pdf]   (AG_CSV from microbench/analysis/build_csv.py)
"""
import argparse
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullLocator

from microbench_style import NIC_BW, NVLINK_BW, ROOT, STYLE, bound, draw, load, mfc, save, series, size_axis

PANELS = [
    ("GB200", "1 node", "GB200: 4 GPUs, 1 Node\n1 NVL domain", 1),
    ("GB200", "4 nodes", "GB200: 16 GPUs, 4 Nodes\n1 NVL domain", 1),
    ("GB200", "18 nodes", "GB200: 72 GPUs, 18 Nodes\n1 NVL domain", 1),
    ("B200", "1 node", "B200: 8 GPUs, 1 Node\n1 NVL domain", 1),
    ("B200", "4 nodes", "B200: 32 GPUs, 4 Nodes\n4 NVL domains", 4),
    ("B200", "8 nodes", "B200: 64 GPUs, 8 Nodes\n8 NVL domains", 8),
]
AG_ARMS = ["NCCL CE", "NCCL STMC", "NCCL legacy", "NCCLX", "MSCCL++"]
NOTE_COLOR = "#c62828"
PANEL_NOTES = {("GB200", "18 nodes"): "MSCCL++ > 16 MiB:\nno valid result"}
LSA_SIZE = {"GB200": None, "B200": 8}   # GB200 panels are one NVLink domain: LSA size = #GPUs
B200_NODE_NIC_BW = 8 * NIC_BW           # 8 x 400 Gb/s rails per node


def ag_network_bound(nodes, gpus_per_node=8):
    """Bus-bandwidth ceiling when each node's NICs receive the (nodes-1)/nodes of the output held by other nodes."""
    n = nodes * gpus_per_node
    return (n - 1) / n * B200_NODE_NIC_BW * nodes / (nodes - 1)


# The "NCCL CE" curve takes, at each size, the measured CE variant that NCCL selects:
#  * multicast vs. unicast: the lower time of the CE cost model (src/tuning/ce_model.cc; sm_100, non-graph,
#    out-of-place parameters);
#  * across NVLink domains: Rail Pipe at >= 4 MiB per rank, Rail Direct below (Rail Ring needs >= 16 nodes).
CE_FLOOR = {"uc": (32.04, 0.130), "mc": (29.14, 0.137)}   # base us, us per LSA rank
CE_NVLINK_BW = 779.0                                        # GB/s
CE_D2H_BW = 53.0 * 2                                        # 2 engines x 53 GB/s (x86 only)
CUDA_DRIVER = 13000                                         # driver of the measurements (CUDA 13.0)
RAIL_PIPE_THRESHOLD = 4 * 2**20


def ce_model_us(multicast, n, per_rank, coherent, driver):
    base, per = CE_FLOOR["mc" if multicast else "uc"]
    floor = base + per * n
    bw = CE_NVLINK_BW
    if multicast and not coherent and driver < 13040:
        bw = min(CE_D2H_BW, CE_NVLINK_BW / n) * n
    t = floor + per_rank * (n if multicast else n - 1) / bw / 1e3
    if not multicast and not coherent:
        t = max(t, floor + per_rank / CE_D2H_BW / 1e3)
    return t


def ce_picks_multicast(platform, lsa_size, per_rank):
    coherent = platform == "GB200"          # Grace: coherent CPU memory
    return (ce_model_us(True, lsa_size, per_rank, coherent, CUDA_DRIVER)
            < ce_model_us(False, lsa_size, per_rank, coherent, CUDA_DRIVER))


def ce_rows(groups, plat, place, lsa_size):
    mc = {int(r["size_bytes"]): r for r in groups.get((plat, place, "NCCL CE multicast"), [])}
    uc = {int(r["size_bytes"]): r for r in groups.get((plat, place, "NCCL CE unicast"), [])}
    pipe = {int(r["size_bytes"]): r for r in groups.get((plat, place, "NCCL CE Rail Pipe"), [])}
    out = []
    for s in sorted(set(mc) | set(uc)):
        if s <= 0:
            continue
        n_gpus = int((mc.get(s) or uc.get(s))["ngpus"])
        per_rank = s / n_gpus
        n_nodes = n_gpus // lsa_size if plat == "B200" else 1
        if n_nodes > 1 and per_rank >= RAIL_PIPE_THRESHOLD and s in pipe:
            out.append(pipe[s])
            continue
        pick = mc if ce_picks_multicast(plat, lsa_size, per_rank) else uc
        if s in pick:
            out.append(pick[s])
    return out


def main(csv_path, out):
    groups = defaultdict(list)
    for r in load(csv_path):
        groups[(r["platform"], r["placement"], r["arm"])].append(r)

    fig, axes = plt.subplots(2, 3, figsize=(7.8, 2.8), sharey=True, sharex="col")
    for i_ax, (ax, (plat, place, title, nodes)) in enumerate(zip(axes.flat, PANELS)):
        ins = ax.inset_axes([0.05, 0.35, 0.34, 0.42])
        missing = []
        for arm in AG_ARMS:
            st = STYLE[arm]
            if arm == "NCCL CE":
                n_gpus = int(title.split(": ")[1].split(" GPUs")[0])
                g, every = ce_rows(groups, plat, place, LSA_SIZE[plat] or n_gpus), 1
            else:
                g, every = groups.get((plat, place, arm)), 2
            if not g:
                if not arm.startswith("NCCL "):
                    missing.append(arm)
                continue
            x, y, sd, t, _ = series(g)
            draw(ax, x, y, sd, st, ls=st["ls"], every=every)
            # the CE inset keeps its breaks at missing sizes
            keep = [i for i, s in enumerate(x) if (arm == "NCCL CE" and math.isnan(s)) or s <= 2**20]
            ins.plot([x[i] for i in keep], [t[i] for i in keep], color=st["color"], ls=st["ls"],
                     lw=0.8, marker=st["marker"], ms=1.8, mfc=mfc(st), mec=st["color"], mew=0.5, zorder=st["z"])
        head, sub = title.split("\n")
        ax.text(0.02, 0.975, head, transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
                fontweight="bold", color="#222222")
        ax.text(0.985, 0.972, sub, transform=ax.transAxes, ha="right", va="top", fontsize=6.2,
                style="italic", color="#333333")
        ax.set_ylim(0, 1000)
        ax.yaxis.set_major_locator(MultipleLocator(200))
        if i_ax % 3:
            ax.tick_params(axis="y", labelleft=False)
        size_axis(ax, 2**10, 4 * 2**30, step=4)
        ax.set_xlim(2**9, 6 * 2**30)
        ax.grid(True, which="major")
        bound(ax, NVLINK_BW, "NVLink5: 900 GB/s", xfrac=0.97, ha="right", va="top")
        if plat == "B200" and nodes > 1:
            nb = ag_network_bound(nodes)
            bound(ax, nb, f"IB: {nb:.0f} GB/s", xfrac=0.97, ha="right", xmin=0.62)
        note = PANEL_NOTES.get((plat, place))
        if note:
            ax.text(0.05, 0.2, note, transform=ax.transAxes, fontsize=5.8, color=NOTE_COLOR,
                    va="top", ha="left", linespacing=1.1)
        if not any(k[:2] == (plat, place) for k in groups):
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, fontsize=8, color="#555555", va="center", ha="center")
            ins.set_visible(False)
        elif missing:
            ax.text(0.97, 0.79, "not measured:\n" + ", ".join(missing), transform=ax.transAxes,
                    fontsize=6, color="#555555", va="center", ha="right")

        size_axis(ins, 2**10, 2**20, step=5)
        ins.set_yscale("log")
        ins.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        ins.yaxis.set_minor_locator(NullLocator())
        ins.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ins.yaxis.tick_right()
        ins.yaxis.set_label_position("right")
        ins.set_ylabel("Latency (µs)", fontsize=5.5, labelpad=1)
        ins.tick_params(labelsize=5, pad=1)
        ins.grid(True, lw=0.3)
        ins.patch.set_alpha(0.92)

    for ax in axes[1]:
        ax.set_xlabel("Message size (bytes)")
    fig.supylabel("Bus bandwidth (GB/s)", fontsize=8, x=0.004)
    handles = [Line2D([], [], color=STYLE[a]["color"], ls=STYLE[a]["ls"], marker=STYLE[a]["marker"], ms=3.5,
                      mfc=mfc(STYLE[a]), mec=STYLE[a]["color"], lw=1.1, label=a) for a in AG_ARMS]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.2, columnspacing=0.8, fontsize=6.5)
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.4, rect=(0, 0, 1, 0.94))
    save(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "figures", "allgather.pdf"))
    a = ap.parse_args()
    main(a.csv, a.out)
