"""Style and drawing helpers shared by figures/p2p_b200.py and figures/allgather.py."""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "grid.color": "#d9d9d9",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# NCCL = greens, other libraries = reds; the split is also carried by line style and marker fill.
STYLE = {
    "NCCL CE":        dict(color="#00441b", marker="o", ls="-",  fill=True,  z=7),
    "NCCL STMC":      dict(color="#238b45", marker="s", ls="-",  fill=True,  z=5),
    "NCCL legacy":    dict(color="#74c476", marker="^", ls="-",  fill=True,  z=4),
    "NCCLX":          dict(color="#e7298a", marker="D", ls="--", fill=False, z=3),
    "MSCCL++":        dict(color="#d94801", marker="v", ls="--", fill=False, z=3),
    "NCCL put":       dict(color="#00441b", marker="o", fill=True,  z=6),
    "NCCL send/recv": dict(color="#74c476", marker="s", fill=True,  z=5),
    "VCCL SM-free":   dict(color="#e7298a", marker="D", fill=False, z=4),
    "VCCL":           dict(color="#d94801", marker="v", fill=False, z=3),
}
BOUND_COLOR = "#555555"
NVLINK_BW = 900.0      # NVLink5, GB/s per direction per GPU
NIC_BW = 50.0          # ConnectX-7 NDR, 400 Gb/s


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def size_label(b, _=None):
    for unit, div in (("G", 2**30), ("M", 2**20), ("K", 2**10)):
        if b >= div:
            v = b / div
            return f"{v:.0f}{unit}" if v == int(v) else f"{v:.1f}{unit}"
    return f"{b:.0f}"


def series(rows, factor=None):
    """Bandwidth from the mean time (factor * bytes / time): factor 1 for point-to-point, (n-1)/n for AllGather.
    The printed busbw columns are rounded to 0.01 GB/s and collapse to 0 at small sizes."""
    xs, ys, sd, ts, tsd = [], [], [], [], []
    for r in sorted(rows, key=lambda r: int(r["size_bytes"])):
        s = int(r["size_bytes"])
        if s <= 0:
            continue
        if xs and s > 2.5 * xs[-1]:          # break the line where sizes are missing
            for lst in (xs, ys, sd, ts, tsd):
                lst.append(math.nan)
        t = float(r["time_us_mean"])
        t_sd = float(r["time_us_std"]) if r.get("time_us_std") else 0.0
        f = factor if factor is not None else (int(r["ngpus"]) - 1) / int(r["ngpus"])
        bw = f * s / t / 1e3
        xs.append(s)
        ys.append(bw)
        sd.append(bw * t_sd / t)
        ts.append(t)
        tsd.append(t_sd)
    return xs, ys, sd, ts, tsd


def mfc(st):
    return st["color"] if st["fill"] else "white"


def draw(ax, x, y, sd, st, ls="-", band=True, every=2):
    ax.plot(x, y, color=st["color"], ls=ls, lw=1.1, marker=st["marker"], ms=3.2,
            mfc=mfc(st), mec=st["color"], mew=0.8, markevery=every, zorder=st["z"])
    if band and any(sd):
        lo = [a - b if not math.isnan(a) else math.nan for a, b in zip(y, sd)]
        hi = [a + b if not math.isnan(a) else math.nan for a, b in zip(y, sd)]
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.15, lw=0, zorder=st["z"] - 1)


def size_axis(ax, lo, hi, step=2):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator([2**k for k in range(0, 34, step) if lo <= 2**k <= hi]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(size_label))


def bound(ax, y, text, xfrac=0.02, va="bottom", ha="left", xmin=0.0):
    ax.axhline(y, xmin=xmin, color=BOUND_COLOR, ls=":", lw=0.9, zorder=1)
    ax.text(xfrac, y, text, transform=ax.get_yaxis_transform(), color=BOUND_COLOR,
            fontsize=6.5, va=va, ha=ha)


def save(fig, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=300)
    if out.endswith(".pdf"):
        fig.savefig(out[:-4] + ".png", dpi=300)
    print("wrote", out)
