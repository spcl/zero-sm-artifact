#!/usr/bin/env python3
"""MSCCL++ tuning selection and trial table.

  select.py screen <dir>            best LL and Simple plan: highest geomean bus bandwidth of the screen logs
  select.py crossover <dir> <sizes> LL->Simple switch: best single split by geomean of per-size medians
  select.py table <dir>             trials.csv (protocol, trial, size_bytes, time_us) from trial_<k>_<ll|simple>.log
Only logs whose process validated every size and exited cleanly count.
"""
import csv
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def parse(path):
    rows, valid, done = [], [], False
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("MSCCLPP_FULL_VALIDATION_PASS,"):
            valid.append(int(dict(x.split("=", 1) for x in line.split(",")[1:])["total_bytes"]))
        elif line.startswith("MSCCLPP_EXACT_TRIAL,"):
            _, trial, size, _, t, _, bw, _ = line.split(",")
            rows.append(dict(trial=int(trial), size=int(size), time_us=float(t), busbw=float(bw)))
        elif line.startswith("MSCCLPP_EXACT_PROCESS_PASS,"):
            done = True
    return rows if rows and done and valid == [r["size"] for r in rows] else None


def geomean(xs):
    return math.exp(st.fmean(math.log(x) for x in xs))


def screen(d):
    best = {}
    for kind in ("ll", "simple"):
        scores = [(geomean([r["busbw"] for r in rows]), p.stem[len(kind) + 1:])
                  for p in sorted(d.glob(f"{kind}_*.log")) if (rows := parse(p))]
        best[kind] = max(scores)[1] if scores else "none"
    (d / "screen.env").write_text(f"ll_plan={best['ll']}\nsimple_plan={best['simple']}\n")


def crossover(d, sizes):
    sizes = [int(s) for s in sizes.split(",")]
    med = {}
    for kind in ("ll", "simple"):
        per = defaultdict(list)
        for p in sorted(d.glob(f"crossover_{kind}_trial*.log")):
            for r in parse(p) or []:
                per[r["size"]].append(r["busbw"])
        med[kind] = {s: st.median(v) for s, v in per.items()}
    if med["simple"]:
        options = []
        for split in range(1, len(sizes) + 1):
            ll, simple = sizes[:split], sizes[split:]
            if set(ll) <= set(med["ll"]) and set(simple) <= set(med["simple"]):
                options.append((geomean([med["ll"][s] for s in ll] + [med["simple"][s] for s in simple]), split))
    else:  # no valid Simple plan: LL over its measured range
        n = next((i for i, s in enumerate(sizes) if s not in med["ll"]), len(sizes))
        options = [(0.0, n)] if n else []
    if not options:
        sys.exit("no complete LL/Simple split")
    split = max(options)[1]
    sel = (d / "screen.env").read_text()
    (d / "selection.env").write_text(sel + f"ll_sizes={','.join(map(str, sizes[:split]))}\n"
                                     f"simple_sizes={','.join(map(str, sizes[split:]))}\n")


def table(d):
    with open(d / "trials.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "trial", "size_bytes", "time_us"])
        for kind, proto in (("ll", "LL"), ("simple", "Simple")):
            for p in sorted(d.glob(f"trial_*_{kind}.log")):
                for r in parse(p) or []:
                    w.writerow([proto, r["trial"], r["size"], r["time_us"]])


if __name__ == "__main__":
    mode, d = sys.argv[1], Path(sys.argv[2])
    {"screen": lambda: screen(d), "crossover": lambda: crossover(d, sys.argv[3]), "table": lambda: table(d)}[mode]()
