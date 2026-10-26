#!/usr/bin/env python3
"""Per-block speedups of the FSDP arms over legacy.

Usage: e2e/fsdp/analyze.py INPUT ... [-o OUT]
INPUT is a sessions.csv (platform,recipe,block,arm,step_ms,points) or a directory searched for sessions.csv files
(e.g. $WORK/results/e2e/fsdp). OUT (default speedup.csv) has one row
per platform x recipe x arm: blocks, mean and SD over blocks of T_legacy/T_arm - 1 (%), and the mean legacy step time.
"""
import argparse
import csv
import glob
import os
import statistics as st
from collections import defaultdict

ARMS = ["symmetric_st", "symmetric_stmc", "ce_unicast", "ce_multicast"]


def read_sessions(inputs):
    files = []
    for p in inputs:
        files += sorted(glob.glob(os.path.join(p, "**", "sessions.csv"), recursive=True)) if os.path.isdir(p) else [p]
    rows = []
    for f in files:
        for r in csv.DictReader(open(f)):
            if r["step_ms"] not in ("", "nan"):
                # results of different allocations of one recipe are separate blocks
                rows.append(dict(r, src=f))
    return rows


def speedups(rows):
    blocks = defaultdict(dict)                      # (platform, recipe, src, block) -> {arm: ms}
    for r in rows:
        blocks[(r["platform"], r["recipe"], r["src"], r["block"])][r["arm"]] = float(r["step_ms"])
    per = defaultdict(list)                         # (platform, recipe, arm) -> [speedup %]
    legacy = defaultdict(list)                      # (platform, recipe) -> [legacy ms]
    for (plat, rec, _, _), arms in blocks.items():
        if "legacy" not in arms:
            continue
        legacy[(plat, rec)].append(arms["legacy"])
        for arm in ARMS:
            if arm in arms:
                per[(plat, rec, arm)].append((arms["legacy"] / arms[arm] - 1) * 100)
    return per, legacy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default="speedup.csv")
    a = ap.parse_args()
    per, legacy = speedups(read_sessions(a.inputs))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["platform", "recipe", "arm", "blocks", "speedup_pct", "sd_pct", "legacy_step_ms"])
        for (plat, rec, arm), v in sorted(per.items(), key=lambda kv: (kv[0][0], kv[0][1], ARMS.index(kv[0][2]))):
            sd = st.stdev(v) if len(v) > 1 else 0.0
            w.writerow([plat, rec, arm, len(v), f"{st.mean(v):.6f}", f"{sd:.6f}", f"{st.mean(legacy[(plat, rec)]):.3f}"])
            print(f"{plat:6} {rec:15} {arm:15} n={len(v):2}  {st.mean(v):+6.2f} +- {sd:4.2f} %")
    print("wrote", os.path.relpath(a.out))


if __name__ == "__main__":
    main()
