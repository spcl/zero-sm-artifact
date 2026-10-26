#!/usr/bin/env python3
"""Per-block speedup of one-sided over two-sided pipeline-parallel P2P.

Usage: e2e/pp/analyze.py INPUT ... [-o OUT]
INPUT is a sessions.csv (platform,recipe,block,arm,step_ms,points) or a directory searched for sessions.csv files
(e.g. $WORK/results/e2e/pp). OUT (default speedup.csv) has one row per
platform x recipe: blocks, mean and SD over blocks of T_two_sided/T_one_sided - 1 (%), the mean two-sided step time,
and the pipeline-bubble share b = (PP - 1) / (GA * VPP + PP - 1) of the recipe.
"""
import argparse
import csv
import glob
import os
import re
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECIPES = os.path.join(ROOT, "e2e", "pp", "recipes")
BASE, ARM = "two_sided", "one_sided"


def recipe_meta(name):
    """LABEL, GPUS, TP, PP, VPP, CP, MBS, GBS, SEQ of a recipe file, plus GA and the bubble share b (%)."""
    text = open(os.path.join(RECIPES, name + ".env")).read()
    m = dict(re.findall(r"\b(LABEL|GPUS|TP|PP|VPP|CP|MBS|GBS|SEQ)=(\S+)", text))
    g = {k: (m[k] if k == "LABEL" else int(m[k])) for k in m}
    dp = g["GPUS"] // (g["TP"] * g["PP"] * g["CP"])
    g["GA"] = g["GBS"] // (g["MBS"] * dp)
    g["bubble_pct"] = 100.0 * (g["PP"] - 1) / (g["GA"] * g["VPP"] + g["PP"] - 1)
    return g


def read_sessions(inputs):
    files = []
    for p in inputs:
        files += sorted(glob.glob(os.path.join(p, "**", "sessions.csv"), recursive=True)) if os.path.isdir(p) else [p]
    rows = []
    for f in files:
        for r in csv.DictReader(open(f)):
            if r["step_ms"] not in ("", "nan"):
                rows.append(dict(r, src=f))     # results of different allocations are separate blocks
    return rows


def speedups(rows):
    """{(platform, recipe): [speedup % per block]}, {(platform, recipe): [two-sided ms per block]}."""
    blocks = defaultdict(dict)
    for r in rows:
        blocks[(r["platform"], r["recipe"], r["src"], r["block"])][r["arm"]] = float(r["step_ms"])
    per, base = defaultdict(list), defaultdict(list)
    for (plat, rec, _, _), arms in blocks.items():
        if BASE in arms and ARM in arms:
            per[(plat, rec)].append((arms[BASE] / arms[ARM] - 1) * 100)
            base[(plat, rec)].append(arms[BASE])
    return per, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default="speedup.csv")
    a = ap.parse_args()
    per, base = speedups(read_sessions(a.inputs))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["platform", "recipe", "blocks", "speedup_pct", "sd_pct", "two_sided_step_ms", "bubble_pct"])
        for (plat, rec), v in sorted(per.items(), key=lambda kv: (kv[0][0], recipe_meta(kv[0][1])["bubble_pct"], kv[0][1])):
            sd = st.stdev(v) if len(v) > 1 else 0.0
            b = recipe_meta(rec)["bubble_pct"]
            w.writerow([plat, rec, len(v), f"{st.mean(v):.6f}", f"{sd:.6f}", f"{st.mean(base[(plat, rec)]):.3f}", f"{b:.3f}"])
            print(f"{plat:6} {rec:12} b={b:4.1f}%  n={len(v)}  {st.mean(v):+6.2f} +- {sd:4.2f} %")
    print("wrote", os.path.relpath(a.out))


if __name__ == "__main__":
    main()
