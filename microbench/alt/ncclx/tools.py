#!/usr/bin/env python3
"""Helpers for run.sh.

  tools.py sizes <world> <min_bytes> <max_bytes>   nccl-tests all_gather_perf sizes (-f 2, float)
  tools.py pilot <sizes>                           first, first >= 16 MiB, last
  tools.py collect <dir> <algo> <sizes> <trials>   trial_<k>.rank0.log -> trials.csv, summary.csv
  tools.py select <tune_dir> <algo>...             geomean of per-size median busbw -> selection.txt
"""
import csv
import io
import json
import math
import statistics
import sys
from pathlib import Path


def parse_bytes(v):
    mult = {"K": 2**10, "M": 2**20, "G": 2**30}
    return int(v[:-1]) * mult[v[-1].upper()] if v[-1].upper() in mult else int(v)


def sizes(world, lo, hi):
    # nccl-tests: per-rank count = (bytes/4/world) rounded down to a multiple of 4 elements
    out, s = [], lo
    while s <= hi:
        c = (s // 4 // world) & ~3
        if c:
            out.append(c * world * 4)
        s *= 2
    return out


def pilot(sz):
    mid = next((s for s in sz if s >= 16 * 2**20), sz[-1])
    return sorted({sz[0], mid, sz[-1]})


def collect(d, algo, sz, trials):
    d = Path(d)
    rows = []
    for t in range(1, trials + 1):
        got = [json.loads(l.split(",", 1)[1]) for l in (d / f"trial_{t}.rank0.log").read_text().splitlines()
               if l.startswith("NCCLX_AGP_TRIAL,")]
        if [r["size_bytes"] for r in got] != sz or any(r["backend"] != f"ncclx_allgatherp_{algo}" for r in got):
            sys.exit(f"{d}: trial {t} incomplete")
        rows += got
    with open(d / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(d / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size_bytes", "trials", "time_us_mean", "time_us_median", "busbw_gbps_mean",
                    "busbw_gbps_median", "busbw_gbps_min", "busbw_gbps_max"])
        for s in sz:
            t = [r["time_us"] for r in rows if r["size_bytes"] == s]
            b = [r["busbw_gbps"] for r in rows if r["size_bytes"] == s]
            w.writerow([s, len(t), statistics.mean(t), statistics.median(t), statistics.mean(b),
                        statistics.median(b), min(b), max(b)])


def select_text(summaries):
    """summaries: {algo: [summary.csv rows]} -> selection.txt content (winner first)."""
    score = {a: math.exp(statistics.mean(math.log(float(r["busbw_gbps_median"])) for r in rows))
             for a, rows in summaries.items()}
    order = sorted(score, key=score.get, reverse=True)
    sz = ",".join(r["size_bytes"] for r in summaries[order[0]])
    return (f"winner={order[0]}\nsizes={sz}\n"
            + "".join(f"candidate={a},geomean_busbw_gbps={score[a]:.9f}\n" for a in order))


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "sizes":
        print(",".join(map(str, sizes(int(a[0]), parse_bytes(a[1]), parse_bytes(a[2])))))
    elif cmd == "pilot":
        print(",".join(map(str, pilot([int(x) for x in a[0].split(",")]))))
    elif cmd == "collect":
        collect(a[0], a[1], [int(x) for x in a[2].split(",")], int(a[3]))
    elif cmd == "select":
        d = Path(a[0])
        txt = select_text({x: list(csv.DictReader(io.StringIO((d / x / "summary.csv").read_text()))) for x in a[1:]})
        (d.parent / "selection.txt").write_text(txt)
        print(txt.splitlines()[0].split("=")[1])
