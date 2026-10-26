#!/usr/bin/env python3
"""Per-GEMM slowdown along the chain, from overlap/run.sh output.

usage: analyze.py SRC [--out DIR] [--compare TSV ...]

SRC is either a run directory written by overlap/run.sh (per-rank measurement
TSVs and CUPTI kernel traces) or a directory holding observations_*.tsv.gz
(a previous --out directory). From a run directory the script first pairs every GEMM of
the overlap pass with the same GEMM of the clean pass (same process, rank,
trial and chain position) and writes observations_<model>.tsv.gz and
kernels.tsv (GEMM and AllGather kernels seen per arm). It then writes
chain_position.tsv: per chain position, the median slowdown over ranks x trials
within each process, then statistics over processes
(process_all_slowdown_pct_median_median is the plotted value).
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

CONFIGS = ("zero", "legacy", "default", "cta_32")
GEMMS_PER_LAUNCH = {"mxfp8_forward": 1, "hybrid_fp8_backward_wgrad": 3}
GEMM_LABELS = {"qwen3_30b_a3b": ("fprop",), "llama3_8b": ("fprop", "dgrad", "wgrad")}
OBS_FIELDS = ("model", "config", "replicate", "sweep", "rank", "trial", "data_send_bytes",
              "global_chain_position", "clean_ns", "overlap_ns")
PREFIX_RE = re.compile(r"(?P<model>[^/]+)/replicate_(?P<rep>\d+)/(?P<sweep>forward|reverse)/"
                       r"(?P<config>[^/.]+)\.rank0\.tsv$")


def open_text(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def read_kernels(path):
    gemms, allgathers = [], []
    with open_text(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {k: header.index(k) for k in ("start_ns", "end_ns", "grid_x", "block_x", "name")}
        for line in f:
            if "nvjet" not in line and "AllGather" not in line:
                continue
            p = line.rstrip("\n").split("\t")
            k = (int(p[col["start_ns"]]), int(p[col["end_ns"]]), int(p[col["grid_x"]]),
                 int(p[col["block_x"]]), p[col["name"]])
            if k[4].lower().startswith("nvjet"):
                gemms.append(k)
            elif "AllGather" in k[4]:
                allgathers.append(k)
    gemms.sort()
    allgathers.sort()
    return gemms, allgathers


def extract_rank(job):
    prefix, model, rep, sweep, config, rank = job
    rows = list(csv.DictReader(open(f"{prefix}.rank{rank}.tsv"), delimiter="\t"))
    trace = f"{prefix}.rank{rank}.kernel_activity.tsv"
    gemms, allgathers = read_kernels(trace + ".gz" if os.path.exists(trace + ".gz") else trace)
    per_launch = GEMMS_PER_LAUNCH[rows[0]["compute_kind"]]
    width = int(rows[0]["compute_repeats"]) * per_launch
    sizes = sorted({int(r["data_send_bytes"]) for r in rows})
    # Untimed start-up in replay.py: one forward-only call, then four full calls.
    startup = 1 + 4 * per_launch
    compute_rows = [r for r in rows if r["mode"] != "comm_alone"]
    warm_groups, rem = divmod(len(gemms) - startup - len(compute_rows) * width, width)
    if rem or warm_groups < 0 or warm_groups % (2 * len(sizes)):
        raise RuntimeError(f"{prefix} rank {rank}: {len(gemms)} GEMM kernels do not match the chain layout")
    body = gemms[startup + warm_groups * width:]
    groups = {}
    for i, r in enumerate(compute_rows):
        groups[(int(r["trial"]), int(r["data_send_bytes"]), r["mode"])] = body[i * width:(i + 1) * width]
    pattern = [k[4] for k in body[:per_launch]]
    out = []
    for (trial, size, mode), clean in groups.items():
        if mode != "compute_alone":
            continue
        overlap = groups[(trial, size, "overlap")]
        for pos, (c, o) in enumerate(zip(clean, overlap), start=1):
            if c[4] != o[4] or c[4] != pattern[(pos - 1) % per_launch]:
                raise RuntimeError(f"{prefix} rank {rank}: GEMM sequence mismatch at position {pos}")
            out.append((model, config, rep, sweep, rank, trial, size, pos, c[1] - c[0], o[1] - o[0]))
    kernels = Counter(("gemm", k[4], k[2], k[3]) for k in body)
    kernels.update(("allgather", k[4], k[2], k[3]) for k in allgathers)
    return out, (model, config, kernels)


def discover(src):
    jobs = []
    for path in sorted(glob.glob(os.path.join(src, "*", "replicate_*", "*", "*.rank0.tsv"))):
        m = PREFIX_RE.search(path)
        if not m or m["config"] not in CONFIGS:
            continue
        prefix = path[: -len(".rank0.tsv")]
        nranks = len(glob.glob(prefix + ".rank*.tsv"))
        jobs += [(prefix, m["model"], int(m["rep"]), m["sweep"], m["config"], r) for r in range(nranks)]
    return jobs


def extract(src, out, workers):
    jobs = discover(src)
    if not jobs:
        return False
    obs = defaultdict(list)
    kernels = defaultdict(Counter)
    with ProcessPoolExecutor(workers) as pool:
        for rows, (model, config, seen) in pool.map(extract_rank, jobs, chunksize=1):
            for r in rows:
                obs[r[0]].append(r)
            kernels[(model, config)].update(seen)
    for model, rows in obs.items():
        rows.sort(key=lambda r: (CONFIGS.index(r[1]), r[2], r[4], r[6], r[5], r[7]))
        with gzip.open(os.path.join(out, f"observations_{model}.tsv.gz"), "wt", compresslevel=9) as f:
            f.write("\t".join(OBS_FIELDS) + "\n")
            f.writelines("\t".join(map(str, r)) + "\n" for r in rows)
    with open(os.path.join(out, "kernels.tsv"), "w") as f:
        f.write("model\tconfig\trole\tkernel\tgrid_x\tblock_x\tlaunches\n")
        for (model, config), counts in sorted(kernels.items()):
            for (role, name, grid, block), n in sorted(counts.items()):
                f.write(f"{model}\t{config}\t{role}\t{name}\t{grid}\t{block}\t{n}\n")
    return True


def describe(prefix, values):
    v = sorted(values)

    def pct(q):
        x = (len(v) - 1) * q / 100.0
        lo, hi = math.floor(x), math.ceil(x)
        return v[lo] + (v[hi] - v[lo]) * (x - lo)

    return {f"{prefix}_n": len(v), f"{prefix}_mean": statistics.fmean(v),
            f"{prefix}_median": statistics.median(v),
            f"{prefix}_sample_sd": statistics.stdev(v) if len(v) > 1 else 0.0,
            f"{prefix}_p5": pct(5), f"{prefix}_p95": pct(95), f"{prefix}_min": v[0], f"{prefix}_max": v[-1]}


def aggregate(src, out):
    per_proc = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(src, "observations_*.tsv.gz"))):
        with gzip.open(path, "rt") as f:
            next(f)
            for line in f:
                model, config, rep, sweep, _rank, _trial, size, pos, clean, overlap = line.split("\t")
                c, o = int(clean), int(overlap)
                per_proc[(model, config, size, int(pos))][(rep, sweep)].append((c, o))
    if not per_proc:
        sys.exit(f"no observations under {src}")
    rows = []
    for (model, config, size, pos), procs in sorted(per_proc.items()):
        slow, cl, ov = [], [], []
        for obs in procs.values():
            slow.append(statistics.median(100.0 * ((o / 1000.0) / (c / 1000.0) - 1.0) for c, o in obs))
            cl.append(statistics.median(c / 1000.0 for c, _ in obs))
            ov.append(statistics.median(o / 1000.0 for _, o in obs))
        labels = GEMM_LABELS.get(model, ("gemm",))
        row = {"model": model, "config": config, "data_send_bytes": size, "global_chain_position": pos,
               "gemm": labels[(pos - 1) % len(labels)], "n_independent_processes": len(procs),
               "sweeps": ",".join(sorted({s for _, s in procs})),
               "all_observations": sum(len(o) for o in procs.values())}
        row.update(describe("process_all_slowdown_pct_median", slow))
        row["process_clean_duration_us_median_median"] = statistics.median(cl)
        row["process_overlap_duration_us_median_median"] = statistics.median(ov)
        rows.append(row)
    path = os.path.join(out, "chain_position.tsv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)
    return rows


def summarize(rows):
    # Same per-panel statistics as the figure: first 100 Qwen GEMMs, all 96 Llama GEMMs.
    shown = {"qwen3_30b_a3b": 100, "llama3_8b": 96}
    print("model\tconfig\tdata_send_bytes\tgemms_slowed_gt_1pct\tmedian_of_those_pct\tmax_pct")
    keys = sorted({(r["model"], r["config"], r["data_send_bytes"]) for r in rows},
                  key=lambda k: (k[0], CONFIGS.index(k[1]), int(k[2])))
    for model, config, size in keys:
        v = [r["process_all_slowdown_pct_median_median"] for r in rows if (r["model"], r["config"],
             r["data_send_bytes"]) == (model, config, size) and r["global_chain_position"] <= shown.get(model, 10**9)]
        hit = [x for x in v if x > 1.0]
        med = f"{statistics.median(hit):.1f}" if hit else "-"
        print(f"{model}\t{config}\t{size}\t{len(hit)}/{len(v)}\t{med}\t{max(v):.1f}")


def compare(rows, paths):
    ref = {}
    for p in paths:
        for r in csv.DictReader(open(p), delimiter="\t"):
            ref[(r["model"], r["config"], r["data_send_bytes"], int(r["global_chain_position"]))] = r
    worst, n = 0.0, 0
    for r in rows:
        k = (r["model"], r["config"], r["data_send_bytes"], r["global_chain_position"])
        if k in ref:
            col = "process_all_slowdown_pct_median_median"
            worst = max(worst, abs(float(ref[k][col]) - r[col]))
            n += 1
    print(f"compared {n} chain positions against {len(ref)} reference rows; "
          f"max |difference| of the plotted median = {worst:.3g} percentage points")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--compare", nargs="*", default=[])
    a = ap.parse_args()
    out = a.out or a.src
    os.makedirs(out, exist_ok=True)
    obs_dir = out if extract(a.src, out, a.workers) else a.src
    rows = aggregate(obs_dir, out)
    summarize(rows)
    if a.compare:
        compare(rows, a.compare)


if __name__ == "__main__":
    main()
