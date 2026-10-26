#!/usr/bin/env python3
"""Reduce per-trial benchmark output to the CSVs plotted by figures/p2p_b200.py and figures/allgather.py.

  python3 microbench/analysis/build_csv.py $WORK/results/microbench   # runs -> CSVs next to them

Each run is <root>/<run>/. Statistics: mean and sample std over trials
(fresh processes) of the out-of-place columns. Missing runs are skipped.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mscclpp  # noqa: E402
import ncclx  # noqa: E402
import vccl  # noqa: E402
from rawio import Run, nccl_tests_trials, nccl_version, summarize  # noqa: E402

P2P_RUNS = [("scale-up", "p2p_b200_1n2g"), ("scale-out", "p2p_b200_2n2g")]
P2P_ARMS = [("sendrecv", "NCCL send/recv"), ("put", "NCCL put")]
AG_RUNS = [("GB200", "1 node", 4, "ag_gb200_1n4g"), ("GB200", "4 nodes", 16, "ag_gb200_4n16g"),
           ("GB200", "18 nodes", 72, "ag_gb200_18n72g"),
           ("B200", "1 node", 8, "ag_b200_1n8g"), ("B200", "4 nodes", 32, "ag_b200_4n32g"),
           ("B200", "8 nodes", 64, "ag_b200_8n64g")]
AG_ARMS = [("legacy", "NCCL legacy"), ("stmc", "NCCL STMC"), ("ce_mc", "NCCL CE multicast"),
           ("ce_uc", "NCCL CE unicast"), ("ce_pipe", "NCCL CE Rail Pipe")]
P2P_FIELDS = ["domain", "arm", "size_bytes", "trials", "busbw_mean", "busbw_std", "time_us_mean", "time_us_std",
              "source"]
AG_FIELDS = ["platform", "placement", "ngpus", "arm", "build", "size_bytes", "trials", "busbw_mean", "busbw_std",
             "time_us_mean", "time_us_std", "stat_note", "source"]


def p2p_nccl(root):
    out = []
    for domain, name in P2P_RUNS:
        run = Run(root, name)
        for tag, arm in P2P_ARMS:
            for r in summarize(nccl_tests_trials(run, f"{tag}.trial*.log")):
                out.append(dict(domain=domain, arm=arm, **r, source=run.source))
    return out


def ag_nccl(root):
    out = []
    for plat, placement, n, name in AG_RUNS:
        run = Run(root, name)
        for tag, arm in AG_ARMS:
            pat = f"ag_{tag}.trial*.log"
            build = nccl_version(run, pat)
            for r in summarize(nccl_tests_trials(run, pat)):
                out.append(dict(platform=plat, placement=placement, ngpus=n, arm=arm, build=build, **r,
                                stat_note="mean", source=run.source))
    return out


def write(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", help="output directory (default: ROOT)")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    out = a.out or root
    os.makedirs(out, exist_ok=True)
    write(os.path.join(out, "p2p_b200.csv"), p2p_nccl(root) + vccl.rows(root), P2P_FIELDS)
    write(os.path.join(out, "ag.csv"), ag_nccl(root) + ncclx.rows(root) + mscclpp.rows(root), AG_FIELDS)
