#!/usr/bin/env python3
"""Mean step time of one training session log.

Usage: steptime.py LOG [--warmup N]   ->  prints "<mean_ms> <points>"

Reads NeMo lines (`iteration I/M ... train_step_timing in s: X`, one per step) or Megatron lines
(`iteration I/ M ... elapsed time per iteration (ms): X`, one per log interval, averaged over the interval).
Iterations I <= N (default 10) are warm-up and discarded.
"""
import argparse
import re
import statistics

NEMO = re.compile(r"iteration +(\d+)/\d+ .*train_step_timing in s: ([0-9.]+)")
MEGATRON = re.compile(r"iteration +(\d+)/ *\d+ .*elapsed time per iteration \(ms\): ([0-9.]+)")


def step_times(path, warmup=10):
    """Retained (iteration, ms) points of a log."""
    pts = {}
    for line in open(path, errors="replace"):
        m = NEMO.search(line)
        if m:
            pts[int(m.group(1))] = float(m.group(2)) * 1000.0
            continue
        m = MEGATRON.search(line)
        if m:
            pts[int(m.group(1))] = float(m.group(2))
    return [(i, v) for i, v in sorted(pts.items()) if i > warmup]


def mean_step_ms(path, warmup=10):
    pts = step_times(path, warmup)
    return (statistics.fmean(v for _, v in pts) if pts else float("nan")), len(pts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args()
    mean, n = mean_step_ms(a.log, a.warmup)
    print(f"{mean:.3f} {n}")
