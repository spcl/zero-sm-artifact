"""MSCCL++ AllGather rows from mscclpp_<platform>_<N>n<G>g runs (microbench/alt/mscclpp)."""
import csv
import io
import re

from rawio import Run, from_times, nccl_tests_trials, summarize

PLACEMENTS = [("b200", "1n8g"), ("b200", "4n32g"), ("b200", "8n64g"),
              ("gb200", "1n4g"), ("gb200", "4n16g"), ("gb200", "18n72g")]


def _native(run, ngpus):
    times = {}
    for r in csv.DictReader(io.StringIO(run.text("trials.csv"))):
        times.setdefault((r["protocol"], int(r["size_bytes"])), []).append(float(r["time_us"]))
    sel = dict(l.split("=", 1) for l in run.text("selection.env").splitlines() if "=" in l) \
        if "selection.env" in run.files else {}
    best = {}
    for proto in sorted({p for p, _ in times}):
        for row in from_times({s: t for (p, s), t in times.items() if p == proto}, ngpus):
            row["stat_note"] = f"mean; {proto} " + (sel.get("ll_plan" if proto == "LL" else "simple_plan", ""))
            s = row["size_bytes"]
            if s not in best or row["time_us_mean"] < best[s]["time_us_mean"]:
                best[s] = row
    return [best[s] for s in sorted(best)]


def rows(root):
    out = []
    for plat, tag in PLACEMENTS:
        run = Run(root, f"mscclpp_{plat}_{tag}")
        if not run.found:
            continue
        nodes, ngpus = map(int, re.match(r"(\d+)n(\d+)g", tag).groups())
        if run.names("interposer.trial*.log"):
            rs = summarize(nccl_tests_trials(run, "interposer.trial*.log"))
            for r in rs:
                r["stat_note"] = "mean; NCCL-API interposer, nccl-tests"
        else:
            rs = _native(run, ngpus)
        for r in rs:
            out.append(dict(platform=plat.upper(), placement=f"{nodes} node" + ("s" if nodes > 1 else ""),
                            ngpus=ngpus, arm="MSCCL++", build="", source=run.source, **r))
    return out
