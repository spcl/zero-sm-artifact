"""VCCL send/recv rows of p2p_b200.csv from vccl_b200_{1n2g,2n2g} (microbench/alt/vccl/run.sh)."""
from rawio import Run, nccl_tests_trials, summarize

RUNS = (("scale-up", "vccl_b200_1n2g"), ("scale-out", "vccl_b200_2n2g"))
MODES = (("normal", "VCCL"), ("smfree", "VCCL SM-free"))


def rows(root):
    out = []
    for domain, name in RUNS:
        run = Run(root, name)
        if not run.found:
            continue
        for mode, arm in MODES:
            for r in summarize(nccl_tests_trials(run, f"{mode}.trial*.log")):
                out.append(dict(domain=domain, arm=arm, **r, source=run.source))
    return out
