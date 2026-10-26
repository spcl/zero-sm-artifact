#!/usr/bin/env python3
"""usage: launcher.py COLLECTOR_SO PREFIX REPLAY_PY [REPLAY_ARGS ...]

Runs REPLAY_PY with CUPTI concurrent-kernel tracing enabled for the whole process and
writes PREFIX.rank<SLURM_PROCID>.kernel_activity.tsv after the replay returns, so no
file I/O falls inside a trial.
"""

import ctypes
import os
import runpy
import sys


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    collector_path, prefix, replay = sys.argv[1:4]
    rank = int(os.environ["SLURM_PROCID"])
    lib = ctypes.CDLL(collector_path, mode=ctypes.RTLD_LOCAL)
    lib.cupti_trace_prepare.argtypes = [ctypes.c_char_p, ctypes.c_int]
    for name in ("cupti_trace_prepare", "cupti_trace_start", "cupti_trace_stop", "cupti_trace_finalize"):
        getattr(lib, name).restype = ctypes.c_int
    if lib.cupti_trace_prepare(os.fsencode(f"{prefix}.rank{rank}.kernel_activity.tsv"), rank) != 0:
        raise RuntimeError("CUPTI prepare failed")

    started = False
    replay_exit = 0
    pending = None
    try:
        if lib.cupti_trace_start() != 0:
            raise RuntimeError("CUPTI start failed")
        started = True
        sys.argv = [replay, *sys.argv[4:]]
        try:
            runpy.run_path(replay, run_name="__main__")
        except SystemExit as exc:
            replay_exit = 0 if exc.code is None else int(exc.code)
    except BaseException as exc:
        pending = exc
    finally:
        if started and lib.cupti_trace_stop() != 0 and pending is None:
            pending = RuntimeError("CUPTI stop failed")
        if lib.cupti_trace_finalize() != 0 and pending is None:
            pending = RuntimeError("CUPTI finalize failed")
    if pending is not None:
        raise pending
    return replay_exit


if __name__ == "__main__":
    sys.exit(main())
