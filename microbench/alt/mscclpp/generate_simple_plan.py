#!/usr/bin/env python3
"""Write the MSCCL++ Simple AllGather plan (language/tests/multi_node/allgather_tbg.py) for one TBG/threads point."""

import argparse
import contextlib
import io
import os
import runpy
import sys
import types
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--num-gpus", type=int, required=True)
parser.add_argument("--gpus-per-node", type=int, required=True)
parser.add_argument("--tbg-size", type=int, required=True)
parser.add_argument("--threads", type=int, required=True)
parser.add_argument("--min-message-size", type=int, default=1 << 10)
parser.add_argument("--max-message-size", type=int, default=4 << 30)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

root = Path(os.environ["MSCCLPP_SRC"])
upstream = root / "python/mscclpp/language/tests/multi_node/allgather_tbg.py"
package_root = root / "python/mscclpp"
package = types.ModuleType("mscclpp")
package.__path__ = [str(package_root)]
sys.modules["mscclpp"] = package
sys.argv = [
    str(upstream),
    "--name",
    args.name,
    "--num_gpus",
    str(args.num_gpus),
    "--gpus_per_node",
    str(args.gpus_per_node),
    "--tbg_size",
    str(args.tbg_size),
    "--num_threads_per_block",
    str(args.threads),
    "--min_message_size",
    str(args.min_message_size),
    "--max_message_size",
    str(args.max_message_size),
]
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    runpy.run_path(str(upstream), run_name="__main__")
payload = captured.getvalue()
if '"protocol": "Simple"' not in payload:
    raise SystemExit("upstream generator did not emit a Simple plan")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(payload)
