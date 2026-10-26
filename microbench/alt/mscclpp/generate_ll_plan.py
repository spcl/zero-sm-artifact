#!/usr/bin/env python3
"""Write the MSCCL++ LL AllGather plan (default_algos.allgather_multi_nodes) for one instances/threads point."""

import argparse
import os
import sys
import types
from pathlib import Path

root = Path(os.environ["MSCCLPP_SRC"])
package_root = root / "python" / "mscclpp"
package = types.ModuleType("mscclpp")
package.__path__ = [str(package_root)]
sys.modules["mscclpp"] = package

from mscclpp.default_algos.allgather_multi_nodes import allgather_multi_nodes
from mscclpp.language.collectives import AllGather
from mscclpp.language.utils import AlgoSpec

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--num-gpus", type=int, required=True)
parser.add_argument("--gpus-per-node", type=int, required=True)
parser.add_argument("--instances", type=int, required=True)
parser.add_argument("--threads", type=int, required=True)
parser.add_argument("--min-message-size", type=int, default=1 << 10)
parser.add_argument("--max-message-size", type=int, default=4 << 30)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

spec = AlgoSpec(
    name=args.name,
    collective=AllGather(args.num_gpus, 1, False),
    nranks_per_node=args.gpus_per_node,
    world_size=args.num_gpus,
    in_place=False,
    instances=args.instances,
    protocol="LL",
    auto_sync=False,
    num_threads_per_block=args.threads,
    reuse_resources=True,
    use_double_scratch_buffer=True,
    min_message_size=args.min_message_size,
    max_message_size=args.max_message_size,
    tags={"default": 1, "tuning_screen": 1},
)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(allgather_multi_nodes(spec).to_json() + "\n")
