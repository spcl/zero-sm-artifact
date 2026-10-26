"""Run a Megatron training script, then destroy the NCCL process group before the interpreter exits."""
import runpy
import sys

import torch

script = sys.argv[1]
sys.argv = sys.argv[1:]
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise
if torch.distributed.is_initialized():
    torch.cuda.synchronize()
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
