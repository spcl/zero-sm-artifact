#!/usr/bin/env python3
"""NCCLX persistent AllGatherP benchmark: FP32, out-of-place, CUDA-event timing, rank-max time per size.

One process per GPU (srun). Handle creation, registration, validation and teardown are outside timing.
Rank 0 prints one NCCLX_AGP_TRIAL,<json> line per size.
"""
import argparse
import json
import os
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", required=True, help="comma-separated total output bytes")
    p.add_argument("--trial", type=int, required=True)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--algo", choices=("ctdirect", "ctpipeline", "ctsrdpipeline"), required=True)
    args = p.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s]

    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    # NCCLX reads the host identity from this file to group ranks by node.
    whoami = Path(f"/tmp/ncclx_agp_whoami_{os.environ['SLURM_JOB_ID']}_{args.trial}_{rank}")
    whoami.write_text(f"DEVICE_NAME={socket.gethostname()}\n")
    try:
        os.environ.update({
            "RANK": str(rank), "WORLD_SIZE": str(world), "LOCAL_RANK": str(local_rank),
            "TORCHCOMM_RANK": str(rank), "TORCHCOMM_SIZE": str(world),
            "NCCL_CTRAN_ENABLE": "1",
            "NCCL_CTRAN_IPC_REGCACHE_ENABLE_ASYNC_SOCKET": "1",
            "NCCL_ALLGATHER_P_ALGO": args.algo,
            "NCCL_TOPO_FILE_PATH": str(whoami),
            "NCCL_IGNORE_TOPO_LOAD_FAILURE": "1",
        })
        import torchcomms
        from torchcomms import new_comm

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        store = dist.TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]), world, rank == 0,
                              timedelta(seconds=180), wait_for_workers=True)
        prefix = f"agp_{args.algo}_trial_{args.trial}"
        comm = new_comm("ncclx", device, store=dist.PrefixStore(f"comm_{prefix}", store), name=prefix,
                        timeout=timedelta(seconds=180))
        sync = dist.PrefixStore(f"sync_{prefix}", store)
        pool = torch.cuda.MemPool(torchcomms.get_mem_allocator(comm.get_backend()))

        def barrier(key):
            sync.set(f"{key}_{rank}", "1")
            for peer in range(world):
                sync.get(f"{key}_{peer}")

        for total in sizes:
            if total % (world * 4):
                raise RuntimeError(f"{total} B is not divisible by world*4")
            count = total // (world * 4)
            # Both buffers come from the NCCLX pool so they are registered once (persistent path).
            with torch.cuda.use_mem_pool(pool):
                src = torch.full((count,), float(rank + 1), dtype=torch.float32, device=device)
                out = torch.empty(count * world, dtype=torch.float32, device=device)
            handle = comm.all_gather_p_init(out)
            comm.barrier(False).wait()
            # First exec completes the lazy key exchange; excluded from timing.
            comm.all_gather_p_exec(handle, src, async_op=True).wait()
            torch.cuda.synchronize(device)
            for peer in range(world):
                if not bool(torch.all(out[peer * count:(peer + 1) * count] == float(peer + 1)).item()):
                    raise RuntimeError(f"validation failure rank={rank} bytes={total}")
            barrier(f"valid_{total}")

            for _ in range(args.warmups):
                comm.all_gather_p_exec(handle, src, async_op=True).wait()
            torch.cuda.synchronize(device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                comm.all_gather_p_exec(handle, src, async_op=True).wait()
            end.record()
            end.synchronize()
            sync.set(f"time_{total}_{rank}", repr(start.elapsed_time(end) * 1000.0 / args.iters))
            if rank == 0:
                t = max(float(sync.get(f"time_{total}_{peer}").decode()) for peer in range(world))
                algbw = total / 1e3 / t
                print("NCCLX_AGP_TRIAL," + json.dumps(dict(
                    backend=f"ncclx_allgatherp_{args.algo}", trial=args.trial, world_size=world, size_bytes=total,
                    time_us=t, algbw_gbps=algbw, busbw_gbps=algbw * (world - 1) / world, wrong=0,
                    warmups=args.warmups, iters=args.iters, dtype="float32", out_of_place=1), sort_keys=True),
                    flush=True)
            barrier(f"done_{total}")
            comm.all_gather_p_free(handle)
            del out, src, start, end, handle
            torch.cuda.empty_cache()

        torch.cuda.synchronize(device)
        barrier("exit")
        sys.stdout.flush()
        if rank == 0:
            time.sleep(2)
        os._exit(0)
    finally:
        whoami.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
