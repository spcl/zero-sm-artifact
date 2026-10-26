#!/usr/bin/env python3
"""Replay one training GEMM chain with a concurrent FSDP AllGather on a second stream.

Each trial runs three modes, in alternating order: compute_alone (the chain only),
comm_alone (the AllGather only) and overlap. The whole chain of Transformer Engine
GEMMs is enqueued behind a device-side gate first; an event recorded right before
GEMM 2 (Qwen: forward GEMM 2; Llama: the second wgrad) releases the AllGather, then
the gate opens. One rank per GPU, eight ranks, launched by Slurm.
"""

import argparse
import csv
import ctypes
import math
import os
import sys
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bridge", required=True)
    p.add_argument("--id-file", required=True)
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--scale-send-bytes", type=int, required=True)
    p.add_argument("--data-send-bytes", type=int, required=True)
    p.add_argument("--compute-kind", choices=("mxfp8_forward", "hybrid_fp8_backward_wgrad"), required=True)
    p.add_argument("--compute-repeats", type=int, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--sweep", choices=("forward", "reverse"), required=True)
    p.add_argument("--warmups", type=int, default=10)
    p.add_argument("--samples", type=int, default=30)
    a = p.parse_args()
    if a.compute_repeats < 2 or a.warmups < 0 or a.samples < 1:
        p.error("need compute-repeats >= 2, warmups >= 0, samples >= 1")
    return a


def bind_bridge(path):
    lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    ull, dbl = ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_double)
    sig = {
        "zs_native_init": [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ull, ull],
        "zs_native_barrier": [],
        "zs_native_begin_trial": [ctypes.c_int],
        "zs_native_arm_compute": [ull],
        "zs_native_mark_phase_ready": [ull],
        "zs_native_mark_compute_done": [ull],
        "zs_native_enqueue_comm": [ctypes.c_int, ull, ull],
        "zs_native_finish_trial": [ctypes.c_int, dbl, dbl, dbl],
        "zs_native_finalize": [],
    }
    for name, argtypes in sig.items():
        getattr(lib, name).argtypes = argtypes
        getattr(lib, name).restype = None
    return lib


def main():
    args = parse_args()
    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    if world != 8:
        raise RuntimeError(f"expected 8 ranks, got {world}")

    import torch

    torch.cuda.set_device(local_rank)
    import transformer_engine
    import transformer_engine.pytorch as te
    import transformer_engine.pytorch.module.linear as te_linear_module
    from transformer_engine.common.recipe import DelayedScaling, Format, MXFP8BlockScaling

    torch.manual_seed(1234 + rank)
    torch.cuda.manual_seed(1234 + rank)
    is_backward = args.compute_kind == "hybrid_fp8_backward_wgrad"
    if not is_backward:
        recipe = MXFP8BlockScaling()
        target_nvtx_suffix = ".gemm"
    else:
        # Megatron's effective delayed-scaling settings for Llama-3-8B FP8 training.
        recipe = DelayedScaling(margin=0, fp8_format=Format.HYBRID, amax_compute_algo="max",
                                amax_history_len=1024, override_linear_precision=(False, False, False))
        target_nvtx_suffix = "wgrad_gemm"
    with te.fp8_model_init(enabled=True, recipe=recipe):
        linear = te.Linear(args.k, args.n, bias=False, params_dtype=torch.bfloat16, device="cuda")
    linear.train(is_backward)
    linear.requires_grad_(is_backward)
    inp = torch.full((args.m, args.k), 1.0 / 16.0, dtype=torch.bfloat16, device="cuda",
                     requires_grad=is_backward)
    compute_stream = torch.cuda.Stream(priority=0)
    phase_stream = torch.cuda.Stream(priority=0)
    stream_value = int(compute_stream.cuda_stream)
    phase_stream_value = int(phase_stream.cuda_stream)
    pre_gemm_event = torch.cuda.Event(enable_timing=False)
    target_gemm_start = torch.cuda.Event(enable_timing=True)
    target_gemm_end = torch.cuda.Event(enable_timing=True)

    output = None
    gradients = None
    grad_output = None

    def launch_forward(first_microbatch=False):
        nonlocal output
        context = torch.enable_grad() if is_backward else torch.no_grad()
        with context, te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            output = linear(inp, is_first_microbatch=first_microbatch)

    def launch(first_microbatch=False):
        nonlocal gradients
        launch_forward(first_microbatch=first_microbatch)
        if is_backward:
            # One fresh forward graph per backward call (fprop, dgrad, wgrad), as in training.
            gradients = torch.autograd.grad(output, (inp, linear.weight), grad_outputs=grad_output)

    # Select the TE path and fill its FP8 weight cache before communication is set up.
    with torch.cuda.stream(compute_stream):
        launch_forward(first_microbatch=True)
        if is_backward:
            grad_output = torch.full_like(output, 1.0 / 32.0)
        for _ in range(4):
            launch()
    compute_stream.synchronize()
    if tuple(output.shape) != (args.m, args.n) or not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("unexpected TE output")

    bridge = bind_bridge(args.bridge)
    bridge.zs_native_init(rank, world, local_rank, os.fsencode(args.id_file),
                          args.scale_send_bytes, args.data_send_bytes)

    # TE brackets every GEMM with nvtx_range_push/pop; use that to mark the target GEMM.
    phase_hook_armed = False
    target_gemm_open = False
    original_push = te_linear_module.nvtx_range_push
    original_pop = te_linear_module.nvtx_range_pop

    def push(message, *a, **kw):
        nonlocal phase_hook_armed, target_gemm_open
        if phase_hook_armed and message.endswith(target_nvtx_suffix):
            target_gemm_start.record(compute_stream)
            pre_gemm_event.record(compute_stream)
            phase_hook_armed = False
            target_gemm_open = True
        return original_push(message, *a, **kw)

    def pop(message, *a, **kw):
        nonlocal target_gemm_open
        if target_gemm_open and message.endswith(target_nvtx_suffix):
            target_gemm_end.record(compute_stream)
            target_gemm_open = False
        return original_pop(message, *a, **kw)

    te_linear_module.nvtx_range_push = push
    te_linear_module.nvtx_range_pop = pop

    if rank == 0:
        env = " ".join(f"{v}={os.environ.get(v, '<unset>')}" for v in (
            "NCCL_CTA_POLICY", "NCCL_SYM_CE_THRESHOLD", "NCCL_SYM_CTAS", "NCCL_WIN_ENABLE"))
        print(f"replay model={args.model} m={args.m} n={args.n} k={args.k} kind={args.compute_kind} "
              f"config={args.config} sweep={args.sweep} warmups={args.warmups} samples={args.samples} "
              f"repeats={args.compute_repeats} scale_bytes={args.scale_send_bytes} "
              f"data_bytes={args.data_send_bytes} torch={torch.__version__} "
              f"transformer_engine={transformer_engine.__version__} {env}", flush=True)

    modes = ((0, "compute_alone"), (1, "comm_alone"), (2, "overlap"))

    def observe(mode, trial):
        nonlocal phase_hook_armed
        bridge.zs_native_begin_trial(1)
        wall_start = time.perf_counter_ns()
        if mode != 1:
            with torch.cuda.stream(compute_stream):
                bridge.zs_native_arm_compute(stream_value)
                launch()
                phase_hook_armed = True
                launch()
                if phase_hook_armed or target_gemm_open:
                    raise RuntimeError("target GEMM was not observed")
                for _ in range(2, args.compute_repeats):
                    launch()
            with torch.cuda.stream(phase_stream):
                phase_stream.wait_event(pre_gemm_event)
                bridge.zs_native_mark_phase_ready(phase_stream_value)
            bridge.zs_native_mark_compute_done(stream_value)
        if mode != 0:
            bridge.zs_native_enqueue_comm(int(mode == 2), args.scale_send_bytes, args.data_send_bytes)
        compute_ms, comm_ms, joined_ms = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        bridge.zs_native_finish_trial(mode, ctypes.byref(compute_ms), ctypes.byref(comm_ms),
                                      ctypes.byref(joined_ms))
        wall_ms = (time.perf_counter_ns() - wall_start) / 1.0e6
        target_ms = 0.0 if mode == 1 else target_gemm_start.elapsed_time(target_gemm_end)
        values = (compute_ms.value, target_ms, comm_ms.value, joined_ms.value, wall_ms)
        if not all(math.isfinite(v) for v in values) or joined_ms.value <= 0:
            raise RuntimeError(f"invalid timing values: {values}")
        return {"trial": trial, "mode": modes[mode][1], "compute_ms": compute_ms.value,
                "target_gemm_ms": target_ms, "comm_ms": comm_ms.value, "joined_ms": joined_ms.value,
                "wall_ms": wall_ms}

    try:
        for warmup in range(1, args.warmups + 1):
            for mode, _ in (modes if warmup % 2 else modes[::-1]):
                observe(mode, warmup)
        rows = []
        for trial in range(1, args.samples + 1):
            for mode, _ in (modes if trial % 2 else modes[::-1]):
                rows.append(observe(mode, trial))
        bridge.zs_native_barrier()

        fields = ("model", "config", "sweep", "rank", "trial", "mode", "compute_ms", "target_gemm_ms",
                  "comm_ms", "joined_ms", "wall_ms", "m", "n", "k", "compute_kind", "compute_repeats",
                  "scale_send_bytes", "data_send_bytes")
        fixed = {"model": args.model, "config": args.config, "sweep": args.sweep, "rank": rank,
                 "m": args.m, "n": args.n, "k": args.k, "compute_kind": args.compute_kind,
                 "compute_repeats": args.compute_repeats, "scale_send_bytes": args.scale_send_bytes,
                 "data_send_bytes": args.data_send_bytes}
        with open(f"{args.output_prefix}.rank{rank}.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
            w.writeheader()
            for row in rows:
                w.writerow({**fixed, **row})
    finally:
        te_linear_module.nvtx_range_push = original_push
        te_linear_module.nvtx_range_pop = original_pop
        bridge.zs_native_finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
