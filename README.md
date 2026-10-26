# Zero-SM NCCL: reproducibility package

Scripts and patches for the three parts of the evaluation: microbenchmarks, the overlap
benchmark, and end-to-end Megatron-LM training.

| Paper figure | Directory |
|---|---|
| Point-to-point bandwidth and latency (B200) | `microbench/` |
| AllGather bandwidth and latency (GB200, B200) | `microbench/` |
| Per-GEMM slowdown along a compute chain (B200) | `overlap/` |
| FSDP step-time speedup (GB200, B200) | `e2e/fsdp/` |
| PP step-time speedup (GB200, B200) | `e2e/pp/` |

## Requirements

- Slurm with pyxis/enroot and PMIx, and a shared filesystem.
- B200 NVL8 nodes (8 GPUs, one NVLink domain per node, 8×400 Gb/s NICs) or GB200 NVL72 racks (4 GPUs per node,
  18 nodes per NVLink domain). Node counts per experiment are listed in each directory's README.
- On the host: CUDA 12.9, an Open MPI build (HPC-X), GCC, Python 3 with matplotlib.
- Containers: `nvcr.io/nvidia/pytorch:26.04-py3`, `nvcr.io/nvidia/nemo:25.11.01`, and `nvcr.io/nvidia/pytorch:26.08-py3`
  for DeepSeek-V3.

## Setup

```bash
cp env/cluster.env.example env/cluster.env    # set account, partition, WORK, toolchain, NICs
sbatch nccl/build.sh                          # NCCL v2.32.3-1 + nccl/patches, nccl-tests v2.21.0
```

Every experiment uses this one NCCL build. Run all commands from the repository root; builds go to `$WORK/build`
and results to `$WORK/results`.

## Figures

Each directory's README lists its run, analysis and figure commands. The figure scripts are in `figures/`.
