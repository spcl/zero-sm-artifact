# End-to-end training

Step time of real training recipes under each NCCL communication arm, measured in paired blocks: within each block, one session per arm runs back to back on the same nodes. Speedup per block is T_legacy / T_arm - 1. Each figure shows the mean and SD over blocks.

Prerequisites, from the repo root:

```bash
cp env/cluster.env.example env/cluster.env   # edit
sbatch nccl/build.sh                          # NCCL used by every arm
sbatch e2e/common/build_shim.sh               # allocator shim, active in every arm
```

`e2e/common/` holds the shim, the paired-block runner and the step-time parser (see `e2e/common/README.md`).

## FSDP

Twelve Megatron FSDP recipes on GB200 NVL72 and B200 NVL8: `e2e/fsdp/README.md`. Figure: `figures/e2e_fsdp.py`.

## Pipeline parallelism

See `e2e/pp/README.md`.
