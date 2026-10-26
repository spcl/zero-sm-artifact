# Overlap benchmark

Measures how much a concurrent FSDP AllGather slows each GEMM of a training compute chain, on one B200 NVL8 node (8 ranks, one per GPU).

| Model | Chain (Transformer Engine `te.Linear`) | AllGather per rank |
|---|---|---|
| `qwen3_30b_a3b` | 128 MXFP8 forward GEMMs, M=4096 N=5120 K=2048 | 144 MiB (150994944 B) |
| `llama3_8b` | 32 FP8 (delayed scaling, hybrid) fprop/dgrad/wgrad triplets, M=N=K=4096 | 26 MiB (27267072 B) + 2 KiB scales |

Each trial runs the chain alone (clean pass), the AllGather alone, and both together (overlap pass), in alternating order. The whole chain is enqueued behind a device-side gate; an event recorded before GEMM 2 (Qwen) or the second wgrad (Llama, chain position 6) releases the AllGather on a high-priority stream. CUPTI records every kernel. Each overlap-pass GEMM is paired with the clean-pass GEMM of the same process, rank, trial and chain position; the plotted value per position is the median over 4 processes of the per-process median over 8 ranks x 30 trials.

| Arm (`config`) | AllGather implementation | NCCL environment |
|---|---|---|
| `zero` | copy engine, 0 CTAs | `NCCL_CTA_POLICY=ZERO NCCL_SYM_CE_THRESHOLD=1099511627776` |
| `legacy` | Ring kernel, default 32 CTAs | `NCCL_WIN_ENABLE=0` (buffers allocated with `ncclMemAlloc`, not registered) |
| `default` | symmetric STMC kernel, 4 CTAs | `NCCL_SYM_CE_THRESHOLD=1099511627776 NCCL_SYM_CTAS=4`, STMC as the only AllGather symmetric kernel |
| `cta_32` | symmetric STMC kernel, 32 CTAs | as `default`, with `NCCL_SYM_CTAS=32` |

## Run

Prerequisites: `env/cluster.env` and the NCCL build (`sbatch nccl/build.sh`). The job builds `src/bridge.cu` (NCCL side) and `src/cupti_trace.cpp` (CUPTI collector) inside `$IMG_PYTORCH` into `$WORK/build/overlap`, then runs every model and arm in fresh processes, alternating the arm order between replicates.

```bash
sbatch overlap/run.sh                                     # paper run, about 40 min
REPLICATES=2 TRIALS=5 WARMUPS=2 MODELS=qwen3_30b_a3b sbatch -t 00:20:00 overlap/run.sh   # short check
```

Overrides: `REPLICATES` (4), `TRIALS` (30), `WARMUPS` (10), `MODELS` (`qwen3_30b_a3b llama3_8b`), `ARMS` (`default cta_32 zero legacy`), `OUT` (`$WORK/results/overlap/<jobid>`).

## Output

`$OUT/<model>/replicate_<r>/<forward|reverse>/<arm>.rank<i>.tsv` (per-trial timings) and `.rank<i>.kernel_activity.tsv.gz` (CUPTI trace), then `overlap/analyze.py` writes into `$OUT`:

- `observations_<model>.tsv.gz`: every paired GEMM (clean and overlap duration in ns).
- `chain_position.tsv`: per chain position, statistics over processes; `process_all_slowdown_pct_median_median` is plotted.
- `kernels.tsv`: GEMM and AllGather kernels seen per arm. Expected: `zero` none, `legacy` `ncclDevKernel_AllGather_RING` grid 32, `default` `ncclSymkDevKernel_AllGather_STMC` grid 4, `cta_32` the same kernel with grid 32.

It also prints the per-panel figure statistics (GEMMs slowed by more than 1% and their median slowdown).

```bash
python3 figures/overlap_timeline.py $OUT/chain_position.tsv   # -> figures/overlap_timeline.pdf
```

The figure scripts need matplotlib (the paper figure was rendered with 3.9.4).
