# Pipeline parallelism

Seven Megatron-LM pipeline-parallel recipes, each run with two pipeline P2P transports:

- `two_sided` (baseline): NCCL Send/Recv between pipeline stages.
- `one_sided`: puts into the receiver's buffer through PyTorch symmetric memory (`put_signal` / `wait_signal`, which
  map to `ncclPutSignal` / `ncclWaitSignal`). The receiver keeps a ring of receive slots sized by the warm-up depth of
  the schedule. Megatron flag `--use-symmetric-memory-p2p`.

All other settings are the same in both arms. DP, TP and EP collectives run on the default NCCL kernels.

| Recipe | GPUs | Parallelism | MBS / GBS (GA) | Seq | B200 nodes | GB200 nodes |
|---|---|---|---|---|---|---|
| `llama405b` Llama-3.1 405B, FP8 | 64 | DP2 TP4 PP8 VPP2 | 1 / 64 (32) | 8K | 8 | 16 |
| `llama70b` Llama-3 70B | 64 | DP2 TP4 PP4 VPP5 CP2 | 1 / 64 (32) | 8K | 8 | 16 |
| `qwen3-32b` Qwen3-32B | 64 | DP4 TP8 PP2 VPP2 | 2 / 64 (8) | 4K | 8 | 16 |
| `nemotron15b` Nemotron-4 15B, FP8 | 32 | DP4 TP4 PP2 VPP2 | 2 / 32 (4) | 4K | 4 | 8 |
| `qwen3-235b` Qwen3-235B-A22B | 64 | DP8 TP1 PP8 VPP2 EP8 | 1 / 64 (8) | 4K | 8 | 16 |
| `qwen3-next` Qwen3-Next 80B-A3B shape | 64 | DP8 TP1 PP8 VPP2 EP8 | 1 / 64 (8) | 4K | 8 | 16 |
| `deepseek-v3` DeepSeek-V3, 16 experts, MXFP8 | 32 | DP4 TP1 PP8 VPP4 EP4 | 1 / 64 (16) | 4K | 4 | 8 |

Recipes are in `recipes/`. They use mock data, except DeepSeek-V3, which uses C4 with full-iteration CUDA graphs. Each
session trains 100 iterations and logs every 10. The step time of a session is the mean over iterations 11–100.

## Setup

Run from the repo root, after the prerequisites in `e2e/README.md`:

```bash
bash e2e/pp/setup_megatron.sh     # $WORK/src/megatron-pp: Megatron-LM 9cfaf94 + megatron/0001-*.patch
sbatch e2e/pp/build_deps.sh       # $WORK/build/pp-deps: HybridEP, nvidia-resiliency-ext; DeepSeek-V3 packages
sbatch e2e/pp/deepseek_data.sh    # $WORK/data/deepseek: C4, tokenizer, model config (DeepSeek-V3 only)
```

The patch makes three changes. It initializes the NCCL user-buffer groups in an order that is the same on every
pipeline stage. It skips the grad-norm skip threshold while optimizer CUDA graphs are active. It adds the DeepSeek-V3
Hydra harness (`examples/zero_sm_pp/deepseek_mlperf`).

The six Megatron recipes run in `$IMG_PYTORCH`. DeepSeek-V3 runs in `nvcr.io/nvidia/pytorch:26.08-py3`, which has the
Transformer Engine fused grouped MLP, with Megatron-Bridge `c705936` on `PYTHONPATH`.

## Run

Paired blocks on one allocation. Within a block both arms run back to back; consecutive blocks alternate which arm
runs first:

```bash
sbatch -N 8 e2e/common/run_blocks.sh e2e/pp/run_session.sh e2e/pp/recipes/qwen3-next.env two_sided one_sided
```

Paper block counts:
- GB200: `BLOCKS=4` for every recipe.
- B200: `BLOCKS=3` for `llama405b`, `llama70b`, `qwen3-235b` and `qwen3-next`; `BLOCKS=5` for the other three.

One session takes from 3 minutes (`nemotron15b`) to about 1 hour (`llama405b` on GB200), so raise `-t` for full
runs. `ITERS` shortens every session. A single session runs inside an allocation with
`bash e2e/pp/run_session.sh <recipe.env> <arm> <dir>`, which writes `<dir>/rank.sh` (the per-rank environment and
command) and `<dir>/train.log`.

## Analyze

```bash
python3 e2e/pp/analyze.py $WORK/results/e2e/pp [-o speedup.csv]
python3 figures/e2e_pp.py speedup.csv
```

`analyze.py` pairs the two sessions of each block and computes the speedup `T_two_sided / T_one_sided - 1`. It writes
the mean and SD over blocks, the mean two-sided step time and the bubble share `b = (PP-1)/(GA*VPP+PP-1)` to
`speedup.csv`. `figures/e2e_pp.py` draws the paper figure from that file. It also estimates the saving of
one-sided puts over a 15T-token pretraining run.
