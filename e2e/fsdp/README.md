# FSDP

Twelve Megatron FSDP training recipes, each run under every AllGather arm in paired blocks. The FSDP parameter AllGather is the only thing that changes between arms.

| Recipe | Model | Parallelism | MBS / GBS | Seq | Nodes GB200 / B200 | Step (s) GB200 / B200 |
|---|---|---|---|---|---|---|
| `llama405b-g` | Llama 3.1 405B, FP8 (96 of 126 layers on B200) | DP32 TP2 | 1 / 64 | 8K | 16 / 8 | 14.3 / 11.4 |
| `llama70b-g` | Llama 3 70B, FP8 | DP64 TP1 | 1 / 128 | 8K | 16 / 8 | 4.4 / 5.5 |
| `llama70b-h` | Llama 3 70B, FP8 | DP32 TP2 | 1 / 64 | 8K | 16 / 8 | 3.5 / 3.1 |
| `qwen3-32b-g1` | Qwen3 32B, FP8 | DP32 TP2 | 8 / 256 | 4K | 16 / 8 | 3.3 / 3.5 |
| `qwen3-32b-g2` | Qwen3 32B, FP8 | DP16 TP2 | 4 / 64 | 4K | 8 / 4 | 1.7 / 1.8 |
| `qwen3-32b-h` | Qwen3 32B, FP8 | DP32 TP2 | 1 / 64 | 4K | 16 / 8 | 2.9 / 1.5 |
| `nemotron15b-g1` | Nemotron-4 15B, FP8 | DP32 TP1 | 4 / 256 | 4K | 8 / 4 | 1.8 / 1.8 |
| `nemotron15b-g2` | Nemotron-4 15B, FP8 | DP8 TP4 | 8 / 128 | 4K | 8 / 4 | 1.0 / 1.1 |
| `nemotron15b-h` | Nemotron-4 15B, FP8 | DP8 TP4 | 2 / 32 | 4K | 8 / 4 | 0.8 / 0.5 |
| `qwen3-235b-g` | Qwen3-235B-A22B, BF16 | DP64 EP2 | 1 / 64 | 16K | 16 / 8 | 18.0 / 6.1 |
| `qwen3-next-h` | Qwen3-Next 80B-A3B, BF16 | DP64 EP2 | 1 / 64 | 4K | 16 / 8 | 27.4 / 6.5 |
| `deepseek-v3-g` | DeepSeek-V3 with 16 experts, MXFP8, full-iteration CUDA graphs | DP32 EP4 | 1 / 64 | 4K | 8 / 4 | 4.8 / 4.9 |

The recipe files in `recipes/` hold every setting, including the platform-specific ones. Data is synthetic, except for DeepSeek-V3, which trains on C4.

Software stacks:
- Dense models: NeMo 2 performance recipes in `$IMG_NEMO` (`pretrain_dense.py`).
- Qwen3 MoE models: Megatron-LM `bf32f44` `pretrain_gpt.py` in `$IMG_PYTORCH`, with the HybridEP dispatcher.
- DeepSeek-V3: the Megatron-Bridge Hydra harness of the pipeline-parallel Megatron tree (`e2e/pp`).

Every arm allocates the FSDP buffers from the `ncclMemAlloc` pool and registers that pool as a symmetric window after the first iteration. The arena shim is active in every arm.

## Arms

| Arm | AllGather | NCCL settings |
|---|---|---|
| `legacy` | ring kernel on the registered buffers | all symmetric kernels excluded (`NCCL_SYM_KERNEL`) |
| `symmetric_st` | symmetric unicast kernel `AllGather_ST` | all other symmetric AllGather kernels excluded |
| `symmetric_stmc` | symmetric multicast kernel (`AllGather_RailRing_LsaSTMC` on B200) | |
| `ce_unicast` | copy engines, no SMs | `NCCL_CTA_POLICY=ZERO`, `NCCL_CE_COLL_PLAN_AG_BATCH=16`, `NCCL_CE_ALLGATHER_PLAN_FUSION=1` |
| `ce_multicast` | copy engines, NVLink multicast | as `ce_unicast`, plus `NCCL_CE_COLL_AG_MULTICAST_THRESHOLD=2^50` |

In every arm the symmetric ReduceScatter and AllReduce kernels are excluded, so these collectives run on NCCL's ring and tree kernels. GB200 runs all five arms. B200 runs `legacy`, `symmetric_stmc` and `ce_multicast`; Nemotron-15B-G2 and -H have no symmetric arm on B200. On B200, the CE arm of the three MoE recipes leaves the unicast/multicast choice to NCCL, as measured. `run_session.sh` writes the exact environment of every session to `<dir>/rank.sh`.

## Setup

Run from the repo root, after the prerequisites in `e2e/README.md`:

```bash
sbatch e2e/fsdp/setup.sh               # Qwen3 MoE: $WORK/src/megatron-fsdp (Megatron-LM bf32f44 + patches/), FLA
bash e2e/pp/setup_megatron.sh          # DeepSeek-V3: $WORK/src/megatron-pp
sbatch e2e/pp/build_deps.sh            # HybridEP, nvidia-resiliency-ext (MoE); Megatron-Bridge and harness packages (DeepSeek-V3)
sbatch e2e/pp/deepseek_data.sh         # DeepSeek-V3: C4, tokenizer, model config into $WORK/data/deepseek
```

The dense recipes need only `$IMG_NEMO`. Patch `0001` adds two flags that skip per-parameter checks during FSDP setup. Patch `0002` initializes the shared-expert gate under meta-device initialization. Qwen3-Next needs both. Neither changes Qwen3-235B.

## Run

Paired blocks of one recipe on one allocation. Each block runs one session per arm (the recipe's `ARMS`), back to back, in rotated order:

```bash
sbatch -N 16 -t 12:00:00 e2e/common/run_blocks.sh e2e/fsdp/run_session.sh e2e/fsdp/recipes/llama70b-h.env
```

Use the node count of the table for your platform. Each session runs 100 iterations; on GB200 the MoE recipes run 50. Iterations 1–10 are discarded (1–20 for the 50-iteration sessions). Session time is about 3–5 minutes of startup plus the iterations. On GB200, Qwen3-Next runs one extra warm-up session first (`WARMUP_SESSION=1`), which compiles its Triton kernels. To run a single session inside an allocation:

```bash
bash e2e/fsdp/run_session.sh e2e/fsdp/recipes/<recipe>.env <arm> <dir>
```

The paper uses these block counts:
- GB200: 3 blocks, except 4 for `qwen3-32b-h` and `nemotron15b-g1`.
- B200: `nemotron15b-h` 19; `qwen3-next-h` 8; `qwen3-32b-g1` and `qwen3-32b-g2` 7; `nemotron15b-g2` 6; `llama70b-h` 5; `llama405b-g` and `deepseek-v3-g` 4; `llama70b-g`, `nemotron15b-g1` and `qwen3-235b-g` 3; `qwen3-32b-h` 2.

Several allocations of one recipe can be combined; `analyze.py` treats each `sessions.csv` as its own set of blocks.

## Analyze

```bash
python3 e2e/fsdp/analyze.py $WORK/results/e2e/fsdp [-o speedup.csv]
python3 figures/e2e_fsdp.py speedup.csv
```

`analyze.py` computes the speedup of each arm in each block as `T_legacy / T_arm - 1`. It writes the mean and SD over blocks and the mean legacy step time to `speedup.csv`. `figures/e2e_fsdp.py` draws the paper figure from that file. The figure includes the saving over a 15T-token pretraining run at CoreWeave on-demand prices ($10.50 per GB200 GPU-hour, $8.60 per B200 GPU-hour).
