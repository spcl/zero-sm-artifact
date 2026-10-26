# MSCCL++ AllGather

MSCCL++ `microsoft/mscclpp@510e829051e5b9e2a71d3e8dbe4fcf2764108a73` with `mscclpp.patch` (operation limit 64 -> 256 so the
multi-node Simple plans fit, and the matching shared-memory attribute), run in `$IMG_PYTORCH`.
The executor `executor_allgather_exact.cc` runs one JSON plan over the nccl-tests sizes: FP32, out-of-place, 5 warmups,
20 timed operations, rank-maximum CUDA-event time, full output validation outside timing, one fresh process per trial.

```bash
sbatch microbench/alt/mscclpp/build.sh                # once, ~15 min
sbatch -N <nodes> microbench/alt/mscclpp/tune.sh      # plan grid + LL->Simple switch -> tuning/selection.env
sbatch -N <nodes> microbench/alt/mscclpp/run.sh       # 10 trials per protocol -> trials.csv
```

Tuning: LL plans (instances 1/2/4/8 x threads 256/512/768/1024) screened at 1 KiB and 16 MiB, Simple plans (thread-block
groups 1/2/4 x threads 256/512/1024) at 256 MiB, 1 GiB, 4 GiB; the best geometric mean of each is kept, and three crossover
trials (LL 1 KiB..16 MiB, 32 MiB on one node; Simple 16 MiB..4 GiB) pick the single LL->Simple switch size.

Paper placements: B200 1/4/8 nodes and GB200 4 nodes use tune.sh + run.sh. GB200 18 nodes uses the untuned default LL plan
(`LL_PLAN=i1_t1024 SIMPLE_PLAN=none sbatch -N 18 .../run.sh`; larger sizes gave no valid result). GB200 1 node uses the
NCCL-API library under nccl-tests: `sbatch -N 1 microbench/alt/mscclpp/interposer.sh`.

Results: `$WORK/results/microbench/mscclpp_<platform>_<N>n<G>g/`. Overrides: `TRIALS`, `MAX_BYTES`, `LL_PLAN`/`SIMPLE_PLAN`
(skip tuning), `LL_INSTANCES`, `LL_THREADS`, `SIMPLE_TBG`, `SIMPLE_THREADS`, `CROSSOVER_TRIALS`.
