# Microbenchmarks

Reproduces Fig. `p2p_b200` (point-to-point on B200) and Fig. `allgather` (AllGather on GB200 and B200).
Every point is the mean of 10 fresh-process trials of nccl-tests (or the library's own benchmark), 20 timed
iterations after 5 warm-up iterations; sizes double from 8 B (P2P) or 1 KiB (AllGather) to 4 GiB.

Prerequisites: `env/cluster.env` and `sbatch nccl/build.sh` (NCCL v2.32.3-1 + patches, nccl-tests v2.21.0).
Run every command from the repository root; results go to `$WORK/results/microbench/<run>/`.

## Figure → commands

| Figure / arm | Platform | Command | Nodes | Time |
|---|---|---|---|---|
| p2p_b200: NCCL put, send/recv | B200 | `sbatch -N 1 microbench/p2p/run.sh` | 1 | 10 min |
| | B200 | `sbatch -N 2 microbench/p2p/run.sh` | 2 | 10 min |
| p2p_b200: VCCL, VCCL SM-free | B200 | `sbatch microbench/alt/vccl/build.sh`, then `sbatch -N 1` / `-N 2 microbench/alt/vccl/run.sh` | 1 / 2 | 15 min each |
| allgather: NCCL CE, STMC, legacy | B200 | `sbatch -N 1` / `-N 4` / `-N 8 microbench/allgather/run.sh` | 1 / 4 / 8 | 10 / 20 / 25 min |
| | GB200 | `sbatch -N 1` / `-N 4` / `-N 18 microbench/allgather/run.sh` | 1 / 4 / 18 | 10 / 10 / 15 min |
| allgather: NCCLX (AllGatherP, best of 3 algorithms) | both | `sbatch microbench/alt/ncclx/build.sh` (1.5 h), then `sbatch -N <n> microbench/alt/ncclx/run.sh` (tune + run) | as above | 15–25 min |
| allgather: MSCCL++ (tuned plans) | B200; GB200 4 nodes | `sbatch microbench/alt/mscclpp/build.sh` (15 min), then `sbatch -N <n> microbench/alt/mscclpp/tune.sh` and `run.sh` | as above | 45–90 min |
| allgather: MSCCL++ | GB200 18 nodes | `LL_PLAN=i1_t1024 SIMPLE_PLAN=none sbatch -N 18 microbench/alt/mscclpp/run.sh` | 18 | 15 min |
| allgather: MSCCL++ (NCCL API) | GB200 1 node | `sbatch -N 1 microbench/alt/mscclpp/interposer.sh` | 1 | 10 min |

Library pins, tuning grids and overrides: `alt/ncclx/README.md`, `alt/mscclpp/README.md`; VCCL is `sii-research/VCCL`
at the commit in `alt/vccl/build.sh` with nccl-tests built against it.

NCCL arms (`microbench/allgather/run.sh`, environment per platform in the script):

| Arm | Selection |
|---|---|
| legacy | B200: `-R 2`, all symmetric AllGather kernels excluded; GB200: no registration |
| stmc | `-R 2`, symmetric kernel pinned (`AllGather_STMC`; `AllGather_RailRing_LsaSTMC` across B200 nodes), CE switch disabled |
| ce_mc / ce_uc | Copy Engine path (`NCCL_CTA_POLICY=ZERO` or `-x 2`), multicast forced on / off; B200 across nodes: Rail Direct |
| ce_pipe | B200 across nodes: Copy Engine Rail Pipe at every size |

The figure's NCCL CE curve takes, per size, the CE variant NCCL's tuner selects: multicast or unicast by the CE cost
model, and Rail Pipe at ≥ 4 MiB per rank across B200 nodes (`ce_rows()` in `figures/allgather.py`).

Short validation run (any script): `TRIALS=2 MAX_BYTES=64M sbatch -N <n> <script>`.

## Analysis and figures

```bash
python3 microbench/analysis/build_csv.py $WORK/results/microbench   # -> p2p_b200.csv, ag.csv
python3 figures/p2p_b200.py $WORK/results/microbench/p2p_b200.csv -o p2p_b200.pdf
python3 figures/allgather.py $WORK/results/microbench/ag.csv -o allgather.pdf
```

Each run is `$WORK/results/microbench/<run>` (`<run>` = `<benchmark>_<platform>_<nodes>n<gpus>g`). Requires Python 3
with matplotlib.
