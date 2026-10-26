# e2e/common

Pieces shared by the FSDP and pipeline-parallel experiments.

| File | Use |
|---|---|
| `build_shim.sh` | `sbatch e2e/common/build_shim.sh` builds the arena shim inside `$IMG_NEMO` and `$IMG_PYTORCH` into `$WORK/build/shim/{nemo,pytorch}/pyarena_malloc_shim.so` (`SHIM_IMAGES` selects images). |
| `shim.sh` | Sourced by session scripts. `shim_env <nemo\|pytorch> [preload]` sets the array `SHIM_ENV` (`LD_PRELOAD` and the glibc `MALLOC_*` settings); put `env "${SHIM_ENV[@]}"` in front of the rank command. |
| `run_blocks.sh` | `sbatch -N <nodes> e2e/common/run_blocks.sh <session-script> <recipe.env> [<arm>...]` (default: the recipe's `ARMS`) runs `BLOCKS` (default 3) paired blocks. Block b runs each arm once, starting at arm (b-1) mod n. |
| `steptime.py` | `steptime.py <log> [--warmup N]` prints `<mean step ms> <points>`. It parses NeMo `train_step_timing` and Megatron `elapsed time per iteration (ms)` lines and drops iterations <= N (default 10). |

The allocator shim makes CPython reuse its arenas instead of unmapping them. Unmapping host memory on GB200 stalls the GPUs, which share the process address space. The shim is active in every arm.

## Session script contract

`run_blocks.sh` calls `bash <session-script> <recipe.env> <arm> <dir>` once per session, inside the allocation. The session script must write the training log to `<dir>/train.log`. `run_blocks.sh` passes `ITERS` and any other environment through. It reads `WARMUP` (discarded iterations) and `WARMUP_SESSION` (1: one discarded session before block 1) from the environment, or else from the recipe file. Without arm arguments it runs the recipe's `ARMS`.

Output: `$WORK/results/e2e/<family>/<recipe>/<jobid>/` (override with `OUT`), where `<family>` is the directory above `recipes/`. It contains `block<b>/<arm>/train.log` and `sessions.csv` with the columns `platform,recipe,block,arm,step_ms,points`.
