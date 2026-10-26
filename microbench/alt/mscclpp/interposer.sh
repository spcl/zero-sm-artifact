#!/bin/bash
#SBATCH -N 1 --exclusive -t 00:30:00 -J mb-mscclpp-nccl-api -o %x-%j.out
# MSCCL++ through its NCCL-API library (built-in AllGather algorithms) under nccl-tests all_gather_perf, TRIALS fresh processes.
# This is the paper's MSCCL++ curve for GB200, 1 node.   sbatch -N 1 microbench/alt/mscclpp/interposer.sh
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/microbench/alt/mscclpp/lib.sh"
for k in $(seq 1 "$TRIALS"); do
  step "$GPUS_PER_NODE" "$OUT/interposer.trial$k.log" -t 15 "${CT[@]}" env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    LD_PRELOAD="$B/lib/libmscclpp_nccl.so" MSCCLPP_NCCL_SYMMETRIC_MEMORY=1 MSCCLPP_DEBUG=WARN \
    "$NT/all_gather_perf" -b 1K -e "$MAX_BYTES" -f 2 -g 1 -w "$WARMUP" -n "$ITERS"
done
