#!/bin/bash
#SBATCH -N 1 --exclusive -t 00:30:00 -J mb-p2p -o %x-%j.out
# Fig. p2p_b200, NCCL arms: two-sided send/recv (SM kernels) vs one-sided put (ncclPutSignal, host RMA, -H 1 -R 2).
#   sbatch -N 1 microbench/p2p/run.sh   2 ranks on one node (NVLink)
#   sbatch -N 2 microbench/p2p/run.sh   1 rank on each of two nodes (GPU 0 and its rail)
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/microbench/common.sh"
TPN=$((2 / NODES))
OUT=${OUT:-$WORK/results/microbench/p2p_${PLATFORM}_${NODES}n2g}
mkdir -p "$OUT"
SR=("$NT/sendrecv_perf" -b 8 -e "$MAX_BYTES" -f 2 -g 1 -w "$WARMUP" -n "$ITERS")

for t in $(seq "$TRIALS"); do step "$TPN" "$OUT/sendrecv.trial$t.log" "${SR[@]}"; done
for t in $(seq "$TRIALS"); do step "$TPN" "$OUT/put.trial$t.log" "${SR[@]}" -H 1 -R 2; done
echo "results: $OUT"
