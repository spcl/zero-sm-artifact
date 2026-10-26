#!/bin/bash
#SBATCH --exclusive -t 01:00:00 -J ncclx-agp -o %x-%j.out
# NCCLX persistent AllGatherP on the allocated nodes (one rank per GPU):
#   tune: each CTran algorithm, 3 trials at 3 pilot sizes; winner = highest geomean of median bus bandwidth
#   production: winner, TRIALS fresh-process trials over the full size sweep.
# Usage: sbatch -N <nodes> microbench/alt/ncclx/run.sh
# Overrides: TRIALS (10), TUNE_TRIALS (3), MIN_BYTES (1K), MAX_BYTES (4G), ALGO (skip tuning), OUT.
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/env/cluster.env"
D=$ROOT/microbench/alt/ncclx
B=$WORK/build/ncclx
N=$SLURM_JOB_NUM_NODES; G=$((N * GPUS_PER_NODE))
OUT=${OUT:-$WORK/results/microbench/ncclx_${PLATFORM}_${N}n${G}g}
TRIALS=${TRIALS:-10}; TUNE_TRIALS=${TUNE_TRIALS:-3}
SIZES=$(python3 "$D/tools.py" sizes "$G" "${MIN_BYTES:-1K}" "${MAX_BYTES:-4G}")
mkdir -p "$OUT"
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
PORT=$((20000 + SLURM_JOB_ID % 20000))

# bench <dir> <algo> <sizes> <trial> <warmups> <iters>
bench() {
  PORT=$((PORT + 1))
  srun --kill-on-bad-exit=1 --mpi=none -N "$N" --ntasks-per-node="$GPUS_PER_NODE" --cpu-bind=none \
    --container-image="$IMG_PYTORCH" --container-writable --no-container-mount-home \
    --container-mounts="$WORK:$WORK,$ROOT:$ROOT,$CUDA_HOME:/opt/cuda-12.9" \
    --output="$1/trial_$4.rank%t.log" \
    env PYTHONPATH="$B/runtime:$B/torch" \
        LD_LIBRARY_PATH="$B/runtime/torchcomms:$B/torch/torch/lib:$B/prefix/lib:/opt/cuda-12.9/lib64:/opt/cuda-12.9/extras/CUPTI/lib64" \
        MASTER_ADDR="$MASTER" MASTER_PORT="$PORT" \
        NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" NCCL_CLIENT_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
        NCCL_CTRAN_ENABLE=1 NCCL_DEBUG=WARN NCCL_DEBUG_SUBSYS=INIT,COLL \
    python3 "$D/agp_bench.py" --algo "$2" --sizes "$3" --trial "$4" --warmups "$5" --iters "$6"
}

if [ -z "${ALGO:-}" ]; then
  PILOT=$(python3 "$D/tools.py" pilot "$SIZES")
  ALGOS="ctdirect ctpipeline"
  (( (N & (N - 1)) == 0 )) && ALGOS="$ALGOS ctsrdpipeline"   # ctsrdpipeline needs a power-of-two node count
  for a in $ALGOS; do
    mkdir -p "$OUT/tune/$a"
    for t in $(seq 1 "$TUNE_TRIALS"); do bench "$OUT/tune/$a" "$a" "$PILOT" "$t" 3 10; done
    python3 "$D/tools.py" collect "$OUT/tune/$a" "$a" "$PILOT" "$TUNE_TRIALS"
  done
  ALGO=$(python3 "$D/tools.py" select "$OUT/tune" $ALGOS)
fi
echo "algorithm: $ALGO"
for t in $(seq 1 "$TRIALS"); do bench "$OUT" "$ALGO" "$SIZES" "$t" 5 20; done
python3 "$D/tools.py" collect "$OUT" "$ALGO" "$SIZES" "$TRIALS"
rm -f "$OUT"/trial_*.rank[1-9]*.log "$OUT"/tune/*/trial_*.rank[1-9]*.log
echo "wrote $OUT/trials.csv"
