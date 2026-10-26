#!/bin/bash
#SBATCH -N 1 --exclusive -t 00:30:00 -J vccl-p2p -o %x-%j.out
# VCCL send/recv, default (SM) and SM-free (PSM zero-copy) modes, 10 fresh-process trials each.
#   sbatch -N 1 microbench/alt/vccl/run.sh   # scale-up: 2 ranks on one node (NVLink)
#   sbatch -N 2 microbench/alt/vccl/run.sh   # scale-out: 1 rank per node (one 400 Gb/s rail)
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/env/cluster.env"
TRIALS=${TRIALS:-10}; MIN_BYTES=${MIN_BYTES:-8}; MAX_BYTES=${MAX_BYTES:-4G}; WARMUP=${WARMUP:-5}; ITERS=${ITERS:-20}

NODES=$SLURM_JOB_NUM_NODES
case $NODES in
  1) NTASKS=2; TPN=2 ;;
  2) NTASKS=2; TPN=1 ;;
  *) echo "use -N 1 or -N 2" >&2; exit 1 ;;
esac
OUT=$WORK/results/microbench/vccl_b200_${NODES}n${NTASKS}g
mkdir -p "$OUT"

LIB=$WORK/build/vccl/lib
PERF="$WORK/build/nccl-tests-vccl/sendrecv_perf -b $MIN_BYTES -e $MAX_BYTES -f 2 -g 1 -w $WARMUP -n $ITERS"
BASE="NCCL_DEBUG=WARN NCCL_IB_BOND_LOAD_BALANCE=0 NCCL_ENABLE_FAULT_TOLERANCE=0 NCCL_MNNVL_ENABLE=0 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp OMPI_MCA_coll_hcoll_enable=0 OMPI_MCA_btl_tcp_if_include=$NCCL_SOCKET_IFNAME"
declare -A MODE=(
  [normal]="NCCL_PASS_SM=0 NCCL_PSM_FORCE_ZEROCOPY=0"
  [smfree]="NCCL_PASS_SM=1 NCCL_PSM_FORCE_ZEROCOPY=1 NCCL_PXN_DISABLE=1"
)

for mode in normal smfree; do
  for t in $(seq 1 "$TRIALS"); do
    srun --mpi=pmix -N "$NODES" -n "$NTASKS" --ntasks-per-node="$TPN" --cpu-bind=none \
      --container-image="$IMG_PYTORCH" --no-container-mount-home \
      --container-mounts="$WORK:$WORK,$CUDA_HOME:$CUDA_HOME,$MPI_HOME:$MPI_HOME" \
      bash -c "export LD_LIBRARY_PATH=$LIB:$CUDA_HOME/lib64:$MPI_HOME/lib:\$LD_LIBRARY_PATH; exec env $BASE ${MODE[$mode]} $PERF" \
      > "$OUT/$mode.trial$t.log" 2>&1
  done
done
echo "results in $OUT"
