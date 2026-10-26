# Sourced by the NCCL microbenchmark scripts (bare host, nccl-tests from nccl/build.sh, MPI via srun).
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/env/cluster.env"
NT=$WORK/build/nccl-tests
export LD_LIBRARY_PATH=$WORK/build/nccl/lib:$MPI_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
# MPI only bootstraps nccl-tests: TCP/shared-memory transports, no UCX/HCOLL.
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp OMPI_MCA_coll_hcoll_enable=0 OMPI_MCA_btl_tcp_if_include=$NCCL_SOCKET_IFNAME
export NCCL_DEBUG=WARN

TRIALS=${TRIALS:-10}
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-20}
MAX_BYTES=${MAX_BYTES:-4G}
NODES=$SLURM_JOB_NUM_NODES

# step <ranks per node> <log> <cmd...>: one fresh-process nccl-tests run across all nodes of the allocation
step() {
  local tpn=$1 log=$2; shift 2
  srun --mpi=pmix -N "$NODES" --ntasks-per-node="$tpn" --cpu-bind=none "$@" > "$log" 2>&1 \
    || echo "failed: $log" >&2
}
