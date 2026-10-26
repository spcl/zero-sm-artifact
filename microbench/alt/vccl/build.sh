#!/bin/bash
#SBATCH -N 1 --exclusive -t 01:00:00 -J vccl-build -o %x-%j.out
# Builds VCCL (+ mlx5-lag-weak.patch) into $WORK/build/vccl and nccl-tests against it into $WORK/build/nccl-tests-vccl.
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/env/cluster.env"
src=$WORK/src; out=$WORK/build
mkdir -p "$src" "$out"

if [ ! -d "$src/vccl" ]; then
  git clone -q https://github.com/sii-research/VCCL.git "$src/vccl"
  git -C "$src/vccl" checkout -q 583e314e7888f806202a7be17dfe9402c5a9db67
  # Makes the bonded-port call optional so VCCL loads where libmlx5 lacks it (load balancing is off in the runs).
  git -C "$src/vccl" apply "$ROOT/microbench/alt/vccl/mlx5-lag-weak.patch"
fi
if [ ! -d "$src/nccl-tests-vccl" ]; then
  git clone -q https://github.com/NVIDIA/nccl-tests.git "$src/nccl-tests-vccl"
  git -C "$src/nccl-tests-vccl" checkout -q 717b68318278e93f371d8ffb46b076069d7c7851
fi

srun -N 1 -n 1 --container-image="$IMG_PYTORCH" --no-container-mount-home \
  --container-mounts="$WORK:$WORK,$CUDA_HOME:$CUDA_HOME,$MPI_HOME:$MPI_HOME" \
  bash -c "set -e; export PATH=$CUDA_HOME/bin:\$PATH
    make -C $src/vccl -j\$(nproc) src.build BUILDDIR=$out/vccl CUDA_HOME=$CUDA_HOME NVCC_GENCODE='$NVCC_GENCODE'
    make -C $src/nccl-tests-vccl -j\$(nproc) MPI=1 MPI_HOME=$MPI_HOME CUDA_HOME=$CUDA_HOME \
      NCCL_HOME=$out/vccl BUILDDIR=$out/nccl-tests-vccl NVCC_GENCODE='$NVCC_GENCODE'"
ls "$out/vccl/lib/libnccl.so" "$out/nccl-tests-vccl/sendrecv_perf"
