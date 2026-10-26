#!/bin/bash
#SBATCH -N 1 --exclusive -t 01:00:00 -J nccl-build -o %x-%j.out
# Builds NCCL v2.32.3-1 + patches/ and nccl-tests v2.21.0 into $WORK/build.
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/env/cluster.env"
src=$WORK/src; out=$WORK/build
export LD_LIBRARY_PATH="$MPI_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$src" "$out"

if [ ! -d "$src/nccl" ]; then
  git clone -q --branch v2.32.3-1 https://github.com/NVIDIA/nccl.git "$src/nccl"
  git -C "$src/nccl" -c user.name=artifact -c user.email=artifact@localhost am -q "$ROOT"/nccl/patches/*.patch
fi
make -C "$src/nccl" -j"$(nproc)" src.build BUILDDIR="$out/nccl" CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$NVCC_GENCODE"

if [ ! -d "$src/nccl-tests" ]; then
  git clone -q --branch v2.21.0 https://github.com/NVIDIA/nccl-tests.git "$src/nccl-tests"
fi
make -C "$src/nccl-tests" -j"$(nproc)" MPI=1 MPI_HOME="$MPI_HOME" CUDA_HOME="$CUDA_HOME" \
  NCCL_HOME="$out/nccl" BUILDDIR="$out/nccl-tests" NVCC_GENCODE="$NVCC_GENCODE"
ls "$out/nccl/lib/libnccl.so" "$out/nccl-tests/all_gather_perf" "$out/nccl-tests/sendrecv_perf"
