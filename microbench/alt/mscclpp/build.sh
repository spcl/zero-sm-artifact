#!/bin/bash
#SBATCH -N 1 --exclusive -t 00:30:00 -J mscclpp-build -o %x-%j.out
# Builds MSCCL++ (pinned commit + mscclpp.patch) in the PyTorch container: the AllGather executor
# (executor_allgather_exact.cc) and the NCCL-API library, into $WORK/build/mscclpp; plus the plan generators' Python dependency.
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/env/cluster.env"
D=$ROOT/microbench/alt/mscclpp; src=$WORK/src/mscclpp; B=$WORK/build/mscclpp
if [ ! -d "$src" ]; then
  git clone -q https://github.com/microsoft/mscclpp.git "$src"
  git -C "$src" checkout -q 510e829051e5b9e2a71d3e8dbe4fcf2764108a73
  git -C "$src" apply "$D/mscclpp.patch"     # executor target, MAX_OPERATION 64 -> 256, >48 KiB plan in shared memory
  cp "$D/executor_allgather_exact.cc" "$src/test/"
fi
arch=$(sed -E 's/.*sm_([0-9]+).*/\1/' <<< "$NVCC_GENCODE")
srun -N1 -n1 --container-image="$IMG_PYTORCH" --no-container-mount-home \
  --container-mounts="$WORK:$WORK,$CUDA_HOME:$CUDA_HOME,$MPI_HOME:$MPI_HOME" bash -lc "
  set -e; export CUDA_HOME=$CUDA_HOME MPI_HOME=$MPI_HOME
  export PATH=$CUDA_HOME/bin:$MPI_HOME/bin:\$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$MPI_HOME/lib:\${LD_LIBRARY_PATH:-}
  cmake -S $src -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=$arch -DMSCCLPP_GPU_ARCHS=$arch -DMSCCLPP_BUILD_PYTHON_BINDINGS=OFF -DMSCCLPP_BUILD_TESTS=ON
  cmake --build $B --target mscclpp_nccl executor_allgather_exact -j\$(nproc)
  python3 -m pip install -q --target $B/pydeps sortedcontainers==2.4.0"
ls "$B/bin/executor_allgather_exact" "$B/lib/libmscclpp_nccl.so"
