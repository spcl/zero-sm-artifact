#!/bin/bash
#SBATCH -N 1 --exclusive -t 03:00:00 -J ncclx-build -o %x-%j.out
# Builds TorchComms/NCCLX (meta-pytorch/torchcomms 5e9fc2dd + torchcomms.patch) against torch 2.11.0+cu129
# inside $IMG_PYTORCH, into $WORK/build/ncclx/{torch,prefix,runtime}.
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/env/cluster.env"
SRC=$WORK/src/torchcomms
B=$WORK/build/ncclx
mkdir -p "$WORK/src" "$B"

if [ ! -d "$SRC" ]; then
  git clone -q https://github.com/meta-pytorch/torchcomms.git "$SRC"
  git -C "$SRC" checkout -q 5e9fc2ddb5c993e493953299066568225187fefc
  git -C "$SRC" apply "$ROOT/microbench/alt/ncclx/torchcomms.patch"
fi

srun -N 1 -n 1 --cpu-bind=none --container-image="$IMG_PYTORCH" --container-writable --no-container-mount-home \
  --container-mounts="$WORK:$WORK,$CUDA_HOME:/opt/cuda-12.9" --container-workdir="$SRC" \
  env B="$B" bash -c '
set -euxo pipefail
# torch 2.11.0+cu129 wheel only (no dependencies); its CUDA 12.9 libraries come from /opt/cuda-12.9.
if [ ! -d "$B/torch/torch" ]; then
  pip download -q --no-deps -d "$B/wheel" torch==2.11.0+cu129 --index-url https://download.pytorch.org/whl/cu129
  python3 -m zipfile -e "$B"/wheel/torch-2.11.0+cu129-*.whl "$B/torch"
fi
export PYTHONPATH=$B/torch
export LD_LIBRARY_PATH=$B/torch/torch/lib:$B/prefix/lib:/opt/cuda-12.9/lib64:/opt/cuda-12.9/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}
export CONDA_PREFIX=$B/prefix CMAKE_PREFIX_PATH=$B/prefix
export CUDA_HOME=/opt/cuda-12.9 CUDA_BIN_PATH=/opt/cuda-12.9 CUDACXX=/opt/cuda-12.9/bin/nvcc PATH=/opt/cuda-12.9/bin:$PATH
export CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/opt/cuda-12.9/bin/nvcc -DCUDA_TOOLKIT_ROOT_DIR=/opt/cuda-12.9 -DCUDAToolkit_ROOT=/opt/cuda-12.9"
export BUILDDIR=$B/ncclx NVCC_ARCH=b200 TORCH_CUDA_ARCH_LIST=10.0a NCCL_BUILD_JOBS=$(nproc) MAX_JOBS=$(nproc)
export CXXSTD=-std=c++20 NCCL_PATCH_FMT_NVCC_CXX20=1 NCCL_SKIP_CONDA_INSTALL=1
unset NVCC_GENCODE   # NVCC_ARCH=b200 selects sm_100a
python3 -c "import torch; assert torch.__version__ == \"2.11.0+cu129\""
bash build_ncclx.sh
export NCCL_BUILD_SKIP_DEPS=1 USE_NCCL=OFF USE_NCCLX=ON USE_GLOO=OFF USE_RCCL=OFF USE_RCCLX=OFF USE_XCCL=OFF \
       USE_TRANSPORT=OFF USE_TRANSPORT_CCA_HOOK=OFF USE_TRITON=OFF
python3 setup.py build --build-base "$B/setuptools" bdist_wheel --dist-dir "$B/wheels" --bdist-dir "$B/bdist"
rm -rf "$B/runtime"
python3 -m zipfile -e "$B"/wheels/torchcomms-*.whl "$B/runtime"
ls "$B"/runtime/torchcomms/_comms_ncclx*.so
'
