#!/bin/bash
# usage: build.sh NCCL_DIR OUT_DIR   (run inside $IMG_PYTORCH; uses its CUDA toolkit)
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
NCCL=$1 OUT=$2
CUDA=/usr/local/cuda
mkdir -p "$OUT"
"$CUDA/bin/nvcc" -O3 -std=c++17 -lineinfo ${NVCC_GENCODE:--gencode=arch=compute_100,code=sm_100} \
  -Xcompiler=-Wall,-Wextra --shared -Xcompiler=-fPIC -I"$NCCL/include" -L"$NCCL/lib" \
  -Xlinker -rpath -Xlinker "$NCCL/lib" "$SRC/bridge.cu" -lnccl -lcuda -ldl -lpthread -lrt \
  -o "$OUT/libbridge.so"
g++ -O2 -std=c++17 -Wall -Wextra -shared -fPIC -I"$CUDA/include" "$SRC/cupti_trace.cpp" \
  -L"$CUDA/lib64" -Wl,-rpath,"$CUDA/lib64" -lcupti -o "$OUT/libcupti_trace.so"
ls -l "$OUT/libbridge.so" "$OUT/libcupti_trace.so"
