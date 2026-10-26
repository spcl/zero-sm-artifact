#!/bin/bash
#SBATCH -N 1 -t 00:45:00 -J fsdp-setup -o %x-%j.out
# Software of the Megatron-LM FSDP recipes (Qwen3-235B-G, Qwen3-Next-H), in $IMG_PYTORCH:
#   $WORK/src/megatron-fsdp       NVIDIA/Megatron-LM bf32f4415ed53d8393125ff4473d82135b68d848 + patches/megatron-lm/*.patch,
#                                 with the dataset index helper compiled
#   $WORK/build/fsdp-deps/fla     flash-linear-attention 0.4.2 (Qwen3-Next)
# HybridEP and nvidia-resiliency-ext come from e2e/pp/build_deps.sh (target pytorch).
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/env/cluster.env"
src=$WORK/src/megatron-fsdp
commit=bf32f4415ed53d8393125ff4473d82135b68d848
if [[ ! -d $src/.git ]]; then
  git clone -q https://github.com/NVIDIA/Megatron-LM.git "$src"
  git -C "$src" -c advice.detachedHead=false checkout -q --detach "$commit"
  # 0001: opt-in flags that skip per-parameter DTensor checks at FSDP setup; 0002: init of the shared-expert gate
  # under meta-device init. Neither changes the Qwen3-235B-G configuration.
  git -C "$src" apply "$ROOT"/e2e/fsdp/patches/megatron-lm/*.patch
fi
[[ $(git -C "$src" rev-parse HEAD) == "$commit" ]]
mkdir -p "$WORK/build/fsdp-deps"
srun -N1 -n1 --container-image="$IMG_PYTORCH" --container-mounts="$WORK:$WORK" --no-container-mount-home bash -c "
  set -euo pipefail
  make -C $src/megatron/core/datasets
  pip install -q --no-deps --target $WORK/build/fsdp-deps/fla flash-linear-attention==0.4.2 fla-core==0.4.2
"
ls "$src"/megatron/core/datasets/helpers_cpp*.so "$WORK/build/fsdp-deps/fla"
