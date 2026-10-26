#!/bin/bash
#SBATCH -N 1 --exclusive -t 02:00:00 -J pp-deps -o %x-%j.out
# Python packages the PP recipes add to their container, installed per image into $WORK/build/pp-deps/<target>
# (used through PYTHONUSERBASE). Targets (DEPS_TARGETS, default both):
#   pytorch   in $IMG_PYTORCH: HybridEP (DeepEP) and nvidia-resiliency-ext, for the six Megatron recipes
#   deepseek  in the DeepSeek-V3 image: HybridEP and the harness requirements; Megatron-Bridge in $WORK/src/megatron-bridge
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/env/cluster.env"
src=$WORK/src
mkdir -p "$src"

clone() {   # clone <url> <dir> <commit>
  [[ -d $2 ]] || git clone -q "$1" "$2"
  git -C "$2" -c advice.detachedHead=false checkout -q --detach "$3"
}
clone https://github.com/deepseek-ai/DeepEP.git "$src/deepep" origin/hybrid-ep

build() {   # build <target> <image> <DeepEP commit> <build setup> <pip requirement>...
  local target=$1 image=$2 commit=$3 setup=$4; shift 4
  local out=$WORK/build/pp-deps/$target tree=$WORK/build/pp-deps/deepep-$target
  git -C "$src/deepep" worktree prune
  [[ -d $tree ]] || git -C "$src/deepep" worktree add -q --detach "$tree" "$commit"
  mkdir -p "$out"
  # PIP_CONSTRAINT is cleared as in the image the DeepSeek-V3 results were measured with; installed packages stay.
  srun -N1 -n1 --container-image="$image" --container-mounts="$WORK:$WORK" --no-container-mount-home bash -c "
    set -euo pipefail
    export PYTHONUSERBASE=$out PIP_CACHE_DIR=$out/.pip-cache PIP_CONSTRAINT= MAX_JOBS=24
    export CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl\${CPLUS_INCLUDE_PATH:+:\$CPLUS_INCLUDE_PATH}
    $setup
    cd $tree && pip install --user --no-deps --no-build-isolation .
    [[ \$# -eq 0 ]] || pip install --user \"\$@\"
    cd / && python3 -c 'import deep_ep, hybrid_ep_cpp; print(\"deep_ep\", deep_ep.__file__)'
  " bash "$@"
}

for target in ${DEPS_TARGETS:-pytorch deepseek}; do
  case $target in
    pytorch)
      build pytorch "$IMG_PYTORCH" 17cfb817bccec3a9c247013360cc550c2bac441e '
        pip install --user --no-deps nvidia-nvshmem-cu13==3.4.5
        export NVSHMEM_DIR=$(python3 -c "import nvidia.nvshmem as m; print(list(m.__path__)[0])")
        export TORCH_CUDA_ARCH_LIST=10.0 HYBRID_EP_MULTINODE=0 USE_NIXL=0' \
        nvidia-resiliency-ext==0.6.0 ;;
    deepseek)
      clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git "$src/megatron-bridge" c70593623dd8f25474d58740815e6bf48d4c6ef2
      image=$(source "$ROOT/e2e/pp/recipes/deepseek-v3.env"; echo "$IMAGE")
      # Transformer Engine 2.20, cuDNN frontend 1.27.0 and CUTLASS DSL 4.8.0 (CUDA 13 runtime): the closest public match to the image the DeepSeek-V3 results were measured with.
      build deepseek "$image" 94a9f8f6b146c07d97ec58f67cd6d303296d6098 'export TORCH_CUDA_ARCH_LIST="10.0;10.3;12.0"
        NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS="100a" pip install --user --no-deps --no-build-isolation \
          "git+https://github.com/NVIDIA/TransformerEngine.git@6ea2a74a9e98c99e6d7b164a33775cc457520027"
        pip install --user nvidia-cudnn-frontend==1.27.0 "nvidia-cutlass-dsl[cu13]==4.8.0"
        # The measured TE build ran the fused grouped-MLP backward with its atomic dprob accumulation under
        # NVTE_ALLOW_NONDETERMINISTIC_ALGO=0; TE 2.20 rejects that combination, so drop only that check.
        sed -i "s/deterministic_dactivation = (/deterministic_dactivation = False and (/" \
          $PYTHONUSERBASE/lib/python3.12/site-packages/transformer_engine/pytorch/ops/fused/grouped_mlp.py' \
        "git+https://github.com/mlcommons/logging.git@6.0.0-rc3" \
        "git+https://github.com/NVIDIA/mlperf-common.git@c02c93c0414071400a10716d9cd5cb8f668b78d4" \
        transformers==5.5.4 blobfile==3.0.0 prettytable==3.11.0 omegaconf==2.3.0 hydra-core==1.3.2 wandb==0.23.0 \
        nvidia-resiliency-ext==0.6.0 megatron-energon==6.0.1 bitstring==4.3.1 filetype==1.2.0 datasets==4.4.1 \
        diffusers==0.37.1 einops==0.8.1 accelerate==1.10.1 peft==0.18.1 mistral-common==1.10.0 tensorboard==2.20.0 \
        timm==1.0.20 open-clip-torch==3.2.0 qwen-vl-utils==0.0.14 ;;
    *) echo "unknown target $target" >&2; exit 2 ;;
  esac
done
ls "$WORK"/build/pp-deps
