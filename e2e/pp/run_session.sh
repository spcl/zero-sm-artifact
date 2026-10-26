#!/bin/bash
# One pipeline-parallel training session (one recipe, one arm) on the nodes of the current allocation.
# Usage: bash e2e/pp/run_session.sh e2e/pp/recipes/<recipe>.env <two_sided|one_sided> <dir>
# Env: ITERS (default 100). Writes <dir>/rank.sh (the per-rank command) and <dir>/train.log.
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/env/cluster.env"
source "$ROOT/e2e/common/shim.sh"
recipe=$(realpath "${1:?recipe.env}"); arm=${2:?arm}; dir=$(realpath -m "${3:?session dir}")
ITERS=${ITERS:-100}
LAUNCH=megatron DEPS=pytorch IMAGE=$IMG_PYTORCH
source "$recipe"
MEGATRON=$WORK/src/megatron-pp
NODES=$(( GPUS / GPUS_PER_NODE ))
mkdir -p "$dir"

# Both arms run DP/TP/EP collectives on the default NCCL kernels (symmetric and copy-engine paths off);
# only the pipeline P2P transport differs.
ag=AllGather_LL,AllGather_LLMC,AllGather_TmaST,AllGather_ST,AllGather_TmaSTMC,AllGather_STMC,AllGather_RailRing_LsaSTMC
rs=ReduceScatter_LL,ReduceScatter_TmaLD,ReduceScatter_LD,ReduceScatter_LDMC,ReduceScatter_RailA2A_LsaLD,ReduceScatter_RailA2A_LsaLDMC
ar=AllReduce_AGxLL_R,AllReduce_AGxLLMC_R,AllReduce_RSxLD_AGxST,AllReduce_RSxLDMC_AGxSTMC,AllReduce_RSxTmaLD_AGxTmaST
shim_env pytorch "$WORK/build/nccl/lib/libnccl.so.2"
envs=("${SHIM_ENV[@]}" "LD_LIBRARY_PATH=$WORK/build/nccl/lib"
  "MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)" "MASTER_PORT=${MASTER_PORT:-29500}"
  NCCL_DEBUG=WARN NCCL_WIN_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_CTA_POLICY=DEFAULT NCCL_SYM_CE_THRESHOLD=1125899906842624
  "NCCL_SYM_KERNEL=^$ag,$rs,$ar" "NCCL_IB_HCA=$NCCL_IB_HCA" "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME")
envs+=("PYTHONUSERBASE=$WORK/build/pp-deps/$DEPS")   # from e2e/pp/build_deps.sh
srun_opts=()
[[ $PLATFORM == b200 ]] && srun_opts+=(--distribution=block:block --cpus-per-task="${CPUS_PER_TASK:-16}" --cpu-bind=cores)

case $LAUNCH in
  megatron)
    case $arm in
      two_sided) p2p=() ;;
      one_sided) p2p=(--use-symmetric-memory-p2p) ;;   # PyTorch symmetric memory put_signal/wait_signal
      *) echo "arm must be two_sided or one_sided" >&2; exit 2 ;;
    esac
    envs+=("PYTHONPATH=$MEGATRON" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=0)
    [[ $PLATFORM == b200 ]] && envs+=(OMP_NUM_THREADS=1 PYTHONHASHSEED=0
      NCCL_TUNER_PLUGIN=none NCCL_NET_PLUGIN=none NCCL_GIN_PLUGIN=none NCCL_GIN_TYPE=3)
    cmd=(python3 "$ROOT/e2e/pp/train.py" "$MEGATRON/pretrain_gpt.py"
      --distributed-backend nccl --distributed-timeout-minutes 20
      --use-mcore-models --transformer-impl transformer_engine --use-flash-attn --disable-bias-linear --bf16
      --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather --ddp-bucket-size 40000000
      --use-nccl-ub --mock-data --tokenizer-type NullTokenizer --make-vocab-size-divisible-by 128 --split 100,0,0
      --no-create-attention-mask-in-dataloader --num-workers 0 --init-method-std 0.02 --seed 1234
      --enable-experimental --train-iters "$ITERS" --eval-iters 0 --eval-interval 100000
      --log-interval 10 --log-throughput --timing-log-level 0
      "${ARGS[@]}" "${p2p[@]}")
    workdir=$MEGATRON mounts=$WORK:$WORK,$ROOT:$ROOT
    ;;
  deepseek)
    case $arm in
      two_sided) envs+=(SYM_MEM_PP=False) ;;
      one_sided) envs+=(SYM_MEM_PP=True) ;;
      *) echo "arm must be two_sided or one_sided" >&2; exit 2 ;;
    esac
    data=$WORK/data/deepseek
    envs+=("PYTHONPATH=$MEGATRON:$WORK/src/megatron-bridge/src" "DEEPSEEK_HF_CONFIG=$data/hf-config"
      "HF_HOME=$data/hf" "DGXNNODES=$NODES" "DGXNGPU=$GPUS_PER_NODE")
    # NCCL's built-in network and GIN paths, not plugins bundled in the image
    [[ $PLATFORM == b200 ]] && envs+=(NCCL_TUNER_PLUGIN=none NCCL_NET_PLUGIN=none NCCL_GIN_PLUGIN=none NCCL_GIN_TYPE=3)
    hydra_var="HYDRA_${PLATFORM}[@]"
    cmd=(python3 -u "$ROOT/e2e/pp/deepseek_entry.py" "${!hydra_var}")
    mkdir -p "$data/npy_index" "$dir/results"
    workdir=$MEGATRON/examples/zero_sm_pp/deepseek_mlperf
    mounts=$WORK:$WORK,$ROOT:$ROOT,$data/c4:/preproc_data:ro,$data/tokenizer:/tokenizer:ro,$data/npy_index:/npy_index,$dir/results:/results
    ;;
esac
env_var="ENV_${PLATFORM}[@]"
envs+=("${ENV[@]}" "${!env_var}")

{
  echo "export RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID WORLD_SIZE=\$SLURM_NTASKS LOCAL_WORLD_SIZE=$GPUS_PER_NODE"
  for e in "${envs[@]}"; do printf 'export %q\n' "$e"; done
  printf 'exec'; printf ' %q' "${cmd[@]}"; echo
} > "$dir/rank.sh"

srun --mpi=pmix -N "$NODES" -n "$GPUS" --ntasks-per-node="$GPUS_PER_NODE" "${srun_opts[@]}" --kill-on-bad-exit=1 \
  --container-image="$IMAGE" --container-mounts="$mounts" --container-workdir="$workdir" \
  --no-container-mount-home --container-writable \
  bash "$dir/rank.sh" > "$dir/train.log" 2>&1
