#!/bin/bash
#SBATCH --exclusive -t 02:00:00 -J e2e-fsdp -o %x-%j.out
# One FSDP training session (one recipe, one arm) on the nodes of the current allocation.
# Usage: bash e2e/fsdp/run_session.sh e2e/fsdp/recipes/<recipe>.env <arm> [<dir>]
#        (or sbatch -N <nodes> e2e/fsdp/run_session.sh ...)
# Arms: legacy symmetric_st symmetric_stmc ce_unicast ce_multicast (per platform, see ARMS in the recipe).
# Env: ITERS (default from the recipe). Writes <dir>/rank.sh (per-rank environment and command) and <dir>/train.log.
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/env/cluster.env"
source "$ROOT/e2e/common/shim.sh"
recipe=$(realpath "${1:?recipe.env}"); arm=${2:?arm}
name=$(basename "$recipe" .env)
dir=$(realpath -m "${3:-$WORK/results/e2e/fsdp/$name/session-${SLURM_JOB_ID:-local}/$arm}")
ENV=() ARGS=() CE_AG_MULTICAST_THRESHOLD=1125899906842624
source "$recipe"   # shellcheck disable=SC1090
[[ " $ARMS " == *" $arm "* ]] || { echo "$name on $PLATFORM runs arms: $ARMS" >&2; exit 2; }
(( SLURM_JOB_NUM_NODES >= NODES )) || { echo "$name needs $NODES nodes" >&2; exit 2; }
mkdir -p "$dir"

# AllGather arm. Every arm uses the same registered buffers; symmetric ReduceScatter/AllReduce kernels are
# excluded in every arm, so those run on NCCL's ring/tree kernels.
big=1125899906842624
ag_st=AllGather_ST
ag_other=AllGather_LL,AllGather_LLMC,AllGather_TmaST,AllGather_TmaSTMC,AllGather_STMC,AllGather_RailRing_LsaSTMC
rsar=ReduceScatter_LL,ReduceScatter_TmaLD,ReduceScatter_LD,ReduceScatter_LDMC,ReduceScatter_RailA2A_LsaLD,ReduceScatter_RailA2A_LsaLDMC
rsar+=,AllReduce_AGxLL_R,AllReduce_AGxLLMC_R,AllReduce_RSxLD_AGxST,AllReduce_RSxLDMC_AGxSTMC,AllReduce_RSxTmaLD_AGxTmaST
case $arm in
  legacy)         nccl=(NCCL_CTA_POLICY=DEFAULT NCCL_SYM_CE_THRESHOLD=$big "NCCL_SYM_KERNEL=^$ag_st,$ag_other,$rsar") ;;
  symmetric_st)   nccl=(NCCL_CTA_POLICY=DEFAULT NCCL_SYM_CE_THRESHOLD=$big "NCCL_SYM_KERNEL=^$rsar,$ag_other") ;;
  symmetric_stmc) nccl=(NCCL_CTA_POLICY=DEFAULT NCCL_SYM_CE_THRESHOLD=$big "NCCL_SYM_KERNEL=^$rsar")
                  # B200: across NVLink domains the multicast kernel is the rail-ring variant
                  [[ $PLATFORM == b200 ]] && nccl[2]+=";AllGather:AllGather_RailRing_LsaSTMC" ;;
  ce_unicast|ce_multicast)
                  # Rail Direct: the cross-domain schedule that plan fusion covers
                  nccl=(NCCL_CTA_POLICY=ZERO "NCCL_SYM_KERNEL=^$rsar" NCCL_CE_COLL_PLAN_AG_BATCH=16 NCCL_CE_ALLGATHER_PLAN_FUSION=1
                        NCCL_HIER_CE_COLL_AG_RAIL_RING_ENABLE=0 NCCL_HIER_CE_COLL_AG_RAIL_PIPE_THRESHOLD=$big)
                  [[ $arm == ce_multicast && -n $CE_AG_MULTICAST_THRESHOLD ]] &&
                    nccl+=(NCCL_CE_COLL_AG_MULTICAST_THRESHOLD=$CE_AG_MULTICAST_THRESHOLD) ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
esac

envs=("LD_LIBRARY_PATH=$WORK/build/nccl/lib"
  "MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)" "MASTER_PORT=${MASTER_PORT:-29500}"
  NCCL_DEBUG=WARN NCCL_WIN_ENABLE=1 NCCL_NVLS_ENABLE=1 "NCCL_IB_HCA=$NCCL_IB_HCA" "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME")
mounts=$WORK:$WORK,$ROOT:$ROOT
case $STACK in
  nemo)   # NeMo 2 performance recipes, Megatron-Core from the image
    image=$IMG_NEMO; shim_env nemo "$WORK/build/nccl/lib/libnccl.so.2"
    envs+=(CUDA_DEVICE_MAX_CONNECTIONS=32 TORCH_NCCL_AVOID_RECORD_STREAMS=1 TORCH_NCCL_HIGH_PRIORITY=1
      NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=1 NVTE_NORM_FWD_USE_CUDNN=1 NVTE_NORM_BWD_USE_CUDNN=1
      NVTE_FWD_LAYERNORM_SM_MARGIN=16 NVTE_BWD_LAYERNORM_SM_MARGIN=16 NCCL_NET_GDR_LEVEL=PHB NCCL_NET_GDR_C2C=1
      TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=False)
    unsets="PYTORCH_CUDA_ALLOC_CONF"   # nccl_ub needs the default CUDA allocator
    cmd=(python3 "$ROOT/e2e/fsdp/pretrain_dense.py" --model "$MODEL" --platform "$PLATFORM" --result-dir "$dir/nemo"
      --num-nodes "$NODES" --gpus-per-node "$GPUS_PER_NODE" --tp "$TP" --mbs "$MBS" --gbs "$GBS" --seq "$SEQ"
      --num-layers "$LAYERS" --recompute-layers "$RECOMPUTE" --max-steps "$ITERS" --log-every-n-steps 10)
    workdir=$ROOT ;;
  megatron)   # Megatron-LM pretrain_gpt.py (e2e/fsdp/setup.sh), HybridEP from e2e/pp/build_deps.sh
    image=$IMG_PYTORCH; shim_env pytorch "$WORK/build/nccl/lib/libnccl.so.2"
    megatron=$WORK/src/megatron-fsdp
    envs+=("PYTHONPATH=$megatron:$WORK/build/fsdp-deps/fla" "PYTHONUSERBASE=$WORK/build/pp-deps/pytorch"
      OMP_NUM_THREADS=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
      TORCH_NCCL_AVOID_RECORD_STREAMS=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NVTE_CPU_OFFLOAD_V1=0
      MEGATRON_DEEPEP_ALLOW_MNNVL=0 MEGATRON_DEEPEP_USE_FABRIC=0
      NCCL_TUNER_PLUGIN=none NCCL_NET_PLUGIN=none NCCL_GIN_PLUGIN=none NCCL_GIN_TYPE=3)
    case $PLATFORM in
      gb200) envs+=(GB200_CLUSTER=1 NVLINK_DOMAIN_SIZE=72) ;;
      b200)  envs+=(NVLINK_DOMAIN_SIZE=$GPUS_PER_NODE) ;;
    esac
    unsets="CUDA_DEVICE_MAX_CONNECTIONS TORCH_NCCL_HIGH_PRIORITY NVTE_FWD_LAYERNORM_SM_MARGIN NVTE_BWD_LAYERNORM_SM_MARGIN"
    # Triton kernels (FLA) persist across the sessions of a recipe, so a warm-up session absorbs compilation.
    rank_line="export TRITON_CACHE_DIR=$WORK/cache/e2e-fsdp/$name/rank\$SLURM_PROCID"
    cmd=(python3 "$megatron/pretrain_gpt.py" "${ARGS[@]}" --train-iters "$ITERS" --log-interval 10)
    workdir=$megatron ;;
  deepseek)   # Hydra harness of the PP Megatron tree with Megatron-Bridge (e2e/pp/setup_megatron.sh, build_deps.sh)
    image=$IMAGE; shim_env pytorch "$WORK/build/nccl/lib/libnccl.so.2"
    data=$WORK/data/deepseek   # e2e/pp/deepseek_data.sh
    envs+=("PYTHONPATH=$WORK/src/megatron-pp:$WORK/src/megatron-bridge/src" "PYTHONUSERBASE=$WORK/build/pp-deps/deepseek"
      "DEEPSEEK_HF_CONFIG=$data/hf-config" "HF_HOME=$data/hf" "DGXNNODES=$NODES" "DGXNGPU=$GPUS_PER_NODE" "MAX_STEPS=$ITERS")
    # NCCL's built-in network and GIN paths, not plugins bundled in the image
    [[ $PLATFORM == b200 ]] && envs+=(NCCL_TUNER_PLUGIN=none NCCL_NET_PLUGIN=none NCCL_GIN_PLUGIN=none NCCL_GIN_TYPE=3)
    unsets=""
    cmd=(python3 -u "$ROOT/e2e/pp/deepseek_entry.py" model.virtual_pipeline_model_parallel_size=null
      data.num_workers=0 data.path_to_cache=/npy_index exp_manager.explicit_log_dir=/results)
    mkdir -p "$data/npy_index" "$dir/results"
    workdir=$WORK/src/megatron-pp/examples/zero_sm_pp/deepseek_mlperf
    mounts+=,$data/c4:/preproc_data:ro,$data/tokenizer:/tokenizer:ro,$data/npy_index:/npy_index,$dir/results:/results ;;
  *) echo "unknown STACK $STACK" >&2; exit 2 ;;
esac
envs=("${SHIM_ENV[@]}" "${envs[@]}" "${ENV[@]}" "${nccl[@]}")

{
  echo 'export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS'
  echo "unset NCCL_ALGO NCCL_PROTO $unsets"
  echo "${rank_line:-}"
  for e in "${envs[@]}"; do printf 'export %q\n' "$e"; done
  printf 'exec'; printf ' %q' "${cmd[@]}"; echo
} > "$dir/rank.sh"

opts=()
[[ $PLATFORM == b200 ]] && opts+=(--distribution=block:block --cpus-per-task="${CPUS_PER_TASK:-16}" --cpu-bind=cores)
srun --mpi=pmix -N "$NODES" -n $(( NODES * GPUS_PER_NODE )) --ntasks-per-node="$GPUS_PER_NODE" "${opts[@]}" --kill-on-bad-exit=1 \
  --container-image="$image" --container-mounts="$mounts" --container-workdir="$workdir" \
  --no-container-mount-home --container-writable \
  bash "$dir/rank.sh" > "$dir/train.log" 2>&1
