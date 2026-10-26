#!/bin/bash
#SBATCH -N 1 --exclusive -t 01:30:00 -J overlap -o %x-%j.out
# Overlap benchmark on one 8-GPU node: GEMM chain + concurrent AllGather, CUPTI-timed, four AllGather arms.
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/env/cluster.env"

REPLICATES=${REPLICATES:-4}   # fresh processes per model and arm; odd = forward order, even = reverse
TRIALS=${TRIALS:-30}
WARMUPS=${WARMUPS:-10}
MODELS=${MODELS:-"qwen3_30b_a3b llama3_8b"}
ARMS=${ARMS:-"default cta_32 zero legacy"}
OUT=${OUT:-$WORK/results/overlap/${SLURM_JOB_ID:-manual}}

export NCCL=$WORK/build/nccl SRC=$ROOT/overlap/src BUILD=$WORK/build/overlap
MOUNTS="$ROOT:$ROOT,$WORK:$WORK"
[ -f "$NCCL/lib/libnccl.so.2" ] || { echo "missing $NCCL; run: sbatch nccl/build.sh" >&2; exit 1; }
for arm in $ARMS; do
  case $arm in zero|legacy|default|cta_32) ;; *) echo "unknown arm $arm" >&2; exit 1 ;; esac
done
mkdir -p "$OUT"

srun -N1 -n1 --container-image="$IMG_PYTORCH" --container-mounts="$MOUNTS" \
  bash "$SRC/build.sh" "$NCCL" "$BUILD"

run_one() {
  local model=$1 arm=$2 rep=$3 sweep=$4
  local _m m n k scale bytes repeats kind
  IFS=$'\t' read -r _m m n k scale bytes repeats kind _ < <(awk -F'\t' -v m="$model" '$1 == m' "$SRC/points.tsv")
  export ARM=$arm PREFIX=$OUT/$model/replicate_$rep/$sweep/$arm
  mkdir -p "$(dirname "$PREFIX")"
  rm -f "$PREFIX.id"
  echo "$(date -Is) replicate=$rep sweep=$sweep model=$model arm=$arm"
  timeout --kill-after=30s 900s \
  srun -N1 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=14 --cpu-bind=cores --exact \
    --kill-on-bad-exit=1 --label --container-image="$IMG_PYTORCH" --container-mounts="$MOUNTS" \
    --chdir="$ROOT" bash -c '
      set -euo pipefail
      STMC_ONLY=^AllGather_LL,AllGather_LLMC,AllGather_TmaST,AllGather_TmaSTMC,AllGather_ST,AllGather_RailRing_LsaSTMC
      for v in NCCL_CTA_POLICY NCCL_SYM_CTAS NCCL_SYM_KERNEL NCCL_SYM_CE_THRESHOLD NCCL_WIN_ENABLE \
               NCCL_ALGO NCCL_PROTO NCCL_MIN_CTAS NCCL_MAX_CTAS NCCL_CE_COLL_PLAN_AG_BATCH \
               NCCL_CE_ALLGATHER_PLAN_FUSION NCCL_CE_COLL_AG_MULTICAST_THRESHOLD NCCL_NET_PLUGIN \
               NCCL_TUNER_PLUGIN CUDA_DEVICE_MAX_CONNECTIONS CUDA_SCALE_LAUNCH_QUEUES; do unset $v; done
      export NCCL_DEBUG=WARN
      # 1 TiB symmetric CE threshold: the symmetric arms keep the symmetric kernel at every size.
      case $ARM in
        zero)    export NCCL_CTA_POLICY=ZERO NCCL_SYM_CE_THRESHOLD=1099511627776 ;;
        default) export NCCL_SYM_CE_THRESHOLD=1099511627776 NCCL_SYM_KERNEL="$STMC_ONLY" NCCL_SYM_CTAS=4 ;;
        cta_32)  export NCCL_SYM_CE_THRESHOLD=1099511627776 NCCL_SYM_KERNEL="$STMC_ONLY" NCCL_SYM_CTAS=32 ;;
        legacy)  export NCCL_WIN_ENABLE=0 ;;
      esac
      export LD_LIBRARY_PATH=$NCCL/lib:${LD_LIBRARY_PATH:-} LD_PRELOAD=$NCCL/lib/libnccl.so.2
      exec python3 "$SRC/launcher.py" "$BUILD/libcupti_trace.so" "$PREFIX" "$SRC/replay.py" \
        --bridge "$BUILD/libbridge.so" --id-file "$PREFIX.id" --output-prefix "$PREFIX" "$@"
    ' bash --model "$model" --m "$m" --n "$n" --k "$k" --scale-send-bytes "$scale" \
      --data-send-bytes "$bytes" --compute-repeats "$repeats" --compute-kind "$kind" \
      --config "$arm" --sweep "$sweep" --warmups "$WARMUPS" --samples "$TRIALS" > "$PREFIX.log" 2>&1
  gzip -f -3 "$PREFIX".rank*.kernel_activity.tsv
}

read -r -a arms <<< "$ARMS"
read -r -a models <<< "$MODELS"
for ((rep = 1; rep <= REPLICATES; ++rep)); do
  if ((rep % 2)); then
    sweep=forward order_arms=("${arms[@]}") order_models=("${models[@]}")
  else
    sweep=reverse
    order_arms=(); for ((i = ${#arms[@]} - 1; i >= 0; --i)); do order_arms+=("${arms[i]}"); done
    order_models=(); for ((i = ${#models[@]} - 1; i >= 0; --i)); do order_models+=("${models[i]}"); done
  fi
  for arm in "${order_arms[@]}"; do
    for model in "${order_models[@]}"; do
      run_one "$model" "$arm" "$rep" "$sweep"
    done
  done
done

srun -N1 -n1 --container-image="$IMG_PYTORCH" --container-mounts="$MOUNTS" \
  python3 "$ROOT/overlap/analyze.py" "$OUT"
echo "results: $OUT"
