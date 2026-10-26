# Sourced by the MSCCL++ scripts: container launch, size list, plan generation and one executor process.
source "${SLURM_SUBMIT_DIR:-$PWD}/microbench/common.sh"
D=$ROOT/microbench/alt/mscclpp
B=$WORK/build/mscclpp
G=$((NODES * GPUS_PER_NODE))
OUT=${OUT:-$WORK/results/microbench/mscclpp_${PLATFORM}_${NODES}n${G}g}
MAXB=$(numfmt --from=iec "$MAX_BYTES")
mkdir -p "$OUT"
CT=(--container-image="$IMG_PYTORCH" --container-name=mscclpp --no-container-mount-home
    --container-mounts="$WORK:$WORK,$ROOT:$ROOT,$CUDA_HOME:$CUDA_HOME,$MPI_HOME:$MPI_HOME")
export PATH=$CUDA_HOME/bin:$MPI_HOME/bin:$PATH LD_LIBRARY_PATH=$B/lib:$LD_LIBRARY_PATH
[ -n "${NCCL_SOCKET_IFNAME:-}" ] && export MSCCLPP_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME

# nccl-tests AllGather totals 1 KiB..4 GiB (x2) for this world size: exact powers of two, or nccl-tests' rounded totals at 72 ranks.
if ((G == 72)); then
  ALL=(1152 3456 8064 16128 32256 64512 130176 261504 524160 1048320 2096640 4193280 8387712 16776576 33554304
       67108608 134217216 268434432 536870016 1073741184 2147483520 4294967040)
else
  ALL=(); for k in $(seq 10 32); do ALL+=($((1 << k))); done
fi
csv() { local IFS=,; echo "$*"; }
upto() { local s r=(); for s in "$@"; do ((s <= MAXB)) && r+=("$s"); done; csv "${r[@]}"; }
near() { local s; for s in "${ALL[@]}"; do ((s * 10 >= $1 * 9)) && { echo "$s"; return; }; done; }   # list entry for a nominal size
range() { local s r=(); for s in "${ALL[@]}"; do ((s >= $1 && s <= $2)) && r+=("$s"); done; echo "${r[@]}"; }
# LL beyond 16 MiB (32 MiB on one node) does not terminate across nodes; Simple covers 16 MiB..4 GiB.
((NODES == 1)) && LLTOP=$((32 << 20)) || LLTOP=$((16 << 20))
LL_RANGE=($(range 0 "$(near $LLTOP)"))
SIMPLE_RANGE=($(range "$(near $((16 << 20)))" "${ALL[-1]}"))

# plans <ll|simple> <cfg>...: generate missing JSON plans (cfg i<instances>_t<threads> or tbg<n>_t<threads>) in one container step
plandir=$B/plans/${G}r_${GPUS_PER_NODE}gpn
planfile() { echo "$plandir/$1_$2.json"; }
plans() {
  local kind=$1 cfg n t cmds=""; shift
  mkdir -p "$plandir"
  for cfg in "$@"; do
    [ -s "$(planfile "$kind" "$cfg")" ] && continue
    n=${cfg%%_*}; t=${cfg#*_t}
    [ "$kind" = ll ] && n="--instances ${n#i} --name allgather_${G}r_$cfg" || n="--tbg-size ${n#tbg} --name allgather_simple_${G}r_$cfg"
    cmds+="python3 $D/generate_${kind}_plan.py --num-gpus $G"
    cmds+=" --gpus-per-node $GPUS_PER_NODE --threads $t $n --output $(planfile "$kind" "$cfg") && "
  done
  [ -z "$cmds" ] || srun -N1 -n1 "${CT[@]}" env PYTHONPATH="$B/pydeps" MSCCLPP_SRC="$WORK/src/mscclpp" bash -c "${cmds}true"
}

# xrun <log> <plan> <sizes> <trial> <minutes>: one fresh executor process on all ranks (5 warmups, 20 timed ops per size)
xrun() {
  step "$GPUS_PER_NODE" "$1" -t "$5" "${CT[@]}" env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" MSCCLPP_CACHE_DIR="/tmp/mscclpp_cache_$SLURM_JOB_ID" \
    MSCCLPP_TOTAL_SIZES="$3" MSCCLPP_WARMUPS=5 MSCCLPP_ITERS=20 MSCCLPP_TRIAL_ID="$4" MSCCLPP_LOG_LEVEL=WARN \
    "$B/bin/executor_allgather_exact" "$2"
}
