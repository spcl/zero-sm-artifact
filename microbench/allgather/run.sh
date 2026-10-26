#!/bin/bash
#SBATCH -N 1 --exclusive -t 01:00:00 -J mb-allgather -o %x-%j.out
# Fig. allgather, NCCL arms (legacy, STMC, CE multicast, CE unicast), all_gather_perf 1 KiB..4 GiB, 10 fresh-process trials each.
#   B200:  sbatch -N 1|4|8 microbench/allgather/run.sh      (8, 32, 64 GPUs)
#   GB200: sbatch -N 1|4|18 microbench/allgather/run.sh     (4, 16, 72 GPUs, one NVLink domain)
# B200 across nodes: ce_mc/ce_uc run Rail Direct and ce_pipe runs Rail Pipe; the figure takes Pipe at >= 4 MiB per rank,
# which is the size at which NCCL selects Pipe by default. ARMS overrides the arm list.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/microbench/common.sh"
G=$((NODES * GPUS_PER_NODE))
OUT=${OUT:-$WORK/results/microbench/ag_${PLATFORM}_${NODES}n${G}g}
mkdir -p "$OUT"
AG=("$NT/all_gather_perf" -b 1K -e "$MAX_BYTES" -f 2 -g 1 -w "$WARMUP" -n "$ITERS")
NOCE=1125899906842624   # 2^50: disables the size-based switch (symmetric -> CE, CE multicast -> unicast)

AGK=AllGather_LL,AllGather_LLMC,AllGather_TmaST,AllGather_ST,AllGather_TmaSTMC,AllGather_STMC,AllGather_RailRing_LsaSTMC
RSAR=ReduceScatter_LL,ReduceScatter_TmaLD,ReduceScatter_LD,ReduceScatter_LDMC,AllReduce_AGxLL_R,AllReduce_AGxLLMC_R,AllReduce_RSxLD_AGxST,AllReduce_RSxLDMC_AGxSTMC,AllReduce_RSxTmaLD_AGxTmaST
CEB=(NCCL_CE_COLL_PLAN_AG_BATCH=16 NCCL_CE_ALLGATHER_PLAN_FUSION=1)
declare -A ENV FLAGS
if [ "$PLATFORM" = b200 ]; then
  BASE=(NCCL_TUNER_PLUGIN=none NCCL_NET_PLUGIN=none NCCL_GIN_PLUGIN=none NCCL_GIN_TYPE=3
        NCCL_WIN_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_SYM_CE_THRESHOLD=$NOCE)
  ((NODES == 1)) && STMCK=AllGather_STMC || STMCK=AllGather_RailRing_LsaSTMC
  # Multi-node CE: Rail Direct for ce_mc/ce_uc, Rail Pipe for ce_pipe.
  ((NODES == 1)) && RAIL=() || RAIL=(NCCL_HIER_CE_COLL_AG_RAIL_RING_ENABLE=0 NCCL_HIER_CE_COLL_AG_RAIL_PIPE_THRESHOLD=$NOCE)
  ENV[legacy]="NCCL_SYM_KERNEL=^$AGK,$RSAR NCCL_CTA_POLICY=DEFAULT"
  ENV[stmc]="NCCL_SYM_KERNEL=^$RSAR;AllGather:$STMCK NCCL_CTA_POLICY=DEFAULT"
  ENV[ce_mc]="NCCL_SYM_KERNEL=^$RSAR NCCL_CTA_POLICY=ZERO ${CEB[*]} NCCL_CE_COLL_AG_MULTICAST_THRESHOLD=$NOCE ${RAIL[*]}"
  ENV[ce_uc]="NCCL_SYM_KERNEL=^$RSAR NCCL_CTA_POLICY=ZERO ${CEB[*]} NCCL_CE_COLL_AG_MULTICAST_THRESHOLD=0 ${RAIL[*]}"
  ENV[ce_pipe]="NCCL_HIER_CE_COLL_AG_RAIL_RING_ENABLE=0 NCCL_HIER_CE_COLL_AG_RAIL_PIPE_THRESHOLD=0"
  for a in legacy stmc ce_mc ce_uc; do FLAGS[$a]="-R 2"; done
  FLAGS[ce_pipe]="-R 2 -x 2"
  ((NODES == 1)) && DEFAULT_ARMS="legacy stmc ce_mc ce_uc" || DEFAULT_ARMS="legacy stmc ce_mc ce_uc ce_pipe"
else
  BASE=()
  DEFAULT_ARMS="legacy stmc ce_mc ce_uc"
  ENV[legacy]=""; FLAGS[legacy]=""
  ENV[stmc]="NCCL_SYM_CE_THRESHOLD=$NOCE NCCL_SYM_KERNEL=^AllGather_LL,AllGather_LLMC,AllGather_TmaST,AllGather_TmaSTMC,AllGather_ST,AllGather_RailRing_LsaSTMC"; FLAGS[stmc]="-R 2"
  ENV[ce_mc]="NCCL_CE_COLL_AG_MULTICAST_THRESHOLD=$NOCE ${CEB[*]}"; FLAGS[ce_mc]="-R 2 -x 2"
  ENV[ce_uc]="NCCL_CE_COLL_AG_MULTICAST_THRESHOLD=0 ${CEB[*]}"; FLAGS[ce_uc]="-R 2 -x 2"
fi

for arm in ${ARMS:-$DEFAULT_ARMS}; do
  read -ra e <<< "${ENV[$arm]}"; read -ra f <<< "${FLAGS[$arm]}"
  for t in $(seq "$TRIALS"); do
    step "$GPUS_PER_NODE" "$OUT/ag_$arm.trial$t.log" env "${BASE[@]}" "${e[@]}" "${AG[@]}" "${f[@]}"
  done
done
echo "results: $OUT"
