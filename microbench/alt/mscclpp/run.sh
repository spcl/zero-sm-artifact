#!/bin/bash
#SBATCH -N 1 --exclusive -t 01:00:00 -J mb-mscclpp -o %x-%j.out
# MSCCL++ AllGather production (native executor, FP32 out-of-place, 5 warmups + 20 timed ops, rank-max CUDA-event time,
# full output validation outside timing): TRIALS fresh processes per protocol with the plans and switch size of tune.sh.
#   sbatch -N <nodes> microbench/alt/mscclpp/run.sh
# Without tuning: LL_PLAN=i<n>_t<threads> SIMPLE_PLAN=tbg<n>_t<threads>|none (switch at 32 MiB, or LL only).
# GB200 18 nodes (paper): LL_PLAN=i1_t1024 SIMPLE_PLAN=none, the untuned default LL plan, 1 KiB..16 MiB.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/microbench/alt/mscclpp/lib.sh"
if [ -n "${LL_PLAN:-}" ]; then
  SIMPLE_PLAN=${SIMPLE_PLAN:-none}
  if [ "$SIMPLE_PLAN" = none ]; then ll=($(range 0 "$(near $LLTOP)")); simple=()
  else ll=($(range 0 "$(near $((16 << 20)))")); simple=($(range "$(near $((32 << 20)))" "${ALL[-1]}")); fi
  printf 'll_plan=%s\nsimple_plan=%s\nll_sizes=%s\nsimple_sizes=%s\n' "$LL_PLAN" "$SIMPLE_PLAN" "$(csv "${ll[@]}")" \
    "$(csv "${simple[@]}")" > "$OUT/selection.env"
else
  cp "$OUT/tuning/selection.env" "$OUT/selection.env" || exit 1
fi
source "$OUT/selection.env"
[ "$ll_plan" = none ] || plans ll "$ll_plan"
[ "$simple_plan" = none ] || plans simple "$simple_plan"
llz=$(upto ${ll_sizes//,/ }); sz=$(upto ${simple_sizes//,/ })
for k in $(seq 1 "$TRIALS"); do
  [ -z "$llz" ] || xrun "$OUT/trial_${k}_ll.log" "$(planfile ll "$ll_plan")" "$llz" "$k" 15
  [ -z "$sz" ] || xrun "$OUT/trial_${k}_simple.log" "$(planfile simple "$simple_plan")" "$sz" "$k" 15
done
python3 "$D/select.py" table "$OUT"
