#!/bin/bash
#SBATCH -N 1 --exclusive -t 01:30:00 -J mb-mscclpp-tune -o %x-%j.out
# MSCCL++ AllGather tuning (native executor): screen one trial per plan, then 3 crossover trials fix the LL->Simple switch.
#   LL plans:     instances LL_INSTANCES x threads LL_THREADS, screened at 1 KiB and 16 MiB
#   Simple plans: thread-block groups SIMPLE_TBG x threads SIMPLE_THREADS, screened at 256 MiB, 1 GiB, 4 GiB
# Writes $OUT/tuning/selection.env, read by run.sh.   sbatch -N <nodes> microbench/alt/mscclpp/tune.sh
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/microbench/alt/mscclpp/lib.sh"
T=$OUT/tuning; mkdir -p "$T"
LL_INSTANCES=${LL_INSTANCES:-1 2 4 8}; LL_THREADS=${LL_THREADS:-256 512 768 1024}
SIMPLE_TBG=${SIMPLE_TBG:-1 2 4}; SIMPLE_THREADS=${SIMPLE_THREADS:-256 512 1024}
CROSSOVER_TRIALS=${CROSSOVER_TRIALS:-3}
LLC=(); for i in $LL_INSTANCES; do for t in $LL_THREADS; do LLC+=("i${i}_t$t"); done; done
SC=(); for g in $SIMPLE_TBG; do for t in $SIMPLE_THREADS; do SC+=("tbg${g}_t$t"); done; done
plans ll "${LLC[@]}"; plans simple "${SC[@]}"

# Plans that fail validation or do not finish (e.g. multi-instance LL at 1 KiB on 64 ranks) drop out of the selection.
for c in "${LLC[@]}"; do xrun "$T/ll_$c.log" "$(planfile ll "$c")" "$(csv "${ALL[0]}" "$(near $((16 << 20)))")" 0 7; done
for c in "${SC[@]}"; do xrun "$T/simple_$c.log" "$(planfile simple "$c")" "$(upto "$(near $((256 << 20)))" "$(near $((1 << 30)))" "${ALL[-1]}")" 0 7; done
python3 "$D/select.py" screen "$T"
source "$T/screen.env"
for k in $(seq 1 "$CROSSOVER_TRIALS"); do
  [ "$ll_plan" = none ] || xrun "$T/crossover_ll_trial$k.log" "$(planfile ll "$ll_plan")" "$(upto "${LL_RANGE[@]}")" "$k" 20
  [ "$simple_plan" = none ] || xrun "$T/crossover_simple_trial$k.log" "$(planfile simple "$simple_plan")" "$(upto "${SIMPLE_RANGE[@]}")" "$k" 20
done
python3 "$D/select.py" crossover "$T" "$(upto "${ALL[@]}")"
cat "$T/selection.env"
