#!/bin/bash
#SBATCH --exclusive -t 06:00:00 -J e2e-blocks -o %x-%j.out
# Paired blocks on one allocation: every block runs one session per arm, back to back on the same nodes;
# block b starts at arm (b-1) mod n (rotated order).
# Usage: sbatch -N <nodes> e2e/common/run_blocks.sh <session-script> <recipe.env> [<arm> ...]
#        (no arms: the recipe's ARMS)
# Env: BLOCKS (default 3), FIRST_BLOCK (default 1; to split blocks across allocations), WARMUP (discarded iterations, default from the recipe or 10), WARMUP_SESSION (1: one
#      discarded session of the first arm before block 1; default from the recipe or 0), OUT (result dir);
#      everything else (e.g. ITERS) is passed through to the session script.
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/env/cluster.env"
session=${1:?session script}; recipe=${2:?recipe.env}; shift 2
arms=("$@")
(( ${#arms[@]} )) || read -r -a arms <<< "$(set +u; source "$recipe"; echo "${ARMS:-}")"
(( ${#arms[@]} )) || { echo "no arms given" >&2; exit 2; }
BLOCKS=${BLOCKS:-3}; FIRST_BLOCK=${FIRST_BLOCK:-1}
name=$(basename "$recipe" .env)
family=$(basename "$(dirname "$(dirname "$(realpath "$recipe")")")")
OUT=${OUT:-$WORK/results/e2e/$family/$name/${SLURM_JOB_ID:-$(date +%Y%m%dT%H%M%S)}}
WARMUP=${WARMUP:-$(set +u; source "$recipe"; echo "${WARMUP:-10}")}
WARMUP_SESSION=${WARMUP_SESSION:-$(set +u; source "$recipe"; echo "${WARMUP_SESSION:-0}")}
mkdir -p "$OUT"
csv=$OUT/sessions.csv
[[ -s $csv ]] || echo "platform,recipe,block,arm,step_ms,points" > "$csv"

n=${#arms[@]}
if [[ $WARMUP_SESSION == 1 ]]; then
  mkdir -p "$OUT/warmup"
  bash "$session" "$recipe" "${arms[0]}" "$OUT/warmup/${arms[0]}" || echo "warm-up session failed" >&2
fi
for (( b = FIRST_BLOCK; b < FIRST_BLOCK + BLOCKS; b++ )); do
  for (( i = 0; i < n; i++ )); do
    arm=${arms[$(( (b - 1 + i) % n ))]}
    dir=$OUT/block$b/$arm
    mkdir -p "$dir"
    echo "$(date -Is) block $b arm $arm -> $dir"
    bash "$session" "$recipe" "$arm" "$dir" || echo "session exited non-zero: block $b arm $arm" >&2
    ms= pts=0
    [[ -s $dir/train.log ]] && read -r ms pts < <(python3 "$ROOT/e2e/common/steptime.py" "$dir/train.log" --warmup "$WARMUP")
    echo "$PLATFORM,$name,$b,$arm,$ms,$pts" >> "$csv"
  done
done
cat "$csv"
