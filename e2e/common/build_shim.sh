#!/bin/bash
#SBATCH -N 1 -t 00:20:00 -J e2e-shim -o %x-%j.out
# Builds the arena shim inside each training image (same glibc and CPU architecture as the ranks).
# Output: $WORK/build/shim/<tag>/pyarena_malloc_shim.so for tag in SHIM_IMAGES (default: nemo pytorch).
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/env/cluster.env"
for tag in ${SHIM_IMAGES:-nemo pytorch}; do
  case $tag in
    nemo) img=$IMG_NEMO ;;
    pytorch) img=$IMG_PYTORCH ;;
    *) echo "unknown image tag $tag" >&2; exit 2 ;;
  esac
  out=$WORK/build/shim/$tag
  mkdir -p "$out"
  srun -N1 -n1 --container-image="$img" --container-mounts="$ROOT/e2e/common:/src:ro,$out:/out" \
    cc -shared -fPIC -O2 -Wall -o /out/pyarena_malloc_shim.so /src/pyarena_malloc_shim.c -ldl
  ls -l "$out/pyarena_malloc_shim.so"
done
