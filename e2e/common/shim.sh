# Sourced by session scripts. shim_env <nemo|pytorch> [extra LD_PRELOAD entries]
# sets SHIM_ENV, an array of VAR=value to pass to `env` in front of the rank command inside the container.
shim_env() {
  local so=$WORK/build/shim/$1/pyarena_malloc_shim.so
  [[ -f $so ]] || { echo "missing $so (run: sbatch e2e/common/build_shim.sh)" >&2; return 1; }
  SHIM_ENV=("LD_PRELOAD=$so${2:+:$2}" PYARENA_SHIM_VERBOSE=1
    MALLOC_MMAP_THRESHOLD_=33554432 MALLOC_TRIM_THRESHOLD_=4294967295 MALLOC_TOP_PAD_=268435456)
}
