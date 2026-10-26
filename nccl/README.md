# NCCL

All experiments use one library: NCCL `v2.32.3-1` (GitHub `NVIDIA/nccl`) with the patches in `patches/`.

| Patch | Adds | Used by |
|---|---|---|
| 0001 | `NCCL_CE_COLL_PLAN_AG_BATCH`: pack consecutive CE AllGathers into one plan | CE arms |
| 0002 | `NCCL_CE_ALLGATHER_PLAN_FUSION`: one synchronization envelope per packed plan | CE arms |
| 0003 | `ncclMemAlloc` grants access to the owning GPU only | all |
| 0004 | GIN symmetric kernels across NVLink domains | symmetric arms on B200 |
| 0005 | `NCCL_ALGO` and `NCCL_SYM_KERNEL` clear only their own tuning table | arm selection |
| 0006–0009 | Hierarchical CE AllGather Rail Pipe schedule (`NCCL_HIER_CE_COLL_AG_RAIL_PIPE_THRESHOLD`) and automatic Rail Ring selection | CE across NVLink domains |

Build NCCL and nccl-tests (`v2.21.0`) on one compute node:

```bash
cp env/cluster.env.example env/cluster.env   # edit
sbatch nccl/build.sh
```

Outputs: `$WORK/build/nccl/lib/libnccl.so` and `$WORK/build/nccl-tests/*_perf`.
