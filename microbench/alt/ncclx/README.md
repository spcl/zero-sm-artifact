# NCCLX persistent AllGatherP

NCCLX from TorchComms (`meta-pytorch/torchcomms` @ `5e9fc2ddb5c993e493953299066568225187fefc`) with torch `2.11.0+cu129`, run in `$IMG_PYTORCH`.
`torchcomms.patch` passes the long device-link object list through an nvcc options file (x86 argument-length limit) and treats an all-zero NVLink fabric UUID as "no multi-node NVLink domain", so B200 nodes reach each other over IB.

```bash
sbatch microbench/alt/ncclx/build.sh            # ~1.5 h, -> $WORK/build/ncclx
sbatch -N 1 microbench/alt/ncclx/run.sh         # B200 8 GPUs / GB200 4 GPUs
sbatch -N 4 microbench/alt/ncclx/run.sh         # B200 32 / GB200 16
sbatch -N 8 microbench/alt/ncclx/run.sh         # B200 64
sbatch -N 18 microbench/alt/ncclx/run.sh        # GB200 72
```

`run.sh` tunes the AllGatherP algorithm (`ctdirect`, `ctpipeline`, `ctsrdpipeline` (power-of-two node counts only); 3 trials at 1 KiB, the first size >= 16 MiB and 4 GiB; winner = highest geometric mean of the median bus bandwidth), then runs 10 fresh-process trials of the winner over the nccl-tests size sweep (1 KiB to 4 GiB, factor 2, 5 warm-up + 20 timed iterations, time = max over ranks).
Paper selections: `ctsrdpipeline` on GB200 4 and 16 GPUs and B200 64 GPUs, `ctpipeline` on GB200 72 GPUs and B200 8 and 32 GPUs; set `ALGO=<name>` to skip tuning.

Output: `$WORK/results/microbench/ncclx_<platform>_<N>n<G>g/{trials.csv,summary.csv,selection.txt,tune/}`; about 20 min per placement.
