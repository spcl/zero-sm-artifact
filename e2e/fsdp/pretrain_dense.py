#!/usr/bin/env python3
"""Dense-model Megatron-FSDP pretraining (NeMo 2 performance recipes in nvcr.io/nvidia/nemo:25.11.01).

One process per GPU (srun). FP8 delayed scaling, Megatron-FSDP optim_grads_params, mock data, no checkpointing.
FSDP buffers live in the ncclMemAlloc pool (nccl_ub) and the pool is registered as a symmetric window after the
first batch, in every arm.
"""
import argparse
import ctypes
import functools
import hashlib
import json
import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace

BUFFER_FIELDS = ("model_weight_buffer", "transpose_weight_buffer", "main_weight_buffer", "main_grad_buffer", "hsdp_wbuf")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("llama3_70b", "llama31_405b", "nemotron4_15b", "qwen3_32b"), required=True)
    p.add_argument("--platform", choices=("gb200", "b200"), required=True)
    p.add_argument("--result-dir", required=True)
    p.add_argument("--num-nodes", type=int, required=True)
    p.add_argument("--gpus-per-node", type=int, required=True)
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--mbs", type=int, required=True)
    p.add_argument("--gbs", type=int, required=True)
    p.add_argument("--seq", type=int, required=True)
    p.add_argument("--num-layers", type=int, required=True)
    p.add_argument("--recompute-layers", type=int, required=True)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--log-every-n-steps", type=int, default=10)
    return p.parse_args()


def install_clip_guard():
    """Skip the clip-scale launch on ranks that own no gradients (empty multi-tensor launch)."""
    import torch
    from megatron.core.optimizer import clip_grads
    from megatron.core.optimizer import optimizer as optimizer_module

    original = clip_grads.clip_grad_by_total_norm_fp32

    @torch.no_grad()
    def guarded(parameters, max_norm, total_norm, use_decoupled_grad=False):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        grads = [getattr(p, "decoupled_grad", None) if use_decoupled_grad else p.grad for p in parameters]
        if max_norm / (total_norm + 1.0e-6) < 1.0 and not any(g is not None for g in grads):
            return None
        return original(parameters, max_norm, total_norm, use_decoupled_grad)

    clip_grads.clip_grad_by_total_norm_fp32 = guarded
    optimizer_module.clip_grad_by_total_norm_fp32 = guarded


def register_named_pool(pool, named_tensors, groups):
    """Register every live pool segment as a symmetric window, ordered by the logical tensor it holds
    (virtual-address order differs across ranks)."""
    import torch
    import torch.distributed as dist

    segments = pool.snapshot()
    owners = [[] for _ in segments]
    for name, t in named_tensors:
        if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.numel():
            continue
        ptr = t.data_ptr()
        for i, seg in enumerate(segments):
            if seg["address"] <= ptr and ptr + t.nbytes <= seg["address"] + seg["total_size"]:
                owners[i].append((name, ptr - seg["address"], t.nbytes, str(t.dtype)))
                break
    ordered = []
    for seg, refs in zip(segments, owners):
        if not refs:
            if seg.get("active_size", 0) == 0:
                continue
            raise RuntimeError("live NCCL pool segment without a logical owner")
        ordered.append((sorted(set(refs)), seg))
    ordered.sort(key=lambda item: item[0][0][0])
    digest = hashlib.sha256(json.dumps([(r, s["total_size"]) for r, s in ordered], sort_keys=True).encode()).hexdigest()
    device = torch.device("cuda", torch.cuda.current_device())
    for g in groups:
        lo = torch.tensor([len(ordered), int(digest[:15], 16)], dtype=torch.int64, device=device)
        hi = lo.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN, group=g)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX, group=g)
        if not torch.equal(lo, hi):
            raise RuntimeError("NCCL pool layout differs across ranks")
    lib = ctypes.CDLL("libnccl.so.2")
    lib.ncclCommWindowRegister.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                           ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    lib.ncclCommWindowDeregister.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    handles = []
    for g in groups:
        comm = g._get_backend(device)._comm_ptr()
        for _, seg in ordered:
            win = ctypes.c_void_p()
            rc = lib.ncclCommWindowRegister(comm, seg["address"], seg["total_size"], ctypes.byref(win), 1)  # NCCL_WIN_COLL_SYMMETRIC
            if rc or not win.value:
                raise RuntimeError(f"ncclCommWindowRegister failed: {rc}")
            handles.append((comm, win.value))
    return lib, handles


def install_manual_registration(platform):
    """Allocate all FSDP buffers in the NCCL pool first, register the pool once after the first batch."""
    import torch
    from lightning.pytorch import Callback
    from megatron.core.distributed.fsdp.src.megatron_fsdp import param_and_grad_buffer as pgb

    buffers = []

    class AllocWithoutRegistration:
        def __init__(self, pool):
            self.ctx = torch.cuda.use_mem_pool(pool)

        def __enter__(self):
            return self.ctx.__enter__()

        def __exit__(self, *a):
            return self.ctx.__exit__(*a)

    def get_mem_alloc_context(self, groups=None):
        if not self.ddp_config.nccl_ub:
            return nullcontext
        return functools.partial(AllocWithoutRegistration, pgb.NCCL_MEMORY_POOL)

    original_init = pgb.ParamAndGradBuffer.__init__

    @functools.wraps(original_init)
    def init_and_track(self, *a, **k):
        original_init(self, *a, **k)
        self.already_registered = False
        buffers.append(self)

    if platform == "b200":
        # Buckets outside the double-buffer layout stay in named pool buffers, so every gather is in the window.
        fp_init, fp_alloc, fp_free = (pgb.FixedPoolAllocator.__init__, pgb.FixedPoolAllocator.allocate,
                                      pgb.FixedPoolAllocator.free)

        def fixed_init(self, *a, **k):
            fp_init(self, *a, **k)
            self._persistent_ids = set()

        def fixed_allocate(self, bucket_id, size, dtype, device, mem_alloc_context=None):
            if self.fsdp_param_groups[bucket_id].fsdp_unit_id in self.fsdp_double_buffer_units:
                return fp_alloc(self, bucket_id=bucket_id, size=size, dtype=dtype, device=device,
                                mem_alloc_context=mem_alloc_context)
            if bucket_id not in self._persistent_ids:
                torch.cuda.synchronize()
                self._persistent_ids.add(bucket_id)
            return pgb.Bucket(data=pgb.get_global_memory_buffer().get_tensor(
                [size], dtype=dtype, name=f"{self.name}_persistent_fallback_{bucket_id}",
                mem_alloc_context=mem_alloc_context))

        def fixed_free(self, bucket_id):
            if self.fsdp_param_groups[bucket_id].fsdp_unit_id in self.fsdp_double_buffer_units:
                fp_free(self, bucket_id)

        pgb.FixedPoolAllocator.__init__ = fixed_init
        pgb.FixedPoolAllocator.allocate = fixed_allocate
        pgb.FixedPoolAllocator.free = fixed_free

    def manual_buffer_registration(self):
        if self.already_registered:
            return
        assert self.ddp_config.nccl_ub and self.ddp_config.fsdp_double_buffer
        pool = pgb.NCCL_MEMORY_POOL
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.synchronize()
        device = torch.device("cuda", torch.cuda.current_device())
        if platform == "gb200":
            for g in self.ubr_groups:
                g._get_backend(device).register_mem_pool(pool)
        else:
            groups, seen = [], set()
            cand = list(self.ubr_groups) + [getattr(getattr(pg, f, None), "data_parallel_group", None)
                                            for pg in self.parameter_groups for f in BUFFER_FIELDS]
            for g in cand:
                if g is not None and id(g) not in seen:
                    seen.add(id(g))
                    groups.append(g)
            named = [(f"parameter_group/{i:05d}/{f}", getattr(getattr(pg, f, None), "data", None))
                     for i, pg in enumerate(self.parameter_groups) for f in BUFFER_FIELDS]
            named += [(f"global_buffer/{k}", t) for k, t in pgb.get_global_memory_buffer().buffer.items()]
            self._windows = register_named_pool(pool, named, groups)
        self.already_registered = True

    pgb.ParamAndGradBuffer.get_mem_alloc_context = get_mem_alloc_context
    pgb.ParamAndGradBuffer.__init__ = init_and_track
    pgb.ParamAndGradBuffer.manual_buffer_registration = manual_buffer_registration

    class FSDPManualRegistrationCallback(Callback):
        registered = False

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if not self.registered:
                for b in buffers:
                    b.manual_buffer_registration()
                self.registered = True

        def on_train_end(self, trainer, pl_module):
            if platform != "b200":
                return
            torch.cuda.synchronize()
            torch.distributed.barrier()
            torch.cuda.synchronize()
            for b in buffers:
                lib, handles = getattr(b, "_windows", (None, []))
                for comm, win in reversed(handles):
                    lib.ncclCommWindowDeregister(comm, win)
            torch.distributed.barrier()

    return FSDPManualRegistrationCallback


def common_overrides(recipe):
    import torch
    o = recipe.optim.config
    o.use_precision_aware_optimizer = True
    o.main_params_dtype = torch.float32
    o.main_grads_dtype = torch.bfloat16
    o.exp_avg_dtype = torch.float32
    o.exp_avg_sq_dtype = torch.float32
    o.clip_grad = 1.0
    ddp = recipe.trainer.strategy.ddp
    ddp.grad_reduce_in_fp32 = False
    ddp.preserve_fp32_weights = True
    ddp.nccl_ub = True
    recipe.data.num_workers = 0
    recipe.data.persistent_workers = False


QWEN_COPIED_FIELDS = (
    "tensor_model_parallel_size", "pipeline_model_parallel_size", "virtual_pipeline_model_parallel_size",
    "context_parallel_size", "sequence_parallel", "bf16", "fp16", "params_dtype", "enable_autocast", "autocast_dtype",
    "fp8", "fp8_recipe", "fp8_param", "fp8_margin", "fp8_interval", "fp8_amax_history_len", "fp8_amax_compute_algo",
    "fp8_wgrad", "fp8_dot_product_attention", "fp8_multi_head_attention", "tp_only_amax_red", "first_last_layers_bf16",
    "num_layers_at_start_in_bf16", "num_layers_at_end_in_bf16", "activation_func_fp8_input_store",
    "gradient_accumulation_fusion", "barrier_with_L1_time", "transformer_impl", "cross_entropy_loss_fusion",
    "cross_entropy_fusion_impl", "masked_softmax_fusion", "apply_rope_fusion", "use_te_rng_tracker",
    "enable_cuda_graph", "external_cuda_graph", "cuda_graph_impl")


def build_recipe(a):
    sys.path.insert(0, "/opt/NeMo")
    import torch

    perf = SimpleNamespace(
        gpus_per_node=a.gpus_per_node, max_steps=a.max_steps, use_fsdp_double_buffer=False,
        use_user_buffer_registration=True, use_sharp=False, compute_dtype="fp8", fp8_recipe="ds",
        nccl_communicator_config_path=None, tensorboard=False, wandb=False, wandb_prj_name=None, wandb_job_name=None,
        gpu="gb200", use_hf_tokenizer=False, use_te_op_fuser=False, use_te_act_func=False, act_func_fp8_input_store=False)
    n, mbs, gbs, tp, rec = a.num_nodes, a.mbs, a.gbs, a.tp, a.recompute_layers
    if a.model == "llama31_405b":
        from scripts.performance.llm import pretrain_llama31_405b as m
        recipe = m.override_recipe_configs(perf, n, mbs, gbs, tp, 1, 1, 1, 1, False, True, rec, 0)
        common_overrides(recipe)
    elif a.model == "llama3_70b":
        from scripts.performance.llm import pretrain_llama3_70b as m
        recipe = m.override_recipe_configs(perf, n, mbs, gbs, tp, 1, 1, 1, 1, False, True, rec, 0, [], False, True, False)
        recipe.trainer.strategy.ddp.fsdp_double_buffer = False
        common_overrides(recipe)
    elif a.model == "nemotron4_15b":
        from scripts.performance.llm import pretrain_nemotron4_15b as m
        perf.use_mcore_fsdp = True
        recipe = m.override_recipe_configs(perf, n, mbs, gbs, tp, 1, 1, 1, 1, False)
        if rec:
            c = recipe.model.config
            c.recompute_granularity, c.recompute_method, c.recompute_num_layers = "full", "uniform", rec
        recipe.trainer.strategy.ddp.fsdp_double_buffer = False
        common_overrides(recipe)
    else:
        # Llama-3 70B performance recipe with the NeMo Qwen3-32B model swapped in.
        from scripts.performance.llm import pretrain_llama3_70b as m
        from nemo.collections.llm.recipes import qwen3_32b
        recipe = m.override_recipe_configs(perf, n, mbs, gbs, tp, 1, 1, 1, 1, False, True, rec, 0, [], False, True, False)
        base, qwen = recipe.model.config, qwen3_32b.model()
        q = qwen.config
        for f in QWEN_COPIED_FIELDS:
            setattr(q, f, getattr(base, f))
        q.seq_length = 4096
        q.max_position_embeddings = max(int(getattr(q, "max_position_embeddings", 4096)), 4096)
        q.recompute_granularity, q.recompute_method, q.recompute_num_layers = "full", "block" if tp == 2 else "uniform", rec
        q.tensor_model_parallel_size, q.pipeline_model_parallel_size, q.context_parallel_size = tp, 1, 1
        q.sequence_parallel = bool(getattr(recipe.trainer.strategy, "sequence_parallel", tp > 1))
        q.bf16, q.fp16, q.params_dtype, q.autocast_dtype = True, False, torch.bfloat16, torch.bfloat16
        q.fp8, q.fp8_recipe = "hybrid", "delayed"
        recipe.model = qwen
        o = recipe.optim.config
        o.fp8_recipe, o.bf16, o.fp16, o.params_dtype = "delayed", True, False, torch.bfloat16
        common_overrides(recipe)
        ddp = recipe.trainer.strategy.ddp
        ddp.use_megatron_fsdp = True
        ddp.use_distributed_optimizer = True
        ddp.data_parallel_sharding_strategy = "optim_grads_params"
        ddp.overlap_grad_reduce = True
        ddp.overlap_param_gather = True
        ddp.average_in_collective = False
        ddp.fp8_param_gather = True
        recipe.trainer.strategy.num_distributed_optimizer_instances = 1
    return recipe


def main():
    a = parse_args()
    os.makedirs(a.result_dir, exist_ok=True)
    import fiddle as fdl

    install_clip_guard()
    callback = install_manual_registration(a.platform)
    recipe = build_recipe(a)
    recipe.trainer.strategy.ddp.use_megatron_fsdp = True
    recipe.trainer.strategy.ddp.average_in_collective = False
    recipe.trainer.callbacks = list(recipe.trainer.callbacks or []) + [fdl.Config(callback)]
    recipe.trainer.strategy.num_distributed_optimizer_instances = 1
    recipe.model.config.num_layers = a.num_layers
    recipe.model.config.seq_length = a.seq
    recipe.data.seq_length = a.seq
    recipe.trainer.enable_checkpointing = False
    recipe.trainer.log_every_n_steps = a.log_every_n_steps
    recipe.log.ckpt = None
    recipe.log.log_dir = a.result_dir
    recipe.resume = None
    fdl.build(recipe)()


if __name__ == "__main__":
    main()
