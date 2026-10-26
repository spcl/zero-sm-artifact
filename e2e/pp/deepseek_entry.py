"""DeepSeek-V3 entry: run the Hydra harness of the PP Megatron tree with the model config read from a local directory.

Arguments are Hydra overrides. DEEPSEEK_HF_CONFIG is a directory holding the config.json of deepseek-ai/DeepSeek-V3
(e2e/pp/deepseek_data.sh); no weights are used.
"""
import os
import runpy
import sys
from pathlib import Path

import megatron.core
import torch
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

capture_begin = torch.cuda.CUDAGraph.capture_begin


def capture_begin_with_rng_state(self, *args, **kwargs):
    # Create each CUDA generator's per-capture RNG state at capture begin, as in the PyTorch build the results
    # were measured with.
    capture_begin(self, *args, **kwargs)
    generators = [torch.cuda.default_generators[torch.cuda.current_device()]]
    generators += [g for g in get_cuda_rng_tracker().get_states().values() if isinstance(g, torch.Generator)]
    buf = torch.empty(1, device="cuda")
    for g in generators:
        buf.uniform_(generator=g)


torch.cuda.CUDAGraph.capture_begin = capture_begin_with_rng_state

harness = Path(megatron.core.__file__).resolve().parents[2] / "examples/zero_sm_pp/deepseek_mlperf"
local_config = os.environ["DEEPSEEK_HF_CONFIG"]
from_hf_pretrained = AutoBridge.from_hf_pretrained


def from_local_config(cls, path, **kwargs):
    return from_hf_pretrained(local_config if str(path) == "deepseek-ai/DeepSeek-V3" else path, **kwargs)


AutoBridge.from_hf_pretrained = classmethod(from_local_config)
sys.path.insert(0, str(harness))
sys.argv[0] = str(harness / "pretrain.py")
runpy.run_path(sys.argv[0], run_name="__main__")
