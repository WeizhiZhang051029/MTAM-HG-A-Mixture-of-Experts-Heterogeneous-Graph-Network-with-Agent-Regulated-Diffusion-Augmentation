"""Fixed-batch CUDA compilation; tail batches and evaluation remain eager."""

from __future__ import annotations
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import tempfile
import time
import torch
import config

_cache_fd = None

def configure_compile_cache() -> None:
    global _cache_fd
    if _cache_fd is None:
        cache = Path(config.PROJECT_ROOT) / ".compile_cache"
        cache.mkdir(parents=True, exist_ok=True)

        _cache_fd = os.open(cache, os.O_RDONLY | os.O_DIRECTORY)
        alias = Path(f"/proc/{os.getpid()}/fd/{_cache_fd}")
        for name in ("tmp", "inductor", "triton"):
            (cache / name).mkdir(exist_ok=True)
        os.environ["TMPDIR"] = str(alias / "tmp")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
        os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "4"
        tempfile.tempdir = str(alias / "tmp")


        import torch._inductor.config as inductor_config
        inductor_config.fx_graph_cache = True

def compiled_pretraining(function):
    @wraps(function)
    def run(*args, **kwargs):
        if not getattr(config, "FAST_COMPILE_PRETRAIN", False):
            return function(*args, **kwargs)
        bound = inspect.signature(function).bind(*args, **kwargs)
        model = bound.arguments["model"]
        device = torch.device(bound.arguments["device"])
        if device.type != "cuda":
            raise ValueError("Compiled pretraining profile requires CUDA.")
        if not getattr(config, "FAST_SKIP_REDUNDANT_REAL_FORWARD", False):
            raise ValueError("Compile profile requires redundant-real-forward skipping.")
        configure_compile_cache()
        batch_size = int(getattr(config, "SYNTHETIC_BATCH_SIZE", config.BATCH_SIZE))
        example = bound.arguments["train_tensors"].x[:batch_size].to(device)
        original_forward = model.forward
        had_override = "forward" in model.__dict__
        original_mode = model.training
        model.eval()
        try:
            with torch.no_grad():
                original_forward(example)
        finally:
            model.train(original_mode)
        compiled = torch.compile(original_forward, mode="reduce-overhead")
        from torch._dynamo.utils import counters
        cache_before = {name: counters["inductor"][name]
                        for name in ("fxgraph_cache_hit", "fxgraph_cache_miss")}
        stats = {
            "compiled_calls": 0, "eager_calls": 0, "batch_shape": list(example.shape),
            "first_compiled_forward_seconds": None,
            "note": "First-forward time excludes lazy backward compilation; phase/epoch time includes both.",
            "mode": "reduce-overhead", "status": "running",
        }
        def forward(x, *other_args, **other_kwargs):
            use_compiled = (model.training and torch.is_grad_enabled()
                            and tuple(x.shape) == tuple(example.shape))
            if not use_compiled:
                stats["eager_calls"] += 1
                return original_forward(x, *other_args, **other_kwargs)
            torch.compiler.cudagraph_mark_step_begin()
            first = stats["compiled_calls"] == 0
            if first:
                torch.cuda.synchronize(device)
                started = time.perf_counter()
            outputs = compiled(x, *other_args, **other_kwargs)
            if first:
                torch.cuda.synchronize(device)
                stats["first_compiled_forward_seconds"] = time.perf_counter() - started
            stats["compiled_calls"] += 1
            return outputs
        model.forward = forward
        try:
            value = function(*args, **kwargs)
            stats["status"] = "complete"
            return value
        except BaseException:
            stats["status"] = "failed"
            raise
        finally:
            if had_override:
                model.forward = original_forward
            else:
                del model.forward
            stats["disk_cache"] = {
                name: counters["inductor"][name] - initial
                for name, initial in cache_before.items()
            }
            path = Path(config.RESULT_DIR) / "compilation_stats.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(stats, indent=2))
    return run
