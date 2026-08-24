from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import torch


def test_prepare_cuda_device_binds_rank_before_cuda_helpers(monkeypatch):
    from freetoken.engine import engine

    calls: list[object] = []
    config = SimpleNamespace(tp_info=SimpleNamespace(rank=2, size=4))

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        engine,
        "set_tp_info",
        lambda *, rank, size: calls.append(("tp", rank, size)),
    )
    monkeypatch.setattr(
        engine,
        "bind_assigned_gpu",
        lambda rank: calls.append(("device", rank)) or torch.device(f"cuda:{rank}"),
    )
    monkeypatch.setattr(
        engine,
        "_ensure_expandable_segments",
        lambda: calls.append("allocator"),
    )
    monkeypatch.setattr(
        engine,
        "_adjust_config",
        lambda cfg: calls.append(("adjust", cfg)),
    )

    device = engine._prepare_cuda_device(config)

    assert device == torch.device("cuda:2")
    assert calls == [
        ("tp", 2, 4),
        ("device", 2),
        "allocator",
        ("adjust", config),
    ]


def test_expandable_segments_prefers_accelerator_api(monkeypatch):
    from freetoken.engine import engine

    calls: list[str] = []
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(
        torch._C,
        "_accelerator_setAllocatorSettings",
        lambda value: calls.append(value),
    )
    monkeypatch.setattr(
        torch.cuda.memory,
        "_set_allocator_settings",
        lambda value: (_ for _ in ()).throw(AssertionError("deprecated API used")),
    )

    engine._ensure_expandable_segments()

    assert calls == ["expandable_segments:True"]


def test_triton_sampling_import_does_not_probe_cuda(monkeypatch):
    """Importing a worker-side kernel module must not create a default-GPU context."""
    module_name = "freetoken.kernel.triton.sampling"
    sys.modules.pop(module_name, None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sampling queried CUDA during import")
        ),
    )

    importlib.import_module(module_name)


def test_triton_sampling_plan_probes_tensor_device(monkeypatch):
    sampling = importlib.import_module("freetoken.kernel.triton.sampling")
    seen: list[torch.device] = []

    def properties(device):
        seen.append(device)
        return SimpleNamespace(multi_processor_count=80)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)

    assert sampling._plan(2, 32768, torch.device("cuda:3")) == (8, 4096)
    assert seen == [torch.device("cuda:3")]
