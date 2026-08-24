from __future__ import annotations

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
