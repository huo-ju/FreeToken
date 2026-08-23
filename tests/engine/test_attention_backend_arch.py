"""Arch-family gating for attention backends.

The utils.arch helpers come in two flavors: open-ended ``is_smXX_supported``
(capability >= X, for family-portable features like PDL) and closed
``is_smXX_family`` (major == X, for arch-specific cubins: FA3 is sm_90a-only,
trtllm-gen is sm_100a/103a-only). Auto backend selection must use the closed
checks: consumer Blackwell (sm_120/121) passes ``is_sm100_supported`` but
cannot run either kernel family, so it has to resolve to fi.
"""

from types import SimpleNamespace

import pytest
import torch

from freetoken.utils import arch


@pytest.mark.parametrize(
    "cc, sm90_sup, sm100_sup, sm90_fam, sm100_fam",
    [
        ((8, 6), False, False, False, False),
        ((9, 0), True, False, True, False),
        ((10, 0), True, True, False, True),
        ((10, 3), True, True, False, True),
        ((11, 0), True, True, False, False),
        ((12, 0), True, True, False, False),
        ((12, 1), True, True, False, False),
        (None, False, False, False, False),  # no CUDA device
    ],
)
def test_arch_helper_semantics(monkeypatch, cc, sm90_sup, sm100_sup, sm90_fam, sm100_fam):
    monkeypatch.setattr(arch, "_get_torch_cuda_version", lambda: cc)
    assert arch.is_sm90_supported() is sm90_sup
    assert arch.is_sm100_supported() is sm100_sup
    assert arch.is_sm90_family() is sm90_fam
    assert arch.is_sm100_family() is sm100_fam


@pytest.mark.parametrize(
    "cc, requested, expected",
    [
        ((7, 5), torch.bfloat16, torch.float16),
        ((7, 5), torch.float16, torch.float16),
        ((8, 0), torch.bfloat16, torch.bfloat16),
        (None, torch.bfloat16, torch.bfloat16),
    ],
)
def test_sm75_activation_dtype_policy(monkeypatch, cc, requested, expected):
    monkeypatch.setattr(arch, "_get_torch_cuda_version", lambda: cc)
    assert arch.is_sm75_device() is (cc == (7, 5))
    assert arch.sm75_activation_dtype(requested) == expected


@pytest.mark.parametrize(
    "value, cc, expected",
    [
        (None, (7, 5), True),
        ("auto", (7, 5), True),
        (None, (8, 0), False),
        ("1", (8, 0), True),
        ("true", None, True),
        ("0", (7, 5), False),
        ("off", (7, 5), False),
    ],
)
def test_triton_turing_compat_switch(monkeypatch, value, cc, expected):
    if value is None:
        monkeypatch.delenv("FREETOKEN_TRITON_TURING_COMPAT", raising=False)
    else:
        monkeypatch.setenv("FREETOKEN_TRITON_TURING_COMPAT", value)
    monkeypatch.setattr(arch, "_get_torch_cuda_version", lambda: cc)
    assert arch.triton_turing_compat_enabled() is expected


def test_triton_turing_compat_rejects_bad_value(monkeypatch):
    monkeypatch.setenv("FREETOKEN_TRITON_TURING_COMPAT", "sometimes")
    with pytest.raises(ValueError, match="FREETOKEN_TRITON_TURING_COMPAT"):
        arch.triton_turing_compat_enabled()


def test_sm75_dtype_policy_is_independent_of_compat_switch(monkeypatch):
    monkeypatch.setenv("FREETOKEN_TRITON_TURING_COMPAT", "0")
    monkeypatch.setattr(arch, "_get_torch_cuda_version", lambda: (7, 5))
    assert not arch.triton_turing_compat_enabled()
    assert arch.sm75_activation_dtype(torch.bfloat16) == torch.float16


def _engine_config(**overrides):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/freetoken-test-model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        **overrides,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            has_swa_attention=False,
            has_linear_attention=False,
            is_moe=False,
            num_layers=10,
            expert_quant="none",
        ),
    )
    return config


def _patch_env(monkeypatch, *, major, flashinfer=True, sgl=True):
    from freetoken.engine import engine

    monkeypatch.setattr(engine, "is_sm100_family", lambda: major == 10)
    monkeypatch.setattr(engine, "is_sm90_family", lambda: major == 9)
    monkeypatch.setattr(engine, "_flashinfer_available", lambda: flashinfer)
    monkeypatch.setattr(engine, "_sgl_flash_attn_available", lambda: sgl)


@pytest.mark.parametrize(
    "major, flashinfer, sgl, expected",
    [
        (10, True, True, "trtllm"),  # datacenter Blackwell
        (9, True, True, "fa,fi"),  # Hopper with sgl_kernel
        (9, True, False, "fi"),  # Hopper without sgl_kernel
        (12, True, True, "fi"),  # consumer Blackwell: no FA3/FA4/trtllm-gen kernels
        (11, True, True, "fi"),  # Thor
        (8, True, True, "fi"),  # Ampere
        (12, False, True, "triton"),  # no flashinfer -> only self-contained option
    ],
)
def test_auto_backend_selection_by_arch(monkeypatch, major, flashinfer, sgl, expected):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch, major=major, flashinfer=flashinfer, sgl=sgl)
    config = _engine_config(attention_backend="auto")
    _adjust_config(config)
    assert config.attention_backend == expected


def test_adjust_config_normalizes_sm75_bf16_to_fp16(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch, major=7)
    monkeypatch.setattr(arch, "_get_torch_cuda_version", lambda: (7, 5))
    config = _engine_config(attention_backend="triton")
    _adjust_config(config)
    assert config.dtype == torch.float16


def test_explicit_trtllm_rejected_outside_sm100_family(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch, major=12)
    config = _engine_config(attention_backend="trtllm")
    with pytest.raises(RuntimeError, match="10.x"):
        _adjust_config(config)


def test_explicit_trtllm_allowed_on_sm100_family(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch, major=10)
    config = _engine_config(attention_backend="trtllm")
    _adjust_config(config)
    assert config.attention_backend == "trtllm"
    assert config.page_size in (16, 32, 64)


def test_fa_backend_rejects_cc12(monkeypatch):
    from freetoken.attention.fa import FlashAttentionBackend

    monkeypatch.setattr(arch, "_get_torch_cuda_version", lambda: (12, 0))
    with pytest.raises(RuntimeError, match="12.x"):
        FlashAttentionBackend(SimpleNamespace(head_dim=128))
