from __future__ import annotations

import functools
import os
from typing import Tuple


@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    import torch
    import torch.version

    if not torch.cuda.is_available() or not torch.version.cuda:
        return None
    return torch.cuda.get_device_capability()


def is_arch_supported(major: int, minor: int = 0) -> bool:
    """capability >= (major, minor). Open-ended: newer archs also pass. Only use this
    for family-portable features (e.g. PDL); arch-specific kernels (sm_90a/sm_100a
    cubins) need the closed is_smXX_family checks below."""
    arch = _get_torch_cuda_version()
    if arch is None:
        return False
    return arch >= (major, minor)


def _is_arch_family(major: int) -> bool:
    arch = _get_torch_cuda_version()
    return arch is not None and arch[0] == major


def is_sm90_family() -> bool:
    """Exactly major 9 (Hopper). For sm_90a-only kernels (e.g. FA3)."""
    return _is_arch_family(9)


def is_sm100_family() -> bool:
    """Exactly major 10 (datacenter Blackwell). For sm_100a/103a-only kernels
    (e.g. trtllm-gen) that consumer Blackwell (sm_120/121) cannot run."""
    return _is_arch_family(10)


def is_sm90_supported() -> bool:
    return is_arch_supported(9, 0)


def is_sm100_supported() -> bool:
    return is_arch_supported(10, 0)


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def triton_turing_compat_enabled(device=None) -> bool:
    """Whether FreeToken's conservative Triton-Turing launch configs are enabled.

    ``FREETOKEN_TRITON_TURING_COMPAT`` is tri-state:

    * unset or ``auto``: enable only on SM75 (the NVIDIA Turing compute capability);
    * ``1``/``true``/``on``: force-enable, useful for tests and bring-up;
    * ``0``/``false``/``off``: disable and retain the upstream launch configs.

    This controls FreeToken kernel launch/layout workarounds. It does not install or
    replace the ``triton`` Python distribution; an SM75 process must still be started
    in an environment where the community ``triton-turing`` fork is installed.
    """
    raw = os.environ.get("FREETOKEN_TRITON_TURING_COMPAT", "auto").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    if raw != "auto":
        raise ValueError(
            "FREETOKEN_TRITON_TURING_COMPAT must be auto, 1/true/on, or 0/false/off; "
            f"got {raw!r}"
        )

    if device is None:
        capability = _get_torch_cuda_version()
    else:
        import torch

        if not torch.cuda.is_available():
            return False
        capability = torch.cuda.get_device_capability(device)
    return capability == (7, 5)
