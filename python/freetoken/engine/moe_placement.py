"""Pure startup placement planning for authoritative MoE expert storage.

The planner deliberately has no torch, CUDA, filesystem, or hardware-discovery
side effects.  Callers measure budgets and provider geometry first, then pass
plain integers here.  A successful plan is therefore a capacity proof that can
be produced before any large expert allocation starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence


class ExpertTier(str, Enum):
    GPU_PERMANENT = "gpu_permanent"
    HOST_PINNED = "host_pinned"
    HOST_LOCKED = "host_locked"
    HOST_PAGEABLE = "host_pageable"


@dataclass(frozen=True)
class ExpertPlacementCapabilities:
    allocation_specs: bool = False
    gpu_permanent_streaming: bool = False
    per_layer_host_residency: bool = False
    cpu_executor_layout: bool = False
    mixed_cpu_gpu_layout: bool = False
    ftw_gpu_permanent_streaming: bool = False


@dataclass(frozen=True)
class MoePlacementInputs:
    """Measured per-rank budgets and checkpoint geometry.

    ``available_vram_bytes`` is the baseline VRAM available to the process,
    before fixed model/runtime allocations.  Host and pin budgets apply only to
    expert banks; their caller-owned safety margins must already be deducted.
    """

    available_vram_bytes: int
    fixed_gpu_bytes: int
    activation_graph_reserve_bytes: int
    host_expert_budget_bytes: int
    pin_budget_bytes: int
    kv_reserve_bytes: int
    expert_bytes_per_slot: int
    host_bytes_per_layer: tuple[int, ...]
    gpu_bytes_per_layer: tuple[int, ...]
    num_experts: int
    prefill_overlap: bool = True
    dynamic_cache_slots: int | None = None
    kv_page_bytes: int = 1
    kv_reserve_pages: int | None = None
    max_dynamic_cache_slots: int | None = None

    @property
    def num_layers(self) -> int:
        return len(self.host_bytes_per_layer)


@dataclass(frozen=True)
class MoePlacementPlan:
    layer_tiers: tuple[ExpertTier, ...]
    permanent_layer_ids: tuple[int, ...]
    cpu_layer_ids: frozenset[int]
    permanent_gpu_bytes: int
    retained_host_bytes: int
    pinned_host_bytes: int
    locked_host_bytes: int
    dynamic_cache_slots: int
    dynamic_floor_slots: int
    kv_pages: int
    kv_bytes: int
    elastic_vram_bytes: int
    prefill_overlap: bool
    constraints: tuple[str, ...]
    decisions: tuple[str, ...]
    warnings: tuple[str, ...]

    def tier_layer_ids(self, tier: ExpertTier) -> tuple[int, ...]:
        return tuple(i for i, value in enumerate(self.layer_tiers) if value is tier)


class MoePlacementInfeasible(ValueError):
    """A pre-load capacity or provider-capability failure."""


def _gib(value: int) -> str:
    return f"{value / 2**30:.2f} GiB"


def _validate_layer_ids(ids: Iterable[int], num_layers: int, label: str) -> tuple[int, ...]:
    result = tuple(sorted(set(ids)))
    invalid = [layer_id for layer_id in result if not 0 <= layer_id < num_layers]
    if invalid:
        raise MoePlacementInfeasible(
            f"{label} contains out-of-range layer ids {invalid}; valid range is "
            f"0..{num_layers - 1}"
        )
    return result


def _head_tail_order(layer_ids: Iterable[int]) -> tuple[int, ...]:
    ordered = sorted(layer_ids)
    result: list[int] = []
    lo, hi = 0, len(ordered) - 1
    while lo <= hi:
        result.append(ordered[lo])
        lo += 1
        if lo <= hi:
            result.append(ordered[hi])
            hi -= 1
    return tuple(result)


def _permanent_order(
    candidates: Iterable[int], host_bytes_per_layer: Sequence[int]
) -> tuple[int, ...]:
    # Largest host saving first proves the minimum layer count.  Equal-size
    # layers use the documented deepest-first locality heuristic.
    return tuple(
        sorted(candidates, key=lambda layer_id: (host_bytes_per_layer[layer_id], layer_id), reverse=True)
    )


def _infeasible_capacity(
    inputs: MoePlacementInputs,
    permanent_gpu_bytes: int,
    dynamic_floor_bytes: int,
    reason: str,
) -> MoePlacementInfeasible:
    host_total = sum(inputs.host_bytes_per_layer)
    host_deficit = max(0, host_total - inputs.host_expert_budget_bytes)
    elastic = (
        inputs.available_vram_bytes
        - inputs.fixed_gpu_bytes
        - inputs.activation_graph_reserve_bytes
        - permanent_gpu_bytes
    )
    required = dynamic_floor_bytes + inputs.kv_reserve_bytes
    shortfall = max(0, required - elastic)
    return MoePlacementInfeasible(
        "placement infeasible:\n"
        f"  reason:                       {reason}\n"
        f"  host expert pool:             {_gib(host_total)}\n"
        f"  host expert budget:           {_gib(inputs.host_expert_budget_bytes)}\n"
        f"  must move to GPU:             {_gib(host_deficit)}\n"
        f"  permanent GPU required:       {_gib(permanent_gpu_bytes)} per rank\n"
        f"  elastic VRAM remaining:       {_gib(elastic)} per rank\n"
        f"  required dynamic + KV floor:  {_gib(required)} per rank\n"
        f"  shortfall:                    {_gib(shortfall)} per rank"
    )


def plan_moe_placement(
    inputs: MoePlacementInputs,
    capabilities: ExpertPlacementCapabilities,
    *,
    policy: str = "balanced",
    allow_gpu_permanent: bool = True,
    explicit_permanent_layer_ids: Iterable[int] = (),
    explicit_permanent_layer_count: int | None = None,
    explicit_cpu_layer_ids: Iterable[int] = (),
    allow_additional_cpu_layers: bool = True,
) -> MoePlacementPlan:
    """Solve authoritative placement and the initial elastic-pool geometry.

    Automatic host-capacity placement chooses the minimum number of complete
    layers.  ``gpu-first`` may add permanent layers to solve a pin deficit;
    ``balanced`` and ``cpu-first`` preserve elastic VRAM and use locked host
    banks for that second-stage deficit.
    """

    if policy not in {"balanced", "gpu-first", "cpu-first"}:
        raise ValueError(f"unknown MoE placement policy {policy!r}")
    if inputs.num_layers <= 0 or inputs.num_experts <= 0:
        raise ValueError("MoE placement requires positive layer and expert counts")
    if len(inputs.gpu_bytes_per_layer) != inputs.num_layers:
        raise ValueError("host_bytes_per_layer and gpu_bytes_per_layer must have equal length")
    numeric = (
        inputs.available_vram_bytes,
        inputs.fixed_gpu_bytes,
        inputs.activation_graph_reserve_bytes,
        inputs.host_expert_budget_bytes,
        inputs.pin_budget_bytes,
        inputs.kv_reserve_bytes,
        inputs.expert_bytes_per_slot,
        *inputs.host_bytes_per_layer,
        *inputs.gpu_bytes_per_layer,
    )
    if any(value < 0 for value in numeric) or inputs.expert_bytes_per_slot == 0:
        raise ValueError("placement budgets and layer sizes must be non-negative; slot bytes must be positive")

    permanent = set(
        _validate_layer_ids(explicit_permanent_layer_ids, inputs.num_layers, "permanent layers")
    )
    cpu = set(_validate_layer_ids(explicit_cpu_layer_ids, inputs.num_layers, "CPU layers"))
    if permanent & cpu:
        raise MoePlacementInfeasible(
            f"layers {sorted(permanent & cpu)} cannot be both GPU permanent and CPU resident"
        )
    if permanent and not (
        capabilities.allocation_specs and capabilities.gpu_permanent_streaming
    ):
        raise MoePlacementInfeasible(
            "explicit GPU permanent placement is unsupported by this expert provider"
        )
    if cpu and not (
        capabilities.per_layer_host_residency and capabilities.cpu_executor_layout
    ):
        raise MoePlacementInfeasible(
            "explicit CPU-layer placement is unsupported by this expert provider"
        )
    if explicit_permanent_layer_count is not None:
        if explicit_permanent_layer_count < 0 or explicit_permanent_layer_count > inputs.num_layers:
            raise MoePlacementInfeasible(
                f"permanent layer count must be in 0..{inputs.num_layers}"
            )
        if permanent and len(permanent) != explicit_permanent_layer_count:
            raise MoePlacementInfeasible(
                "explicit permanent layer ids do not match explicit permanent layer count"
            )
        if not permanent:
            permanent.update(
                _permanent_order(range(inputs.num_layers), inputs.host_bytes_per_layer)[
                    :explicit_permanent_layer_count
                ]
            )
    if permanent & cpu:
        raise MoePlacementInfeasible(
            f"layers {sorted(permanent & cpu)} cannot be both GPU permanent and CPU resident"
        )
    if permanent and not (
        capabilities.allocation_specs and capabilities.gpu_permanent_streaming
    ):
        raise MoePlacementInfeasible(
            "explicit GPU permanent placement is unsupported by this expert provider"
        )

    decisions: list[str] = []
    constraints = [
        f"host expert budget {_gib(inputs.host_expert_budget_bytes)}",
        f"pin budget {_gib(inputs.pin_budget_bytes)}",
        f"KV reserve {_gib(inputs.kv_reserve_bytes)}",
    ]
    warnings: list[str] = []

    host_total = sum(inputs.host_bytes_per_layer)
    host_deficit = max(0, host_total - inputs.host_expert_budget_bytes)
    host_removed = sum(inputs.host_bytes_per_layer[i] for i in permanent)
    if host_removed < host_deficit:
        fixed_count = explicit_permanent_layer_count is not None
        if not allow_gpu_permanent or fixed_count:
            raise _infeasible_capacity(
                inputs,
                sum(inputs.gpu_bytes_per_layer[i] for i in permanent),
                0,
                "explicit policy leaves more host expert data than the host budget permits",
            )
        if not (capabilities.allocation_specs and capabilities.gpu_permanent_streaming):
            raise MoePlacementInfeasible(
                "placement infeasible: host RAM deficit requires GPU permanent layers, "
                "but the expert provider lacks allocation specs or streaming support"
            )
        for layer_id in _permanent_order(
            (i for i in range(inputs.num_layers) if i not in permanent and i not in cpu),
            inputs.host_bytes_per_layer,
        ):
            permanent.add(layer_id)
            host_removed += inputs.host_bytes_per_layer[layer_id]
            if host_removed >= host_deficit:
                break
        if host_removed < host_deficit:
            raise MoePlacementInfeasible(
                "placement infeasible: explicit CPU layers prevent enough host data from moving to GPU"
            )
        decisions.append(
            f"selected {len(permanent)} GPU permanent layers to cover {_gib(host_deficit)} host deficit"
        )

    permanent_gpu = sum(inputs.gpu_bytes_per_layer[i] for i in permanent)
    host_backed = set(range(inputs.num_layers)) - permanent
    # Locked layers cannot use the asynchronous double buffer in phase one.
    overlap = inputs.prefill_overlap and not cpu
    if inputs.prefill_overlap and cpu:
        decisions.append("disabled prefill overlap because locked CPU layers are present")
    dynamic_floor_slots = 0 if not host_backed else inputs.num_experts * (2 if overlap else 1)
    dynamic_floor_bytes = dynamic_floor_slots * inputs.expert_bytes_per_slot

    def elastic_bytes() -> int:
        return (
            inputs.available_vram_bytes
            - inputs.fixed_gpu_bytes
            - inputs.activation_graph_reserve_bytes
            - sum(inputs.gpu_bytes_per_layer[i] for i in permanent)
        )

    if elastic_bytes() < dynamic_floor_bytes + inputs.kv_reserve_bytes:
        raise _infeasible_capacity(
            inputs, permanent_gpu, dynamic_floor_bytes,
            "permanent placement leaves less than the dynamic-cache and KV floors",
        )

    pinned = host_backed - cpu
    pinned_bytes = sum(inputs.host_bytes_per_layer[i] for i in pinned)
    pin_deficit = max(0, pinned_bytes - inputs.pin_budget_bytes)
    if pin_deficit:
        can_lock = (
            allow_additional_cpu_layers
            and capabilities.per_layer_host_residency
            and capabilities.cpu_executor_layout
        )
        mixed_lock = (
            capabilities.per_layer_host_residency
            and capabilities.cpu_executor_layout
            and capabilities.mixed_cpu_gpu_layout
        )
        try_gpu = policy == "gpu-first" or (policy == "balanced" and not can_lock)
        if (
            try_gpu
            and allow_gpu_permanent
            and explicit_permanent_layer_count is None
        ):
            if not (capabilities.allocation_specs and capabilities.gpu_permanent_streaming):
                warnings.append("gpu-first unavailable: provider cannot stream permanent layers")
            else:
                for layer_id in _permanent_order(pinned, inputs.host_bytes_per_layer):
                    candidate_floor_slots = 0 if len(permanent) + 1 == inputs.num_layers else dynamic_floor_slots
                    candidate_required = (
                        candidate_floor_slots * inputs.expert_bytes_per_slot
                        + inputs.kv_reserve_bytes
                    )
                    candidate_elastic = elastic_bytes() - inputs.gpu_bytes_per_layer[layer_id]
                    if candidate_elastic < candidate_required:
                        continue
                    permanent.add(layer_id)
                    pinned.remove(layer_id)
                    pinned_bytes -= inputs.host_bytes_per_layer[layer_id]
                    if pinned_bytes <= inputs.pin_budget_bytes:
                        break
                if pinned_bytes <= inputs.pin_budget_bytes:
                    decisions.append("used additional GPU permanent layers to satisfy the pin budget")

        if pinned_bytes > inputs.pin_budget_bytes:
            if not can_lock:
                raise MoePlacementInfeasible(
                    "placement infeasible: pin budget requires locked host layers, but the "
                    "provider lacks per-layer residency or a compatible CPU executor"
                )
            for layer_id in _head_tail_order(pinned):
                cpu.add(layer_id)
                pinned.remove(layer_id)
                pinned_bytes -= inputs.host_bytes_per_layer[layer_id]
                if pinned_bytes <= inputs.pin_budget_bytes and mixed_lock:
                    break
            decisions.append("selected head/tail HOST_LOCKED layers to satisfy the pin budget")
            if overlap:
                overlap = False
                dynamic_floor_slots = 0 if not pinned and not cpu else inputs.num_experts
                dynamic_floor_bytes = dynamic_floor_slots * inputs.expert_bytes_per_slot
                decisions.append("disabled prefill overlap because locked CPU layers are present")

    host_backed = set(range(inputs.num_layers)) - permanent
    overlap = inputs.prefill_overlap and not cpu
    dynamic_floor_slots = 0 if not host_backed else inputs.num_experts * (2 if overlap else 1)
    dynamic_floor_bytes = dynamic_floor_slots * inputs.expert_bytes_per_slot
    permanent_gpu = sum(inputs.gpu_bytes_per_layer[i] for i in permanent)
    elastic = elastic_bytes()
    if elastic < dynamic_floor_bytes + inputs.kv_reserve_bytes:
        raise _infeasible_capacity(
            inputs, permanent_gpu, dynamic_floor_bytes,
            "pin-budget placement leaves less than the dynamic-cache and KV floors",
        )

    if inputs.max_dynamic_cache_slots is None:
        max_dynamic = max(
            dynamic_floor_slots,
            len(pinned) * inputs.num_experts,
        )
    else:
        max_dynamic = inputs.max_dynamic_cache_slots
    if max_dynamic < dynamic_floor_slots:
        raise MoePlacementInfeasible(
            f"dynamic cache cap {max_dynamic} is below the floor {dynamic_floor_slots}"
        )
    if inputs.dynamic_cache_slots is None:
        dynamic_slots = min(
            max_dynamic,
            (elastic - inputs.kv_reserve_bytes) // inputs.expert_bytes_per_slot,
        )
        dynamic_slots = max(dynamic_floor_slots, dynamic_slots)
    else:
        dynamic_slots = inputs.dynamic_cache_slots
        if not dynamic_floor_slots and not host_backed:
            dynamic_slots = 0
        if dynamic_slots < dynamic_floor_slots:
            raise MoePlacementInfeasible(
                f"dynamic cache has {dynamic_slots} slots; floor is {dynamic_floor_slots}"
            )
        if dynamic_slots > max_dynamic:
            raise MoePlacementInfeasible(
                f"dynamic cache has {dynamic_slots} slots; maximum useful host-backed cache is {max_dynamic}"
            )

    remaining_for_kv = elastic - dynamic_slots * inputs.expert_bytes_per_slot
    kv_page_bytes = inputs.kv_page_bytes
    if kv_page_bytes <= 0:
        raise ValueError("kv_page_bytes must be positive")
    reserve_pages = (
        inputs.kv_reserve_pages
        if inputs.kv_reserve_pages is not None
        else (inputs.kv_reserve_bytes + kv_page_bytes - 1) // kv_page_bytes
    )
    kv_pages = remaining_for_kv // kv_page_bytes
    if kv_pages < reserve_pages or remaining_for_kv < inputs.kv_reserve_bytes:
        raise _infeasible_capacity(
            inputs, permanent_gpu, dynamic_floor_bytes,
            "requested dynamic cache leaves less than the KV reserve",
        )
    kv_bytes = kv_pages * kv_page_bytes

    layer_tiers = tuple(
        ExpertTier.GPU_PERMANENT
        if i in permanent
        else ExpertTier.HOST_LOCKED
        if i in cpu
        else ExpertTier.HOST_PINNED
        for i in range(inputs.num_layers)
    )
    retained_host = sum(inputs.host_bytes_per_layer[i] for i in range(inputs.num_layers) if i not in permanent)
    locked_host = sum(inputs.host_bytes_per_layer[i] for i in cpu)
    pinned_host = retained_host - locked_host
    if retained_host > inputs.host_expert_budget_bytes:
        raise AssertionError("planner produced a plan above the host budget")
    if pinned_host > inputs.pin_budget_bytes:
        raise AssertionError("planner produced a plan above the pin budget")

    return MoePlacementPlan(
        layer_tiers=layer_tiers,
        permanent_layer_ids=tuple(sorted(permanent)),
        cpu_layer_ids=frozenset(cpu),
        permanent_gpu_bytes=permanent_gpu,
        retained_host_bytes=retained_host,
        pinned_host_bytes=pinned_host,
        locked_host_bytes=locked_host,
        dynamic_cache_slots=dynamic_slots,
        dynamic_floor_slots=dynamic_floor_slots,
        kv_pages=kv_pages,
        kv_bytes=kv_bytes,
        elastic_vram_bytes=elastic,
        prefill_overlap=overlap,
        constraints=tuple(constraints),
        decisions=tuple(decisions),
        warnings=tuple(warnings),
    )


__all__ = [
    "ExpertPlacementCapabilities",
    "ExpertTier",
    "MoePlacementInfeasible",
    "MoePlacementInputs",
    "MoePlacementPlan",
    "plan_moe_placement",
]
