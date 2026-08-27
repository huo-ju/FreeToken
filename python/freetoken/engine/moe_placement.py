"""Pure startup placement planning for authoritative MoE expert storage.

The planner deliberately has no torch, CUDA, filesystem, or hardware-discovery
side effects.  Callers measure budgets and provider geometry first, then pass
plain integers here.  A successful plan is therefore a capacity proof that can
be produced before any large expert allocation starts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Sequence

from freetoken.moe.execution_plan import RoutedTpLayout


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
    kv_min_pages: int
    kv_target_pages: int | None
    kv_page_bytes: int
    expert_bytes_per_slot: int
    host_bytes_per_layer: tuple[int, ...]
    gpu_bytes_per_layer: tuple[int, ...]
    num_experts: int
    prefill_overlap: bool = True
    dynamic_cache_slots: int | None = None
    max_dynamic_cache_slots: int | None = None

    @property
    def num_layers(self) -> int:
        return len(self.host_bytes_per_layer)

    @property
    def kv_mode(self) -> str:
        return "fixed" if self.kv_target_pages is not None else "auto"

    @property
    def kv_required_pages(self) -> int:
        return self.kv_target_pages if self.kv_target_pages is not None else self.kv_min_pages

    @property
    def kv_required_bytes(self) -> int:
        return self.kv_required_pages * self.kv_page_bytes


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
    fixed_gpu_bytes: int
    activation_graph_reserve_bytes: int
    dynamic_cache_bytes: int
    kv_pages: int
    kv_bytes: int
    elastic_vram_bytes: int
    vram_headroom_bytes: int
    prefill_overlap: bool
    constraints: tuple[str, ...]
    decisions: tuple[str, ...]
    warnings: tuple[str, ...]

    def tier_layer_ids(self, tier: ExpertTier) -> tuple[int, ...]:
        return tuple(i for i, value in enumerate(self.layer_tiers) if value is tier)


class MoePlacementInfeasible(ValueError):
    """A pre-load capacity or provider-capability failure."""


@dataclass(frozen=True)
class RankTopology:
    rank: int
    pci_bdf: str
    numa_node: int
    compute_capability: str
    vram_bytes: int
    pcie_generation: int
    pcie_width: int

    def __post_init__(self) -> None:
        if self.rank < 0 or not self.pci_bdf:
            raise ValueError("rank topology requires a non-negative rank and PCI BDF")
        if self.vram_bytes <= 0 or self.pcie_generation <= 0 or self.pcie_width <= 0:
            raise ValueError("rank topology VRAM and negotiated PCIe link must be positive")


@dataclass(frozen=True)
class NodeMoePlacementInputs:
    """All-rank inputs for one immutable startup placement decision."""

    rank_inputs: tuple[MoePlacementInputs, ...]
    rank_topology: tuple[RankTopology, ...]
    routed_tp_layout: RoutedTpLayout
    host_expert_budget_bytes: int
    pin_budget_bytes: int
    checkpoint_layout: str = "generic"
    runtime_disk_tier: bool = False


@dataclass(frozen=True)
class NodeMoePlacementPlan:
    routed_tp_layout: RoutedTpLayout
    rank_plans: tuple[MoePlacementPlan, ...]
    rank_topology: tuple[RankTopology, ...]
    retained_host_bytes: int
    pinned_host_bytes: int
    locked_host_bytes: int
    permanent_gpu_bytes: int
    fixed_gpu_bytes: int
    activation_graph_reserve_bytes: int
    dynamic_cache_bytes: int
    kv_bytes: int
    vram_headroom_bytes: tuple[int, ...]
    kv_mode: str
    kv_min_pages: int
    kv_target_pages: int | None
    kv_pages: int
    runtime_disk_tier: bool
    constraints: tuple[str, ...]
    checksum: str

    def canonical_dict(self) -> dict:
        return {
            "routed_tp_layout": self.routed_tp_layout.canonical_dict(),
            "rank_topology": [topology.__dict__ for topology in self.rank_topology],
            "rank_plans": [
                {
                    "layer_tiers": [tier.value for tier in plan.layer_tiers],
                    "permanent_layer_ids": list(plan.permanent_layer_ids),
                    "cpu_layer_ids": sorted(plan.cpu_layer_ids),
                    "permanent_gpu_bytes": plan.permanent_gpu_bytes,
                    "retained_host_bytes": plan.retained_host_bytes,
                    "pinned_host_bytes": plan.pinned_host_bytes,
                    "locked_host_bytes": plan.locked_host_bytes,
                    "dynamic_cache_slots": plan.dynamic_cache_slots,
                    "dynamic_floor_slots": plan.dynamic_floor_slots,
                    "fixed_gpu_bytes": plan.fixed_gpu_bytes,
                    "activation_graph_reserve_bytes": plan.activation_graph_reserve_bytes,
                    "dynamic_cache_bytes": plan.dynamic_cache_bytes,
                    "kv_pages": plan.kv_pages,
                    "kv_bytes": plan.kv_bytes,
                    "elastic_vram_bytes": plan.elastic_vram_bytes,
                    "vram_headroom_bytes": plan.vram_headroom_bytes,
                    "prefill_overlap": plan.prefill_overlap,
                    "constraints": list(plan.constraints),
                    "decisions": list(plan.decisions),
                    "warnings": list(plan.warnings),
                }
                for plan in self.rank_plans
            ],
            "retained_host_bytes": self.retained_host_bytes,
            "pinned_host_bytes": self.pinned_host_bytes,
            "locked_host_bytes": self.locked_host_bytes,
            "permanent_gpu_bytes": self.permanent_gpu_bytes,
            "fixed_gpu_bytes": self.fixed_gpu_bytes,
            "activation_graph_reserve_bytes": self.activation_graph_reserve_bytes,
            "dynamic_cache_bytes": self.dynamic_cache_bytes,
            "kv_bytes": self.kv_bytes,
            "vram_headroom_bytes": list(self.vram_headroom_bytes),
            "kv_mode": self.kv_mode,
            "kv_min_pages": self.kv_min_pages,
            "kv_target_pages": self.kv_target_pages,
            "kv_pages": self.kv_pages,
            "runtime_disk_tier": self.runtime_disk_tier,
            "constraints": list(self.constraints),
        }

    def verify_checksum(self) -> None:
        actual = _node_plan_checksum(self.canonical_dict())
        if actual != self.checksum:
            raise MoePlacementInfeasible(
                f"node placement checksum mismatch: got {self.checksum}, recomputed {actual}"
            )


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
    required = dynamic_floor_bytes + inputs.kv_required_bytes
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
        inputs.kv_min_pages,
        inputs.kv_target_pages if inputs.kv_target_pages is not None else 0,
        inputs.kv_page_bytes,
        inputs.expert_bytes_per_slot,
        *inputs.host_bytes_per_layer,
        *inputs.gpu_bytes_per_layer,
    )
    if any(value < 0 for value in numeric):
        raise ValueError("placement budgets, page counts, and layer sizes must be non-negative")
    if inputs.expert_bytes_per_slot == 0 or inputs.kv_page_bytes == 0:
        raise ValueError("expert slot bytes and KV page bytes must be positive")
    if inputs.kv_target_pages is not None and inputs.kv_target_pages == 0:
        raise ValueError("fixed KV target pages must be positive")

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
        (
            f"KV fixed {inputs.kv_target_pages} pages / "
            f"{_gib(inputs.kv_required_bytes)}"
            if inputs.kv_target_pages is not None
            else f"KV minimum {inputs.kv_min_pages} pages / {_gib(inputs.kv_required_bytes)}"
        ),
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

    if elastic_bytes() < dynamic_floor_bytes + inputs.kv_required_bytes:
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
                        + inputs.kv_required_bytes
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
    if elastic < dynamic_floor_bytes + inputs.kv_required_bytes:
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
            (elastic - inputs.kv_required_bytes) // inputs.expert_bytes_per_slot,
        )
        dynamic_slots = max(dynamic_floor_slots, dynamic_slots)
    else:
        dynamic_slots = inputs.dynamic_cache_slots
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
    required_pages = inputs.kv_required_pages
    kv_pages = (
        inputs.kv_target_pages
        if inputs.kv_target_pages is not None
        else remaining_for_kv // kv_page_bytes
    )
    if kv_pages < required_pages or remaining_for_kv < inputs.kv_required_bytes:
        raise _infeasible_capacity(
            inputs, permanent_gpu, dynamic_floor_bytes,
            "requested dynamic cache leaves less than the KV reserve",
        )
    kv_bytes = kv_pages * kv_page_bytes
    vram_headroom = remaining_for_kv - kv_bytes

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
        fixed_gpu_bytes=inputs.fixed_gpu_bytes,
        activation_graph_reserve_bytes=inputs.activation_graph_reserve_bytes,
        dynamic_cache_bytes=dynamic_slots * inputs.expert_bytes_per_slot,
        kv_pages=kv_pages,
        kv_bytes=kv_bytes,
        elastic_vram_bytes=elastic,
        vram_headroom_bytes=vram_headroom,
        prefill_overlap=overlap,
        constraints=tuple(constraints),
        decisions=tuple(decisions),
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class _PermanentState:
    layer_ids: tuple[int, ...]
    gpu_bytes: int
    host_bytes_removed: int


def _permanent_state_frontier(
    inputs: MoePlacementInputs,
    *,
    explicit_permanent_layer_ids: Iterable[int],
    explicit_permanent_layer_count: int | None,
    explicit_cpu_layer_ids: Iterable[int],
    allow_gpu_permanent: bool,
) -> tuple[_PermanentState, ...]:
    """Generate deterministic rank-local authoritative GPU choices.

    DSV4 has equal-sized layer shards, so its production frontier is the small
    ``k=0..L`` prefix family.  The generic path retains a Pareto frontier by
    (layer count, GPU bytes, host bytes removed), which keeps non-uniform test
    providers correct without imposing a proportional node budget.
    """
    base = set(
        _validate_layer_ids(
            explicit_permanent_layer_ids, inputs.num_layers, "permanent layers"
        )
    )
    cpu = set(_validate_layer_ids(explicit_cpu_layer_ids, inputs.num_layers, "CPU layers"))
    if base & cpu:
        raise MoePlacementInfeasible(
            f"layers {sorted(base & cpu)} cannot be both GPU permanent and CPU resident"
        )
    if explicit_permanent_layer_count is not None:
        count = explicit_permanent_layer_count
        if count < 0 or count > inputs.num_layers:
            raise MoePlacementInfeasible(
                f"permanent layer count must be in 0..{inputs.num_layers}"
            )
        if base and len(base) != count:
            raise MoePlacementInfeasible(
                "explicit permanent layer ids do not match explicit permanent layer count"
            )
        if not base:
            eligible = [layer for layer in range(inputs.num_layers) if layer not in cpu]
            if len(eligible) < count:
                raise MoePlacementInfeasible(
                    "explicit permanent layer count overlaps the required CPU layer set"
                )
            base.update(
                _permanent_order(eligible, inputs.host_bytes_per_layer)[:count]
            )
        ids = tuple(sorted(base))
        return (
            _PermanentState(
                ids,
                sum(inputs.gpu_bytes_per_layer[layer] for layer in ids),
                sum(inputs.host_bytes_per_layer[layer] for layer in ids),
            ),
        )

    optional = tuple(
        layer
        for layer in _permanent_order(
            (
                layer
                for layer in range(inputs.num_layers)
                if layer not in base and layer not in cpu
            ),
            inputs.host_bytes_per_layer,
        )
    )
    if not allow_gpu_permanent:
        optional = ()

    def state(ids: Iterable[int]) -> _PermanentState:
        ordered = tuple(sorted(ids))
        return _PermanentState(
            ordered,
            sum(inputs.gpu_bytes_per_layer[layer] for layer in ordered),
            sum(inputs.host_bytes_per_layer[layer] for layer in ordered),
        )

    # The serving provider's normal case: one deterministic prefix per count.
    geometries = {
        (inputs.host_bytes_per_layer[layer], inputs.gpu_bytes_per_layer[layer])
        for layer in optional
    }
    if len(geometries) <= 1:
        return tuple(state(base | set(optional[:count])) for count in range(len(optional) + 1))

    states = [state(base)]
    for layer in optional:
        expanded = states + [state((*candidate.layer_ids, layer)) for candidate in states]
        # Exact resource duplicates cannot differ in capacity. Prefer the stable
        # deepest-first layer order so caller iteration order never affects output.
        deduplicated: dict[tuple[int, int, int], _PermanentState] = {}
        for candidate in expanded:
            key = (
                len(candidate.layer_ids),
                candidate.gpu_bytes,
                candidate.host_bytes_removed,
            )
            previous = deduplicated.get(key)
            if previous is None or tuple(reversed(candidate.layer_ids)) > tuple(
                reversed(previous.layer_ids)
            ):
                deduplicated[key] = candidate
        values = tuple(deduplicated.values())
        states = []
        for candidate in values:
            dominated = any(
                other is not candidate
                and len(other.layer_ids) == len(candidate.layer_ids)
                and other.gpu_bytes <= candidate.gpu_bytes
                and other.host_bytes_removed >= candidate.host_bytes_removed
                and (
                    other.gpu_bytes < candidate.gpu_bytes
                    or other.host_bytes_removed > candidate.host_bytes_removed
                )
                for other in values
            )
            if not dominated:
                states.append(candidate)
    return tuple(
        sorted(states, key=lambda value: (value.gpu_bytes, -value.host_bytes_removed, value.layer_ids))
    )


def _cpu_state_frontier(
    inputs: MoePlacementInputs,
    capabilities: ExpertPlacementCapabilities,
    permanent_layer_ids: Iterable[int],
    *,
    explicit_cpu_layer_ids: Iterable[int],
    allow_additional_cpu_layers: bool,
) -> tuple[tuple[int, ...], ...]:
    base = set(_validate_layer_ids(explicit_cpu_layer_ids, inputs.num_layers, "CPU layers"))
    permanent = set(permanent_layer_ids)
    if base & permanent:
        return ()
    if not allow_additional_cpu_layers or not (
        capabilities.per_layer_host_residency and capabilities.cpu_executor_layout
    ):
        return (tuple(sorted(base)),)
    eligible = _head_tail_order(
        layer
        for layer in range(inputs.num_layers)
        if layer not in permanent and layer not in base
    )
    if not capabilities.mixed_cpu_gpu_layout:
        return tuple(
            dict.fromkeys((tuple(sorted(base)), tuple(sorted(base | set(eligible)))))
        )
    return tuple(
        tuple(sorted(base | set(eligible[:count]))) for count in range(len(eligible) + 1)
    )


def _rank_candidate_frontier(
    inputs: MoePlacementInputs,
    capabilities: ExpertPlacementCapabilities,
    *,
    policy: str,
    allow_gpu_permanent: bool,
    explicit_permanent_layer_ids: Iterable[int],
    explicit_permanent_layer_count: int | None,
    explicit_cpu_layer_ids: Iterable[int],
    allow_additional_cpu_layers: bool,
) -> tuple[MoePlacementPlan, ...]:
    permanent_states = _permanent_state_frontier(
        inputs,
        explicit_permanent_layer_ids=explicit_permanent_layer_ids,
        explicit_permanent_layer_count=explicit_permanent_layer_count,
        explicit_cpu_layer_ids=explicit_cpu_layer_ids,
        allow_gpu_permanent=allow_gpu_permanent,
    )
    candidates: dict[tuple, MoePlacementPlan] = {}
    failures: list[str] = []
    local_capacity = replace(
        inputs,
        host_expert_budget_bytes=sum(inputs.host_bytes_per_layer),
        pin_budget_bytes=sum(inputs.host_bytes_per_layer),
    )
    for permanent in permanent_states:
        cpu_states = _cpu_state_frontier(
            inputs,
            capabilities,
            permanent.layer_ids,
            explicit_cpu_layer_ids=explicit_cpu_layer_ids,
            allow_additional_cpu_layers=allow_additional_cpu_layers,
        )
        for cpu_ids in cpu_states:
            try:
                plan = plan_moe_placement(
                    local_capacity,
                    capabilities,
                    policy=policy,
                    allow_gpu_permanent=bool(permanent.layer_ids),
                    explicit_permanent_layer_ids=permanent.layer_ids,
                    explicit_permanent_layer_count=len(permanent.layer_ids),
                    explicit_cpu_layer_ids=cpu_ids,
                    allow_additional_cpu_layers=False,
                )
            except MoePlacementInfeasible as error:
                failures.append(str(error))
                continue
            key = (
                plan.permanent_gpu_bytes,
                plan.retained_host_bytes,
                plan.pinned_host_bytes,
                plan.locked_host_bytes,
                plan.dynamic_cache_slots,
                plan.kv_pages,
                plan.vram_headroom_bytes,
                plan.permanent_layer_ids,
                tuple(sorted(plan.cpu_layer_ids)),
            )
            candidates[key] = plan
    if not candidates:
        detail = failures[0] if failures else "no policy-compatible layer tier candidate"
        raise MoePlacementInfeasible(f"rank frontier is empty: {detail}")
    return tuple(
        sorted(
            candidates.values(),
            key=lambda plan: (
                plan.permanent_gpu_bytes,
                plan.locked_host_bytes,
                -plan.retained_host_bytes,
                plan.permanent_layer_ids,
                tuple(sorted(plan.cpu_layer_ids)),
            ),
        )
    )


def _node_plan_checksum(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def plan_node_moe_placement(
    inputs: NodeMoePlacementInputs,
    capabilities: Sequence[ExpertPlacementCapabilities],
    *,
    policy: str = "balanced",
    allow_gpu_permanent: bool = True,
    explicit_permanent_layer_ids: Iterable[int] = (),
    explicit_permanent_layer_count: int | None = None,
    explicit_cpu_layer_ids: Iterable[int] = (),
    allow_additional_cpu_layers: bool = True,
) -> NodeMoePlacementPlan:
    """Jointly prove aggregate host/pin and every rank's VRAM/KV capacity.

    Rank frontiers contain only locally feasible VRAM/cache/KV states.  This
    function combines them under the node-shared RAM and pin constraints; no
    budget is divided or apportioned to ranks before the solve.
    """
    rank_count = len(inputs.rank_inputs)
    if rank_count == 0:
        raise ValueError("node placement requires at least one rank")
    if len(inputs.rank_topology) != rank_count or len(capabilities) != rank_count:
        raise ValueError("rank inputs, topology and capabilities must have equal length")
    if len(inputs.routed_tp_layout.widths) != rank_count:
        raise ValueError("routed TP layout rank count does not match placement inputs")
    if tuple(topology.rank for topology in inputs.rank_topology) != tuple(range(rank_count)):
        raise ValueError("rank topology must be ordered and cover ranks 0..tp_size-1")
    if inputs.host_expert_budget_bytes < 0 or inputs.pin_budget_bytes < 0:
        raise ValueError("aggregate host and pin budgets must be non-negative")
    if policy not in {"balanced", "gpu-first", "cpu-first"}:
        raise ValueError(f"unknown MoE placement policy {policy!r}")

    equal_widths = len(set(inputs.routed_tp_layout.widths)) == 1
    if not equal_widths and inputs.checkpoint_layout not in {
        "dsv4_ds_fp4_safetensors", "synthetic"
    }:
        raise MoePlacementInfeasible(
            "weighted routed TP requires original DSV4 DS-FP4 safetensors; "
            f"checkpoint layout {inputs.checkpoint_layout!r} is unsupported"
        )

    kv_modes = {rank.kv_mode for rank in inputs.rank_inputs}
    kv_page_sizes = {rank.kv_page_bytes for rank in inputs.rank_inputs}
    kv_minima = {rank.kv_min_pages for rank in inputs.rank_inputs}
    kv_targets = {rank.kv_target_pages for rank in inputs.rank_inputs}
    if len(kv_modes) != 1 or len(kv_page_sizes) != 1:
        raise MoePlacementInfeasible(
            "all TP ranks must use one KV mode and the same KV page byte geometry"
        )
    kv_mode = next(iter(kv_modes))
    kv_page_bytes = next(iter(kv_page_sizes))
    if kv_mode == "fixed":
        if len(kv_targets) != 1:
            raise MoePlacementInfeasible(
                "all TP ranks must use the same fixed KV target"
            )
        kv_min_pages = 0
        kv_target_pages = next(iter(kv_targets))
    else:
        if len(kv_minima) != 1:
            raise MoePlacementInfeasible(
                "all TP ranks must use the same auto KV minimum"
            )
        kv_min_pages = next(iter(kv_minima))
        kv_target_pages = None

    full_host_per_rank = tuple(sum(rank.host_bytes_per_layer) for rank in inputs.rank_inputs)
    full_host = sum(full_host_per_rank)
    # When the aggregate pin budget retains the complete host pool there is no
    # reason to enumerate HOST_LOCKED variants. Explicit CPU layers remain exact.
    enumerate_cpu_options = (
        allow_additional_cpu_layers and inputs.pin_budget_bytes < full_host
    )
    frontier_values: list[tuple[MoePlacementPlan, ...]] = []
    frontier_failures: list[str] = []
    for rank, rank_input in enumerate(inputs.rank_inputs):
        try:
            frontier_values.append(
                _rank_candidate_frontier(
                    rank_input,
                    capabilities[rank],
                    policy=policy,
                    allow_gpu_permanent=allow_gpu_permanent,
                    explicit_permanent_layer_ids=explicit_permanent_layer_ids,
                    explicit_permanent_layer_count=explicit_permanent_layer_count,
                    explicit_cpu_layer_ids=explicit_cpu_layer_ids,
                    allow_additional_cpu_layers=enumerate_cpu_options,
                )
            )
        except MoePlacementInfeasible as error:
            frontier_values.append(())
            dynamic = (
                "auto"
                if rank_input.dynamic_cache_slots is None
                else str(rank_input.dynamic_cache_slots)
            )
            frontier_failures.append(
                f"  rank {rank}: {error}; available={_gib(rank_input.available_vram_bytes)}, "
                f"fixed={_gib(rank_input.fixed_gpu_bytes)}, "
                f"activation/graph={_gib(rank_input.activation_graph_reserve_bytes)}, "
                f"dynamic slots={dynamic}, KV {rank_input.kv_mode} "
                f"{rank_input.kv_required_pages} pages"
            )
    if frontier_failures:
        raise MoePlacementInfeasible(
            "node placement infeasible before aggregate combination:\n"
            + "\n".join(frontier_failures)
        )
    frontiers = tuple(frontier_values)

    host_deficit = max(0, full_host - inputs.host_expert_budget_bytes)

    max_link = max(
        topology.pcie_generation * topology.pcie_width
        for topology in inputs.rank_topology
    )
    best_score: tuple | None = None
    best_plans: tuple[MoePlacementPlan, ...] | None = None

    def consider(plans: tuple[MoePlacementPlan, ...]) -> None:
        nonlocal best_score, best_plans
        retained = sum(plan.retained_host_bytes for plan in plans)
        pinned = sum(plan.pinned_host_bytes for plan in plans)
        if retained > inputs.host_expert_budget_bytes or pinned > inputs.pin_budget_bytes:
            return
        permanent = sum(plan.permanent_gpu_bytes for plan in plans)
        locked = sum(plan.locked_host_bytes for plan in plans)
        selection_headroom = min(
            plan.elastic_vram_bytes
            - plan.dynamic_cache_bytes
            - inputs.rank_inputs[rank].kv_required_bytes
            for rank, plan in enumerate(plans)
        )
        common_kv_pages = (
            kv_target_pages
            if kv_target_pages is not None
            else min(plan.kv_pages for plan in plans)
        )
        pcie_benefit = sum(
            (max_link - topology.pcie_generation * topology.pcie_width)
            * plan.permanent_gpu_bytes
            for topology, plan in zip(inputs.rank_topology, plans)
        )
        stable = tuple(
            (
                tuple(-layer for layer in reversed(plan.permanent_layer_ids)),
                tuple(sorted(plan.cpu_layer_ids)),
            )
            for plan in plans
        )
        capacity_score = (
            (locked, max(0, permanent - host_deficit), permanent)
            if policy == "gpu-first"
            else (max(0, permanent - host_deficit), permanent, locked)
        )
        score = (
            *capacity_score,
            -selection_headroom,
            -pcie_benefit,
            -common_kv_pages,
            stable,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_plans = plans

    # Multi-choice DP collapses combinations with identical aggregate resource
    # coordinates. This keeps pin-limited CPU frontiers tractable without losing
    # any state that can affect feasibility or the documented final objective.
    states: dict[
        tuple[int, int, int],
        tuple[tuple[MoePlacementPlan, ...], tuple],
    ] = {(0, 0, 0): ((), ())}
    for rank, frontier in enumerate(frontiers):
        merged: dict[
            tuple[int, int, int],
            tuple[tuple[MoePlacementPlan, ...], tuple],
        ] = {}
        for (retained, pinned, permanent), (chosen, _quality) in states.items():
            for plan in frontier:
                key = (
                    retained + plan.retained_host_bytes,
                    pinned + plan.pinned_host_bytes,
                    permanent + plan.permanent_gpu_bytes,
                )
                if key[0] > inputs.host_expert_budget_bytes:
                    continue
                if key[1] > inputs.pin_budget_bytes:
                    continue
                plans = (*chosen, plan)
                selection_headroom = min(
                    candidate.elastic_vram_bytes
                    - candidate.dynamic_cache_bytes
                    - inputs.rank_inputs[index].kv_required_bytes
                    for index, candidate in enumerate(plans)
                )
                pcie_benefit = sum(
                    (max_link - inputs.rank_topology[index].pcie_generation
                     * inputs.rank_topology[index].pcie_width)
                    * candidate.permanent_gpu_bytes
                    for index, candidate in enumerate(plans)
                )
                common_kv = (
                    kv_target_pages
                    if kv_target_pages is not None
                    else min(candidate.kv_pages for candidate in plans)
                )
                stable = tuple(
                    (
                        tuple(-layer for layer in reversed(candidate.permanent_layer_ids)),
                        tuple(sorted(candidate.cpu_layer_ids)),
                    )
                    for candidate in plans
                )
                quality = (-selection_headroom, -pcie_benefit, -common_kv, stable)
                previous = merged.get(key)
                if previous is None or quality < previous[1]:
                    merged[key] = (plans, quality)
        states = merged
        if not states:
            break
    for plans, _quality in states.values():
        if len(plans) == rank_count:
            consider(plans)
    if best_plans is None:
        max_removed = tuple(
            full_host_per_rank[rank]
            - min(plan.retained_host_bytes for plan in frontier)
            for rank, frontier in enumerate(frontiers)
        )
        aggregate_removable = sum(max_removed)
        shortfall = max(0, host_deficit - aggregate_removable)
        rank_lines = "\n".join(
            f"  rank {rank} frontier max removable: {_gib(removable)} "
            f"(full host {_gib(full_host_per_rank[rank])})"
            for rank, removable in enumerate(max_removed)
        )
        raise MoePlacementInfeasible(
            "node placement infeasible:\n"
            f"  aggregate host pool:          {_gib(full_host)}\n"
            f"  aggregate host budget:        {_gib(inputs.host_expert_budget_bytes)}\n"
            f"  aggregate host deficit:       {_gib(host_deficit)}\n"
            f"  frontier removable capacity:  {_gib(aggregate_removable)}\n"
            f"  aggregate shortfall:          {_gib(shortfall)}\n"
            f"  aggregate pin budget:         {_gib(inputs.pin_budget_bytes)}\n"
            f"  KV mode:                      {kv_mode} "
            f"({kv_target_pages if kv_target_pages is not None else kv_min_pages} pages)\n"
            f"{rank_lines}\n"
            "  fixed dynamic cache, fixed/minimum KV, and activation/graph reserves "
            "were enforced on every rank frontier"
        )

    kv_pages = (
        kv_target_pages
        if kv_target_pages is not None
        else min(plan.kv_pages for plan in best_plans)
    )
    rank_plans = tuple(
        replace(
            plan,
            kv_pages=kv_pages,
            kv_bytes=kv_pages * kv_page_bytes,
            vram_headroom_bytes=(
                plan.elastic_vram_bytes
                - plan.dynamic_cache_bytes
                - kv_pages * kv_page_bytes
            ),
            decisions=(
                *plan.decisions,
                "selected by node-global aggregate host/pin frontier",
            ),
        )
        for plan in best_plans
    )
    retained = sum(plan.retained_host_bytes for plan in rank_plans)
    pinned = sum(plan.pinned_host_bytes for plan in rank_plans)
    locked = sum(plan.locked_host_bytes for plan in rank_plans)
    permanent = sum(plan.permanent_gpu_bytes for plan in rank_plans)
    fixed = sum(plan.fixed_gpu_bytes for plan in rank_plans)
    graph = sum(plan.activation_graph_reserve_bytes for plan in rank_plans)
    dynamic = sum(plan.dynamic_cache_bytes for plan in rank_plans)
    kv_bytes = sum(plan.kv_bytes for plan in rank_plans)
    headroom = tuple(plan.vram_headroom_bytes for plan in rank_plans)
    if retained > inputs.host_expert_budget_bytes or pinned > inputs.pin_budget_bytes:
        raise AssertionError("joint planner exceeded an aggregate host constraint")
    if any(value < 0 for value in headroom):
        raise AssertionError("joint planner produced negative rank VRAM headroom")
    constraints = (
        f"aggregate host expert bytes {retained} <= {inputs.host_expert_budget_bytes}",
        f"aggregate pinned expert bytes {pinned} <= {inputs.pin_budget_bytes}",
        (
            f"KV fixed at {kv_pages} pages on every rank"
            if kv_mode == "fixed"
            else f"KV auto-selected at all-rank lower bound {kv_pages} pages"
        ),
        "each rank satisfies fixed + permanent + dynamic + KV + graph reserve <= VRAM",
        (
            "runtime expert disk tier explicitly enabled"
            if inputs.runtime_disk_tier
            else "runtime expert disk tier disabled"
        ),
    )
    provisional = NodeMoePlacementPlan(
        routed_tp_layout=inputs.routed_tp_layout,
        rank_plans=rank_plans,
        rank_topology=inputs.rank_topology,
        retained_host_bytes=retained,
        pinned_host_bytes=pinned,
        locked_host_bytes=locked,
        permanent_gpu_bytes=permanent,
        fixed_gpu_bytes=fixed,
        activation_graph_reserve_bytes=graph,
        dynamic_cache_bytes=dynamic,
        kv_bytes=kv_bytes,
        vram_headroom_bytes=headroom,
        kv_mode=kv_mode,
        kv_min_pages=kv_min_pages,
        kv_target_pages=kv_target_pages,
        kv_pages=kv_pages,
        runtime_disk_tier=inputs.runtime_disk_tier,
        constraints=constraints,
        checksum="",
    )
    checksum = _node_plan_checksum(provisional.canonical_dict())
    result = replace(provisional, checksum=checksum)
    result.verify_checksum()
    return result


__all__ = [
    "ExpertPlacementCapabilities",
    "ExpertTier",
    "MoePlacementInfeasible",
    "MoePlacementInputs",
    "MoePlacementPlan",
    "NodeMoePlacementInputs",
    "NodeMoePlacementPlan",
    "RankTopology",
    "plan_moe_placement",
    "plan_node_moe_placement",
]
