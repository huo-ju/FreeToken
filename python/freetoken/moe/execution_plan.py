"""Topology-neutral contracts for heterogeneous MoE execution.

The objects in this module are deliberately torch-free.  Python plans are used
for startup validation, debug dumps and tests; the decode hot path consumes
preallocated tensors with the same fields rather than constructing these
objects for every token.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class AuthoritativeSource(str, Enum):
    GPU_PERMANENT = "gpu_permanent"
    HOST_PINNED = "host_pinned"
    HOST_LOCKED = "host_locked"


class RouteRepresentation(str, Enum):
    SHARDED = "sharded"
    FULL_OWNER = "full_owner"


class WeightRepresentation(str, Enum):
    TP_SHARD = "tp_shard"
    FULL_EXPERT = "full_expert"


class ComputeAction(str, Enum):
    GPU_SHARD = "gpu_shard"
    CPU_SHARD = "cpu_shard"
    GPU_FULL_OWNER = "gpu_full_owner"


class WeightTransport(str, Enum):
    NONE = "none"
    LOCAL_H2D = "local_h2d"


class Aggregation(str, Enum):
    LOCAL_SUM_THEN_TP_ALL_REDUCE = "local_sum_then_tp_all_reduce"


class ExecutionPolicy(str, Enum):
    COMPATIBILITY = "compatibility"
    FORCE_EQUAL_TP = "force_equal_tp"
    FORCE_WEIGHTED_TP = "force_weighted_tp"
    FORCE_CPU = "force_cpu"
    FORCE_RESIDENT_EP = "force_resident_ep"
    FORCE_STATIC_MIXED = "force_static_mixed"
    AUTO = "auto"


@dataclass(frozen=True)
class ExecutionPolicyAdapter:
    """Startup-resolved bridge from forced policies to legacy executors.

    The policy is immutable for an engine generation. Compatibility delegates
    to the existing backend decision tree; forced policies select one sharded
    action for every routed work item and reject contradictory residency before
    decode or CUDA-graph capture begins.
    """

    policy: ExecutionPolicy
    plan_epoch: int
    tp_size: int

    def __post_init__(self) -> None:
        if self.plan_epoch < 0:
            raise ValueError("plan epoch must be non-negative")
        if self.tp_size <= 0:
            raise ValueError("execution policy adapter requires a positive TP size")

    @property
    def forced_action(self) -> ComputeAction | None:
        if self.policy in {
            ExecutionPolicy.FORCE_EQUAL_TP,
            ExecutionPolicy.FORCE_WEIGHTED_TP,
        }:
            return ComputeAction.GPU_SHARD
        if self.policy is ExecutionPolicy.FORCE_CPU:
            return ComputeAction.CPU_SHARD
        return None

    def validate_backend(
        self,
        *,
        decode_target: str,
        cpu_layer_ids: Sequence[int] = (),
        gpu_resident_layer_ids: Sequence[int] = (),
    ) -> None:
        if self.policy in {
            ExecutionPolicy.FORCE_EQUAL_TP,
            ExecutionPolicy.FORCE_WEIGHTED_TP,
        }:
            if decode_target != "gpu" or cpu_layer_ids:
                raise ValueError(
                    f"{self.policy.value} requires the GPU-shard offload path for every layer"
                )
        elif self.policy is ExecutionPolicy.FORCE_CPU:
            if decode_target != "cpu" or gpu_resident_layer_ids:
                raise ValueError(
                    "force_cpu requires the CPU-shard path and forbids GPU permanent layers"
                )
        elif self.policy is not ExecutionPolicy.COMPATIBILITY:
            raise ValueError(f"execution policy {self.policy.value!r} is not implemented")

    def reference_plan(
        self, *, layer_id: int, capacity: int, valid_routes: int
    ) -> "ExpertExecutionPlan":
        action = self.forced_action
        if action is None:
            raise ValueError("compatibility policy has no single forced reference plan")
        return forced_sharded_plan(
            plan_epoch=self.plan_epoch,
            layer_id=layer_id,
            tp_size=self.tp_size,
            capacity=capacity,
            valid_routes=valid_routes,
            action=action,
        )


def compatibility_policy(moe_backend: str) -> ExecutionPolicy:
    """Map legacy CLI values without changing their externally visible semantics."""
    if moe_backend in {"offload", "cpu", "hybrid"}:
        return ExecutionPolicy.COMPATIBILITY
    if moe_backend == "fused":
        return ExecutionPolicy.FORCE_EQUAL_TP
    raise ValueError(f"unknown MoE backend {moe_backend!r}")


@dataclass(frozen=True)
class RoutedTpLayout:
    """Startup-fixed routed-expert partition across TP ranks."""

    widths: tuple[int, ...]
    offsets: tuple[int, ...]
    alignment: int
    total_intermediate_size: int

    def __post_init__(self) -> None:
        if not self.widths:
            raise ValueError("routed TP layout requires at least one rank")
        if len(self.widths) != len(self.offsets):
            raise ValueError("routed TP widths and offsets must have equal length")
        if self.alignment <= 0:
            raise ValueError("routed TP alignment must be positive")
        if self.total_intermediate_size <= 0:
            raise ValueError("routed intermediate size must be positive")
        cursor = 0
        for rank, (width, offset) in enumerate(zip(self.widths, self.offsets)):
            if width <= 0:
                raise ValueError(f"routed TP rank {rank} has non-positive width {width}")
            if width % self.alignment:
                raise ValueError(
                    f"routed TP rank {rank} width {width} is not aligned to {self.alignment}"
                )
            if offset != cursor:
                raise ValueError(
                    f"routed TP rank {rank} starts at {offset}, expected {cursor}; "
                    "layout has a gap or overlap"
                )
            cursor += width
        if cursor != self.total_intermediate_size:
            raise ValueError(
                f"routed TP widths cover {cursor}, expected {self.total_intermediate_size}"
            )

    @classmethod
    def from_widths(cls, widths: Sequence[int], *, alignment: int) -> "RoutedTpLayout":
        offsets: list[int] = []
        cursor = 0
        for width in widths:
            offsets.append(cursor)
            cursor += width
        return cls(tuple(widths), tuple(offsets), alignment, cursor)

    @classmethod
    def equal(
        cls, intermediate_size: int, tp_size: int, *, alignment: int = 1
    ) -> "RoutedTpLayout":
        if tp_size <= 0 or intermediate_size % tp_size:
            raise ValueError(
                f"cannot evenly split intermediate size {intermediate_size} across TP{tp_size}"
            )
        return cls.from_widths((intermediate_size // tp_size,) * tp_size, alignment=alignment)

    def local_slice(self, rank: int) -> slice:
        return slice(self.offsets[rank], self.offsets[rank] + self.widths[rank])

    def canonical_dict(self) -> dict:
        return {
            "widths": list(self.widths),
            "offsets": list(self.offsets),
            "alignment": self.alignment,
            "total_intermediate_size": self.total_intermediate_size,
        }

    def checksum(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def resolve_routed_tp_layout(
    setting: str,
    *,
    total_intermediate_size: int,
    tp_size: int,
    alignment: int = 1,
) -> RoutedTpLayout:
    """Resolve the startup CLI value into one validated routed-expert layout."""
    if setting == "equal":
        return RoutedTpLayout.equal(
            total_intermediate_size, tp_size, alignment=alignment
        )
    try:
        widths = tuple(int(value.strip()) for value in setting.split(","))
    except ValueError as error:
        raise ValueError(
            f"invalid routed TP layout {setting!r}; expected comma-separated integers"
        ) from error
    if len(widths) != tp_size:
        raise ValueError(
            f"routed TP layout has {len(widths)} widths, expected TP size {tp_size}"
        )
    layout = RoutedTpLayout.from_widths(widths, alignment=alignment)
    if layout.total_intermediate_size != total_intermediate_size:
        raise ValueError(
            f"routed TP layout covers {layout.total_intermediate_size}, "
            f"expected routed intermediate size {total_intermediate_size}"
        )
    return layout


@dataclass(frozen=True)
class ExpertExecutionPlan:
    """Debug/reference form of one layer's fixed-capacity execution plan.

    ``valid_routes`` is the live prefix of every route-oriented tuple.  Padding
    beyond it is ignored, mirroring the device valid-count ABI used by graph
    replay.  A FULL_OWNER route uses no shard target; a SHARDED route has exactly
    one GPU/CPU action for every rank.
    """

    plan_epoch: int
    layer_id: int
    route_representation: tuple[RouteRepresentation, ...]
    full_owner_by_route: tuple[int, ...]
    shard_target_by_rank: tuple[tuple[ComputeAction | None, ...], ...]
    valid_routes: int
    aggregation: Aggregation = Aggregation.LOCAL_SUM_THEN_TP_ALL_REDUCE

    @property
    def capacity(self) -> int:
        return len(self.route_representation)

    @property
    def tp_size(self) -> int:
        return len(self.shard_target_by_rank)

    def validate(self, *, expected_epoch: int | None = None) -> None:
        if self.plan_epoch < 0:
            raise ValueError("plan epoch must be non-negative")
        if expected_epoch is not None and self.plan_epoch != expected_epoch:
            raise ValueError(
                f"plan epoch mismatch: got {self.plan_epoch}, expected {expected_epoch}"
            )
        if self.layer_id < 0:
            raise ValueError("layer id must be non-negative")
        if not 0 <= self.valid_routes <= self.capacity:
            raise ValueError("valid route count is outside fixed plan capacity")
        if len(self.full_owner_by_route) != self.capacity:
            raise ValueError("full-owner vector does not match route capacity")
        if not self.shard_target_by_rank:
            raise ValueError("execution plan requires at least one TP rank")
        if any(len(row) != self.capacity for row in self.shard_target_by_rank):
            raise ValueError("shard-target matrix does not match route capacity")

        for route in range(self.valid_routes):
            representation = self.route_representation[route]
            owner = self.full_owner_by_route[route]
            targets = tuple(row[route] for row in self.shard_target_by_rank)
            if representation is RouteRepresentation.SHARDED:
                if owner != -1:
                    raise ValueError(f"sharded route {route} unexpectedly has owner {owner}")
                invalid = [target for target in targets if target not in {
                    ComputeAction.GPU_SHARD, ComputeAction.CPU_SHARD
                }]
                if invalid:
                    raise ValueError(
                        f"sharded route {route} must have exactly one GPU/CPU shard action per rank"
                    )
            elif representation is RouteRepresentation.FULL_OWNER:
                if not 0 <= owner < self.tp_size:
                    raise ValueError(f"full-owner route {route} has invalid owner {owner}")
                if any(target is not None for target in targets):
                    raise ValueError(
                        f"full-owner route {route} also contains a TP shard action"
                    )
            else:
                raise ValueError(f"route {route} has unknown representation {representation!r}")

    def local_worklists(self, rank: int) -> dict[ComputeAction, tuple[int, ...]]:
        self.validate()
        if not 0 <= rank < self.tp_size:
            raise ValueError(f"rank {rank} outside TP{self.tp_size}")
        result = {action: [] for action in ComputeAction}
        for route in range(self.valid_routes):
            if self.route_representation[route] is RouteRepresentation.FULL_OWNER:
                if self.full_owner_by_route[route] == rank:
                    result[ComputeAction.GPU_FULL_OWNER].append(route)
            else:
                target = self.shard_target_by_rank[rank][route]
                assert target is not None
                result[target].append(route)
        return {action: tuple(routes) for action, routes in result.items()}

    def canonical_dict(self) -> dict:
        return {
            "plan_epoch": self.plan_epoch,
            "layer_id": self.layer_id,
            "valid_routes": self.valid_routes,
            "route_representation": [value.value for value in self.route_representation],
            "full_owner_by_route": list(self.full_owner_by_route),
            "shard_target_by_rank": [
                [value.value if value is not None else None for value in row]
                for row in self.shard_target_by_rank
            ],
            "aggregation": self.aggregation.value,
        }

    def checksum(self) -> str:
        self.validate()
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def forced_sharded_plan(
    *,
    plan_epoch: int,
    layer_id: int,
    tp_size: int,
    capacity: int,
    valid_routes: int,
    action: ComputeAction,
) -> ExpertExecutionPlan:
    if action not in {ComputeAction.GPU_SHARD, ComputeAction.CPU_SHARD}:
        raise ValueError("forced sharded plan requires GPU_SHARD or CPU_SHARD")
    plan = ExpertExecutionPlan(
        plan_epoch=plan_epoch,
        layer_id=layer_id,
        route_representation=(RouteRepresentation.SHARDED,) * capacity,
        full_owner_by_route=(-1,) * capacity,
        shard_target_by_rank=tuple((action,) * capacity for _ in range(tp_size)),
        valid_routes=valid_routes,
    )
    plan.validate()
    return plan


__all__ = [
    "Aggregation",
    "AuthoritativeSource",
    "ComputeAction",
    "ExecutionPolicy",
    "ExecutionPolicyAdapter",
    "ExpertExecutionPlan",
    "RouteRepresentation",
    "RoutedTpLayout",
    "WeightRepresentation",
    "WeightTransport",
    "compatibility_policy",
    "forced_sharded_plan",
    "resolve_routed_tp_layout",
]
