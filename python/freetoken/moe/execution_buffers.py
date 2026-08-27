"""Preallocated tensor ABI for a rank-local MoE execution plan."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .execution_plan import ComputeAction, ExpertExecutionPlan, RouteRepresentation


_REPRESENTATION_ID = {
    RouteRepresentation.SHARDED: 0,
    RouteRepresentation.FULL_OWNER: 1,
}
_ACTION_ID = {
    ComputeAction.GPU_SHARD: 0,
    ComputeAction.CPU_SHARD: 1,
    ComputeAction.GPU_FULL_OWNER: 2,
}


@dataclass
class LocalExecutionPlanBuffers:
    """Fixed-shape buffers whose storage survives capture/replay.

    ``load_reference_plan`` is a test/debug adapter.  Production kernels write
    the same masks/worklists/counts device-side from live routing state.
    """

    route_capacity: int
    token_capacity: int
    hidden_size: int
    plan_epoch: torch.Tensor
    valid_routes: torch.Tensor
    route_representation: torch.Tensor
    full_owner_by_route: torch.Tensor
    local_shard_target: torch.Tensor
    route_masks: torch.Tensor
    worklists: torch.Tensor
    valid_counts: torch.Tensor
    output_accumulation: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        route_capacity: int,
        token_capacity: int,
        hidden_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "LocalExecutionPlanBuffers":
        if route_capacity <= 0 or token_capacity <= 0 or hidden_size <= 0:
            raise ValueError("execution buffer capacities and hidden size must be positive")
        actions = len(ComputeAction)
        return cls(
            route_capacity=route_capacity,
            token_capacity=token_capacity,
            hidden_size=hidden_size,
            plan_epoch=torch.zeros((), dtype=torch.int64, device=device),
            valid_routes=torch.zeros((), dtype=torch.int32, device=device),
            route_representation=torch.empty(route_capacity, dtype=torch.int8, device=device),
            full_owner_by_route=torch.empty(route_capacity, dtype=torch.int16, device=device),
            local_shard_target=torch.empty(route_capacity, dtype=torch.int8, device=device),
            route_masks=torch.zeros(actions, route_capacity, dtype=torch.bool, device=device),
            worklists=torch.empty(actions, route_capacity, dtype=torch.int32, device=device),
            valid_counts=torch.zeros(actions, dtype=torch.int32, device=device),
            output_accumulation=torch.zeros(
                token_capacity, hidden_size, dtype=dtype, device=device
            ),
        )

    def load_reference_plan(self, plan: ExpertExecutionPlan, *, rank: int) -> None:
        """Populate in place without replacing any captured storage."""
        plan.validate()
        if plan.capacity > self.route_capacity:
            raise ValueError(
                f"plan capacity {plan.capacity} exceeds buffer capacity {self.route_capacity}"
            )
        worklists = plan.local_worklists(rank)
        self.plan_epoch.fill_(plan.plan_epoch)
        self.valid_routes.fill_(plan.valid_routes)
        self.route_representation.fill_(-1)
        self.full_owner_by_route.fill_(-1)
        self.local_shard_target.fill_(-1)
        self.route_masks.zero_()
        self.worklists.fill_(-1)
        self.valid_counts.zero_()
        self.output_accumulation.zero_()

        representations = [
            _REPRESENTATION_ID[value]
            for value in plan.route_representation[: plan.valid_routes]
        ]
        if representations:
            self.route_representation[: plan.valid_routes].copy_(
                torch.tensor(representations, dtype=torch.int8, device=self.plan_epoch.device)
            )
            self.full_owner_by_route[: plan.valid_routes].copy_(
                torch.tensor(
                    plan.full_owner_by_route[: plan.valid_routes],
                    dtype=torch.int16,
                    device=self.plan_epoch.device,
                )
            )
        local_targets = plan.shard_target_by_rank[rank][: plan.valid_routes]
        encoded_targets = [
            _ACTION_ID[target] if target is not None else -1 for target in local_targets
        ]
        if encoded_targets:
            self.local_shard_target[: plan.valid_routes].copy_(
                torch.tensor(encoded_targets, dtype=torch.int8, device=self.plan_epoch.device)
            )
        for action, routes in worklists.items():
            action_id = _ACTION_ID[action]
            self.valid_counts[action_id] = len(routes)
            if routes:
                indices = torch.tensor(routes, dtype=torch.int32, device=self.plan_epoch.device)
                self.worklists[action_id, : len(routes)].copy_(indices)
                self.route_masks[action_id, indices.to(torch.int64)] = True


__all__ = ["LocalExecutionPlanBuffers"]
