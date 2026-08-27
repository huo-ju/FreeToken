from __future__ import annotations

import torch

from freetoken.moe.execution_buffers import LocalExecutionPlanBuffers
from freetoken.moe.execution_plan import ComputeAction, forced_sharded_plan


def test_reference_replay_changes_valid_counts_without_reallocating_storage():
    buffers = LocalExecutionPlanBuffers.allocate(
        route_capacity=8, token_capacity=2, hidden_size=16,
        device="cpu", dtype=torch.float32,
    )
    pointers = {
        name: getattr(buffers, name).data_ptr()
        for name in (
            "route_representation", "route_masks", "worklists",
            "valid_counts", "output_accumulation",
        )
    }
    gpu = forced_sharded_plan(
        plan_epoch=1, layer_id=0, tp_size=2, capacity=8, valid_routes=7,
        action=ComputeAction.GPU_SHARD,
    )
    cpu = forced_sharded_plan(
        plan_epoch=2, layer_id=0, tp_size=2, capacity=8, valid_routes=3,
        action=ComputeAction.CPU_SHARD,
    )
    buffers.load_reference_plan(gpu, rank=1)
    assert buffers.valid_counts.tolist() == [7, 0, 0]
    buffers.load_reference_plan(cpu, rank=1)
    assert buffers.valid_counts.tolist() == [0, 3, 0]
    assert int(buffers.plan_epoch) == 2
    assert all(getattr(buffers, name).data_ptr() == pointer for name, pointer in pointers.items())
