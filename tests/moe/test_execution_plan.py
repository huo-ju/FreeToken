from __future__ import annotations

import pytest

from freetoken.moe.execution_plan import (
    ComputeAction,
    ExecutionPolicy,
    ExecutionPolicyAdapter,
    ExpertExecutionPlan,
    RouteRepresentation,
    RoutedTpLayout,
    compatibility_policy,
    forced_sharded_plan,
    resolve_routed_tp_layout,
)


def test_equal_layout_matches_legacy_partition():
    layout = RoutedTpLayout.equal(2048, 4, alignment=256)
    assert layout.widths == (512, 512, 512, 512)
    assert layout.offsets == (0, 512, 1024, 1536)
    assert layout.local_slice(3) == slice(1536, 2048)


def test_weighted_layout_covers_intermediate_exactly():
    layout = RoutedTpLayout.from_widths((512, 512, 768, 256), alignment=256)
    assert layout.offsets == (0, 512, 1024, 1792)
    assert layout.total_intermediate_size == 2048


def test_resolve_routed_tp_layout_accepts_equal_and_explicit_forms():
    equal = resolve_routed_tp_layout(
        "equal", total_intermediate_size=2048, tp_size=4, alignment=256
    )
    weighted = resolve_routed_tp_layout(
        "512, 512, 768, 256",
        total_intermediate_size=2048,
        tp_size=4,
        alignment=256,
    )
    assert equal.widths == (512,) * 4
    assert weighted.widths == (512, 512, 768, 256)


@pytest.mark.parametrize(
    "setting, match",
    [
        ("512,nope,768,256", "comma-separated integers"),
        ("1024,1024", "expected TP size 4"),
        ("512,512,512,256", "covers 1792"),
        ("512,512,640,384", "not aligned"),
    ],
)
def test_resolve_routed_tp_layout_rejects_invalid_startup_values(setting, match):
    with pytest.raises(ValueError, match=match):
        resolve_routed_tp_layout(
            setting, total_intermediate_size=2048, tp_size=4, alignment=256
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"widths": (512, 0, 1536), "alignment": 256}, "non-positive"),
        ({"widths": (512, 513, 1023), "alignment": 256}, "not aligned"),
    ],
)
def test_layout_rejects_zero_width_and_bad_tiles(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RoutedTpLayout.from_widths(**kwargs)


def test_forced_gpu_and_cpu_plans_cover_every_rank_local_shard():
    for action in (ComputeAction.GPU_SHARD, ComputeAction.CPU_SHARD):
        plan = forced_sharded_plan(
            plan_epoch=7,
            layer_id=3,
            tp_size=4,
            capacity=8,
            valid_routes=5,
            action=action,
        )
        for rank in range(4):
            assert plan.local_worklists(rank)[action] == (0, 1, 2, 3, 4)


def test_static_three_way_plan_has_mutually_exclusive_route_representation():
    plan = ExpertExecutionPlan(
        plan_epoch=4,
        layer_id=2,
        route_representation=(
            RouteRepresentation.FULL_OWNER,
            RouteRepresentation.SHARDED,
            RouteRepresentation.SHARDED,
            RouteRepresentation.SHARDED,  # fixed-capacity padding
        ),
        full_owner_by_route=(1, -1, -1, -1),
        shard_target_by_rank=(
            (None, ComputeAction.GPU_SHARD, ComputeAction.CPU_SHARD, None),
            (None, ComputeAction.CPU_SHARD, ComputeAction.CPU_SHARD, None),
        ),
        valid_routes=3,
    )
    plan.validate(expected_epoch=4)
    assert plan.local_worklists(0) == {
        ComputeAction.GPU_SHARD: (1,),
        ComputeAction.CPU_SHARD: (2,),
        ComputeAction.GPU_FULL_OWNER: (),
    }
    assert plan.local_worklists(1)[ComputeAction.GPU_FULL_OWNER] == (0,)


def test_full_owner_route_cannot_also_execute_shards():
    plan = ExpertExecutionPlan(
        plan_epoch=0,
        layer_id=0,
        route_representation=(RouteRepresentation.FULL_OWNER,),
        full_owner_by_route=(0,),
        shard_target_by_rank=((ComputeAction.GPU_SHARD,), (None,)),
        valid_routes=1,
    )
    with pytest.raises(ValueError, match="also contains"):
        plan.validate()


def test_plan_epoch_mismatch_fails_fast_and_checksum_is_deterministic():
    plan = forced_sharded_plan(
        plan_epoch=9,
        layer_id=1,
        tp_size=2,
        capacity=4,
        valid_routes=0,
        action=ComputeAction.GPU_SHARD,
    )
    assert plan.checksum() == plan.checksum()
    with pytest.raises(ValueError, match="epoch mismatch"):
        plan.validate(expected_epoch=8)


def test_legacy_cli_mapping_does_not_redefine_hybrid():
    for backend in ("offload", "cpu", "hybrid"):
        assert compatibility_policy(backend) is ExecutionPolicy.COMPATIBILITY


def test_forced_policy_adapter_validates_actual_executor_and_residency():
    gpu = ExecutionPolicyAdapter(ExecutionPolicy.FORCE_EQUAL_TP, plan_epoch=3, tp_size=4)
    assert gpu.forced_action is ComputeAction.GPU_SHARD
    gpu.validate_backend(decode_target="gpu")
    with pytest.raises(ValueError, match="GPU-shard"):
        gpu.validate_backend(decode_target="gpu", cpu_layer_ids=(2,))

    cpu = ExecutionPolicyAdapter(ExecutionPolicy.FORCE_CPU, plan_epoch=3, tp_size=4)
    assert cpu.forced_action is ComputeAction.CPU_SHARD
    cpu.validate_backend(decode_target="cpu")
    with pytest.raises(ValueError, match="forbids GPU permanent"):
        cpu.validate_backend(decode_target="cpu", gpu_resident_layer_ids=(41,))


def test_forced_policy_adapter_builds_the_same_fixed_plan_contract():
    adapter = ExecutionPolicyAdapter(
        ExecutionPolicy.FORCE_CPU, plan_epoch=5, tp_size=2
    )
    plan = adapter.reference_plan(layer_id=7, capacity=8, valid_routes=3)
    assert plan.plan_epoch == 5
    assert plan.local_worklists(0)[ComputeAction.CPU_SHARD] == (0, 1, 2)


def test_weighted_tp_policy_uses_gpu_shard_action():
    adapter = ExecutionPolicyAdapter(
        ExecutionPolicy.FORCE_WEIGHTED_TP, plan_epoch=1, tp_size=4
    )
    assert adapter.forced_action is ComputeAction.GPU_SHARD
    adapter.validate_backend(decode_target="gpu")
