from __future__ import annotations

import random

import pytest

from freetoken.engine.moe_placement import (
    ExpertPlacementCapabilities,
    ExpertTier,
    MoePlacementInfeasible,
    MoePlacementInputs,
    plan_moe_placement,
)


CAPS = ExpertPlacementCapabilities(
    allocation_specs=True,
    gpu_permanent_streaming=True,
    per_layer_host_residency=True,
    cpu_executor_layout=True,
    mixed_cpu_gpu_layout=True,
)


def _inputs(**overrides) -> MoePlacementInputs:
    values = dict(
        available_vram_bytes=2_000,
        fixed_gpu_bytes=200,
        activation_graph_reserve_bytes=100,
        host_expert_budget_bytes=400,
        pin_budget_bytes=400,
        kv_reserve_bytes=200,
        expert_bytes_per_slot=10,
        host_bytes_per_layer=(100, 100, 100, 100),
        gpu_bytes_per_layer=(80, 80, 80, 80),
        num_experts=8,
        prefill_overlap=True,
        kv_page_bytes=20,
    )
    values.update(overrides)
    return MoePlacementInputs(**values)


def test_sufficient_host_and_pin_budget_preserves_all_host_authority():
    plan = plan_moe_placement(_inputs(), CAPS)

    assert plan.permanent_layer_ids == ()
    assert plan.cpu_layer_ids == frozenset()
    assert plan.layer_tiers == (ExpertTier.HOST_PINNED,) * 4
    assert plan.dynamic_floor_slots == 16
    assert plan.dynamic_cache_slots == 32
    assert plan.kv_bytes >= 200


def test_host_deficit_uses_minimum_deep_layers():
    plan = plan_moe_placement(_inputs(host_expert_budget_bytes=250), CAPS)

    assert plan.permanent_layer_ids == (2, 3)
    assert plan.retained_host_bytes == 200
    assert plan.permanent_gpu_bytes == 160
    assert plan.dynamic_floor_slots == 16


def test_non_layer_aligned_host_deficit_rounds_up_to_one_layer():
    plan = plan_moe_placement(_inputs(host_expert_budget_bytes=399), CAPS)
    assert plan.permanent_layer_ids == (3,)


def test_balanced_pin_deficit_uses_head_tail_locked_layers():
    plan = plan_moe_placement(_inputs(pin_budget_bytes=150), CAPS, policy="balanced")

    assert plan.permanent_layer_ids == ()
    assert plan.cpu_layer_ids == frozenset({0, 1, 3})
    assert plan.pinned_host_bytes == 100
    assert plan.locked_host_bytes == 300
    assert not plan.prefill_overlap
    assert plan.dynamic_floor_slots == 8


def test_gpu_first_pin_deficit_uses_spare_vram():
    plan = plan_moe_placement(_inputs(pin_budget_bytes=150), CAPS, policy="gpu-first")

    assert plan.permanent_layer_ids == (1, 2, 3)
    assert plan.cpu_layer_ids == frozenset()
    assert plan.pinned_host_bytes == 100


def test_balanced_uses_gpu_when_cpu_executor_is_unavailable():
    caps = ExpertPlacementCapabilities(
        allocation_specs=True,
        gpu_permanent_streaming=True,
        per_layer_host_residency=True,
        cpu_executor_layout=False,
        mixed_cpu_gpu_layout=False,
    )
    plan = plan_moe_placement(_inputs(pin_budget_bytes=150), caps, policy="balanced")

    assert plan.permanent_layer_ids == (1, 2, 3)
    assert plan.cpu_layer_ids == frozenset()


def test_all_permanent_has_no_dynamic_floor():
    plan = plan_moe_placement(
        _inputs(host_expert_budget_bytes=0, pin_budget_bytes=0),
        CAPS,
        explicit_permanent_layer_count=4,
    )

    assert plan.permanent_layer_ids == (0, 1, 2, 3)
    assert plan.dynamic_floor_slots == 0
    assert plan.dynamic_cache_slots == 0
    assert plan.retained_host_bytes == 0


def test_vram_one_byte_below_floor_has_reproducible_report():
    inputs = _inputs(
        available_vram_bytes=739,
        host_expert_budget_bytes=300,
    )
    with pytest.raises(MoePlacementInfeasible) as error:
        plan_moe_placement(inputs, CAPS)

    message = str(error.value)
    assert "required dynamic + KV floor" in message
    assert "shortfall" in message


def test_explicit_unsupported_tier_fails_instead_of_degrading():
    with pytest.raises(MoePlacementInfeasible, match="unsupported"):
        plan_moe_placement(
            _inputs(),
            ExpertPlacementCapabilities(),
            explicit_permanent_layer_count=1,
        )


def test_explicit_permanent_count_is_exact_under_gpu_first():
    plan = plan_moe_placement(
        _inputs(pin_budget_bytes=150),
        CAPS,
        policy="gpu-first",
        explicit_permanent_layer_count=1,
    )

    assert plan.permanent_layer_ids == (3,)
    assert plan.cpu_layer_ids == frozenset({0, 2})


def test_explicit_cpu_set_is_not_silently_extended():
    with pytest.raises(MoePlacementInfeasible, match="pin budget"):
        plan_moe_placement(
            _inputs(pin_budget_bytes=150),
            CAPS,
            explicit_cpu_layer_ids=(0,),
            allow_gpu_permanent=False,
            allow_additional_cpu_layers=False,
        )


def test_explicit_count_cannot_overlap_explicit_cpu_layers():
    with pytest.raises(MoePlacementInfeasible, match="both GPU permanent and CPU"):
        plan_moe_placement(
            _inputs(),
            CAPS,
            explicit_permanent_layer_count=1,
            explicit_cpu_layer_ids=(3,),
        )


def test_host_deficit_cannot_be_solved_by_locked_layers():
    with pytest.raises(MoePlacementInfeasible, match="host expert"):
        plan_moe_placement(
            _inputs(host_expert_budget_bytes=300),
            CAPS,
            allow_gpu_permanent=False,
            explicit_cpu_layer_ids=(0,),
        )


def test_random_successes_obey_all_capacity_invariants():
    rng = random.Random(41)
    for _ in range(400):
        layers = rng.randint(1, 12)
        experts = rng.randint(1, 16)
        host_sizes = tuple(rng.randint(20, 200) for _ in range(layers))
        gpu_sizes = tuple(rng.randint(10, 160) for _ in range(layers))
        inputs = MoePlacementInputs(
            available_vram_bytes=rng.randint(500, 10_000),
            fixed_gpu_bytes=rng.randint(0, 300),
            activation_graph_reserve_bytes=rng.randint(0, 300),
            host_expert_budget_bytes=rng.randint(0, sum(host_sizes)),
            pin_budget_bytes=rng.randint(0, sum(host_sizes)),
            kv_reserve_bytes=rng.randint(1, 300),
            expert_bytes_per_slot=rng.randint(1, 20),
            host_bytes_per_layer=host_sizes,
            gpu_bytes_per_layer=gpu_sizes,
            num_experts=experts,
            prefill_overlap=bool(rng.getrandbits(1)),
            kv_page_bytes=rng.randint(1, 20),
        )
        try:
            plan = plan_moe_placement(inputs, CAPS, policy=rng.choice(("balanced", "gpu-first")))
        except MoePlacementInfeasible:
            continue

        assert plan.retained_host_bytes <= inputs.host_expert_budget_bytes
        assert plan.pinned_host_bytes <= inputs.pin_budget_bytes
        assert plan.dynamic_cache_slots >= plan.dynamic_floor_slots
        assert plan.kv_bytes >= inputs.kv_reserve_bytes
        assert (
            plan.permanent_gpu_bytes
            + plan.dynamic_cache_slots * inputs.expert_bytes_per_slot
            + plan.kv_bytes
            + inputs.fixed_gpu_bytes
            + inputs.activation_graph_reserve_bytes
            <= inputs.available_vram_bytes
        )
        assert len(plan.layer_tiers) == layers
        assert all(
            tier is ExpertTier.GPU_PERMANENT
            for layer_id, tier in enumerate(plan.layer_tiers)
            if layer_id in plan.permanent_layer_ids
        )
