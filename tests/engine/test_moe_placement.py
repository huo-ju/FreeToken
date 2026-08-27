from __future__ import annotations

import random
from dataclasses import replace

import pytest

from freetoken.engine.moe_placement import (
    ExpertPlacementCapabilities,
    ExpertTier,
    MoePlacementInfeasible,
    MoePlacementInputs,
    NodeMoePlacementInputs,
    RankTopology,
    plan_moe_placement,
    plan_node_moe_placement,
)
from freetoken.moe.execution_plan import RoutedTpLayout


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
        kv_min_pages=10,
        kv_target_pages=None,
        kv_page_bytes=20,
        expert_bytes_per_slot=10,
        host_bytes_per_layer=(100, 100, 100, 100),
        gpu_bytes_per_layer=(80, 80, 80, 80),
        num_experts=8,
        prefill_overlap=True,
    )
    values.update(overrides)
    return MoePlacementInputs(**values)


def _topology(rank: int, *, width: int = 16, vram: int = 24_000) -> RankTopology:
    return RankTopology(
        rank=rank,
        pci_bdf=f"0000:{0x40 + rank:02x}:00.0",
        numa_node=rank // 2,
        compute_capability="7.5",
        vram_bytes=vram,
        pcie_generation=3,
        pcie_width=width,
    )


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


def test_fixed_kv_target_is_not_expanded_into_free_vram():
    plan = plan_moe_placement(
        _inputs(
            available_vram_bytes=20_000,
            kv_min_pages=0,
            kv_target_pages=128,
            kv_page_bytes=20,
        ),
        CAPS,
    )

    assert plan.kv_pages == 128
    assert plan.kv_bytes == 2_560
    assert plan.vram_headroom_bytes > 0


def test_auto_kv_uses_vram_left_after_authoritative_and_dynamic_placement():
    plan = plan_moe_placement(
        _inputs(kv_min_pages=3, kv_target_pages=None, dynamic_cache_slots=16),
        CAPS,
    )

    assert plan.dynamic_cache_slots == 16
    assert plan.kv_pages > 3
    assert 0 <= plan.vram_headroom_bytes < 20


def test_fixed_dynamic_cache_below_prefill_floor_is_rejected():
    with pytest.raises(MoePlacementInfeasible, match="floor is 16"):
        plan_moe_placement(_inputs(dynamic_cache_slots=15), CAPS)


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
            kv_min_pages=rng.randint(1, 30),
            kv_target_pages=None,
            kv_page_bytes=rng.randint(1, 20),
            expert_bytes_per_slot=rng.randint(1, 20),
            host_bytes_per_layer=host_sizes,
            gpu_bytes_per_layer=gpu_sizes,
            num_experts=experts,
            prefill_overlap=bool(rng.getrandbits(1)),
        )
        try:
            plan = plan_moe_placement(inputs, CAPS, policy=rng.choice(("balanced", "gpu-first")))
        except MoePlacementInfeasible:
            continue

        assert plan.retained_host_bytes <= inputs.host_expert_budget_bytes
        assert plan.pinned_host_bytes <= inputs.pin_budget_bytes
        assert plan.dynamic_cache_slots >= plan.dynamic_floor_slots
        assert plan.kv_pages >= inputs.kv_min_pages
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


def test_node_planner_uses_global_budget_instead_of_equal_rank_split():
    # Rank 0 owns a wider physical shard and legitimately retains three times
    # as many host bytes.  An old host_budget/tp_size split would force an
    # unnecessary permanent layer there despite sufficient aggregate RAM.
    rank0 = _inputs(
        available_vram_bytes=5_000,
        host_expert_budget_bytes=0,
        pin_budget_bytes=0,
        host_bytes_per_layer=(150,) * 4,
        gpu_bytes_per_layer=(120,) * 4,
    )
    rank1 = _inputs(
        available_vram_bytes=5_000,
        host_expert_budget_bytes=0,
        pin_budget_bytes=0,
        host_bytes_per_layer=(50,) * 4,
        gpu_bytes_per_layer=(40,) * 4,
    )
    plan = plan_node_moe_placement(
        NodeMoePlacementInputs(
            rank_inputs=(rank0, rank1),
            rank_topology=(_topology(0), _topology(1, width=4)),
            routed_tp_layout=RoutedTpLayout.equal(1024, 2, alignment=256),
            host_expert_budget_bytes=800,
            pin_budget_bytes=800,
        ),
        (CAPS, CAPS),
    )

    assert plan.rank_plans[0].retained_host_bytes == 600
    assert plan.rank_plans[1].retained_host_bytes == 200
    assert plan.retained_host_bytes == 800
    assert plan.rank_plans[0].permanent_layer_ids == ()
    assert plan.kv_pages == min(rank_plan.kv_pages for rank_plan in plan.rank_plans)
    plan.verify_checksum()


def test_node_global_solver_finds_weighted_asymmetric_capacity_proof():
    # Full host is 1,376 units and the node can retain only 896, so 480 must
    # move to GPU. Proportional budget splitting requires rank 2 to move ten
    # 12-unit layers, but its VRAM frontier stops at nine. The aggregate solve
    # transfers the remaining 12 units to three extra rank-3 layers instead.
    shard_sizes = (8, 8, 12, 4)
    max_permanent = (15, 15, 9, 33)
    rank_inputs = tuple(
        _inputs(
            available_vram_bytes=limit * shard + shard + 20,
            fixed_gpu_bytes=0,
            activation_graph_reserve_bytes=0,
            host_expert_budget_bytes=43 * shard,
            pin_budget_bytes=43 * shard,
            host_bytes_per_layer=(shard,) * 43,
            gpu_bytes_per_layer=(shard,) * 43,
            num_experts=1,
            expert_bytes_per_slot=shard,
            prefill_overlap=False,
            dynamic_cache_slots=1,
            kv_min_pages=0,
            kv_target_pages=2,
            kv_page_bytes=10,
        )
        for shard, limit in zip(shard_sizes, max_permanent)
    )
    plan = plan_node_moe_placement(
        NodeMoePlacementInputs(
            rank_inputs=rank_inputs,
            rank_topology=tuple(
                _topology(rank, width=(8, 8, 16, 4)[rank]) for rank in range(4)
            ),
            routed_tp_layout=RoutedTpLayout.from_widths(
                (512, 512, 768, 256), alignment=256
            ),
            host_expert_budget_bytes=896,
            pin_budget_bytes=1_376,
            checkpoint_layout="synthetic",
        ),
        (CAPS,) * 4,
        allow_additional_cpu_layers=False,
    )

    assert [len(rank.permanent_layer_ids) for rank in plan.rank_plans] == [15, 15, 9, 33]
    assert plan.retained_host_bytes == 896
    assert plan.permanent_gpu_bytes == 480
    assert plan.kv_mode == "fixed"
    assert plan.kv_pages == 2
    assert all(rank.kv_pages == 2 for rank in plan.rank_plans)
    assert plan.runtime_disk_tier is False


def test_equal_tp_host_deficit_prefers_symmetric_headroom():
    plan = plan_node_moe_placement(
        NodeMoePlacementInputs(
            rank_inputs=(
                _inputs(dynamic_cache_slots=16),
                _inputs(dynamic_cache_slots=16),
            ),
            rank_topology=(_topology(0), _topology(1)),
            routed_tp_layout=RoutedTpLayout.equal(1024, 2, alignment=256),
            host_expert_budget_bytes=600,
            pin_budget_bytes=800,
        ),
        (CAPS, CAPS),
        allow_additional_cpu_layers=False,
    )

    assert [len(rank.permanent_layer_ids) for rank in plan.rank_plans] == [1, 1]
    assert plan.vram_headroom_bytes[0] == plan.vram_headroom_bytes[1]


def test_nonuniform_layer_frontier_keeps_lower_gpu_feasible_subset():
    rank = _inputs(
        available_vram_bytes=1_000,
        host_expert_budget_bytes=22,
        pin_budget_bytes=22,
        host_bytes_per_layer=(9, 8, 5),
        gpu_bytes_per_layer=(9, 4, 2),
        num_experts=1,
        expert_bytes_per_slot=1,
        prefill_overlap=False,
        dynamic_cache_slots=1,
        kv_min_pages=0,
        kv_target_pages=2,
        kv_page_bytes=10,
    )
    plan = plan_node_moe_placement(
        NodeMoePlacementInputs(
            rank_inputs=(rank,),
            rank_topology=(_topology(0),),
            routed_tp_layout=RoutedTpLayout.equal(1024, 1, alignment=256),
            host_expert_budget_bytes=9,
            pin_budget_bytes=22,
        ),
        (CAPS,),
        allow_additional_cpu_layers=False,
    )

    assert plan.rank_plans[0].permanent_layer_ids == (1, 2)
    assert plan.permanent_gpu_bytes == 6


def test_infeasible_node_report_lists_aggregate_and_each_rank_frontier():
    tiny = _inputs(available_vram_bytes=700, dynamic_cache_slots=16)
    with pytest.raises(MoePlacementInfeasible) as error:
        plan_node_moe_placement(
            NodeMoePlacementInputs(
                rank_inputs=(tiny, tiny),
                rank_topology=(_topology(0), _topology(1)),
                routed_tp_layout=RoutedTpLayout.equal(1024, 2, alignment=256),
                host_expert_budget_bytes=0,
                pin_budget_bytes=800,
            ),
            (CAPS, CAPS),
            allow_additional_cpu_layers=False,
        )

    message = str(error.value)
    assert "aggregate host deficit" in message
    assert "aggregate shortfall" in message
    assert "rank 0 frontier max removable" in message
    assert "rank 1 frontier max removable" in message


def test_node_plan_checksum_rejects_rank_divergence():
    inputs = NodeMoePlacementInputs(
        rank_inputs=(_inputs(), _inputs()),
        rank_topology=(_topology(0), _topology(1)),
        routed_tp_layout=RoutedTpLayout.equal(1024, 2, alignment=256),
        host_expert_budget_bytes=800,
        pin_budget_bytes=800,
    )
    plan = plan_node_moe_placement(inputs, (CAPS, CAPS))
    with pytest.raises(MoePlacementInfeasible, match="checksum mismatch"):
        replace(plan, checksum="stale").verify_checksum()


def test_node_explicit_permanent_count_is_exact_on_every_rank():
    inputs = NodeMoePlacementInputs(
        rank_inputs=(
            _inputs(dynamic_cache_slots=16),
            _inputs(dynamic_cache_slots=16),
        ),
        rank_topology=(_topology(0), _topology(1)),
        routed_tp_layout=RoutedTpLayout.equal(1024, 2, alignment=256),
        host_expert_budget_bytes=600,
        pin_budget_bytes=800,
    )
    plan = plan_node_moe_placement(
        inputs,
        (CAPS, CAPS),
        explicit_permanent_layer_count=1,
        allow_additional_cpu_layers=False,
    )

    assert [rank.permanent_layer_ids for rank in plan.rank_plans] == [(3,), (3,)]


def test_node_checksum_is_independent_of_explicit_layer_id_iteration_order():
    inputs = NodeMoePlacementInputs(
        rank_inputs=(_inputs(dynamic_cache_slots=16),),
        rank_topology=(_topology(0),),
        routed_tp_layout=RoutedTpLayout.equal(1024, 1, alignment=256),
        host_expert_budget_bytes=200,
        pin_budget_bytes=400,
    )
    first = plan_node_moe_placement(
        inputs,
        (CAPS,),
        explicit_permanent_layer_ids=(3, 1),
        explicit_permanent_layer_count=2,
        allow_additional_cpu_layers=False,
    )
    second = plan_node_moe_placement(
        inputs,
        (CAPS,),
        explicit_permanent_layer_ids=(1, 3),
        explicit_permanent_layer_count=2,
        allow_additional_cpu_layers=False,
    )

    assert first.checksum == second.checksum


def test_node_gpu_only_zero_policy_reports_host_shortfall_without_fallback():
    with pytest.raises(MoePlacementInfeasible, match="aggregate host deficit"):
        plan_node_moe_placement(
            NodeMoePlacementInputs(
                rank_inputs=(_inputs(), _inputs()),
                rank_topology=(_topology(0), _topology(1)),
                routed_tp_layout=RoutedTpLayout.equal(1024, 2, alignment=256),
                host_expert_budget_bytes=799,
                pin_budget_bytes=800,
            ),
            (CAPS, CAPS),
            allow_gpu_permanent=False,
            allow_additional_cpu_layers=False,
        )


def test_node_planner_preserves_forced_policy_constraints_on_every_rank():
    inputs = NodeMoePlacementInputs(
        rank_inputs=(_inputs(), _inputs()),
        rank_topology=(_topology(0), _topology(1)),
        routed_tp_layout=RoutedTpLayout.equal(1024, 2, alignment=256),
        host_expert_budget_bytes=800,
        pin_budget_bytes=0,
    )
    plan = plan_node_moe_placement(
        inputs,
        (CAPS, CAPS),
        allow_gpu_permanent=False,
        explicit_cpu_layer_ids=range(4),
        allow_additional_cpu_layers=False,
    )
    assert all(rank.permanent_layer_ids == () for rank in plan.rank_plans)
    assert all(rank.cpu_layer_ids == frozenset(range(4)) for rank in plan.rank_plans)


def test_node_gpu_first_uses_permanent_capacity_before_locked_host():
    inputs = NodeMoePlacementInputs(
        rank_inputs=(_inputs(), _inputs()),
        rank_topology=(_topology(0), _topology(1)),
        routed_tp_layout=RoutedTpLayout.equal(1024, 2, alignment=256),
        host_expert_budget_bytes=800,
        pin_budget_bytes=200,
    )
    plan = plan_node_moe_placement(inputs, (CAPS, CAPS), policy="gpu-first")

    assert plan.locked_host_bytes == 0
    assert plan.pinned_host_bytes == 200
    assert plan.permanent_gpu_bytes == 480


def test_weighted_node_layout_rejects_unknown_checkpoint_before_allocation():
    with pytest.raises(MoePlacementInfeasible, match="safetensors"):
        plan_node_moe_placement(
            NodeMoePlacementInputs(
                rank_inputs=(_inputs(),) * 4,
                rank_topology=tuple(_topology(rank) for rank in range(4)),
                routed_tp_layout=RoutedTpLayout.from_widths(
                    (512, 512, 768, 256), alignment=256
                ),
                host_expert_budget_bytes=1_600,
                pin_budget_bytes=1_600,
                checkpoint_layout="ftw",
            ),
            (CAPS,) * 4,
        )
