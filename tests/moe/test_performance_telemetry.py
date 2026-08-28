from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.moe.telemetry import (
    MoeTelemetryError,
    aggregate_rank_snapshots,
    subtract_snapshots,
)
from freetoken.server.stats import StatsTracker, build_stats


def _rank(rank: int, *, epoch: int = 7, checksum: str = "abc") -> dict:
    return {
        "rank": rank,
        "plan_epoch": epoch,
        "placement_checksum": checksum,
        "local_width": 256 * (rank + 1),
        "layers": [
            {
                "layer": 0,
                "residency": "host_backed",
                "calls": 2,
                "active": 8,
                "missing": rank + 1,
                "fetched": rank + 1,
            },
            {
                "layer": 1,
                "residency": "gpu_permanent",
                "calls": 2,
                "active": 8,
                "missing": 0,
                "fetched": 0,
            },
        ],
        "prefill": {
            "full_layer_rows": 256,
            "cache_hit_d2d_rows": rank,
            "h2d_rows": 256 - rank,
        },
    }


def test_aggregate_keeps_raw_rank_layer_data_and_builds_totals():
    doc = aggregate_rank_snapshots([_rank(1), _rank(0)], expected_ranks=2)
    assert doc["schema"] == "freetoken.moe-performance-telemetry"
    assert [row["rank"] for row in doc["ranks"]] == [0, 1]
    assert doc["totals"]["calls"] == 8
    assert doc["totals"]["active"] == 32
    assert doc["totals"]["missing"] == 3
    assert doc["totals"]["fetched"] == 3
    assert doc["totals"]["prefill_h2d_rows"] == 511


def test_aggregate_reports_each_layers_critical_timing_rank():
    ranks = [_rank(0), _rank(1)]
    ranks[0]["timing"] = {
        "sample": "latest_completed_forward",
        "layers": [
            {"layer": 0, "layer_wall_ms": 4.0},
            {"layer": 1, "layer_wall_ms": 9.0},
        ],
    }
    ranks[1]["timing"] = {
        "sample": "latest_completed_forward",
        "layers": [
            {"layer": 0, "layer_wall_ms": 7.0},
            {"layer": 1, "layer_wall_ms": 5.0},
        ],
    }

    timing = aggregate_rank_snapshots(ranks, expected_ranks=2)["timing"]

    assert timing["critical_rank_by_layer"] == [
        {"layer": 0, "rank": 1, "layer_wall_ms": 7.0},
        {"layer": 1, "rank": 0, "layer_wall_ms": 9.0},
    ]
    assert timing["critical_layer_counts_by_rank"] == [1, 1]


def test_aggregate_rejects_partial_rank_timing():
    ranks = [_rank(0), _rank(1)]
    ranks[0]["timing"] = {"layers": [{"layer": 0, "layer_wall_ms": 4.0}]}
    with pytest.raises(MoeTelemetryError, match="timing requires 2 ranks"):
        aggregate_rank_snapshots(ranks, expected_ranks=2)


@pytest.mark.parametrize(
    "rows, message",
    [
        ([_rank(0), None], "requires 2 ranks"),
        ([_rank(0), _rank(0)], "do not match"),
        ([_rank(0), _rank(1, epoch=8)], "epochs disagree"),
        ([_rank(0), _rank(1, checksum="def")], "checksums disagree"),
    ],
)
def test_aggregate_fails_loud_on_incoherent_tp_snapshots(rows, message):
    with pytest.raises(MoeTelemetryError, match=message):
        aggregate_rank_snapshots(rows, expected_ranks=2)


def test_stats_tracker_retains_only_an_explicit_telemetry_snapshot():
    class Reply:
        completion_tokens_delta = 0
        prompt_tokens_delta = 0
        finished = False

    tracker = StatsTracker()
    tracker.observe(Reply())
    assert tracker.moe_telemetry is None

    Reply.moe_telemetry = {"schema": "freetoken.moe-performance-telemetry"}
    tracker.observe(Reply())
    assert tracker.moe_telemetry == Reply.moe_telemetry


def test_stats_schema_is_additive_only_when_telemetry_exists():
    tracker = StatsTracker()
    state = SimpleNamespace(
        stats=tracker,
        config=SimpleNamespace(
            model_config=SimpleNamespace(
                has_linear_attention=False, has_swa_attention=False, is_moe=True
            ),
            served_model_name="model",
            max_seq_len=1024,
            page_size=1,
        ),
    )
    assert "moe_telemetry" not in build_stats(state, 0, 0)
    tracker.moe_telemetry = {"schema": "freetoken.moe-performance-telemetry"}
    assert build_stats(state, 0, 0)["moe_telemetry"] == tracker.moe_telemetry


def test_subtract_snapshots_excludes_the_warmup_window():
    before = aggregate_rank_snapshots([_rank(0)], expected_ranks=1)
    after_rank = _rank(0)
    after_rank["layers"][0].update(calls=5, steps=5, active=20, missing=4, fetched=4)
    after_rank["prefill"]["h2d_rows"] = 512
    after = aggregate_rank_snapshots([after_rank], expected_ranks=1)

    delta = subtract_snapshots(after, before)

    layer = delta["ranks"][0]["layers"][0]
    assert (layer["calls"], layer["active"], layer["missing"], layer["fetched"]) == (3, 12, 3, 3)
    assert layer["miss_rate"] == 0.25
    assert delta["ranks"][0]["prefill"]["h2d_rows"] == 256


def test_counter_reset_does_not_change_cache_mapping():
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=3,
        cache_size=3,
        device=torch.device("cpu"),
    )
    cache.slot_for_id[0, 1] = 2
    cache.id_of_slot[2] = 1
    cache.usage[2] = 99
    mapping = (cache.slot_for_id.clone(), cache.id_of_slot.clone(), cache.usage.clone())
    cache.lru_stats.fill_(4)
    cache.prefill_lru_stats.fill_(5)
    cache.prefill_full_layer_rows = 6
    cache.prefill_h2d_rows = 7

    cache.reset_stats()

    torch.testing.assert_close(cache.slot_for_id, mapping[0])
    torch.testing.assert_close(cache.id_of_slot, mapping[1])
    torch.testing.assert_close(cache.usage, mapping[2])
    assert not cache.lru_stats.any()
    assert not cache.prefill_lru_stats.any()
    assert cache.prefill_full_layer_rows == cache.prefill_h2d_rows == 0
