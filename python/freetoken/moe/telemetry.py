"""Schema and strict TP aggregation for opt-in MoE performance telemetry.

Workers build one local snapshot after a request has completed. Keeping the
normalization and validation here (with no torch dependency) makes the wire
contract testable on CPU and prevents a partially reported TP run from looking
like a valid benchmark artifact.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


SCHEMA = "freetoken.moe-performance-telemetry"
VERSION = 1


class MoeTelemetryError(RuntimeError):
    """A multi-rank telemetry snapshot is incomplete or internally inconsistent."""


def aggregate_rank_snapshots(
    snapshots: Iterable[dict[str, Any] | None], *, expected_ranks: int
) -> dict[str, Any]:
    """Validate and aggregate one snapshot from every TP rank."""
    rows = list(snapshots)
    if len(rows) != expected_ranks or any(row is None for row in rows):
        present = [row.get("rank") for row in rows if isinstance(row, dict)]
        raise MoeTelemetryError(
            f"MoE telemetry requires {expected_ranks} ranks; received {len(rows)} "
            f"snapshots with ranks {present}"
        )
    ranks = [dict(row) for row in rows if row is not None]
    actual = sorted(int(row.get("rank", -1)) for row in ranks)
    wanted = list(range(expected_ranks))
    if actual != wanted:
        raise MoeTelemetryError(f"MoE telemetry ranks {actual} do not match {wanted}")

    epochs = {int(row.get("plan_epoch", -1)) for row in ranks}
    if len(epochs) != 1:
        raise MoeTelemetryError(f"MoE telemetry plan epochs disagree: {sorted(epochs)}")
    checksums = {row.get("placement_checksum") for row in ranks}
    if len(checksums) != 1:
        raise MoeTelemetryError(
            f"MoE telemetry placement checksums disagree: {sorted(map(str, checksums))}"
        )

    ranks.sort(key=lambda row: int(row["rank"]))
    totals: dict[str, int | float] = {
        "calls": 0,
        "active": 0,
        "missing": 0,
        "fetched": 0,
        "prefill_full_layer_rows": 0,
        "prefill_cache_hit_d2d_rows": 0,
        "prefill_h2d_rows": 0,
    }
    for rank in ranks:
        for layer in rank.get("layers", []):
            for field in ("calls", "active", "missing", "fetched"):
                totals[field] += int(layer.get(field, 0))
        prefill = rank.get("prefill", {})
        for field in ("full_layer_rows", "cache_hit_d2d_rows", "h2d_rows"):
            totals[f"prefill_{field}"] += int(prefill.get(field, 0))
    totals["miss_rate"] = (
        totals["missing"] / totals["active"] if totals["active"] else 0.0
    )
    result = {
        "schema": SCHEMA,
        "version": VERSION,
        "plan_epoch": epochs.pop(),
        "placement_checksum": checksums.pop(),
        "tp_size": expected_ranks,
        "totals": totals,
        "ranks": ranks,
    }
    timing_ranks = [rank for rank in ranks if rank.get("timing") is not None]
    if timing_ranks:
        if len(timing_ranks) != expected_ranks:
            raise MoeTelemetryError(
                f"MoE timing requires {expected_ranks} ranks; received {len(timing_ranks)}"
            )
        critical = []
        counts = [0] * expected_ranks
        num_timing_layers = len(timing_ranks[0]["timing"].get("layers", []))
        for rank in timing_ranks[1:]:
            actual_layers = len(rank["timing"].get("layers", []))
            if actual_layers != num_timing_layers:
                raise MoeTelemetryError(
                    f"rank {rank['rank']} timing has {actual_layers} layers; "
                    f"expected {num_timing_layers}"
                )
        for layer in range(num_timing_layers):
            candidates = []
            for rank in timing_ranks:
                timing_layers = rank["timing"].get("layers", [])
                wall = timing_layers[layer].get("layer_wall_ms")
                if wall is not None:
                    candidates.append(
                        (float(wall), int(rank["rank"]), timing_layers[layer])
                    )
            if candidates:
                wall_ms, critical_rank, layer_timing = max(candidates)
                counts[critical_rank] += 1
                critical.append({
                    "layer": layer,
                    "rank": critical_rank,
                    **{
                        field: float(value)
                        for field, value in layer_timing.items()
                        if field.endswith("_ms")
                    },
                    "layer_wall_ms": wall_ms,
                })
        result["timing"] = {
            "sample": "latest_completed_forward",
            "critical_rank_by_layer": critical,
            "critical_layer_counts_by_rank": counts,
        }
    return result


def subtract_snapshots(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    """Return raw per-rank counter deltas for one benchmark measurement window."""
    for field in ("schema", "version", "tp_size", "plan_epoch", "placement_checksum"):
        if after.get(field) != before.get(field):
            raise MoeTelemetryError(
                f"cannot subtract MoE telemetry with different {field}: "
                f"{before.get(field)!r} -> {after.get(field)!r}"
            )
    before_ranks = {int(row["rank"]): row for row in before.get("ranks", [])}
    delta_ranks = []
    counter_fields = ("calls", "active", "missing", "fetched", "steps")
    for after_rank in after.get("ranks", []):
        rank_id = int(after_rank["rank"])
        if rank_id not in before_ranks:
            raise MoeTelemetryError(f"rank {rank_id} is missing from baseline telemetry")
        before_rank = before_ranks[rank_id]
        delta_rank = deepcopy(after_rank)
        before_layers = {int(row["layer"]): row for row in before_rank.get("layers", [])}
        for layer in delta_rank.get("layers", []):
            layer_id = int(layer["layer"])
            if layer_id not in before_layers:
                raise MoeTelemetryError(
                    f"rank {rank_id} layer {layer_id} is missing from baseline telemetry"
                )
            baseline = before_layers[layer_id]
            for field in counter_fields:
                value = int(layer.get(field, 0)) - int(baseline.get(field, 0))
                if value < 0:
                    raise MoeTelemetryError(
                        f"rank {rank_id} layer {layer_id} counter {field} decreased"
                    )
                layer[field] = value
            calls, active, missing, fetched = (
                int(layer[field]) for field in ("calls", "active", "missing", "fetched")
            )
            layer["active_per_step"] = active / calls if calls else 0.0
            layer["missing_per_step"] = missing / calls if calls else 0.0
            layer["miss_rate"] = missing / active if active else 0.0
            layer["fetched_per_step"] = fetched / calls if calls else 0.0
        for field in ("full_layer_rows", "cache_hit_d2d_rows", "h2d_rows"):
            value = int(delta_rank.get("prefill", {}).get(field, 0)) - int(
                before_rank.get("prefill", {}).get(field, 0)
            )
            if value < 0:
                raise MoeTelemetryError(f"rank {rank_id} prefill counter {field} decreased")
            delta_rank["prefill"][field] = value
        before_prefill_layers = {
            int(row["layer"]): row
            for row in before_rank.get("prefill", {}).get("on_demand_layers", [])
        }
        for layer in delta_rank.get("prefill", {}).get("on_demand_layers", []):
            layer_id = int(layer["layer"])
            baseline = before_prefill_layers.get(layer_id)
            if baseline is None:
                raise MoeTelemetryError(
                    f"rank {rank_id} prefill layer {layer_id} is missing from baseline telemetry"
                )
            for field in ("calls", "active", "missing", "fetched"):
                value = int(layer.get(field, 0)) - int(baseline.get(field, 0))
                if value < 0:
                    raise MoeTelemetryError(
                        f"rank {rank_id} prefill layer {layer_id} counter {field} decreased"
                    )
                layer[field] = value
        for field, value in tuple(delta_rank.get("storage", {}).items()):
            if isinstance(value, (int, float)):
                delta_rank["storage"][field] = (
                    value - before_rank.get("storage", {}).get(field, 0)
                )
        delta_ranks.append(delta_rank)
    return aggregate_rank_snapshots(delta_ranks, expected_ranks=int(after["tp_size"]))


__all__ = [
    "MoeTelemetryError",
    "SCHEMA",
    "VERSION",
    "aggregate_rank_snapshots",
    "subtract_snapshots",
]
