"""Request-boundary, CUDA-graph-safe MoE component profiling.

Each layer/component owns one fixed pair of external CUDA events. Event records
become graph nodes during capture and are updated on every replay. The server
reads them only after a request boundary synchronization, so the hot path never
performs device-to-host timing reads or allocates events. The event-record graph
nodes still have measurable cost, so this recorder is diagnostic-only.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


COMPONENTS = (
    "ensure_experts",
    "miss_copy",
    "routed_gate_up",
    "routed_activation",
    "routed_down",
    "routed_total",
    "shared_expert",
    "all_reduce",
    "layer_wall",
)


class MoeCudaTiming:
    """Fixed-lifetime CUDA event pairs for one rank's MoE layers."""

    def __init__(self, *, num_layers: int, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("MoE CUDA timing requires a CUDA device")
        self.num_layers = int(num_layers)
        self.device = device
        self._pairs = {
            (layer, component): (
                torch.cuda.Event(enable_timing=True, external=True),
                torch.cuda.Event(enable_timing=True, external=True),
            )
            for layer in range(self.num_layers)
            for component in COMPONENTS
        }
        self._recorded: set[tuple[int, str]] = set()

    @contextmanager
    def measure(self, layer: int, component: str) -> Iterator[None]:
        if not 0 <= layer < self.num_layers:
            raise ValueError(f"timing layer {layer} outside [0, {self.num_layers})")
        if component not in COMPONENTS:
            raise ValueError(f"unknown MoE timing component {component!r}")
        key = (int(layer), component)
        start, end = self._pairs[key]
        stream = torch.cuda.current_stream(self.device)
        start.record(stream)
        self._recorded.add(key)
        try:
            yield
        finally:
            end.record(stream)

    def snapshot(self) -> dict:
        """Read the latest completed invocation for each recorded component."""
        layers = []
        for layer in range(self.num_layers):
            row: dict[str, int | float] = {"layer": layer}
            for component in COMPONENTS:
                key = (layer, component)
                if key in self._recorded:
                    start, end = self._pairs[key]
                    row[f"{component}_ms"] = float(start.elapsed_time(end))
            layers.append(row)
        return {
            "sample": "latest_completed_forward",
            "components": list(COMPONENTS),
            "layers": layers,
        }


__all__ = ["COMPONENTS", "MoeCudaTiming"]
