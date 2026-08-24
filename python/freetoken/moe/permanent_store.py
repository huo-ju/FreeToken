"""Immutable GPU-authoritative expert banks, physically separate from the LRU."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import torch


@dataclass
class PermanentExpertStore:
    num_layers: int
    num_experts: int
    layer_ids: tuple[int, ...]
    bank_schema: tuple[str, ...]
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]]
    device: torch.device
    banks: dict[str, torch.Tensor] = field(init=False)
    _layer_index: dict[int, int] = field(init=False, repr=False)
    _loaded: set[int] = field(default_factory=set, init=False, repr=False)
    _copy_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        ids = tuple(sorted(set(int(layer_id) for layer_id in self.layer_ids)))
        if ids != self.layer_ids:
            raise ValueError("permanent layer ids must be sorted and unique")
        if any(layer_id < 0 or layer_id >= self.num_layers for layer_id in ids):
            raise ValueError(f"permanent layer ids {ids} outside [0, {self.num_layers})")
        if set(self.specs) != set(self.bank_schema):
            raise ValueError(
                f"permanent bank specs {sorted(self.specs)} do not match schema "
                f"{self.bank_schema}"
            )
        self._layer_index = {layer_id: index for index, layer_id in enumerate(ids)}
        self.banks = {}
        for name in self.bank_schema:
            shape, dtype = self.specs[name]
            if not shape or shape[0] != self.num_experts:
                raise ValueError(
                    f"bank {name!r} spec must start with num_experts={self.num_experts}; "
                    f"got {shape}"
                )
            self.banks[name] = torch.empty(
                (len(ids), *shape), dtype=dtype, device=self.device
            )

    @property
    def loaded_layer_ids(self) -> frozenset[int]:
        return frozenset(self._loaded)

    @property
    def nbytes(self) -> int:
        return sum(bank.numel() * bank.element_size() for bank in self.banks.values())

    @property
    def expert_bytes_per_slot(self) -> int:
        return sum(
            math.prod(shape[1:]) * torch.empty((), dtype=dtype).element_size()
            for shape, dtype in self.specs.values()
        )

    def contains(self, layer_id: int) -> bool:
        return layer_id in self._layer_index

    def stage(self, layer_id: int, tensors: dict[str, torch.Tensor]) -> None:
        if layer_id not in self._layer_index:
            raise ValueError(f"layer {layer_id} was not reserved as GPU permanent")
        if set(tensors) != set(self.bank_schema):
            raise ValueError(
                f"layer {layer_id} banks {sorted(tensors)} do not match {self.bank_schema}"
            )
        index = self._layer_index[layer_id]
        with self._copy_lock:
            if layer_id in self._loaded:
                raise RuntimeError(f"GPU permanent expert layer {layer_id} was staged twice")
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)
            for name in self.bank_schema:
                source = tensors[name]
                target = self.banks[name][index]
                if source.shape != target.shape or source.dtype != target.dtype:
                    raise ValueError(
                        f"GPU permanent bank {name!r} layer {layer_id}: source "
                        f"{source.shape}/{source.dtype}, target {target.shape}/{target.dtype}"
                    )
                target.copy_(source, non_blocking=False)
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            self._loaded.add(layer_id)

    def validate_complete(self) -> None:
        missing = sorted(set(self.layer_ids) - self._loaded)
        if missing:
            raise RuntimeError(f"GPU permanent expert layers were not staged: {missing}")

    def views(self, layer_id: int) -> tuple[torch.Tensor, ...]:
        if layer_id not in self._layer_index:
            raise ValueError(f"layer {layer_id} is not GPU permanent")
        if layer_id not in self._loaded:
            raise RuntimeError(f"GPU permanent expert layer {layer_id} has not been staged")
        index = self._layer_index[layer_id]
        return tuple(self.banks[name][index] for name in self.bank_schema)


__all__ = ["PermanentExpertStore"]
