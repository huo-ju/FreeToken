"""Weight loading for DeepSeek-V4-Flash (engine path).

  - :func:`iter_weights` streams resident (non-expert) tensors keyed to engine param
    names; model's ``load_state_dict`` casts each. ``wo_a`` dequantized to bf16 to match
    the reference bf16 einsum.
  - :func:`load_dsfp4_expert_sources` packs routed FP4 experts into pinned CPU banks for
    the offload cache. DeepSeek FP4: e8m0 per-32 block scale, no global scale.
"""

from __future__ import annotations

import collections
import json
import os
import re
import time
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.moe.execution_plan import RoutedTpLayout
from tqdm import tqdm

from freetoken.models.loader import drop_page_cache
from freetoken.utils import div_ceil, div_even

from .args import DeepseekV4Args, load_args


class _ShardReader:
    def __init__(self, folder: str, weight_map: dict, device):
        self._folder = folder
        self._weight_map = weight_map
        self._device = str(device)
        self._handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self._weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(
                os.path.join(self._folder, shard), framework="pt", device=self._device
            ).__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        for shard, handle in self._handles.items():
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass
            drop_page_cache(os.path.join(self._folder, shard))
        self._handles.clear()


def _weight_map(model_path: str) -> dict:
    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def _dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Dequantize 128x128 block-scaled FP8 (e4m3) to bf16.

    scale is e8m0 exponent codes, ``value = 2^(code-127)`` (Triton FP8 GEMM convention).
    Used for ``wo_a`` to match the reference's bf16 einsum.
    """
    n, k = weight.shape
    codes = scale.view(torch.uint8).to(torch.float32)
    s = torch.exp2(codes - 127.0)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:n, :k]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def _vocab_tp_shard(
    tensor: torch.Tensor, vocab_size: int, *, tp_rank: int, tp_size: int
) -> torch.Tensor:
    """Slice embedding/head rows using the engine's padded vocab partition."""
    if tp_size == 1:
        return tensor
    rows = div_ceil(vocab_size, tp_size)
    start = rows * tp_rank
    end = min(start + rows, vocab_size)
    local = tensor[start:end].contiguous()
    if local.shape[0] == rows:
        return local
    # Only the final rank can need padding. The forward path masks those rows,
    # but keeping a rectangular shard lets all_gather use identical shapes.
    padded = tensor.new_zeros((rows, *tensor.shape[1:]))
    padded[: local.shape[0]].copy_(local)
    return padded


def _shared_expert_tp_shard(
    tensor: torch.Tensor,
    proj: str,
    kind: str,
    intermediate_size: int,
    *,
    tp_rank: int,
    tp_size: int,
    block: int = 128,
) -> torch.Tensor:
    """Shard the resident shared expert over its SwiGLU intermediate axis."""
    if tp_size == 1:
        return tensor
    local_i = div_even(intermediate_size, tp_size)
    i0, i1 = tp_rank * local_i, (tp_rank + 1) * local_i
    if kind == "scale":
        i0, i1 = i0 // block, i1 // block
    axis = 0 if proj in ("w1", "w3") else 1
    return tensor.narrow(axis, i0, i1 - i0).contiguous()


def iter_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
):
    """Stream resident (non-expert) weights as ``(name, tensor)`` keyed to engine params.

    Routed FP4 experts come from the offload cache, so ``include_moe_experts`` must be
    False (DeepSeek-V4 only runs ``--moe-backend offload``). Tensors yielded in checkpoint
    dtype (fp8 + e8m0 preserved); ``wo_a`` dequantized to bf16 to match the reference einsum.
    """
    if include_moe_experts:
        raise ValueError(
            "DeepSeek-V4 routed experts are served from the offload cache; "
            "run with --moe-backend offload (include_moe_experts must be False)."
        )
    if not include_non_moe:
        return

    args = load_args(model_path, max_batch_size=1)
    reader = _ShardReader(model_path, _weight_map(model_path), device)
    tp = get_tp_info()

    def get(name: str) -> torch.Tensor:
        return reader.get(name)

    def linear(prefix: str):
        yield f"{prefix}.weight", get(f"{prefix}.weight")
        if reader.has(f"{prefix}.scale"):
            yield f"{prefix}.scale", get(f"{prefix}.scale")

    def shared_expert_linear(prefix: str, proj: str):
        for kind in ("weight", "scale"):
            key = f"{prefix}.{kind}"
            if reader.has(key):
                yield key, _shared_expert_tp_shard(
                    get(key), proj, kind, args.moe_inter_dim,
                    tp_rank=tp.rank, tp_size=tp.size,
                )

    try:
        yield "embed.weight", _vocab_tp_shard(
            get("embed.weight"), args.vocab_size,
            tp_rank=tp.rank, tp_size=tp.size,
        )
        yield "norm.weight", get("norm.weight")
        yield "head", _vocab_tp_shard(
            get("head.weight"), args.vocab_size,
            tp_rank=tp.rank, tp_size=tp.size,
        )
        for nm in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            yield nm, get(nm)

        for L in range(args.n_layers):
            a = f"layers.{L}.attn"
            yield from linear(f"{a}.wq_a")
            yield f"{a}.q_norm.weight", get(f"{a}.q_norm.weight")
            yield from linear(f"{a}.wq_b")
            yield from linear(f"{a}.wkv")
            yield f"{a}.kv_norm.weight", get(f"{a}.kv_norm.weight")
            # wo_a: FP8 in the checkpoint, dequantized to bf16 (reference bf16 einsum).
            yield f"{a}.wo_a", _dequant_fp8_block(
                get(f"{a}.wo_a.weight"), get(f"{a}.wo_a.scale")
            )
            yield from linear(f"{a}.wo_b")
            yield f"{a}.attn_sink", get(f"{a}.attn_sink")

            ratio = args.compress_ratios[L]
            if ratio:
                c = f"{a}.compressor"
                yield f"{c}.ape", get(f"{c}.ape")
                yield f"{c}.wkv.weight", get(f"{c}.wkv.weight")
                yield f"{c}.wgate.weight", get(f"{c}.wgate.weight")
                yield f"{c}.norm.weight", get(f"{c}.norm.weight")
                if ratio == 4:
                    idx = f"{a}.indexer"
                    yield from linear(f"{idx}.wq_b")
                    yield f"{idx}.weights_proj.weight", get(f"{idx}.weights_proj.weight")
                    ic = f"{idx}.compressor"
                    yield f"{ic}.ape", get(f"{ic}.ape")
                    yield f"{ic}.wkv.weight", get(f"{ic}.wkv.weight")
                    yield f"{ic}.wgate.weight", get(f"{ic}.wgate.weight")
                    yield f"{ic}.norm.weight", get(f"{ic}.norm.weight")

            yield f"layers.{L}.attn_norm.weight", get(f"layers.{L}.attn_norm.weight")
            yield f"layers.{L}.ffn_norm.weight", get(f"layers.{L}.ffn_norm.weight")

            g = f"layers.{L}.ffn.gate"
            yield f"{g}.weight", get(f"{g}.weight")
            if L < args.n_hash_layers:
                yield f"{g}.tid2eid", get(f"{g}.tid2eid")
            else:
                yield f"{g}.bias", get(f"{g}.bias")
            for proj in ("w1", "w2", "w3"):
                yield from shared_expert_linear(
                    f"layers.{L}.ffn.shared_experts.{proj}", proj
                )

            for nm in (
                "hc_attn_fn", "hc_ffn_fn", "hc_attn_base",
                "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale",
            ):
                yield f"layers.{L}.{nm}", get(f"layers.{L}.{nm}")
    finally:
        reader.close()


# --------------------------------------------------------------------------------------
# Routed FP4 expert pinned banks.
# --------------------------------------------------------------------------------------
_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|scale)$"
)


def dsfp4_expert_bank_specs(
    args: DeepseekV4Args,
    *,
    routed_tp_widths: tuple[int, ...] | None = None,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Per-rank host/cache shapes for DeepSeek-FP4 routed experts.

    Expert tensor parallelism splits the SwiGLU intermediate dimension. Every
    rank retains all expert ids (so routing is identical), computes its local
    intermediate slice, and :class:`OffloadMoELayer` all-reduces the partial
    ``[T, H]`` outputs. Consequently one GPU slot and one rank's host backing are
    ``1 / tp_size`` of the unsharded expert payload.
    """
    tp = get_tp_info()
    E, H = args.n_routed_experts, args.dim
    layout = (
        RoutedTpLayout.equal(args.moe_inter_dim, tp.size, alignment=256)
        if routed_tp_widths is None
        else RoutedTpLayout.from_widths(routed_tp_widths, alignment=256)
    )
    if len(layout.widths) != tp.size or layout.total_intermediate_size != args.moe_inter_dim:
        raise ValueError("routed TP layout does not match DSV4 intermediate size / TP size")
    I = layout.widths[tp.rank]
    e8m0 = torch.float8_e8m0fnu
    return {
        "gate_up_packed": ((E, 2 * I, H // 2), torch.uint8),
        "gate_up_scale": ((E, 2 * I, H // 32), e8m0),
        "down_packed": ((E, H, I // 2), torch.uint8),
        "down_scale": ((E, H, I // 32), e8m0),
    }


def _load_with_residency(load, num_layers: int, *, layer_sink, layer_residency, gpu_sink):
    """Run a DSV4 bank fill through conversion or mixed serving residency."""
    from freetoken.moe.host_banks import HostResidency, ServingLayerPipeline

    if layer_sink is not None:
        if layer_residency is not None or gpu_sink is not None:
            raise ValueError("conversion layer_sink cannot be combined with serving residency")
        return load(layer_sink)

    residency = layer_residency or [HostResidency.PINNED.value] * num_layers
    if len(residency) != num_layers:
        raise ValueError(
            f"layer_residency has {len(residency)} entries; expected {num_layers}"
        )
    with ServingLayerPipeline(residency, gpu_sink) as serving:
        return load(serving)


def _ordered_expert_shard_work(shards, layer_residency: list[str] | None):
    """Return serial shard work with GPU-only layers first.

    The published DSV4 checkpoint stores routed experts roughly in layer order.
    Reading it naively therefore faults and pins every host-backed shallow layer
    before reaching the deep layers selected for GPU-only residency. On a
    host-RAM constrained machine that recreates the all-host startup peak even
    though the final layout fits.

    Split a shard into two work items when it contains both residency classes.
    That can reopen a boundary shard once, but guarantees every GPU-only layer
    is staged and released before any pinned layer is retained. Conversion and
    ordinary all-pinned serving retain the original lexical shard order.
    """
    ordered = [(shard, shards[shard]) for shard in sorted(shards)]
    if not layer_residency or "gpu_only" not in layer_residency:
        return ordered

    gpu_only = {
        layer_id
        for layer_id, residency in enumerate(layer_residency)
        if residency == "gpu_only"
    }
    work = []
    for gpu_phase in (True, False):
        for shard, entries in ordered:
            selected = [
                entry
                for entry in entries
                if (int(entry[1].group("layer")) in gpu_only) is gpu_phase
            ]
            if selected:
                work.append((shard, selected))
    return work


def load_dsfp4_expert_sources(
    model_path: str,
    args: DeepseekV4Args,
    *,
    layer_sink=None,
    layer_residency: list[str] | None = None,
    gpu_sink=None,
    routed_tp_widths: tuple[int, ...] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Build pinned CPU DeepSeek-FP4 banks for the routed experts.

    4 banks, one tensor per layer (independent allocations). With
    ``I_local = I / tp_size``: ``gate_up_packed/scale`` are
    ``[E, 2I_local, H//2]`` uint8 / ``[..., H//32]`` e8m0, and
    ``down_packed/scale`` are ``[E, H, I_local//2]`` /
    ``[..., I_local//32]``.

    ``layer_sink=None`` (serving): pin each layer as its writes complete, via an
    internally-owned :class:`PinPipeline`. ``layer_sink`` given (converter): the
    completion tracker fires into it instead -- nothing here is pinned, and the sink
    may release banks it has written out, so the returned tensors are only valid
    until then (the caller owns that tradeoff).
    """
    from freetoken.moe.host_banks import LayerCompletionTracker, alloc_layer_banks

    folder = model_path
    weight_map = _weight_map(folder)
    L, E = args.n_layers, args.n_routed_experts
    I = args.moe_inter_dim
    tp = get_tp_info()

    for shard in sorted(set(weight_map.values())):
        drop_page_cache(os.path.join(folder, shard))

    shards: dict[str, list[tuple[str, re.Match]]] = collections.defaultdict(list)
    for name, shard in weight_map.items():
        m = _EXPERT_RE.match(name)
        if m is None:
            continue
        if int(m.group("layer")) >= L:  # skip the MTP layer (index L)
            continue
        shards[shard].append((name, m))

    # Allocate only this rank's intermediate slice. This is the host-RAM half of
    # expert TP; the matching GPU kernels infer local I from these bank shapes.
    specs = dsfp4_expert_bank_specs(args, routed_tp_widths=routed_tp_widths)
    hb = alloc_layer_banks(specs, L)
    banks = {name: [b.tensor for b in hb[name]] for name in specs}

    def _load(sink) -> int:
        tracker = LayerCompletionTracker(E * 6, hb, sink)  # {w1,w2,w3} x {weight,scale} x experts
        placed = 0
        work = _ordered_expert_shard_work(shards, layer_residency)
        for shard, entries in tqdm(work, desc="Loading DSV4 FP4 experts"):
            path = os.path.join(folder, shard)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for name, m in entries:
                    layer = _place_dsfp4(
                        banks, name, f.get_tensor(name), I,
                        tp_rank=tp.rank, tp_size=tp.size,
                        routed_tp_widths=routed_tp_widths,
                    )
                    tracker.note(layer)
                    placed += 1
            drop_page_cache(path)
        return placed

    placed = _load_with_residency(
        _load, L, layer_sink=layer_sink,
        layer_residency=layer_residency, gpu_sink=gpu_sink,
    )

    expected = L * E * 6  # {w1,w2,w3} x {weight, scale}
    assert placed == expected, f"loaded {placed} expert tensors, expected {expected}"
    return banks


def dummy_dsfp4_expert_sources(
    args: DeepseekV4Args,
    *,
    layer_residency: list[str] | None = None,
    gpu_sink=None,
    routed_tp_widths: tuple[int, ...] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Fabricate the 4 ds_fp4 banks for --dummy-weight (no checkpoint on disk)."""
    from freetoken.moe.host_banks import alloc_layer_banks

    L, E = args.n_layers, args.n_routed_experts
    specs = dsfp4_expert_bank_specs(args, routed_tp_widths=routed_tp_widths)
    hb = alloc_layer_banks(specs, L)
    banks = {name: [b.tensor for b in hb[name]] for name in specs}

    def _finish(sink):
        for layer_id in range(L):
            # Preserve the production loader's peak-RAM behavior: fault and
            # finish one layer at a time so a GPU-only sink can discard it
            # before the next layer is materialized. Scales stay zero (valid
            # e8m0); packed e2m1 payloads are randomized.
            banks["gate_up_packed"][layer_id].random_(0, 256)
            banks["down_packed"][layer_id].random_(0, 256)
            sink(layer_id, {name: per_layer[layer_id] for name, per_layer in hb.items()})

    _load_with_residency(
        _finish, L, layer_sink=None,
        layer_residency=layer_residency, gpu_sink=gpu_sink,
    )
    return banks


def is_expert_tensor(name: str) -> bool:
    """Predicate for the common parallel reader: is this a routed-expert tensor?"""
    return _EXPERT_RE.match(name) is not None


def _place_dsfp4(
    banks: dict,
    name: str,
    t: torch.Tensor,
    I: int,
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    routed_tp_widths: tuple[int, ...] | None = None,
    bank_layer_id: int | None = None,
) -> int:
    """Copy one expert tensor into its layer/expert slot (shared by serial + parallel
    readers): w1->gate_up[:,:I], w3->gate_up[:,I:], w2->down. Returns the layer index."""
    m = _EXPERT_RE.match(name)
    layer, expert = int(m.group("layer")), int(m.group("expert"))
    destination_layer = layer if bank_layer_id is None else bank_layer_id
    proj, kind = m.group("proj"), m.group("kind")
    # The serving bank specs validate the production 256-value tile.  Keep the
    # low-level placement helper geometry-neutral for tiny loader unit fixtures.
    layout = (
        RoutedTpLayout.equal(I, tp_size)
        if routed_tp_widths is None
        else RoutedTpLayout.from_widths(routed_tp_widths, alignment=1)
    )
    if len(layout.widths) != tp_size or layout.total_intermediate_size != I:
        raise ValueError("routed TP layout does not match expert tensor / TP size")
    local_i = layout.widths[tp_rank]
    i0, i1 = layout.offsets[tp_rank], layout.offsets[tp_rank] + local_i
    if kind == "weight":
        t = t.view(torch.uint8)
        if proj == "w1":
            banks["gate_up_packed"][destination_layer][expert, :local_i] = t[i0:i1]
        elif proj == "w3":
            banks["gate_up_packed"][destination_layer][expert, local_i:] = t[i0:i1]
        else:  # w2 -> down
            banks["down_packed"][destination_layer][expert] = t[:, i0 // 2:i1 // 2]
    else:  # scale (e8m0)
        if proj == "w1":
            banks["gate_up_scale"][destination_layer][expert, :local_i] = t[i0:i1]
        elif proj == "w3":
            banks["gate_up_scale"][destination_layer][expert, local_i:] = t[i0:i1]
        else:
            banks["down_scale"][destination_layer][expert] = t[:, i0 // 32:i1 // 32]
    return layer


class Dsfp4DiskExpertSource:
    """Original-safetensors source with one bounded pinned staging layer."""

    def __init__(
        self,
        model_path: str,
        args: DeepseekV4Args,
        *,
        routed_tp_widths: tuple[int, ...],
        pin: bool = True,
    ) -> None:
        from freetoken.moe.host_banks import alloc_banks

        self.args = args
        self.routed_tp_widths = tuple(routed_tp_widths)
        self.num_layers = args.n_layers
        self.num_experts = args.n_routed_experts
        self._tp = get_tp_info()
        specs = dsfp4_expert_bank_specs(
            args, routed_tp_widths=self.routed_tp_widths
        )
        self._host_banks = alloc_banks(specs)
        if pin:
            for bank in self._host_banks.values():
                bank.pin()
        self._banks = {
            name: [bank.tensor] for name, bank in self._host_banks.items()
        }
        # OffloadMoeCache expects one tensor per logical layer. They deliberately
        # alias the same staging storage; load_rows refreshes it before every miss copy.
        self.sources = {
            name: [bank.tensor] * self.num_layers
            for name, bank in self._host_banks.items()
        }
        self.staging_bytes = sum(bank.nbytes for bank in self._host_banks.values())
        self.expert_bytes_per_slot = self.staging_bytes // self.num_experts
        self.disk_refill_calls = 0
        self.disk_refill_experts = 0
        self.disk_refill_bytes = 0
        self.disk_refill_seconds = 0.0
        self._reader = _ShardReader(model_path, _weight_map(model_path), "cpu")

    def load_rows(self, layer_id: int, expert_ids: list[int] | tuple[int, ...]) -> int:
        if not 0 <= layer_id < self.num_layers:
            raise ValueError(f"DSV4 disk source layer {layer_id} is out of range")
        experts = tuple(dict.fromkeys(int(expert) for expert in expert_ids))
        if any(expert < 0 or expert >= self.num_experts for expert in experts):
            raise ValueError(f"DSV4 disk source expert ids are out of range: {experts}")
        started = time.perf_counter()
        placed = 0
        loaded_bytes = 0
        for expert in experts:
            for proj in ("w1", "w2", "w3"):
                for kind in ("weight", "scale"):
                    name = (
                        f"layers.{layer_id}.ffn.experts.{expert}."
                        f"{proj}.{kind}"
                    )
                    if not self._reader.has(name):
                        raise KeyError(f"checkpoint is missing routed expert tensor {name!r}")
                    tensor = self._reader.get(name)
                    loaded_bytes += tensor.numel() * tensor.element_size()
                    _place_dsfp4(
                        self._banks,
                        name,
                        tensor,
                        self.args.moe_inter_dim,
                        tp_rank=self._tp.rank,
                        tp_size=self._tp.size,
                        routed_tp_widths=self.routed_tp_widths,
                        bank_layer_id=0,
                    )
                    placed += 1
        self.disk_refill_calls += 1
        self.disk_refill_experts += len(experts)
        self.disk_refill_bytes += loaded_bytes
        self.disk_refill_seconds += time.perf_counter() - started
        return placed

    def reset_stats(self) -> None:
        self.disk_refill_calls = 0
        self.disk_refill_experts = 0
        self.disk_refill_bytes = 0
        self.disk_refill_seconds = 0.0

    def stats(self) -> dict[str, int | float]:
        return {
            "runtime_expert_disk_reads": self.disk_refill_calls,
            "disk_refill_experts": self.disk_refill_experts,
            "disk_refill_bytes": self.disk_refill_bytes,
            "disk_refill_seconds": self.disk_refill_seconds,
        }

    def close(self) -> None:
        self._reader.close()


def load_dsfp4_expert_sources_parallel(
    model_path: str, args: DeepseekV4Args, *, workers: int = 8, chunk: int = 8 << 20,
    layer_sink=None, layer_residency: list[str] | None = None, gpu_sink=None,
    routed_tp_widths: tuple[int, ...] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """parallel path: same banks as load_dsfp4_expert_sources, filled from the common
    chunked multi-threaded O_DIRECT reader instead of serial per-shard safe_open.
    ``layer_sink``: see :func:`load_dsfp4_expert_sources`."""
    from freetoken.models.weight import iter_expert_tensors_parallel
    from freetoken.moe.host_banks import LayerCompletionTracker, alloc_layer_banks

    L, E = args.n_layers, args.n_routed_experts
    I = args.moe_inter_dim
    tp = get_tp_info()
    specs = dsfp4_expert_bank_specs(args, routed_tp_widths=routed_tp_widths)
    hb = alloc_layer_banks(specs, L)  # lazy host banks (unpinned)
    banks = {name: [b.tensor for b in hb[name]] for name in specs}

    def _is_expert(name: str) -> bool:
        m = _EXPERT_RE.match(name)
        return m is not None and int(m.group("layer")) < L  # skip the MTP layer (index L)

    def _load(sink) -> int:
        tracker = LayerCompletionTracker(E * 6, hb, sink)
        placed = 0
        for name, t in iter_expert_tensors_parallel(model_path, _is_expert, workers=workers, chunk=chunk):
            layer = _place_dsfp4(
                banks, name, t, I, tp_rank=tp.rank, tp_size=tp.size,
                routed_tp_widths=routed_tp_widths,
            )
            tracker.note(layer)
            placed += 1
        return placed

    placed = _load_with_residency(
        _load, L, layer_sink=layer_sink,
        layer_residency=layer_residency, gpu_sink=gpu_sink,
    )

    expected = L * E * 6
    assert placed == expected, f"loaded {placed} expert tensors, expected {expected}"
    return banks


__all__ = [
    "Dsfp4DiskExpertSource",
    "iter_weights",
    "load_dsfp4_expert_sources",
    "load_dsfp4_expert_sources_parallel",
    "dsfp4_expert_bank_specs",
    "is_expert_tensor",
]
