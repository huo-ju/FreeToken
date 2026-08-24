from __future__ import annotations

import ctypes
import os

import pytest
import torch


def test_gpu_only_layer_plan_preserves_dynamic_prefill_floor():
    from freetoken.engine.cache_budget import plan_gpu_only_layers

    # E=8, cache=42: two 8-slot prefill buffers + three complete resident
    # layers, with two extra dynamic LRU slots left over.
    assert plan_gpu_only_layers(
        cache_size=42,
        num_experts=8,
        num_layers=7,
        prefill_overlap=True,
        requested=-1,
    ) == (4, 5, 6)

    with pytest.raises(ValueError, match="exceeds the 3 complete layers"):
        plan_gpu_only_layers(
            cache_size=42,
            num_experts=8,
            num_layers=7,
            prefill_overlap=True,
            requested=4,
        )


def test_gpu_only_layer_plan_can_cover_every_expert():
    from freetoken.engine.cache_budget import plan_gpu_only_layers

    assert plan_gpu_only_layers(
        cache_size=24,
        num_experts=8,
        num_layers=3,
        prefill_overlap=True,
        requested=-1,
    ) == (0, 1, 2)
    assert plan_gpu_only_layers(
        cache_size=24,
        num_experts=8,
        num_layers=3,
        prefill_overlap=True,
        requested=3,
    ) == (0, 1, 2)

    # With the exact same cache, keeping one host-backed layer would require
    # two dynamic prefill buffers and therefore does not fit.
    with pytest.raises(ValueError, match="dynamic cache floor"):
        plan_gpu_only_layers(
            cache_size=24,
            num_experts=8,
            num_layers=3,
            prefill_overlap=True,
            requested=2,
        )


def test_dsv4_tp4_bank_specs_are_quarter_sized(monkeypatch):
    import freetoken.distributed.info as info
    from freetoken.distributed import DistributedInfo
    from freetoken.models.deepseek_v4.args import DeepseekV4Args
    from freetoken.models.deepseek_v4.weight import dsfp4_expert_bank_specs

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank=2, size=4))
    args = DeepseekV4Args(
        n_layers=1,
        n_routed_experts=2,
        dim=64,
        moe_inter_dim=1024,
        compress_ratios=(0,),
    )

    specs = dsfp4_expert_bank_specs(args)

    assert specs["gate_up_packed"][0] == (2, 512, 32)
    assert specs["gate_up_scale"][0] == (2, 512, 2)
    assert specs["down_packed"][0] == (2, 64, 128)
    assert specs["down_scale"][0] == (2, 64, 8)


def test_dsv4_tp4_placement_slices_gate_output_and_down_input():
    from freetoken.models.deepseek_v4.weight import _place_dsfp4

    E, H, I, local_i = 1, 64, 128, 32
    banks = {
        "gate_up_packed": [torch.empty(E, 2 * local_i, H // 2, dtype=torch.uint8)],
        "gate_up_scale": [torch.empty(E, 2 * local_i, H // 32, dtype=torch.uint8)],
        "down_packed": [torch.empty(E, H, local_i // 2, dtype=torch.uint8)],
        "down_scale": [torch.empty(E, H, local_i // 32, dtype=torch.uint8)],
    }
    rank = 2
    row = torch.arange(I, dtype=torch.uint8).view(I, 1).expand(I, H // 2).clone()
    scale = torch.arange(I, dtype=torch.uint8).view(I, 1).expand(I, H // 32).clone()
    down = torch.arange(H * (I // 2), dtype=torch.int64).view(H, I // 2).to(torch.uint8)
    down_scale = torch.arange(H * (I // 32), dtype=torch.int64).view(H, I // 32).to(torch.uint8)

    base = "layers.0.ffn.experts.0"
    _place_dsfp4(banks, f"{base}.w1.weight", row, I, tp_rank=rank, tp_size=4)
    _place_dsfp4(banks, f"{base}.w3.weight", row + 1, I, tp_rank=rank, tp_size=4)
    _place_dsfp4(banks, f"{base}.w1.scale", scale, I, tp_rank=rank, tp_size=4)
    _place_dsfp4(banks, f"{base}.w3.scale", scale + 1, I, tp_rank=rank, tp_size=4)
    _place_dsfp4(banks, f"{base}.w2.weight", down, I, tp_rank=rank, tp_size=4)
    _place_dsfp4(banks, f"{base}.w2.scale", down_scale, I, tp_rank=rank, tp_size=4)

    lo, hi = rank * local_i, (rank + 1) * local_i
    torch.testing.assert_close(banks["gate_up_packed"][0][0, :local_i], row[lo:hi])
    torch.testing.assert_close(banks["gate_up_packed"][0][0, local_i:], row[lo:hi] + 1)
    torch.testing.assert_close(banks["gate_up_scale"][0][0, :local_i], scale[lo:hi])
    torch.testing.assert_close(banks["gate_up_scale"][0][0, local_i:], scale[lo:hi] + 1)
    torch.testing.assert_close(
        banks["down_packed"][0][0], down[:, lo // 2:hi // 2]
    )
    torch.testing.assert_close(
        banks["down_scale"][0][0], down_scale[:, lo // 32:hi // 32]
    )


def test_dsv4_tp4_vocab_and_shared_expert_storage_are_sharded(monkeypatch):
    import freetoken.distributed.info as info
    from freetoken.distributed import DistributedInfo
    from freetoken.models.deepseek_v4.args import DeepseekV4Args
    from freetoken.models.deepseek_v4.model import Transformer

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank=2, size=4))
    args = DeepseekV4Args(
        vocab_size=18,
        dim=128,
        moe_inter_dim=1024,
        n_layers=1,
        n_hash_layers=0,
        n_heads=4,
        q_lora_rank=128,
        head_dim=128,
        rope_head_dim=64,
        o_groups=2,
        o_lora_rank=128,
        index_n_heads=4,
        index_head_dim=128,
        compress_ratios=(0,),
    )
    with torch.device("meta"):
        model = Transformer(args)

    shared = model.layers[0].ffn.shared_experts
    assert model.embed.weight.shape == (5, 128)
    assert model.head.shape == (5, 128)
    assert shared.w1.weight.shape == (256, 128)
    assert shared.w3.weight.shape == (256, 128)
    assert shared.w2.weight.shape == (128, 256)


def test_dsv4_tp_weight_slices_match_vocab_and_shared_intermediate_partitions():
    from freetoken.models.deepseek_v4.weight import (
        _shared_expert_tp_shard,
        _vocab_tp_shard,
    )

    vocab = torch.arange(18 * 2).view(18, 2)
    vocab_rank3 = _vocab_tp_shard(vocab, 18, tp_rank=3, tp_size=4)
    torch.testing.assert_close(vocab_rank3[:3], vocab[15:18])
    assert vocab_rank3[3:].count_nonzero() == 0

    I, H, rank = 1024, 8, 2
    row_weight = torch.arange(I * H).view(I, H)
    row_scale = torch.arange((I // 128) * (H // 128 or 1)).view(I // 128, -1)
    down_weight = torch.arange(H * I).view(H, I)
    down_scale = torch.arange((H // 128 or 1) * (I // 128)).view(-1, I // 128)

    torch.testing.assert_close(
        _shared_expert_tp_shard(
            row_weight, "w1", "weight", I, tp_rank=rank, tp_size=4
        ),
        row_weight[512:768],
    )
    torch.testing.assert_close(
        _shared_expert_tp_shard(
            row_scale, "w3", "scale", I, tp_rank=rank, tp_size=4
        ),
        row_scale[4:6],
    )
    torch.testing.assert_close(
        _shared_expert_tp_shard(
            down_weight, "w2", "weight", I, tp_rank=rank, tp_size=4
        ),
        down_weight[:, 512:768],
    )
    torch.testing.assert_close(
        _shared_expert_tp_shard(
            down_scale, "w2", "scale", I, tp_rank=rank, tp_size=4
        ),
        down_scale[:, 4:6],
    )


def test_dsv4_tp4_shared_expert_partials_sum_to_unsharded_output():
    torch.manual_seed(17)
    x = torch.randn(3, 16)
    w1 = torch.randn(32, 16)
    w3 = torch.randn(32, 16)
    w2 = torch.randn(16, 32)

    full = torch.nn.functional.silu(x @ w1.T) * (x @ w3.T) @ w2.T
    partials = []
    for rank in range(4):
        lo, hi = rank * 8, (rank + 1) * 8
        partials.append(
            (torch.nn.functional.silu(x @ w1[lo:hi].T) * (x @ w3[lo:hi].T))
            @ w2[:, lo:hi].T
        )

    torch.testing.assert_close(torch.stack(partials).sum(0), full)


def test_dsv4_serial_load_orders_gpu_only_layers_before_pinned_layers():
    from freetoken.models.deepseek_v4.weight import (
        _EXPERT_RE,
        _ordered_expert_shard_work,
    )

    def entry(layer: int, expert: int = 0):
        name = f"layers.{layer}.ffn.experts.{expert}.w1.weight"
        match = _EXPERT_RE.match(name)
        assert match is not None
        return name, match

    # The boundary shard intentionally mixes both classes. It must be reopened
    # so its GPU-only tensor is handled in phase one and its pinned tensor only
    # after every GPU-only work item has been staged.
    shards = {
        "model-00001.safetensors": [entry(0)],
        "model-00002.safetensors": [entry(1), entry(3)],
        "model-00003.safetensors": [entry(2)],
    }
    residency = ["pinned", "pinned", "gpu_only", "gpu_only"]

    work = _ordered_expert_shard_work(shards, residency)
    layers = [int(match.group("layer")) for _, entries in work for _, match in entries]

    assert layers == [3, 2, 0, 1]
    assert [shard for shard, _ in work] == [
        "model-00002.safetensors",
        "model-00003.safetensors",
        "model-00001.safetensors",
        "model-00002.safetensors",
    ]


@pytest.mark.skipif(not hasattr(os, "sysconf"), reason="needs Unix mincore")
def test_gpu_only_host_bank_release_drops_resident_private_pages():
    from freetoken.moe.host_banks import HostBank

    bank = HostBank((8 << 20,), torch.uint8)
    bank.tensor.fill_(0x5A)

    page = os.sysconf("SC_PAGE_SIZE")
    pages = (bank.nbytes + page - 1) // page
    libc = ctypes.CDLL(None, use_errno=True)

    def resident_pages() -> int:
        vec = (ctypes.c_ubyte * pages)()
        rc = libc.mincore(
            ctypes.c_void_p(bank.addr), ctypes.c_size_t(bank.nbytes), vec
        )
        if rc != 0:
            errno = ctypes.get_errno()
            pytest.skip(f"mincore unavailable: errno={errno}")
        return sum(value & 1 for value in vec)

    before = resident_pages()
    assert before >= pages * 0.9
    bank.release()
    assert resident_pages() <= pages * 0.1


def test_gpu_only_cache_drops_host_reference_and_keeps_protected_mapping():
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    E, L = 2, 3
    specs = {
        "gate_up": ((E, 4, 3), torch.float16),
        "down": ((E, 3, 2), torch.float16),
    }
    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=4,
        device=torch.device("cpu"),
        prefill_overlap=False,
    )
    cache.prepare_gpu_resident_layers(specs, [2])
    resident = {
        "gate_up": torch.arange(E * 4 * 3, dtype=torch.float16).view(E, 4, 3),
        "down": torch.arange(E * 3 * 2, dtype=torch.float16).view(E, 3, 2),
    }
    cache.stage_gpu_resident_layer(2, resident)
    sources = {
        name: [torch.zeros(shape, dtype=dtype) for _ in range(L)]
        for name, (shape, dtype) in specs.items()
    }
    sources["gate_up"][2].copy_(resident["gate_up"])
    sources["down"][2].copy_(resident["down"])
    cache.set_bank_sources(
        sources,
        layer_residency=[
            HostResidency.PINNED.value,
            HostResidency.PINNED.value,
            HostResidency.GPU_ONLY.value,
        ],
    )

    assert cache.dynamic_cache_size == E
    assert cache.bank_sources["gate_up"][2] is None
    assert cache.slot_for_id[2].tolist() == [2, 3]
    assert cache.id_of_slot.tolist() == [-1, -1, 4, 5]
    assert cache.usage[2:].tolist() == [torch.iinfo(torch.int64).max] * E
    views = cache.gpu_resident_views(2)
    torch.testing.assert_close(views[0], resident["gate_up"])
    torch.testing.assert_close(views[1], resident["down"])

    ids = torch.tensor([[1, 0]], dtype=torch.int32)
    cache.map_gpu_resident_experts(2, ids)
    assert ids.tolist() == [[3, 2]]
    assert cache.usage[2:].tolist() == [torch.iinfo(torch.int64).max] * E

    # Normal admissions may recycle only the dynamic prefix. The permanent
    # max-int usage sentinel keeps the hostless layer out of the victim set.
    from freetoken.moe.offload_kernels import _ensure_experts_hybrid_cpu

    dynamic_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    _ensure_experts_hybrid_cpu(cache, 0, dynamic_ids, max_fetch=2, frac_q16=0)
    assert sorted(dynamic_ids.reshape(-1).tolist()) == [0, 1]
    assert cache.slot_for_id[2].tolist() == [2, 3]
    assert cache.id_of_slot[2:].tolist() == [4, 5]
    assert cache.usage[2:].tolist() == [torch.iinfo(torch.int64).max] * E

    with pytest.raises(ValueError, match="cannot be changed"):
        cache.validate_rebuild(5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_flashlib_lru_does_not_evict_gpu_only_slots():
    """Validate the max-int sentinel against the production GPU LRU kernel."""
    from freetoken.moe.offload_cache import OffloadMoeCache

    E, L = 2, 3
    specs = {
        "gate_up": ((E, 4, 4), torch.float16),
        "down": ((E, 4, 2), torch.float16),
    }
    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=4,
        device=torch.device("cuda:0"),
        prefill_overlap=False,
    )
    cache.prepare_gpu_resident_layers(specs, [2])
    # The admission kernel only needs the maps; staging/data movement is covered
    # separately by the cache residency test above.
    cache._gpu_resident_loaded.add(2)
    cache._restore_gpu_resident_mappings()

    for layer_id in (0, 1):
        ids = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda:0")
        cache.ensure_experts(layer_id, ids)
        torch.cuda.synchronize()
        assert sorted(ids.cpu().reshape(-1).tolist()) == [0, 1]
        assert cache.slot_for_id[2].cpu().tolist() == [2, 3]
        assert cache.id_of_slot[2:].cpu().tolist() == [4, 5]
        assert cache.usage[2:].cpu().tolist() == [torch.iinfo(torch.int64).max] * E


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_dsv4_tp4_partial_expert_outputs_sum_to_unsharded():
    """Exercise the TP=4 local-intermediate geometry.

    A SwiGLU intermediate shard is independent until the down projection, so
    summing four local routed outputs must reproduce one unsharded expert. This
    pins both the gate/up row split and down input-column split used by the
    checkpoint loader.
    """
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

    torch.manual_seed(11)
    device = torch.device("cuda:0")
    E, H, I, top_k = 2, 256, 2048, 2

    def packed(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device=device)

    def scales(*shape):
        return torch.full(shape, 121, dtype=torch.uint8, device=device).view(
            torch.float8_e8m0fnu
        )

    full = (
        packed(E, 2 * I, H // 2),
        scales(E, 2 * I, H // 32),
        packed(E, H, I // 2),
        scales(E, H, I // 32),
    )
    x = torch.randn(3, H, dtype=torch.bfloat16, device=device) * 0.1
    ids = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int32, device=device)
    weights = torch.rand(3, top_k, dtype=torch.float32, device=device)

    unsharded = routed_experts_fp4(
        x.clone(), ids.clone(), weights, *full, swiglu_limit=10.0
    ).float()
    partials = []
    local_i = I // 4
    gate_scale_u8 = full[1].view(torch.uint8)
    down_scale_u8 = full[3].view(torch.uint8)
    for rank in range(4):
        lo, hi = rank * local_i, (rank + 1) * local_i
        local = (
            torch.cat(
                (full[0][:, lo:hi], full[0][:, I + lo:I + hi]), dim=1
            ).contiguous(),
            torch.cat(
                (
                    gate_scale_u8[:, lo:hi],
                    gate_scale_u8[:, I + lo:I + hi],
                ),
                dim=1,
            ).contiguous().view(torch.float8_e8m0fnu),
            full[2][:, :, lo // 2:hi // 2].contiguous(),
            down_scale_u8[:, :, lo // 32:hi // 32]
            .contiguous()
            .view(torch.float8_e8m0fnu),
        )
        partials.append(
            routed_experts_fp4(
                x.clone(), ids.clone(), weights, *local, swiglu_limit=10.0
            ).float()
        )

    reduced = torch.stack(partials).sum(0)
    torch.testing.assert_close(reduced, unsharded, rtol=2e-2, atol=2e-2)
