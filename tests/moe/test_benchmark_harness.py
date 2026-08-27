from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_PATH = Path(__file__).parents[2] / "benchmarks" / "bench_decode_moe.py"
_SPEC = importlib.util.spec_from_file_location("bench_decode_moe", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_BENCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCH)


def test_eval_plan_case_can_supply_a_reproducible_prompt(tmp_path):
    plan = {
        "warmup": {"id": "warmup", "prompt": "warm"},
        "decode": [{"id": "decode_short", "prompt": "decode prompt", "expected": "ok"}],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    assert _BENCH.load_problem(
        None, 0, eval_plan=str(path), case_id="decode_short"
    ) == ("decode prompt", "ok")


def test_tp4_server_command_preserves_graph_and_cache_configuration():
    args = _BENCH.parse_args([
        "--model", "/model", "--tp-size", "4", "--cache", "512",
        "--gpu", "0,1,2,3", "--no-prefill-overlap", "--repetitions", "3",
    ])
    command = _BENCH.serve_cmd(args, "offload", 19000)
    assert command[command.index("--tp-size") + 1] == "4"
    assert command[command.index("--distributed-timeout") + 1] == "86400"
    assert command[command.index("--moe-cache-size") + 1] == "512"
    assert command[command.index("--gpu") + 1] == "0,1,2,3"
    assert "--disable-moe-prefill-overlap" in command


def test_server_readiness_has_no_default_deadline(monkeypatch):
    args = _BENCH.parse_args(["--model", "/model"])
    assert args.server_timeout == 0

    class Proc:
        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(
        _BENCH, "get_json", lambda *_args, **_kwargs: {"maintenance": "serving"}
    )
    _BENCH.wait_ready("http://localhost", Proc(), "/unused", args.server_timeout)


def test_topology_uses_upstream_gpu_uuid_order(monkeypatch):
    from freetoken import gpu_select

    monkeypatch.setattr(
        gpu_select,
        "resolve_gpu_uuids",
        lambda specs: ("GPU-b", "GPU-a"),
    )
    monkeypatch.setattr(
        _BENCH.subprocess,
        "check_output",
        lambda *args, **kwargs: (
            "0, GPU-a, A, 00000000:01:00.0, 3, 16, 8192, 600.0\n"
            "1, GPU-b, B, 00000000:02:00.0, 4, 16, 16384, 600.0\n"
        ),
    )

    rows = _BENCH._gpu_topology("1,0", 2)

    assert [row["uuid"] for row in rows] == ["GPU-b", "GPU-a"]
    assert [row["rank"] for row in rows] == [0, 1]


def test_cache_sequence_starts_largest_geometry_with_fixed_kv_capacity():
    args = _BENCH.parse_args([
        "--model", "/model", "--tp-size", "1",
        "--cache-sequence", "768", "512", "256", "--num-tokens", "16384",
    ])
    command = _BENCH.serve_cmd(args, "offload", 19000)
    assert command[command.index("--moe-cache-size") + 1] == "768"
    assert command[command.index("--num-tokens") + 1] == "16384"


def test_cache_sequence_rejects_runtime_rebuild_under_tp4():
    with pytest.raises(SystemExit):
        _BENCH.parse_args([
            "--model", "/model", "--tp-size", "4",
            "--cache-sequence", "768", "512",
        ])


def test_weighted_tp_benchmark_flags_reach_server_before_model_startup():
    args = _BENCH.parse_args([
        "--model", "/model", "--tp-size", "4", "--backend", "offload",
        "--execution-policy", "force_weighted_tp",
        "--moe-tp-layout", "512,512,768,256",
    ])
    command = _BENCH.serve_cmd(args, "offload", 19000)
    assert command[command.index("--moe-execution-policy") + 1] == "force_weighted_tp"
    assert command[command.index("--moe-tp-layout") + 1] == "512,512,768,256"


def test_weighted_tp_benchmark_rejects_missing_layout_without_starting_server():
    with pytest.raises(SystemExit):
        _BENCH.parse_args([
            "--model", "/model", "--tp-size", "4",
            "--execution-policy", "force_weighted_tp",
        ])


def test_disk_backed_benchmark_reaches_server_only_in_eager_mode():
    args = _BENCH.parse_args([
        "--model", "/model", "--tp-size", "4", "--backend", "offload",
        "--execution-policy", "force_weighted_tp",
        "--moe-tp-layout", "512,512,768,256",
        "--moe-disk-backed", "--no-graph", "--no-prefill-overlap",
    ])
    command = _BENCH.serve_cmd(args, "offload", 19000)
    assert "--moe-disk-backed" in command
    assert command[command.index("--cuda-graph-max-bs") + 1] == "0"

    equal = _BENCH.parse_args([
        "--model", "/model", "--tp-size", "4", "--backend", "offload",
        "--execution-policy", "force_equal_tp",
        "--moe-disk-backed", "--no-graph", "--no-prefill-overlap",
    ])
    assert "--moe-disk-backed" in _BENCH.serve_cmd(equal, "offload", 19000)

    with pytest.raises(SystemExit):
        _BENCH.parse_args([
            "--model", "/model", "--tp-size", "4", "--backend", "offload",
            "--execution-policy", "force_weighted_tp",
            "--moe-tp-layout", "512,512,768,256",
            "--moe-disk-backed",
        ])
    with pytest.raises(SystemExit):
        _BENCH.parse_args([
            "--model", "/model", "--tp-size", "4", "--backend", "offload",
            "--execution-policy", "force_weighted_tp",
            "--moe-tp-layout", "512,512,768,256",
            "--moe-disk-backed", "--no-graph",
        ])


def test_benchmark_extracts_placement_and_runtime_storage_telemetry(tmp_path):
    log = tmp_path / "serve.log"
    log.write_text(
        "MoE authoritative placement (checksum abcdef012345)\n"
        "  mode:                VRAM+RAM (runtime disk disabled)\n"
        "  KV:                  fixed 128 pages / 16384 tokens\n"
        "MoE disk refill telemetry: reads=3 experts=7 bytes=4096 seconds=0.125000\n"
    )

    metadata = _BENCH._placement_log_metadata(str(log))

    assert metadata["placement_checksum"] == "abcdef012345"
    assert metadata["placement_mode"] == "VRAM+RAM (runtime disk disabled)"
    assert metadata["kv_plan"] == "fixed 128 pages / 16384 tokens"
    assert metadata["runtime_expert_disk_reads"] == 3
    assert metadata["disk_refill_experts"] == 7
    assert metadata["disk_refill_bytes"] == 4096
    assert metadata["disk_refill_seconds"] == 0.125
