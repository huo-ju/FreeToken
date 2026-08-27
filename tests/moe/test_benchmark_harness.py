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
        "--no-prefill-overlap", "--repetitions", "3",
    ])
    command = _BENCH.serve_cmd(args, "offload", 19000)
    assert command[command.index("--tp-size") + 1] == "4"
    assert command[command.index("--distributed-timeout") + 1] == "86400"
    assert command[command.index("--moe-cache-size") + 1] == "512"
    assert "--disable-moe-prefill-overlap" in command


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
