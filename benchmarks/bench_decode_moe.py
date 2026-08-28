"""Single-stream (bs=1) decode benchmark for any MoE model on any offload backend.

Measures through the real serving path: for each backend the bench spawns ``ft serve``,
sends a warmed chat request over /v1/chat/completions with ``stream=true``, and
timestamps every SSE event as it arrives. Numbers therefore include the scheduler,
detokenizer, and HTTP/SSE hop -- what a client actually sees -- not bare engine forwards.

Method -- at bs=1 the server emits one delta event per decode step, and the final chunk
(``stream_options.include_usage``) reports exact token counts, so

    decode_tok_s = (completion_tokens - 1) / (t_last_event - t_first_event)

which stays correct even when the detokenizer coalesces a few tokens into one event
(multibyte characters): the window is still anchored on the first and last token's
arrival. ``ignore_eos`` keeps the step count at exactly ``D`` regardless of sampling.
TTFT is the measured run's warm first-token latency (template rendering + prefill
included). With ``--moe-telemetry``, graph-safe per-rank/per-layer routing and copy
counters are persisted from the server's live /v1/stats document.

Prompt: an AIME-25 problem sent as a chat message with thinking enabled -- a real
reasoning workload, so expert routing is representative. The server renders the chat
template (including checkpoint-shipped encoders like DSV4's ``encoding_dsv4.py``). The
problems come from the ``math-ai/aime25`` dataset on the Hub, downloaded into the usual
HF cache on first run; ``--aime`` points at a local jsonl instead.

Sampling: the checkpoint's recommended params (``generation_config.json``), falling back
to temperature 1.0 / top_p 0.95 / top_k 64 for fields the checkpoint does not specify --
resolved here and sent explicitly, because the server's own unspecified-field defaults
are greedy and would silently degrade the routing workload for checkpoints without a
full sampling recommendation. The generated text is per-server-process deterministic
(fresh server, fixed request sequence), so one text sha1 per backend is a real
cross-backend check; token ids are not visible over the API, so this is a weaker
invariant than the old in-process id hash. ``--greedy`` sends temperature 0 for the
stricter comparison.

Run (one backend):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python benchmarks/bench_decode_moe.py \
        --model /path/to/model

Run (all three backends, one server per backend):
    ... --model /path/to/model --backend offload,cpu,hybrid --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# Applied for every field the checkpoint's generation_config.json does not specify.
FALLBACK_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}

# AIME-25 problems, pulled from the Hub into the usual HF cache on first run.
AIME_REPO = "math-ai/aime25"
AIME_FILE = "test.jsonl"
# Reasoning models need the answer format spelled out; the boxed answer is also what makes
# a run spot-checkable by eye.
BOXED_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="checkpoint dir (or .ftw)")
    p.add_argument(
        "--backend",
        default="offload",
        help="comma list of offload|cpu|hybrid; one server per backend",
    )
    p.add_argument(
        "--aime",
        default=os.environ.get("FREETOKEN_AIME25_JSONL"),
        help=f"local jsonl instead of downloading {AIME_REPO}; default $FREETOKEN_AIME25_JSONL",
    )
    p.add_argument("--problem", type=int, default=0, help="0-based AIME problem index")
    p.add_argument(
        "--eval-plan",
        default=None,
        help="JSON test plan containing warmup/prefill/decode/quality cases instead of AIME",
    )
    p.add_argument(
        "--case-id",
        default="decode_64_r0",
        help="case id selected from --eval-plan",
    )
    p.add_argument("--decode", type=int, default=256, help="decode tokens to measure (D)")
    p.add_argument("--tp-size", type=int, default=1, help="tensor-parallel server ranks")
    p.add_argument(
        "--execution-policy",
        choices=("compatibility", "force_equal_tp", "force_weighted_tp"),
        default="compatibility",
        help="experimental startup-fixed MoE execution policy",
    )
    p.add_argument(
        "--moe-tp-layout",
        default="equal",
        help="routed intermediate widths in TP-rank order, or equal",
    )
    p.add_argument(
        "--moe-disk-backed",
        action="store_true",
        help=(
            "run the separately labelled bounded-memory DSV4 emergency path; runtime "
            "checkpoint reads make it ineligible for the normal P3 baseline"
        ),
    )
    p.add_argument(
        "--repetitions", type=int, default=3,
        help="steady-state measured requests per server (default: 3)",
    )
    p.add_argument(
        "--cache",
        type=int,
        default=0,
        help="GPU expert cache slots; 0 = auto-size from free VRAM",
    )
    p.add_argument(
        "--cache-sequence",
        type=int,
        nargs="+",
        default=None,
        help="measure several cache sizes in one server using safe runtime rebuilds",
    )
    p.add_argument("--cache-rate", type=float, default=None, help="cache slots as a fraction of L*E")
    p.add_argument(
        "--hybrid-fetch",
        type=int,
        default=-1,
        help="hybrid: max PCIe fetches/layer; -1 = auto (benched pcie/cpu bandwidth fraction)",
    )
    p.add_argument("--mem-ratio", type=float, default=0.9, help="target VRAM utilization")
    p.add_argument(
        "--moe-host-budget-gb",
        type=float,
        default=None,
        help="freeze aggregate expert host budget instead of using the startup snapshot",
    )
    p.add_argument(
        "--gpu",
        default=None,
        help="GPU for the serve: a UUID or nvidia-smi index (as ft serve --gpu)",
    )
    p.add_argument(
        "--num-tokens", type=int, default=None,
        help="fixed KV capacity in tokens (passed as --num-tokens)",
    )
    p.add_argument(
        "--expert-load", choices=("auto", "serial", "parallel"), default="auto",
        help="expert-bank checkpoint loading mode passed to the server",
    )
    p.add_argument("--no-graph", action="store_true", help="eager decode instead of CUDA graph")
    p.add_argument(
        "--no-prefill-overlap", action="store_true",
        help="disable the MoE prefill copy double buffer",
    )
    p.add_argument(
        "--moe-telemetry", action="store_true",
        help="enable per-rank/per-layer MoE counters and persist them in benchmark JSON",
    )
    p.add_argument(
        "--moe-timing", action="store_true",
        help=(
            "record fixed CUDA-event MoE component timings and persist one snapshot "
            "per measured repetition (diagnostic profile, not a scoreable baseline)"
        ),
    )
    p.add_argument(
        "--greedy",
        action="store_true",
        help="force temperature 0 (ignore the checkpoint's sampling) so ids are comparable",
    )
    p.add_argument(
        "--server-timeout",
        type=float,
        default=0,
        help="seconds to wait for server readiness; 0 disables the deadline (default: 0)",
    )
    p.add_argument(
        "--distributed-timeout",
        type=float,
        default=86400,
        help="seconds TP collectives may wait during asymmetric model loading (default: 24h)",
    )
    p.add_argument("--json", dest="json_out", default=None, help="append the result rows here")
    args = p.parse_args(argv)
    if args.tp_size < 1 or args.repetitions < 1:
        p.error("--tp-size and --repetitions must be positive")
    if args.distributed_timeout <= 0:
        p.error("--distributed-timeout must be positive")
    backends = tuple(value.strip() for value in args.backend.split(","))
    if args.execution_policy == "force_weighted_tp":
        if args.moe_tp_layout == "equal":
            p.error("force_weighted_tp requires an explicit --moe-tp-layout")
        if any(backend != "offload" for backend in backends):
            p.error("force_weighted_tp benchmark rows require --backend offload")
    elif args.moe_tp_layout != "equal":
        p.error("an explicit --moe-tp-layout requires force_weighted_tp")
    if args.execution_policy == "force_equal_tp" and any(
        backend != "offload" for backend in backends
    ):
        p.error("force_equal_tp benchmark rows require --backend offload")
    if args.moe_disk_backed:
        if args.execution_policy not in ("force_equal_tp", "force_weighted_tp"):
            p.error("--moe-disk-backed requires a forced TP execution policy")
        if not args.no_graph:
            p.error("--moe-disk-backed requires --no-graph")
        if not args.no_prefill_overlap:
            p.error("--moe-disk-backed requires --no-prefill-overlap")
    if args.cache_sequence is not None:
        if any(size <= 0 for size in args.cache_sequence):
            p.error("--cache-sequence values must be positive")
        if args.cache > 0 or args.cache_rate is not None:
            p.error("--cache-sequence cannot be combined with --cache or --cache-rate")
        if args.tp_size > 1 and len(args.cache_sequence) > 1:
            p.error(
                "runtime cache rebuild is unsupported under TP > 1; run one "
                "--cache value per benchmark invocation"
            )
    if args.num_tokens is not None and args.num_tokens <= 0:
        p.error("--num-tokens must be positive")
    return args


def load_problem(
    path: str | None,
    index: int,
    *,
    eval_plan: str | None = None,
    case_id: str = "decode_64_r0",
) -> tuple[str, str]:
    """One AIME-25 (problem, answer). Downloads the dataset unless ``path`` overrides it.

    Accepts both the Hub schema (``problem``) and the pre-formatted jsonl some local copies
    use (``prompt``, answer instruction already appended)."""
    if eval_plan is not None:
        plan = json.loads(Path(eval_plan).read_text())
        cases = []
        for section in ("warmup", "decode_warmup", "prefill", "decode", "quality"):
            value = plan.get(section, [])
            cases.extend(value if isinstance(value, list) else [value])
        matches = [case for case in cases if isinstance(case, dict) and case.get("id") == case_id]
        if len(matches) != 1:
            sys.exit(
                f"--case-id {case_id!r} matched {len(matches)} cases in {eval_plan}; expected one"
            )
        case = matches[0]
        return case["prompt"], str(case.get("expected", ""))
    if not path:
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(AIME_REPO, AIME_FILE, repo_type="dataset")
        except Exception as e:  # offline, rate-limited, repo moved
            sys.exit(f"could not fetch {AIME_REPO}/{AIME_FILE} ({e}); pass --aime <local jsonl>")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not 0 <= index < len(rows):
        sys.exit(f"--problem {index} out of range ({len(rows)} problems available)")
    row = rows[index]
    text = row.get("problem") or row["prompt"]
    if "boxed" not in text:
        text = f"{text}\n{BOXED_INSTRUCTION}"
    return text, str(row.get("answer", ""))


def resolve_sampling(model_path: str, greedy: bool) -> tuple[dict, str]:
    """Checkpoint-recommended sampling with per-field fallback; returns (params, source).

    Resolved client-side and sent explicitly: the server fills unspecified fields with
    its framework defaults (temperature 0 / no filtering), not with these fallbacks."""
    if greedy:
        return {"temperature": 0.0, "top_p": 1.0, "top_k": -1}, "greedy (--greedy)"
    recommended: dict = {}
    cfg = Path(model_path) / "generation_config.json"
    if cfg.is_file():
        raw = json.loads(cfg.read_text())
        recommended = {k: raw[k] for k in FALLBACK_SAMPLING if raw.get(k) is not None}
        if raw.get("do_sample") is False or recommended.get("temperature") == 0.0:
            return {"temperature": 0.0, "top_p": 1.0, "top_k": -1}, "checkpoint (greedy)"
    params = {**FALLBACK_SAMPLING, **recommended}
    if params["top_k"] == 0:
        params["top_k"] = -1  # HF spells "no top-k filtering" as 0; the API as -1
    taken = sorted(recommended)
    source = f"checkpoint{taken} + fallback" if taken else "fallback (no generation_config)"
    return params, source


def get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def post_json(url: str, payload: dict, timeout: float = 300) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        # Cache rebuild intentionally uses non-2xx responses for recoverable
        # rejections. Preserve the JSON reason instead of replacing it with the
        # unhelpful generic ``HTTP Error 503`` exception.
        try:
            detail = json.loads(error.read())
        except (json.JSONDecodeError, UnicodeDecodeError):
            detail = {"error": str(error)}
        raise RuntimeError(
            f"POST {url} failed with HTTP {error.code}: {detail}"
        ) from error


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve_cmd(args: argparse.Namespace, backend: str, port: int) -> list[str]:
    cmd = [
        sys.executable, "-m", "freetoken.cli", "serve",
        "--model", args.model,
        "--host", "127.0.0.1", "--port", str(port),
        "--moe-backend", backend,
        "--moe-execution-policy", args.execution_policy,
        "--moe-tp-layout", args.moe_tp_layout,
        "--tp-size", str(args.tp_size),
        "--distributed-timeout", str(args.distributed_timeout),
        "--max-running-requests", "1",
        "--max-seq-len-override", str(8192 + args.decode),
        "--memory-ratio", str(args.mem_ratio),
        "--expert-load", args.expert_load,
        "--cuda-graph-max-bs", "0" if args.no_graph else "1",
        "--moe-hybrid-max-fetch", str(args.hybrid_fetch),
    ]
    if args.gpu:
        cmd += ["--gpu", args.gpu]
    if args.moe_host_budget_gb is not None:
        cmd += ["--moe-host-budget-gb", str(args.moe_host_budget_gb)]
    if args.moe_disk_backed:
        cmd.append("--moe-disk-backed")
    if args.no_prefill_overlap:
        cmd.append("--disable-moe-prefill-overlap")
    if args.moe_telemetry or args.moe_timing:
        cmd.append("--moe-collect-stats")
    if args.moe_timing:
        cmd.append("--moe-collect-timing")
    if args.num_tokens is not None:
        cmd += ["--num-tokens", str(args.num_tokens)]
    if args.cache_sequence is not None:
        cmd += ["--moe-cache-size", str(args.cache_sequence[0])]
    elif args.cache > 0:
        cmd += ["--moe-cache-size", str(args.cache)]
    elif args.cache_rate is not None:
        cmd += ["--moe-cache-rate", str(args.cache_rate)]
    else:
        cmd.append("--moe-cache-auto")
    return cmd


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _git_worktree_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _placement_log_metadata(log_path: str) -> dict:
    """Extract durable placement/storage facts from the completed server log."""
    try:
        log = Path(log_path).read_text(errors="replace")
    except OSError:
        return {}
    checksum = re.search(r"MoE authoritative placement \(checksum ([0-9a-f]+)\)", log)
    mode = re.search(r"^\s*mode:\s*(.+)$", log, flags=re.MULTILINE)
    kv = re.search(r"^\s*KV:\s*(.+)$", log, flags=re.MULTILINE)
    disk = re.search(
        r"MoE disk refill telemetry: reads=(\d+) experts=(\d+) bytes=(\d+) "
        r"seconds=([0-9.]+)",
        log,
    )
    result = {
        "placement_checksum": checksum.group(1) if checksum else None,
        "placement_mode": mode.group(1).strip() if mode else None,
        "kv_plan": kv.group(1).strip() if kv else None,
        "runtime_expert_disk_reads": 0,
        "disk_refill_experts": 0,
        "disk_refill_bytes": 0,
        "disk_refill_seconds": 0.0,
    }
    if disk:
        result.update(
            runtime_expert_disk_reads=int(disk.group(1)),
            disk_refill_experts=int(disk.group(2)),
            disk_refill_bytes=int(disk.group(3)),
            disk_refill_seconds=float(disk.group(4)),
        )
    return result


def _meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemAvailable", "Mlocked", "SwapFree"}:
                result[f"{key.lower()}_bytes"] = int(value.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return result


def _gpu_topology(gpu_spec: str | None = None, tp_size: int = 1) -> list[dict]:
    """Best-effort rank identity in the upstream server's ``--gpu`` order."""
    fields = (
        "index,uuid,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,"
        "memory.total,driver_version"
    )
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    from freetoken.gpu_select import parse_gpu_spec, resolve_gpu_uuids

    try:
        requested = (
            parse_gpu_spec(gpu_spec)
            if gpu_spec
            else tuple(str(rank) for rank in range(tp_size))
        )
        selected = resolve_gpu_uuids(requested) or requested
    except ValueError:
        return []

    rows = []
    for line in raw.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 8:
            continue
        bdf_parts = values[3].lower().split(":")
        bdf = (
            f"{bdf_parts[-3][-4:]}:{bdf_parts[-2]}:{bdf_parts[-1]}"
            if len(bdf_parts) >= 3 else values[3].lower()
        )
        numa_path = Path("/sys/bus/pci/devices") / bdf / "numa_node"
        try:
            numa = int(numa_path.read_text().strip())
        except (OSError, ValueError):
            numa = -1
        rows.append({
            "index": int(values[0]),
            "uuid": values[1],
            "name": values[2],
            "pci_bdf": bdf,
            "pcie_generation": int(values[4]), "pcie_width": int(values[5]),
            "memory_total_mib": int(values[6]),
            "driver_version": values[7],
            "numa_node": numa,
        })

    ordered = []
    for rank, spec in enumerate(selected):
        if spec.upper().startswith("GPU-"):
            matches = [row for row in rows if row["uuid"].upper().startswith(spec.upper())]
        else:
            matches = [row for row in rows if row["index"] == int(spec)]
        if len(matches) == 1:
            ordered.append({**matches[0], "rank": rank})
    return ordered


class ResourceMonitor:
    """Low-frequency host/link sampler for startup and request peak accounting."""

    def __init__(
        self, interval_s: float = 1.0, *, gpu_spec: str | None = None, tp_size: int = 1
    ):
        self.interval_s = interval_s
        self.gpu_spec = gpu_spec
        self.tp_size = tp_size
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples = 0
        self.min_mem_available = 2**63 - 1
        self.max_mlocked = 0
        self.min_swap_free = 2**63 - 1
        self.gpus: dict[int, dict] = {}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            host = _meminfo()
            self.min_mem_available = min(
                self.min_mem_available, host.get("memavailable_bytes", self.min_mem_available)
            )
            self.max_mlocked = max(self.max_mlocked, host.get("mlocked_bytes", 0))
            self.min_swap_free = min(
                self.min_swap_free, host.get("swapfree_bytes", self.min_swap_free)
            )
            for gpu in _gpu_topology(self.gpu_spec, self.tp_size):
                peak = self.gpus.setdefault(gpu["rank"], dict(gpu))
                peak["pcie_generation"] = max(
                    peak.get("pcie_generation", 0), gpu["pcie_generation"]
                )
                peak["pcie_width"] = max(peak.get("pcie_width", 0), gpu["pcie_width"])
            self.samples += 1
            self._stop.wait(self.interval_s)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 2))
        return {
            "samples": self.samples,
            "min_mem_available_bytes": (
                None if self.min_mem_available == 2**63 - 1 else self.min_mem_available
            ),
            "max_mlocked_bytes": self.max_mlocked,
            "min_swap_free_bytes": (
                None if self.min_swap_free == 2**63 - 1 else self.min_swap_free
            ),
            "gpu_topology_peak_link": [self.gpus[rank] for rank in sorted(self.gpus)],
        }


def die_with_log(msg: str, log_path: str) -> None:
    tail = "".join(Path(log_path).read_text().splitlines(keepends=True)[-30:])
    sys.exit(f"[bench] {msg}\n[bench] server log tail ({log_path}):\n{tail}")


def wait_ready(origin: str, proc: subprocess.Popen, log_path: str, timeout: float) -> None:
    deadline = None if timeout <= 0 else time.monotonic() + timeout
    while deadline is None or time.monotonic() < deadline:
        if proc.poll() is not None:
            die_with_log(f"server exited with code {proc.returncode} during startup", log_path)
        try:
            health = get_json(f"{origin}/health", timeout=5)
        except (OSError, ValueError):  # not bound yet / reset / partial response
            time.sleep(1.0)
            continue
        if health.get("status") == "error":
            die_with_log(f"server reported startup error: {health}", log_path)
        if health.get("maintenance") == "serving":
            return
        time.sleep(1.0)
    die_with_log(f"server not ready after {timeout:.0f}s", log_path)


def pump_output(src, log_f) -> None:
    """Mirror the server's output to our terminal while keeping the log file complete.

    Raw byte chunks (read1, not line-buffered) so \\r progress bars render live."""
    for chunk in iter(lambda: src.read1(65536), b""):
        log_f.write(chunk)
        log_f.flush()
        sys.stdout.buffer.write(chunk)
        sys.stdout.flush()


def stop_server(proc: subprocess.Popen) -> None:
    """SIGTERM the whole session (frontend + scheduler/tokenizer workers), escalate.

    Best-effort by design: it runs in ``finally`` and must not mask the real error.
    killpg runs even when the frontend already exited -- a crashed frontend leaves live
    non-daemon workers in the group, and they hold the GPU."""
    for sig, wait_s in ((signal.SIGTERM, 90), (signal.SIGKILL, 30)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:  # whole group already gone
            pass
        try:
            proc.wait(timeout=wait_s)
            break
        except subprocess.TimeoutExpired:
            continue
    time.sleep(3)  # let the driver reclaim VRAM before the next backend's server


def stream_generate(origin: str, model_id: str, problem: str, sampling: dict,
                    args: argparse.Namespace) -> dict:
    """One streamed chat completion; returns per-token arrival stamps, text, and usage."""
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": problem}],
        "max_tokens": args.decode,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": True},
        **sampling,
    }
    req = urllib.request.Request(
        f"{origin}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    stamps: list[float] = []
    pieces: list[str] = []
    usage: dict | None = None
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=1800)
    except urllib.error.HTTPError as e:
        sys.exit(f"[bench] request failed: HTTP {e.code}: {e.read()[:500]!r}")
    # Iterate the SSE stream line by line as bytes; json.loads decodes UTF-8 itself.
    # (A text-mode reader keyed off the content-type would decode latin-1: the server
    # sends ensure_ascii=False JSON with no charset on text/event-stream.)
    with resp:
        for raw in resp:
            line = raw.strip()
            if not line or not line.startswith(b"data:"):
                continue  # blank separators between events
            payload = line[len(b"data:"):].strip()
            if payload == b"[DONE]":
                break
            now = time.perf_counter()
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                text = delta.get("reasoning_content") or delta.get("content")
                if text:
                    stamps.append(now)
                    pieces.append(text)
    if usage is None:
        sys.exit("[bench] stream ended without a usage chunk; is this a FreeToken server?")
    return {"t0": t0, "stamps": stamps, "text": "".join(pieces), "usage": usage}


def _summarize_measurements(
    *,
    args: argparse.Namespace,
    backend: str,
    problem: str,
    measured: list[dict],
    stats: dict,
    stats_before: dict,
    cache_status: dict,
    cache_size: int | None,
    command: list[str],
    topology: list[dict],
    host_before: dict,
    host_after: dict,
    resource_summary: dict,
    server_log: str,
) -> dict:
    runs = []
    for repetition, result in enumerate(measured):
        stamps, usage = result["stamps"], result["usage"]
        if len(stamps) < 2:
            raise RuntimeError(f"need >=2 token events to measure decode, got {len(stamps)}")
        completion = usage["completion_tokens"]
        if completion != args.decode:
            print(
                f"[bench] WARNING: completion_tokens={completion} != --decode {args.decode}",
                flush=True,
            )
        steps = completion - 1
        decode_time = stamps[-1] - stamps[0]
        gaps = sorted((b - a) * 1e3 for a, b in zip(stamps, stamps[1:]))
        runs.append({
            "repetition": repetition,
            "decode_steps": steps,
            "decode_time_s": decode_time,
            "decode_tok_s": steps / decode_time if decode_time > 0 else 0.0,
            "ms_per_token": decode_time / steps * 1e3 if steps > 0 else 0.0,
            "event_ms_p50": gaps[len(gaps) // 2],
            "event_ms_p99": gaps[min(len(gaps) - 1, int(len(gaps) * 0.99))],
            "ttft_ms": (stamps[0] - result["t0"]) * 1e3,
            "events": len(stamps),
            "completion_tokens": completion,
            "output_sha1": hashlib.sha1(result["text"].encode()).hexdigest()[:12],
        })
    result = measured[-1]
    usage = result["usage"]

    def median(field: str) -> float:
        return statistics.median(run[field] for run in runs)

    telemetry = stats.get("moe_telemetry")
    telemetry_before = stats_before.get("moe_telemetry")
    if telemetry is not None and telemetry_before is not None:
        from freetoken.moe.telemetry import subtract_snapshots

        telemetry = subtract_snapshots(telemetry, telemetry_before)

    row = {
        "schema": "freetoken.moe-server-benchmark",
        "version": 1,
        "commit": _git_commit(),
        "worktree_dirty": _git_worktree_dirty(),
        "command": shlex.join(command),
        "model": args.model,
        "backend": backend,
        "tp_size": args.tp_size,
        "execution_policy": args.execution_policy,
        "moe_tp_layout": args.moe_tp_layout,
        "moe_disk_backed": args.moe_disk_backed,
        "memory_ratio": args.mem_ratio,
        "moe_host_budget_gb": args.moe_host_budget_gb,
        "moe_cache_size": cache_size,
        "kv_token_override": args.num_tokens,
        "problem": args.problem,
        "eval_case_id": args.case_id if args.eval_plan else None,
        "prompt_sha256": hashlib.sha256(problem.encode()).hexdigest(),
        "prompt_tokens": usage["prompt_tokens"],
        "decode_steps": runs[-1]["decode_steps"],
        "decode_tok_s": median("decode_tok_s"),
        "ms_per_token": median("ms_per_token"),
        "event_ms_p50": median("event_ms_p50"),
        "event_ms_p99": median("event_ms_p99"),
        "ttft_ms": median("ttft_ms"),
        "events": runs[-1]["events"],
        "completion_tokens": runs[-1]["completion_tokens"],
        "vram_gib": stats.get("vram_bytes", 0) / 2**30,
        "cuda_graph": not args.no_graph,
        "prefill_overlap": not args.no_prefill_overlap,
        "cache_initial_state": "one warm-up request after startup/rebuild",
        "cache_status": cache_status,
        # Counter delta covering only the measured repetitions (warm-up excluded).
        "moe_telemetry": telemetry,
        "moe_timing_runs": (
            [result.get("moe_timing") for result in measured]
            if args.moe_timing
            else None
        ),
        "gpu_topology": topology,
        "host_memory_before": host_before,
        "host_memory_after": host_after,
        "resource_summary": resource_summary,
        "swap_delta_bytes": (
            host_before.get("swapfree_bytes", 0) - host_after.get("swapfree_bytes", 0)
        ),
        "sampling": resolve_sampling(args.model, args.greedy)[0],
        "output_sha1": hashlib.sha1(result["text"].encode()).hexdigest()[:12],
        "runs": runs,
        "server_log": server_log,
        **_placement_log_metadata(server_log),
    }

    print(
        f"\n==== decode bs=1 [{backend}] cache={cache_size} via /v1/chat/completions ====",
        flush=True,
    )
    print(
        f"  decode throughput : {row['decode_tok_s']:8.2f} tok/s  "
        f"({row['ms_per_token']:.3f} ms/token)"
    )
    print(f"  TTFT (warm)       : {row['ttft_ms']:8.1f} ms  (prompt {row['prompt_tokens']} tok)")
    spread = max(run["decode_tok_s"] for run in runs) - min(
        run["decode_tok_s"] for run in runs
    )
    print(
        f"  decode measured   : {len(runs)} x {row['decode_steps']} steps; "
        f"event p50 {row['event_ms_p50']:.3f} / p99 {row['event_ms_p99']:.3f} ms"
    )
    print(f"  repetition spread : {spread:.3f} tok/s (max - min)")
    print(f"  vram (server)     : {row['vram_gib']:8.2f} GiB")
    sha_note = "greedy" if args.greedy else "sampled, per-server deterministic"
    print(f"  output sha1       : {row['output_sha1']}  ({sha_note}; compare across backends)")
    print(f"  output sample     : {result['text'][:240]!r}")
    return row


def run_one(args: argparse.Namespace, backend: str) -> list[dict]:
    problem, answer = load_problem(
        args.aime, args.problem, eval_plan=args.eval_plan, case_id=args.case_id
    )
    sampling, sampling_src = resolve_sampling(args.model, args.greedy)
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    if args.json_out:
        artifact = Path(args.json_out)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        fd, log_path = tempfile.mkstemp(
            prefix=f"{artifact.name}.{backend}.",
            suffix=".server.log",
            dir=artifact.parent,
        )
    else:
        fd, log_path = tempfile.mkstemp(prefix=f"bench-serve-{backend}-", suffix=".log")
    cmd = serve_cmd(args, backend, port)
    host_before = _meminfo()
    topology = _gpu_topology(args.gpu, args.tp_size)
    resource_monitor = ResourceMonitor(gpu_spec=args.gpu, tp_size=args.tp_size)
    resource_monitor.start()

    print(
        f"[bench] model={args.model}\n"
        f"[bench] backend={backend} "
        f"cache={args.cache_sequence or args.cache or args.cache_rate or 'auto'} "
        f"mem_ratio={args.mem_ratio} decode={args.decode} graph={not args.no_graph}\n"
        f"[bench] sampling={sampling} <- {sampling_src}\n"
        f"[bench] server log: {log_path}",
        flush=True,
    )

    with os.fdopen(fd, "wb") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
        )
        pump = threading.Thread(target=pump_output, args=(proc.stdout, log_f), daemon=True)
        pump.start()
        try:
            wait_ready(origin, proc, log_path, args.server_timeout)
            model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]
            print(f"[bench] model_id={model_id}", flush=True)
            label = f"eval case {args.case_id}" if args.eval_plan else f"AIME25 #{args.problem}"
            print(f"[bench] {label} (answer {answer})", flush=True)

            cache_sizes = args.cache_sequence or [args.cache if args.cache > 0 else None]
            measurement_sets = []
            for index, cache_size in enumerate(cache_sizes):
                if index:
                    rebuild = post_json(
                        f"{origin}/v1/cache/rebuild",
                        {"moe_cache_size": cache_size, "mode": "if_idle", "timeout": 300},
                        timeout=360,
                    )
                    if rebuild.get("status") != "ok":
                        raise RuntimeError(f"cache rebuild to {cache_size} failed: {rebuild}")
                # Re-establish a comparable warm state after startup or rebuild.
                stream_generate(origin, model_id, problem, sampling, args)
                stats_before = get_json(f"{origin}/v1/stats")
                measured = []
                stats = stats_before
                for _ in range(args.repetitions):
                    result = stream_generate(origin, model_id, problem, sampling, args)
                    stats = get_json(f"{origin}/v1/stats")
                    if args.moe_timing:
                        snapshot = stats.get("moe_telemetry") or {}
                        result["moe_timing"] = {
                            "aggregate": snapshot.get("timing"),
                            "ranks": [
                                {"rank": rank.get("rank"), "timing": rank.get("timing")}
                                for rank in snapshot.get("ranks", [])
                            ],
                        }
                    measured.append(result)
                measurement_sets.append((
                    cache_size,
                    measured,
                    stats_before,
                    stats,
                    get_json(f"{origin}/v1/cache/status"),
                ))
        finally:
            stop_server(proc)
            pump.join(timeout=10)
            resource_summary = resource_monitor.stop()
    host_after = _meminfo()
    return [
        _summarize_measurements(
            args=args,
            backend=backend,
            problem=problem,
            measured=measured,
            stats=stats,
            stats_before=stats_before,
            cache_status=cache_status,
            cache_size=cache_size,
            command=cmd,
            topology=topology,
            host_before=host_before,
            host_after=host_after,
            resource_summary=resource_summary,
            server_log=log_path,
        )
        for cache_size, measured, stats_before, stats, cache_status in measurement_sets
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    backends = [b.strip() for b in args.backend.split(",") if b.strip()]
    unknown = [b for b in backends if b not in ("offload", "cpu", "hybrid")]
    if unknown:
        sys.exit(f"unknown backend(s): {unknown}")

    failed = []
    for backend in backends:
        try:
            rows = run_one(args, backend)
        # SystemExit inherits BaseException, not Exception, so name both: a mid-decode
        # connection drop (server crash) must not abort the remaining backends either.
        except (SystemExit, Exception) as e:
            if len(backends) == 1:
                raise
            print(f"\n[bench] backend {backend} failed: {e!r}", flush=True)
            failed.append(backend)
            continue
        if args.json_out:
            with open(args.json_out, "a") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
    if failed:
        print(f"\n[bench] backends that failed: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
