# benchmarks

Run from the repo root with `PYTHONPATH=python:.`. Each script's `--help` /
docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

The heterogeneous-MoE baseline uses the same entry point with TP4. It performs
one cache warm-up followed by three measured repetitions and emits a versioned
artifact containing the commit, full server command, prompt hash, graph/cache
settings, per-GPU PCI BDF/NUMA/negotiated PCIe link, host memory/swap deltas and
each repetition (the top-level latency fields are medians):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=python:. \
python benchmarks/bench_decode_moe.py \
  --model /path/to/DeepSeek-V4-Flash-0731 --tp-size 4 \
  --backend offload --cache 512 --num-tokens 16384 \
  --eval-plan /path/to/test_plan.json --case-id decode_64_r0 \
  --decode 64 --greedy --repetitions 3 --json baseline.jsonl
```

Run the cache/prefill matrix with `--cache 256`, `--cache 512`, and a value
above 512, adding `--no-prefill-overlap` for the overlap-off rows. Keep model,
prompt, decode length, graph setting and memory ratio identical across rows.
Runtime cache rebuild currently supports TP1 only, so TP4 matrix rows must use
separate invocations. `--cache-sequence` is available for TP1 and rejects a
multi-value sequence before model startup when `--tp-size` is greater than one.

P3's startup-fixed weighted routed-expert baseline is opt-in. The widths are in
TP-rank order and must be positive 256-value tiles whose sum is the checkpoint's
routed intermediate size. Shared experts remain equal TP:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=python:. \
python benchmarks/bench_decode_moe.py \
  --model /path/to/DeepSeek-V4-Flash-0731 --tp-size 4 \
  --backend offload --cache 768 --num-tokens 16384 \
  --execution-policy force_weighted_tp \
  --moe-tp-layout 512,512,768,256 \
  --eval-plan /path/to/test_plan.json --case-id decode_64_r0 \
  --decode 64 --greedy --repetitions 3 --moe-telemetry --json weighted.jsonl
```

`--moe-telemetry` is opt-in. It persists the complete TP-rank/layer counter
matrix from `/v1/stats`; the artifact subtracts the post-warm-up snapshot, so
calls, misses, fetches, and prefill movement cover only measured repetitions.
Missing ranks or mismatched placement epochs/checksums fail the run. When
`--json` is set, the complete server log is retained beside the JSONL artifact.
Add `--moe-timing` to capture fixed external CUDA events into the decode graph;
the artifact stores one per-rank/per-layer component snapshot for every measured
repetition and identifies the critical rank for each layer. Use
`--moe-host-budget-gb` for A/B runs so startup memory noise cannot change the
permanent-layer placement. Component timing is a profiling mode, not a scoreable
baseline mode: on the SM75 TP4 reference run its event nodes reduced throughput
by about 3.8%. Counter-only telemetry remained below the 1% perturbation limit.

Weighted TP currently accepts only original DSV4 DS-FP4 safetensors. It rejects
FTW banks before expert allocation because those banks encode the older TP1 layout.
The P3 capacity/performance baseline above intentionally does not use
`--moe-disk-backed`: normal startup must prove a joint VRAM+RAM placement with zero
runtime checkpoint reads. `--moe-disk-backed --no-graph --no-prefill-overlap` remains
available only as a separately labelled low-memory emergency/diagnostic run; its results
must not be mixed into the P3 baseline.

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

For the DSV4 heterogeneous-layout calibration points (local intermediate width
256/512/768/2048), select the explicit DS-FP4 profiles and sweep miss counts
from one through top-k by choosing matching batch/miss-rate combinations:

```bash
python benchmarks/bench_offload_cache_copy.py \
  --models dsv4-dsfp4-i256 dsv4-dsfp4-i512 dsv4-dsfp4-i768 dsv4-dsfp4-i2048 \
  --cache-slots 256 512 --batch-sizes 1 4 16 --miss-counts 1 2 3 4 5 6
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.
