# Supported models

FreeToken loads HF safetensors checkpoints directly (plus native GGUF for
Gemma-4). The checkpoints below are known-good — the prebuilt kernels are tuned
for them; other checkpoints of the same architectures work too.

| Model | HF checkpoints |
|---|---|
| DeepSeek-V4 | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| GLM-5.2 | [nvidia/GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4) |
| GLM-4.7 | [nvidia/GLM-4.7-NVFP4](https://huggingface.co/nvidia/GLM-4.7-NVFP4) |
| Qwen3.6 / Qwen3.5 MoE | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)), [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4), [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)) |
| Qwen3.6 dense | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)), [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) |
| Qwen3-MoE | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| gpt-oss | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| Gemma-4 | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4), [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it), [nvidia/Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) .. |
| MiniMax-M2.5 | [nvidia/MiniMax-M2.5-NVFP4](https://huggingface.co/nvidia/MiniMax-M2.5-NVFP4) |
| Muse-Glimmer | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B), [RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4) |

## MoE backends

`ft serve --moe-backend {auto,fused,offload,cpu,hybrid}`:

- **fused** — experts resident on GPU (needs the VRAM); never auto-selected.
- **offload** — experts live in host RAM, an LRU cache of expert slots on GPU;
  misses stream over PCIe.
- **cpu** — misses are computed on the CPU instead of fetched.
- **hybrid** — per step, fetches some misses over PCIe and computes the rest on
  CPU, overlapped. Run `ft bench bw` once per machine to calibrate the split.
- **auto** — dense models always resolve to `fused`; MoE models resolve to
  `offload`, upgraded to `hybrid` when a cached `ft bench bw` profile
  recommends it.

## Notes

- `ft checkpoint` conversion is optional — it pre-converts a checkpoint into
  FreeToken's fast-load format, and `ft serve --model` auto-detects the result.
- DeepSeek-V4 checkpoints must keep the `inference/config.json` subdir — the
  authoritative model args are read from there.
- Multimodal checkpoints are served text-only.

## DeepSeek-V4 on multiple GPUs

For DeepSeek-V4 FP4, tensor parallelism keeps every expert id routable on every
rank but shards each expert's SwiGLU intermediate dimension. A TP=4 process
therefore holds one quarter of every host expert bank and one quarter of every
GPU expert slot; the partial routed outputs are summed across the four GPUs.

The placement planner also has a GPU-permanent tier. Complete expert layers assigned to
that tier are copied into a separate immutable allocation as soon as the layer finishes
loading, then their anonymous host pages are discarded. With `R` resident
layers, aggregate expert host backing is approximately:

```
(num_moe_layers - R) * num_experts * unsharded_expert_bytes
```

At the published V4-Flash geometry (`43` layers, `256` experts, expert
intermediate size `2048`), TP=4 uses 0.796875 GiB per complete expert layer on
each GPU. Every layer moved to the GPU-only tier removes 3.1875 GiB of aggregate
host backing; keeping all layers host-backed would require about 137.06 GiB for
the routed experts alone.

`--moe-gpu-only-layers auto` is the default compatibility setting. It permits
the planner to use the minimum number of complete layers required by the host
and pin budgets; it no longer consumes every layer that happens to fit in VRAM.
`--moe-gpu-only-layers N` makes the requested count strict, and `0` keeps the
traditional all-host-backed layout (or reports the host budget as infeasible).

For a low-concurrency TP=4 deployment, an explicit layout can be selected with:

```bash
ft serve \
  --model /data/models/DeepSeek-V4-Flash-0731 \
  --tp-size 4 \
  --moe-backend offload \
  --nvfp4-backend triton \
  --moe-placement auto \
  --moe-host-budget-gb 96 \
  --moe-cache-size 256 \
  --moe-gpu-only-layers auto \
  --expert-load serial \
  --disable-moe-prefill-overlap \
  --max-running-requests 1 \
  --cuda-graph-max-bs 0 \
  --max-seq-len-override 512 \
  --num-tokens 1536 \
  --max-prefill-length 128 \
  --memory-ratio 1.0
```

This asks for one rebuildable dynamic layer and caps retained expert backing at
96 GiB aggregate. At the published geometry the planner needs 13 permanent
layers to cover the roughly 41.06 GiB host deficit, subject to the fixed-model,
KV and activation VRAM floors. Treat this as capacity arithmetic rather than a
portable preset: the planner rejects the configuration before expert loading if
the measured GPU budget cannot prove it fits.

`--disable-moe-prefill-overlap` trades prefill H2D/GEMM overlap for one fewer
dynamic layer buffer. `--moe-cache-auto` jointly sizes the dynamic cache and KV
pool after subtracting the permanent floor; explicit host and pin budgets remain
hard constraints.

Multi-GPU DSV4 currently requires the original safetensors checkpoint. Existing
FTW expert banks contain a TP=1 physical layout, so the server rejects them for
TP>1 instead of silently duplicating experts and over-summing their outputs.
