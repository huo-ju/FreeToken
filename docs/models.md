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

The offload cache also has a GPU-only tier. Complete expert layers assigned to
that tier are copied into protected cache slots as soon as the layer finishes
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

`--moe-gpu-only-layers auto` is the default. It uses every complete layer that
fits after reserving one dynamic layer, or two while prefill-copy overlap is
enabled. `--moe-gpu-only-layers N` makes the requested count strict, and `0`
keeps the traditional all-host-backed layout.

On four 24 GiB Turing GPUs with about 128 GiB of host RAM, the following
low-concurrency bring-up configuration was validated against the original
`DeepSeek-V4-Flash-0731` safetensors checkpoint:

```bash
ft serve \
  --model /data/models/DeepSeek-V4-Flash-0731 \
  --dtype float16 \
  --tp-size 4 \
  --moe-backend offload \
  --nvfp4-backend triton \
  --moe-cache-size 4352 \
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

This allocates 17 layer-equivalents of cache: one remains dynamic and 16 complete
layers (27 through 42) become GPU-only. It releases 51.00 GiB of aggregate host
pages and retains about 86.06 GiB of routed-expert backing. The measured ready
state was about 22.5--23.2 GiB per GPU and 4.9 GiB host memory available, so this
is deliberately a tight starting point, not a high-concurrency preset. Reduce
the cache or token budget on smaller GPUs; add more resident layers only when
both the VRAM budget and one dynamic layer still fit.

SM75 automatically uses FP16 weights/activations with FP32 accumulation.
`--disable-moe-prefill-overlap` trades prefill H2D/GEMM overlap for one fewer
dynamic layer buffer, making one additional complete layer eligible for the
GPU-only tier. `--moe-cache-auto` remains useful when VRAM is the only
constraint, but it does not infer how much host RAM the checkpoint needs; use
an explicit cache size when GPU residency is required to make the model fit in
host memory.

Multi-GPU DSV4 currently requires the original safetensors checkpoint. Existing
FTW expert banks contain a TP=1 physical layout, so the server rejects them for
TP>1 instead of silently duplicating experts and over-summing their outputs.
