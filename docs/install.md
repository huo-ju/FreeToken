# Install

## Requirements

- Linux x86_64, NVIDIA GPU, driver r580+ (CUDA 13)
- Python >= 3.10, with [uv](https://docs.astral.sh/uv/) recommended (plain
  `pip` + `venv` works too)

## Method 1: Install from PyPI

```bash
uv venv && source .venv/bin/activate
uv pip install "freetoken[accel]"
```

CUDA kernels are JIT-compiled on first use, need a CUDA 13 toolkit with `nvcc` on PATH.

## Method 2: Install from source

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

### Turing / SM75

Official Triton no longer supports Turing. Install the community fork in the same
environment before installing FreeToken:

```bash
git clone https://github.com/Chennesxu/triton-turing.git ../triton-turing
uv pip install -e ../triton-turing
uv pip install --no-deps -e ../triton-turing/python/triton_kernels
uv pip install -e ".[accel]"
```

FreeToken automatically enables its conservative SM75 launch configurations when the
active GPU has compute capability 7.5. The tri-state switch is useful for diagnosis:

```bash
export FREETOKEN_TRITON_TURING_COMPAT=auto  # default: enabled only on SM75
export FREETOKEN_TRITON_TURING_COMPAT=1     # force-enable
export FREETOKEN_TRITON_TURING_COMPAT=0     # disable and use upstream launch configs
```

The switch controls FreeToken's kernel workarounds; it does not dynamically replace the
installed `triton` package. Turing processes still require `triton-turing` in their Python
environment.

SM75 has FP16 Tensor Cores but no native BF16 Tensor Core mode. FreeToken therefore treats
BF16 as a checkpoint/storage dtype on Turing and serves it with **FP16 weights and
activations plus FP32 accumulation**. This conversion is automatic even when
`--dtype=bfloat16` was requested explicitly; it avoids the much slower software preservation of
BF16 arithmetic semantics. `FREETOKEN_TRITON_TURING_COMPAT=0` disables launch workarounds,
not this hardware dtype policy.

## Verify

```bash
source .venv/bin/activate
ft --version
ft serve --model ~/path/to/Qwen3.6-35B-A3B
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}]}'
```

Then head to [quickstart.md](quickstart.md).
