"""Native-16-bit execution checks for the SM75 FP16/FP32 policy."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_e4m3_emulation_buffer_is_fp16_on_sm75():
    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("SM75-specific dtype policy")

    from freetoken.kernel.triton.e4m3_compat import e4m3_act_dtype, e4m3_native

    assert not e4m3_native()
    assert e4m3_act_dtype() == torch.float16


@pytest.mark.parametrize("m", [1, 33, 257])
def test_mxfp8_linear_accepts_fp16_activations(m: int):
    from freetoken.kernel.triton.mxfp8_linear import mxfp8_dequant, mxfp8_linear

    torch.manual_seed(m)
    n, k = 127, 512
    codes = torch.randint(122, 131, (n, k // 32), device="cuda", dtype=torch.uint8)
    scale = torch.exp2(codes.float() - 127.0)
    source = torch.randn(n, k, device="cuda") * 0.05
    weight = (
        (source.view(n, -1, 32) / scale.unsqueeze(-1))
        .clamp(-448, 448)
        .view(n, k)
        .to(torch.float8_e4m3fn)
    )
    x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1

    got = mxfp8_linear(x, weight, codes)
    ref_weight = mxfp8_dequant(weight, codes, dtype=torch.float32)
    want = (x.float() @ ref_weight.t()).to(torch.float16)

    assert got.dtype == torch.float16
    torch.testing.assert_close(got, want, rtol=3e-2, atol=5e-2)


def test_mxfp4_splitk_preserves_fp16_activation_dtype():
    from freetoken.kernel import mxfp4_splitk_gemv_triton

    routes, experts, n, k = 2, 2, 16, 32
    x = torch.randn(routes, k, device="cuda", dtype=torch.float16)
    blocks = torch.zeros(experts, k // 2, n, device="cuda", dtype=torch.uint8)
    scales = torch.full(
        (experts, k // 32, n), 127, device="cuda", dtype=torch.uint8
    )
    bias = torch.randn(experts, n, device="cuda", dtype=torch.float16)
    ids = torch.tensor([0, 1], device="cuda", dtype=torch.int64)

    got = mxfp4_splitk_gemv_triton(
        x,
        blocks,
        scales,
        bias,
        ids,
        N=n,
        K=k,
        stride_xe=x.stride(0),
        num_splits=1,
        block_n=16,
    )

    assert got.dtype == torch.float16
    torch.testing.assert_close(got, bias.to(torch.float16), rtol=0, atol=0)


def test_df11_decodes_directly_to_fp16():
    from freetoken.kernel.triton.df11 import df11_compress, df11_compress_rows
    from freetoken.kernel.triton.df11_decode import df11_decompress, df11_gather_decode

    torch.manual_seed(7)
    weight = torch.randn(37, 129, device="cuda", dtype=torch.bfloat16)
    bundle = df11_compress(weight)
    got = df11_decompress(bundle, dtype=torch.float16)
    assert got.dtype == torch.float16
    assert torch.equal(got, weight.to(torch.float16))

    row_bundle = df11_compress_rows(weight)
    ids = torch.tensor([36, 0, 11], device="cuda", dtype=torch.int64)
    gathered = df11_gather_decode(row_bundle, ids, dtype=torch.float16)
    assert gathered.dtype == torch.float16
    assert torch.equal(gathered, weight.index_select(0, ids).to(torch.float16))
