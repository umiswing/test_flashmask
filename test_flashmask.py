import os
import math
import itertools
import pytest
from einops import rearrange, repeat
import paddle
try:
    from flash_mask.cute.interface import flashmask_attention
except (ImportError, ModuleNotFoundError):
    # Note(umiswing): comment it out if you really want to test in the old way
    assert False
    from paddle.nn.functional.flash_attention import flashmask_attention

from generate_startend_row_indices import (
  startend_row_indices_to_attn_bias,
  generate_none_mask,
  generate_sliding_window_mask,
  generate_causal_document_mask,
  generate_document_mask,
  generate_share_question_mask,
  generate_global_sliding_window_mask,
  generate_causal_blockwise_mask,
  generate_prefix_lm_document_mask,
  generate_prefix_lm_causal_mask,
  generate_qk_sparse_mask,
  generate_random_eviction_mask,
  generate_empty_mask,
)
from functools import partial
from test_util import attention_ref, detect_fa_versions

# batch_size, seqlen_q, seqlen_k, nheads, nheads_kv
shape_cases = (
    [
        (2840, 32, 32, 16, 4),
        (1, 300, 300, 16, 16),
        # (2, 8192, 32768, 32, 4), # this will oom
        # (2, 8192, 8192, 32, 4),  # this will oom
        (2, 8192, 8192, 14, 1),
        (2, 16384, 16384, 4, 1),
        (1, 1, 127, 1, 1),
        (1, 128, 127, 1, 1),
        (1, 127, 128, 1, 1),
        (2, 16383, 16384, 4, 1),
        (2, 16384, 16383, 4, 1),
        (2, 1000, 1000, 4, 1),
        (2, 2000, 2000, 4, 1),
        (2, 3000, 3000, 4, 1),
        (1, 4000, 4000, 1, 1),
        (1, 8192, 32768+1024, 2, 1),
        (1, 8192, 16384+1024, 2, 1),
        (2, 7600, 7600, 32, 8),
    ]
    # tridao case
    + list(itertools.product(
        [9],                # batch_size
        [1, 64,  128, 256, 239, 799, 113, 113, 128, 113, 108, 256, 384, 640, 512, 1024, 1023, 1024,],       # seqlen_q
        [128, 192, 256,   203, 128, 217, 211, 256, 512, 256, 128, 256, 1024, 1024, 1023,],      # seqlen_k
        [6],                # nheads
        [6, 2, 1],          # nheads_kv
    ))
    + list(itertools.product(
        [2],                # batch_size
        [4096, 4224],       # seqlen_q
        [4096, 4224],       # seqlen_k
        [6],                # nheads
        [6, 2, 1],          # nheads_kv
    ))
)

_shape_cases_before = len(shape_cases)
shape_cases = list(dict.fromkeys(shape_cases))  # 保序去重
print(f"{'='*60}")
print(f"[test_flashmask] Shape Cases Summary:")
print(f"  - Original Count: {_shape_cases_before}")
print(f"  - Unique Count:   {len(shape_cases)}")
print(f"  - Removed:        {_shape_cases_before - len(shape_cases)}")
print(f"{'='*60}")

d_dv_cases = [
    (32, 32),
    (64, 64),
    (80, 80),
    (128, 128),
    (192, 128),
    (192, 192),
    (256, 256),
]

fa_versions = detect_fa_versions()

# Generate all combinations for second param
def generate_shapes():
    for batch_size, seqlen_q, seqlen_k, nheads, nheads_kv in shape_cases:
        if nheads_kv == 1:
          nheads_startend_row_indices_values = [1]
        else:
          nheads_startend_row_indices_values = [1, nheads_kv]
        for nheads_startend_row_indices in nheads_startend_row_indices_values:
            yield (
                batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices
            )

@pytest.mark.parametrize("dtype", [paddle.bfloat16])
@pytest.mark.parametrize("fa_version", fa_versions)
@pytest.mark.parametrize(
    "d, dv",
    d_dv_cases,
    ids=[f"d{c[0]}-dv{c[1]}" for c in d_dv_cases]
)
@pytest.mark.parametrize(
    "batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices",
    list(generate_shapes())
)
@pytest.mark.parametrize(
    "gen_startend_row_indices",
    [
        partial(generate_none_mask, causal=False), # full
        partial(generate_none_mask, causal=True), # causal
        partial(generate_sliding_window_mask), # sliding window
        partial(generate_causal_document_mask), # causal document mask
        partial(generate_document_mask), # document mask
        partial(generate_share_question_mask), # share question mask
        partial(generate_global_sliding_window_mask), # global sliding window
        partial(generate_causal_blockwise_mask), # causal blockwise mask
        partial(generate_prefix_lm_document_mask), # prefix lm document mask
        partial(generate_prefix_lm_causal_mask), # prefix lm causal mask
        partial(generate_qk_sparse_mask), # qk-sparse mask
        partial(generate_random_eviction_mask), # random eviction mask
        partial(generate_empty_mask),
    ],
)
def test_flashmask(
    batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, d, dv, nheads_startend_row_indices, fa_version, dtype, gen_startend_row_indices, softcap=0.0
):
    paddle.seed(2026)
    assert nheads % nheads_kv == 0

    flashmask_impl = (fa_version == 3 or fa_version == 4)

    startend_row_indices, causal = gen_startend_row_indices(batch_size, seqlen_q, seqlen_k, nheads_startend_row_indices)

    if (batch_size, seqlen_q, seqlen_k, nheads, nheads_kv) == (2, 7600, 7600, 32, 8) and fa_version == 3:
        pytest.skip("Skipping (2,7600,7600,32,8) on fa3 due to OOM")

    if (fa_version == 2 or (d == 192 and dv == 192)) and seqlen_q != seqlen_k and causal:
        # fa3/fa4 fallback to fa2
        pytest.skip(f"Skipping because running fa2 in causal when seqlen_q != seqlen_k")

    if fa_version == 4 and startend_row_indices is not None and startend_row_indices.shape[-1] == 4:
        pytest.skip(f"Skipping because running fa4 when startend_row_indices.shape[-1] == 4")

    use_sink = flashmask_impl and not (d == 192 and dv == 192)

    q_ref = paddle.randn(shape=[batch_size, seqlen_q, nheads, d], dtype=dtype)
    k_ref = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, d], dtype=dtype)
    v_ref = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, dv], dtype=dtype)

    q_ref.stop_gradient = False
    k_ref.stop_gradient = False
    v_ref.stop_gradient = False

    q_bf16, k_bf16, v_bf16 = [x.detach().clone() for x in (q_ref, k_ref, v_ref)]

    q_bf16.stop_gradient = False
    k_bf16.stop_gradient = False
    v_bf16.stop_gradient = False

    q, k, v = [x.detach().clone() for x in (q_ref, k_ref, v_ref)]

    q.stop_gradient = False
    k.stop_gradient = False
    v.stop_gradient = False

    attn_bias = startend_row_indices_to_attn_bias(startend_row_indices, seqlen_q, nheads, dtype, causal)

    if use_sink:
        sink_ref = paddle.randn(shape=[nheads], dtype=dtype)
        sink_bf16 = sink_ref.detach().clone()
        sink = sink_ref.detach().clone()

        sink_bf16.stop_gradient = False
        sink_ref.stop_gradient = False
        sink.stop_gradient = False
    else:
        sink_ref = None
        sink_bf16 = None
        sink = None

    out_ref, attn_ref = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        causal=causal,
        attn_bias=attn_bias,
        learnable_sink=sink_ref,
    )

    out_bf16, attn_bf16 = attention_ref(
        q_bf16,
        k_bf16,
        v_bf16,
        causal=causal,
        attn_bias=attn_bias,
        upcast=False,
        reorder_ops=True,
        learnable_sink=sink_bf16,
    )

    # # Numerical error if we just do any arithmetic on out_ref
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    assert softcap == 0.0
    rtol = 2 if softcap == 0.0 else 3

    print(f"Paddle naive bf16 Output max diff: {(out_bf16 - out_ref).abs().max().item()}")
    print(f"Paddle naive bf16 Output mean diff: {(out_bf16 - out_ref).abs().mean().item()}")

    if fa_version == 2:
        paddle.set_flags({'FLAGS_flash_attn_version': 2})
    elif fa_version == 3:
        paddle.set_flags({'FLAGS_flash_attn_version': 3})
    elif fa_version == 4:
        paddle.set_flags({'FLAGS_flash_attn_version': 4})
    else:
        raise ValueError(
            f"Invalid flash attention version: {fa_version}"
        )

    out, lse = flashmask_attention(
        q,
        k,
        v,
        startend_row_indices=startend_row_indices,
        causal=causal,
        return_softmax_lse=True,
        learnable_sink=sink,
    )
    print(f"flashmask Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"flashmask Output mean diff: {(out - out_ref).abs().mean().item()}")
    # if not causal:
    #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
    # breakpoint()

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.

    assert (out - out_ref).abs().max().item() <= rtol * (out_bf16 - out_ref).abs().max().item() + fwd_atol

    g = paddle.randn(shape=out.shape, dtype=out.dtype)
    out.backward(g)
    out_ref.backward(g)
    out_bf16.backward(g)

    print(f"flashmask dQ max diff: {(q.grad - q_ref.grad).abs().max().item()}")
    print(f"flashmask dK max diff: {(k.grad - k_ref.grad).abs().max().item()}")
    print(f"flashmask dV max diff: {(v.grad - v_ref.grad).abs().max().item()}")
    print(f"flashmask dQ mean diff: {(q.grad - q_ref.grad).abs().mean().item()}")
    print(f"flashmask dK mean diff: {(k.grad - k_ref.grad).abs().mean().item()}")
    print(f"flashmask dV mean diff: {(v.grad - v_ref.grad).abs().mean().item()}")

    print(f"Paddle naive bf16 dQ max diff: {(q_bf16.grad - q_ref.grad).abs().max().item()}")
    print(f"Paddle naive bf16 dK max diff: {(k_bf16.grad - k_ref.grad).abs().max().item()}")
    print(f"Paddle naive bf16 dV max diff: {(v_bf16.grad - v_ref.grad).abs().max().item()}")
    print(f"Paddle naive bf16 dQ mean diff: {(q_bf16.grad - q_ref.grad).abs().mean().item()}")
    print(f"Paddle naive bf16 dK mean diff: {(k_bf16.grad - k_ref.grad).abs().mean().item()}")
    print(f"Paddle naive bf16 dV mean diff: {(v_bf16.grad - v_ref.grad).abs().mean().item()}")

    dq_atol = 2 * (q_ref.grad + 0.3 - 0.3 - q_ref.grad).abs().max().item() + (0 if softcap == 0 else 3e-4)
    assert (q.grad - q_ref.grad).abs().max().item() <= rtol * (q_bf16.grad - q_ref.grad).abs().max().item() + dq_atol
    dk_atol = 2 * (k_ref.grad + 0.3 - 0.3 - k_ref.grad).abs().max().item() + (0 if softcap == 0 else 3e-4)
    assert (k.grad - k_ref.grad).abs().max().item() <= rtol * (k_bf16.grad - k_ref.grad).abs().max().item() + dk_atol
    dv_atol = 2 * (v_ref.grad + 0.3 - 0.3 - v_ref.grad).abs().max().item() + (0 if softcap == 0 else 3e-4)
    assert (v.grad - v_ref.grad).abs().max().item() <= rtol * (v_bf16.grad - v_ref.grad).abs().max().item() + dv_atol

    if use_sink:
        print(f"flashmask dSink max diff: {(sink.grad - sink_ref.grad).abs().max().item()}")
        print(f"flashmask dSink mean diff: {(sink.grad - sink_ref.grad).abs().mean().item()}")
        print(f"Paddle naive bf16 dSink max diff: {(sink_bf16.grad - sink_ref.grad).abs().max().item()}")
        print(f"Paddle naive bf16 dSink mean diff: {(sink_bf16.grad - sink_ref.grad).abs().mean().item()}")

        delta = (out_ref * g).sum(-1)                                      # (b, sq, h)
        delta = delta.transpose([0, 2, 1])                                 # (b, h, sq)
        p_sink = 1.0 - attn_ref.sum(-1)

        dsink_ref_analytic = -(p_sink * delta).sum(axis=[0, 2])            # (h,)
        err_scale = (p_sink * delta.abs()).sum(axis=[0, 2])                # (h,)

        dsink_diff = (sink.grad - sink_ref.grad).abs()
        dsink_tol = 1e-2 + rtol * 2**-8 * err_scale
        assert bool((dsink_diff <= dsink_tol).all().item()), (
            f"dsink mismatch: max_diff={dsink_diff.max().item():.6f}, "
            f"max_tol={dsink_tol.max().item():.6f}, "
            f"err_scale_max={err_scale.max().item():.4f}"
        )

    paddle.device.cuda.empty_cache()
