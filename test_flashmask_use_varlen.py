"""
Test: FlashMask (Paddle) vs flash_attn_varlen_func (PyTorch) via convert_to_varlen.

Workflow:
  1. Generate q, k, v, causal, startend_row_indices (Paddle tensors, padded layout).
  2. Call convert_to_varlen() to transform startend_row_indices into varlen format:
       - q_varlen, k_varlen, v_varlen: concatenated Paddle tensors (total_q, nheads, d)
       - cu_seqlens_q, cu_seqlens_k: cumulative sequence lengths (Paddle, int32)
       - max_seqlen_q, max_seqlen_k: maximum sequence lengths (int)
  3. Call Paddle's flashmask_attention with the original padded input.
  4. Convert varlen tensors from Paddle to PyTorch, then call PyTorch's
     flash_attn_varlen_func.
  5. Compare the two outputs via np.allclose.
"""

import os
import glob
import math
import itertools
import pytest
import numpy as np
from functools import partial

import paddle

# ── Paddle: flashmask_attention ──────────────────────────────────────────────
from flash_mask import flashmask_attention
# from paddlefleet.ops.flash_mask import flashmask_attention

# ── Mask generators (Paddle) ────────────────────────────────────────────────
from generate_startend_row_indices import (
    startend_row_indices_to_attn_bias,
    generate_causal_document_mask,
    generate_document_mask,
    generate_causal_document_mask_diff_batch,
    generate_document_mask_diff_batch,
    generate_document_mask_simu,
    generate_document_mask_diff_batch_simu,
)

from test_util import attention_ref

# ─────────────────────────────────────────────────────────────────────────────
# Test parameters
# ─────────────────────────────────────────────────────────────────────────────

# (batch_size, seqlen_q, seqlen_k, nheads, nheads_kv)
shape_cases = [
    (2840, 32, 32, 16, 4),
    (1, 300, 300, 16, 16),
    (1, 256, 256, 4, 4),
    (2, 512, 512, 8, 2),
    (1, 1024, 1024, 4, 1),
    (2, 300, 300, 6, 2),
    (1, 128, 128, 1, 1),
    (2, 1000, 1000, 4, 1),
    (2, 8192, 8192, 4, 1),
    (2, 8192, 8192, 14, 1),
    (2, 16384, 16384, 4, 1),
    (2, 1000, 1000, 4, 1),
    (2, 2000, 2000, 4, 1),
    (2, 3000, 3000, 4, 1),
    (1, 4000, 4000, 1, 1),
    (2, 7600, 7600, 32, 8),
]


def generate_shapes():
    for batch_size, seqlen_q, seqlen_k, nheads, nheads_kv in shape_cases:
        nheads_startend_row_indices = 1
        yield (batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices)


# Only test mask types that are compatible with varlen (causal-style masks).
mask_generators = [
    partial(generate_document_mask),                # document
    partial(generate_causal_document_mask),                # causal document
    partial(generate_document_mask_diff_batch),                # document
    partial(generate_causal_document_mask_diff_batch),                # causal document
    partial(generate_document_mask_simu),                # simu causal document
    partial(generate_document_mask_diff_batch_simu),                # simu causal document diff batch
]


# ─────────────────────────────────────────────────────────────────────────────
# The test
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", [paddle.bfloat16])
@pytest.mark.parametrize("d, dv", [(192, 128), (256, 256)])
@pytest.mark.parametrize(
    "batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices",
    list(generate_shapes()),
)
@pytest.mark.parametrize("softmax_scale", [None, 1.0 / math.sqrt(64)])
@pytest.mark.parametrize("gen_startend_row_indices", mask_generators)
def test_flashmask_to_varlen(
    batch_size,
    seqlen_q,
    seqlen_k,
    nheads,
    nheads_kv,
    d,
    dv,
    softmax_scale,
    nheads_startend_row_indices,
    dtype,
    gen_startend_row_indices,
):
    """
    Compare Paddle flashmask_attention output with PyTorch flash_attn_varlen_func output
    after converting startend_row_indices to varlen format via convert_to_varlen().
    """
    paddle.seed(2024)
    assert nheads % nheads_kv == 0

    # ── 1. Generate padded Q, K, V (Paddle) ─────────────────────────────────
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

    # Generate mask
    startend_row_indices, causal = gen_startend_row_indices(
        batch_size, seqlen_q, seqlen_k, nheads_startend_row_indices
    )

    # ── 3. Call naive ref ─────────────────────────────────
    attn_bias = startend_row_indices_to_attn_bias(startend_row_indices, seqlen_q, nheads, dtype, causal)

    out_ref, attn_ref = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        causal=causal,
        attn_bias=attn_bias,
        softmax_scale=softmax_scale,
    )

    out_bf16, attn_bf16 = attention_ref(
        q_bf16,
        k_bf16,
        v_bf16,
        causal=causal,
        attn_bias=attn_bias,
        upcast=False,
        reorder_ops=True,
        softmax_scale=softmax_scale,
    )

    # # Numerical error if we just do any arithmetic on out_ref
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    rtol = 2

    print(f"Paddle naive bf16 Output max diff: {(out_bf16 - out_ref).abs().max().item()}")
    print(f"Paddle naive bf16 Output mean diff: {(out_bf16 - out_ref).abs().mean().item()}")


    # ── 4. Call flashmask with use_varlen ─────────────────────────────
    # Convert Paddle varlen tensors to PyTorch CUDA tensors
    paddle.set_flags({"FLAGS_flash_attn_version": 4})
    out = flashmask_attention(
        q,
        k,
        v,
        startend_row_indices=startend_row_indices,
        causal=causal,
        return_softmax_lse=False,
        use_varlen=True,
        softmax_scale=softmax_scale,
    )
    print(f"flashmask Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"flashmask Output mean diff: {(out - out_ref).abs().mean().item()}")

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

    dq_atol = 2 * (q_ref.grad + 0.3 - 0.3 - q_ref.grad).abs().max().item()
    assert (q.grad - q_ref.grad).abs().max().item() <= rtol * (q_bf16.grad - q_ref.grad).abs().max().item() + dq_atol
    dk_atol = 2 * (k_ref.grad + 0.3 - 0.3 - k_ref.grad).abs().max().item()
    assert (k.grad - k_ref.grad).abs().max().item() <= rtol * (k_bf16.grad - k_ref.grad).abs().max().item() + dk_atol
    dv_atol = 2 * (v_ref.grad + 0.3 - 0.3 - v_ref.grad).abs().max().item()
    assert (v.grad - v_ref.grad).abs().max().item() <= rtol * (v_bf16.grad - v_ref.grad).abs().max().item() + dv_atol
