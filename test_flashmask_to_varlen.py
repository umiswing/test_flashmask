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
import math
import itertools
import pytest
import numpy as np
from functools import partial

import paddle
import torch

# ── Paddle: flashmask_attention ──────────────────────────────────────────────
try:
    from flash_mask.cute.interface import flashmask_attention
except (ImportError, ModuleNotFoundError):
    from paddle.nn.functional.flash_attention import flashmask_attention

# ── PyTorch: flash_attn_varlen_func (FA4 cute) ──────────────────────────────
from flash_attn.cute.interface import flash_attn_varlen_func

# ── convert_to_varlen (fake implementation in varlen_utils.py) ───────────────
from varlen_utils import convert_to_varlen

# ── Mask generators (Paddle) ────────────────────────────────────────────────
from generate_startend_row_indices import (
    generate_causal_document_mask,
    generate_document_mask,
    generate_causal_document_mask_diff_batch,
    generate_document_mask_diff_batch,
    generate_document_mask_simu,
    generate_document_mask_diff_batch_simu,
)

from test_util import attention_ref

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def paddle_to_torch(t: paddle.Tensor) -> torch.Tensor:
    """Convert a Paddle tensor to a PyTorch CUDA tensor.

    For bf16 tensors we view as int16 before going through numpy (which
    doesn't support bf16), then reinterpret back to bfloat16 on the
    PyTorch side.
    """
    if t.dtype == paddle.bfloat16:
        np_arr = t.view(paddle.int16).numpy()
        return torch.from_numpy(np_arr).view(torch.bfloat16).cuda()
    return torch.from_numpy(t.numpy()).cuda()


def torch_to_paddle(t: torch.Tensor) -> paddle.Tensor:
    """Convert a PyTorch CUDA tensor to a Paddle tensor.

    For bf16 tensors we view as int16 before going through numpy, then
    reinterpret back to bfloat16 on the Paddle side.
    """
    if t.dtype == torch.bfloat16:
        np_arr = t.cpu().view(torch.int16).numpy()
        return paddle.to_tensor(np_arr).view(paddle.bfloat16)
    return paddle.to_tensor(t.cpu().numpy())


# ─────────────────────────────────────────────────────────────────────────────
# Test parameters
# ─────────────────────────────────────────────────────────────────────────────

# (batch_size, seqlen_q, seqlen_k, nheads, nheads_kv)
shape_cases = [
    (1, 256, 256, 4, 4),
    (2, 512, 512, 8, 2),
    (1, 1024, 1024, 4, 1),
    (2, 300, 300, 6, 2),
    (1, 128, 128, 1, 1),
    (2, 1000, 1000, 4, 1),
]


def generate_shapes():
    for batch_size, seqlen_q, seqlen_k, nheads, nheads_kv in shape_cases:
        if nheads_kv == 1:
            nheads_startend_row_indices_values = [1]
        else:
            nheads_startend_row_indices_values = [1, nheads_kv]
        for nheads_sri in nheads_startend_row_indices_values:
            yield (batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_sri)


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
@pytest.mark.parametrize("d, dv", [(64, 64), (128, 128)])
@pytest.mark.parametrize(
    "batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices",
    list(generate_shapes()),
)
@pytest.mark.parametrize("gen_startend_row_indices", mask_generators)
def test_flashmask_to_varlen(
    batch_size,
    seqlen_q,
    seqlen_k,
    nheads,
    nheads_kv,
    d,
    dv,
    nheads_startend_row_indices,
    dtype,
    gen_startend_row_indices,
):
    """
    Compare Paddle flashmask_attention output with PyTorch flash_attn_varlen_func output
    after converting startend_row_indices to varlen format via convert_to_varlen().
    """
    paddle.seed(2024)
    torch.manual_seed(2024)
    assert nheads % nheads_kv == 0

    # ── 1. Generate padded Q, K, V (Paddle) ─────────────────────────────────
    q_paddle = paddle.randn(shape=[batch_size, seqlen_q, nheads, d], dtype=dtype)
    k_paddle = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, d], dtype=dtype)
    v_paddle = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, dv], dtype=dtype)

    # Generate mask
    startend_row_indices, causal = gen_startend_row_indices(
        batch_size, seqlen_q, seqlen_k, nheads_startend_row_indices
    )

    # ── 2. Convert to varlen format ──────────────────────────────────────────
    # convert_to_varlen returns Paddle tensors in a dict:
    #   "q":             (total_q, nheads, d)        Paddle
    #   "k":             (total_k, nheads_kv, d)     Paddle
    #   "v":             (total_k, nheads_kv, dv)    Paddle
    #   "cu_seqlens_q":  (num_seqs + 1,)             Paddle int32
    #   "cu_seqlens_k":  (num_seqs + 1,)             Paddle int32
    #   "max_seqlen_q":  int
    #   "max_seqlen_k":  int
    #   "causal":        bool

    varlen = convert_to_varlen(
        q_paddle, k_paddle, v_paddle, causal, startend_row_indices
    )

    q_vl_paddle = varlen["q"]
    k_vl_paddle = varlen["k"]
    v_vl_paddle = varlen["v"]
    cu_seqlens_q_paddle = varlen["cu_seqlens_q"]
    cu_seqlens_k_paddle = varlen["cu_seqlens_k"]
    max_seqlen_q = varlen["max_seqlen_q"]
    max_seqlen_k = varlen["max_seqlen_k"]
    varlen_causal = varlen.get("causal", causal)

    # ── 3. Call Paddle's flashmask_attention ─────────────────────────────────
    paddle.set_flags({"FLAGS_flash_attn_version": 4})

    # Skip if FA4 doesn't support this configuration
    if startend_row_indices is not None and startend_row_indices.shape[-1] == 4:
        pytest.skip("FA4 does not support startend_row_indices with last dim == 4")

    q_fm = q_paddle.detach().clone()
    k_fm = k_paddle.detach().clone()
    v_fm = v_paddle.detach().clone()
    q_fm.stop_gradient = False
    k_fm.stop_gradient = False
    v_fm.stop_gradient = False

    out_fm, lse_fm = flashmask_attention(
        q_fm,
        k_fm,
        v_fm,
        startend_row_indices=startend_row_indices,
        causal=causal,
        return_softmax_lse=True,
    )

    # ── 4. Call PyTorch's flash_attn_varlen_func ─────────────────────────────
    # Convert Paddle varlen tensors to PyTorch CUDA tensors
    q_varlen_pt = paddle_to_torch(q_vl_paddle).contiguous()
    k_varlen_pt = paddle_to_torch(k_vl_paddle).contiguous()
    v_varlen_pt = paddle_to_torch(v_vl_paddle).contiguous()
    cu_seqlens_q_pt = paddle_to_torch(cu_seqlens_q_paddle).contiguous().to(torch.int32)
    cu_seqlens_k_pt = paddle_to_torch(cu_seqlens_k_paddle).contiguous().to(torch.int32)

    out_varlen_pt, lse_varlen_pt = flash_attn_varlen_func(
        q_varlen_pt,
        k_varlen_pt,
        v_varlen_pt,
        cu_seqlens_q=cu_seqlens_q_pt,
        cu_seqlens_k=cu_seqlens_k_pt,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=varlen_causal,
        return_lse=True,
    )

    # ── 5. Compare outputs ───────────────────────────────────────────────────
    # Map varlen output back to padded layout for comparison.
    # If convert_to_varlen provides an "output_to_padded" callable, use it;
    # otherwise reshape the flat (total_q, nheads, dv) back to
    # (batch_size, seqlen_q, nheads, dv).
    if "output_to_padded" in varlen:
        out_varlen_padded_pt = varlen["output_to_padded"](out_varlen_pt)
    else:
        out_varlen_padded_pt = out_varlen_pt.reshape(batch_size, seqlen_q, nheads, dv)

    # Convert both outputs to float32 numpy for comparison
    out_fm_np = paddle.cast(out_fm, paddle.float32).numpy()
    out_vl_np = torch_to_paddle(out_varlen_padded_pt).cast(paddle.float32).numpy()

    max_diff = np.max(np.abs(out_fm_np - out_vl_np))
    mean_diff = np.mean(np.abs(out_fm_np - out_vl_np))
    print(f"\n[flashmask vs varlen] max diff: {max_diff:.6e}, mean diff: {mean_diff:.6e}")

    assert np.allclose(out_fm_np, out_vl_np, rtol=1e-2, atol=1e-2), (
        f"Output mismatch: max diff {max_diff:.6e}, mean diff {mean_diff:.6e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke test: single config, causal document mask
    test_flashmask_to_varlen(
        batch_size=2,
        seqlen_q=512,
        seqlen_k=512,
        nheads=4,
        nheads_kv=2,
        d=128,
        dv=128,
        nheads_startend_row_indices=1,
        dtype=paddle.bfloat16,
        gen_startend_row_indices=partial(generate_causal_document_mask),
    )
    print("\nSmoke test passed!")
