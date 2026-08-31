"""
Comprehensive test for flashmask_attention(..., use_varlen=True).

Covers the following scenarios systematically:
1. Multiple calls (dual-split) vs single call
2. Zero Q documents at head/middle/tail of cu_seqlens
3. Q padding (last doc lts < seqlen_q)
4. Cross multiple batch (different doc boundaries per batch)
5. Padding between two batches (dead K columns)
6. Causal vs non-causal
7. Context parallel (dual chunk CP)
8. Dual chunk (exercised via CP simulation)
9. Many different seqlens and cp_sizes

Test structure:
- Part A: Non-CP tests (single-call path, bound1 and bound2)
- Part B: Dual-chunk CP tests (multi-call path)
"""

import sys
import os
import math
import numpy as np
import pytest
from functools import partial

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "flash-attention", "flashmask"))

import paddle

from flash_mask import flashmask_attention
from generate_startend_row_indices import (
    startend_row_indices_to_attn_bias,
    generate_causal_document_mask,
    generate_document_mask,
    generate_causal_document_mask_diff_batch,
    generate_document_mask_diff_batch,
    generate_document_mask_simu,
    generate_document_mask_diff_batch_simu,
)
from context_parallel_utils import preprocess_index_dual_chunks
from test_util import attention_ref


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_forward_backward_check(
    q, k, v, startend_row_indices, causal, softmax_scale, rtol=2,
    label="",
):
    """Run flashmask use_varlen forward+backward and compare with reference.

    Returns (fwd_pass, bwd_pass) booleans.
    """
    batch_size, seqlen_q, nheads, d = q.shape
    seqlen_k = k.shape[1]
    dv = v.shape[-1]
    dtype = q.dtype

    # Reference
    q_ref = q.detach().clone()
    k_ref = k.detach().clone()
    v_ref = v.detach().clone()
    q_ref.stop_gradient = False
    k_ref.stop_gradient = False
    v_ref.stop_gradient = False

    attn_bias = startend_row_indices_to_attn_bias(
        startend_row_indices, seqlen_q, nheads, dtype, causal
    )

    out_ref, _ = attention_ref(
        q_ref, k_ref, v_ref,
        causal=causal,
        attn_bias=attn_bias,
        softmax_scale=softmax_scale,
    )

    # bf16 reference for tolerance
    q_bf16 = q.detach().clone()
    k_bf16 = k.detach().clone()
    v_bf16 = v.detach().clone()
    q_bf16.stop_gradient = False
    k_bf16.stop_gradient = False
    v_bf16.stop_gradient = False
    out_bf16, _ = attention_ref(
        q_bf16, k_bf16, v_bf16,
        causal=causal,
        attn_bias=attn_bias,
        upcast=False,
        reorder_ops=True,
        softmax_scale=softmax_scale,
    )

    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    bf16_diff = (out_bf16 - out_ref).abs().max().item()

    # flashmask with use_varlen
    q_test = q.detach().clone()
    k_test = k.detach().clone()
    v_test = v.detach().clone()
    q_test.stop_gradient = False
    k_test.stop_gradient = False
    v_test.stop_gradient = False

    paddle.set_flags({"FLAGS_flash_attn_version": 4})
    out = flashmask_attention(
        q_test, k_test, v_test,
        startend_row_indices=startend_row_indices,
        causal=causal,
        return_softmax_lse=False,
        use_varlen=True,
        softmax_scale=softmax_scale,
    )

    max_diff = (out - out_ref).abs().max().item()
    print(f"  [{label}] fwd: max_diff={max_diff:.6f}, bf16_diff={bf16_diff:.6f}, fwd_atol={fwd_atol:.6f}")

    assert max_diff <= rtol * bf16_diff + fwd_atol, (
        f"[{label}] fwd: max_diff={max_diff} > rtol*bf16_diff+atol={rtol * bf16_diff + fwd_atol}"
    )

    # Backward
    g = paddle.randn(shape=out.shape, dtype=out.dtype)
    out.backward(g)
    out_ref.backward(g)
    out_bf16.backward(g)

    dq_atol = 2 * (q_ref.grad + 0.3 - 0.3 - q_ref.grad).abs().max().item()
    dk_atol = 2 * (k_ref.grad + 0.3 - 0.3 - k_ref.grad).abs().max().item()
    dv_atol = 2 * (v_ref.grad + 0.3 - 0.3 - v_ref.grad).abs().max().item()

    dq_diff = (q_test.grad - q_ref.grad).abs().max().item()
    dk_diff = (k_test.grad - k_ref.grad).abs().max().item()
    dv_diff = (v_test.grad - v_ref.grad).abs().max().item()

    dq_bf16_diff = (q_bf16.grad - q_ref.grad).abs().max().item()
    dk_bf16_diff = (k_bf16.grad - k_ref.grad).abs().max().item()
    dv_bf16_diff = (v_bf16.grad - v_ref.grad).abs().max().item()

    print(f"    dQ: diff={dq_diff:.6f}, bf16={dq_bf16_diff:.6f}")
    print(f"    dK: diff={dk_diff:.6f}, bf16={dk_bf16_diff:.6f}")
    print(f"    dV: diff={dv_diff:.6f}, bf16={dv_bf16_diff:.6f}")

    assert dq_diff <= rtol * dq_bf16_diff + dq_atol, f"[{label}] dQ too large"
    assert dk_diff <= rtol * dk_bf16_diff + dk_atol, f"[{label}] dK too large"
    assert dv_diff <= rtol * dv_bf16_diff + dv_atol, f"[{label}] dV too large"


# ─────────────────────────────────────────────────────────────────────────────
# Mask generators with zero-Q-document support
# ─────────────────────────────────────────────────────────────────────────────

def generate_causal_doc_mask_with_zero_q_head(batch_size, seqlen_q, seqlen_k, h):
    """Causal document mask with a zero-length Q document at the HEAD.

    Dead K columns at the start mean the first document has zero Q rows attending.
    Achieved by setting lts[0:dead_k] to 0 (ute >= lts → dead).
    We create this via bound_num=2 where the first few K columns are dead.
    """
    # Create 3 docs: first doc is dead (zero Q length), then 2 normal docs
    dead_k_len = max(1, seqlen_k // 8)
    remaining = seqlen_k - dead_k_len
    doc1_len = remaining // 2
    doc2_len = remaining - doc1_len

    lts = np.zeros(seqlen_k, dtype=np.int32)
    ute = np.zeros(seqlen_k, dtype=np.int32)

    # Dead K columns at head: lts=0, ute=0 → ute >= lts → dead
    # Actually for dead columns: ute[j] >= lts[j]. Set lts=0 and ute=0 → 0>=0 → dead
    for j in range(dead_k_len):
        lts[j] = 0
        ute[j] = 0

    # Doc 1 (active): K[dead_k_len : dead_k_len+doc1_len], Q[0:doc1_len]
    q_offset = 0
    k_start = dead_k_len
    for j in range(doc1_len):
        lts[k_start + j] = doc1_len
        ute[k_start + j] = q_offset + j  # causal pattern

    # Doc 2 (active): K[dead_k_len+doc1_len : seqlen_k], Q[doc1_len:doc1_len+doc2_len]
    q_offset2 = doc1_len
    k_start2 = dead_k_len + doc1_len
    for j in range(doc2_len):
        lts[k_start2 + j] = doc1_len + doc2_len
        ute[k_start2 + j] = q_offset2 + j  # causal pattern

    # Clip to seqlen_q
    lts = np.clip(lts, 0, seqlen_q)
    ute = np.clip(ute, 0, seqlen_q)

    lts_t = paddle.to_tensor(lts).reshape([1, 1, seqlen_k, 1])
    ute_t = paddle.to_tensor(ute).reshape([1, 1, seqlen_k, 1])
    sri = paddle.concat([lts_t, ute_t], axis=-1)
    sri = sri.expand([batch_size, 1, seqlen_k, 2])

    causal = False
    return sri, causal


def generate_causal_doc_mask_with_zero_q_middle(batch_size, seqlen_q, seqlen_k, h):
    """Causal document mask with a zero-length Q document in the MIDDLE.

    Dead K columns in the middle create a zero-Q doc between two active docs.
    """
    doc1_len = seqlen_k // 4
    dead_k_len = max(1, seqlen_k // 8)
    doc2_len = seqlen_k - doc1_len - dead_k_len

    lts = np.zeros(seqlen_k, dtype=np.int32)
    ute = np.zeros(seqlen_k, dtype=np.int32)

    # Doc 1 (active): K[0:doc1_len], Q[0:doc1_len]
    for j in range(doc1_len):
        lts[j] = doc1_len
        ute[j] = j  # causal

    # Dead K columns in middle
    k_dead_start = doc1_len
    for j in range(dead_k_len):
        lts[k_dead_start + j] = doc1_len  # lts = doc1_len
        ute[k_dead_start + j] = doc1_len  # ute >= lts → dead

    # Doc 2 (active): K[doc1_len+dead_k_len:], Q[doc1_len:doc1_len+doc2_len]
    q_offset2 = doc1_len
    k_start2 = doc1_len + dead_k_len
    for j in range(doc2_len):
        lts[k_start2 + j] = doc1_len + doc2_len
        ute[k_start2 + j] = q_offset2 + j  # causal

    lts = np.clip(lts, 0, seqlen_q)
    ute = np.clip(ute, 0, seqlen_q)

    lts_t = paddle.to_tensor(lts).reshape([1, 1, seqlen_k, 1])
    ute_t = paddle.to_tensor(ute).reshape([1, 1, seqlen_k, 1])
    sri = paddle.concat([lts_t, ute_t], axis=-1)
    sri = sri.expand([batch_size, 1, seqlen_k, 2])

    causal = False
    return sri, causal


def generate_causal_doc_mask_with_zero_q_tail(batch_size, seqlen_q, seqlen_k, h):
    """Causal document mask with a zero-length Q document at the TAIL.

    Dead K columns at the end.
    """
    dead_k_len = max(1, seqlen_k // 8)
    remaining = seqlen_k - dead_k_len
    doc1_len = remaining // 2
    doc2_len = remaining - doc1_len

    lts = np.zeros(seqlen_k, dtype=np.int32)
    ute = np.zeros(seqlen_k, dtype=np.int32)

    # Doc 1
    for j in range(doc1_len):
        lts[j] = doc1_len
        ute[j] = j

    # Doc 2
    q_offset2 = doc1_len
    k_start2 = doc1_len
    for j in range(doc2_len):
        lts[k_start2 + j] = doc1_len + doc2_len
        ute[k_start2 + j] = q_offset2 + j

    # Dead tail
    k_dead_start = doc1_len + doc2_len
    for j in range(dead_k_len):
        lts[k_dead_start + j] = doc1_len + doc2_len
        ute[k_dead_start + j] = doc1_len + doc2_len  # ute >= lts → dead

    lts = np.clip(lts, 0, seqlen_q)
    ute = np.clip(ute, 0, seqlen_q)

    lts_t = paddle.to_tensor(lts).reshape([1, 1, seqlen_k, 1])
    ute_t = paddle.to_tensor(ute).reshape([1, 1, seqlen_k, 1])
    sri = paddle.concat([lts_t, ute_t], axis=-1)
    sri = sri.expand([batch_size, 1, seqlen_k, 2])

    causal = False
    return sri, causal


def generate_doc_mask_with_q_padding(batch_size, seqlen_q, seqlen_k, h):
    """Non-causal document mask where last doc's lts < seqlen_q (Q padding).

    The last document doesn't cover all Q positions, exercising the Q padding path.
    """
    assert seqlen_q == seqlen_k
    # Only use 3/4 of Q for actual documents, leaving 1/4 as padding
    effective_q = (seqlen_q * 3) // 4
    doc1_len = effective_q // 2
    doc2_len = effective_q - doc1_len

    lts = np.zeros(seqlen_k, dtype=np.int32)
    ute = np.zeros(seqlen_k, dtype=np.int32)

    # Doc 1: K[0:doc1_len], Q[0:doc1_len]
    for j in range(doc1_len):
        lts[j] = doc1_len
        ute[j] = j

    # Doc 2: K[doc1_len:effective_q], Q[doc1_len:effective_q]
    # lts = effective_q (< seqlen_q → triggers Q padding)
    for j in range(doc2_len):
        lts[doc1_len + j] = effective_q
        ute[doc1_len + j] = doc1_len + j

    # Remaining K columns [effective_q:seqlen_k] - dead
    for j in range(effective_q, seqlen_k):
        lts[j] = effective_q
        ute[j] = effective_q  # dead

    lts = np.clip(lts, 0, seqlen_q)
    ute = np.clip(ute, 0, seqlen_q)

    lts_t = paddle.to_tensor(lts).reshape([1, 1, seqlen_k, 1])
    ute_t = paddle.to_tensor(ute).reshape([1, 1, seqlen_k, 1])
    sri = paddle.concat([lts_t, ute_t], axis=-1)
    sri = sri.expand([batch_size, 1, seqlen_k, 2])

    causal = False
    return sri, causal


def generate_doc_mask_diff_batch_with_zero_q(batch_size, seqlen_q, seqlen_k, h):
    """Different doc boundaries per batch, with zero-Q docs in some batches.

    Exercises cross-batch variation AND dead columns.
    """
    assert batch_size >= 2

    all_lts = []
    all_ute = []

    for bi in range(batch_size):
        lts = np.zeros(seqlen_k, dtype=np.int32)
        ute = np.zeros(seqlen_k, dtype=np.int32)

        if bi % 2 == 0:
            # Even batch: dead K at head, then one normal doc
            dead_len = max(1, seqlen_k // 8)
            active_len = seqlen_k - dead_len
            for j in range(dead_len):
                lts[j] = 0
                ute[j] = 0
            for j in range(active_len):
                lts[dead_len + j] = active_len
                ute[dead_len + j] = j
        else:
            # Odd batch: normal doc then dead K at tail
            active_len = (seqlen_k * 3) // 4
            dead_len = seqlen_k - active_len
            for j in range(active_len):
                lts[j] = active_len
                ute[j] = j
            for j in range(dead_len):
                lts[active_len + j] = active_len
                ute[active_len + j] = active_len

        lts = np.clip(lts, 0, seqlen_q)
        ute = np.clip(ute, 0, seqlen_q)
        all_lts.append(lts)
        all_ute.append(ute)

    lts_np = np.stack(all_lts, axis=0)  # (batch, seqlen_k)
    ute_np = np.stack(all_ute, axis=0)

    lts_t = paddle.to_tensor(lts_np).reshape([batch_size, 1, seqlen_k, 1])
    ute_t = paddle.to_tensor(ute_np).reshape([batch_size, 1, seqlen_k, 1])
    sri = paddle.concat([lts_t, ute_t], axis=-1)

    causal = False
    return sri, causal


def generate_doc_mask_with_padding_between_batches(batch_size, seqlen_q, seqlen_k, h):
    """Each batch has docs that leave padding at the end (dead K between batches conceptually).

    This tests that batch items with different Q padding are handled correctly.
    """
    assert batch_size >= 2

    all_lts = []
    all_ute = []

    for bi in range(batch_size):
        lts = np.zeros(seqlen_k, dtype=np.int32)
        ute = np.zeros(seqlen_k, dtype=np.int32)

        # Use different effective lengths per batch to create varying padding
        effective_len = seqlen_k - (bi + 1) * max(1, seqlen_k // (batch_size * 4))
        effective_len = max(seqlen_k // 2, effective_len)  # at least half

        # Two docs that together fill effective_len
        doc1_len = effective_len // 3
        doc2_len = effective_len - doc1_len

        # Doc 1
        for j in range(doc1_len):
            lts[j] = doc1_len
            ute[j] = j

        # Doc 2
        for j in range(doc2_len):
            lts[doc1_len + j] = doc1_len + doc2_len
            ute[doc1_len + j] = doc1_len + j

        # Dead tail (padding)
        for j in range(effective_len, seqlen_k):
            lts[j] = effective_len
            ute[j] = effective_len  # dead

        lts = np.clip(lts, 0, seqlen_q)
        ute = np.clip(ute, 0, seqlen_q)
        all_lts.append(lts)
        all_ute.append(ute)

    lts_np = np.stack(all_lts, axis=0)
    ute_np = np.stack(all_ute, axis=0)

    lts_t = paddle.to_tensor(lts_np).reshape([batch_size, 1, seqlen_k, 1])
    ute_t = paddle.to_tensor(ute_np).reshape([batch_size, 1, seqlen_k, 1])
    sri = paddle.concat([lts_t, ute_t], axis=-1)

    causal = False
    return sri, causal


# ─────────────────────────────────────────────────────────────────────────────
# Part A: Non-CP tests (single-call path)
# ─────────────────────────────────────────────────────────────────────────────

# Test cases covering: causal/non-causal, various seqlens, zero-Q, Q-padding, cross-batch
part_a_shape_cases = [
    # (batch_size, seqlen_q, seqlen_k, nheads, nheads_kv)
    # Small shapes
    (1, 128, 128, 4, 4),
    (2, 256, 256, 4, 2),
    (2, 512, 512, 8, 2),
    # Medium shapes
    (1, 1024, 1024, 4, 1),
    (2, 1000, 1000, 4, 1),
    (2, 2000, 2000, 4, 1),
    # Large shapes
    (2, 4096, 4096, 4, 1),
    (2, 8192, 8192, 4, 1),
]

part_a_mask_generators = [
    # Causal masks (bound_num=1) - exercises _convert_to_varlen_bound1
    ("causal_doc", partial(generate_causal_document_mask)),
    ("causal_doc_diff_batch", partial(generate_causal_document_mask_diff_batch)),
    # Non-causal masks (bound_num=2) - exercises _convert_to_varlen_bound2
    ("noncausal_doc", partial(generate_document_mask)),
    ("noncausal_doc_diff_batch", partial(generate_document_mask_diff_batch)),
    # Simulated causal via bound2 (tests causal detection in bound2 path)
    ("simu_causal", partial(generate_document_mask_simu)),
    ("simu_causal_diff_batch", partial(generate_document_mask_diff_batch_simu)),
    # Zero-Q document at head/middle/tail
    ("zero_q_head", generate_causal_doc_mask_with_zero_q_head),
    ("zero_q_middle", generate_causal_doc_mask_with_zero_q_middle),
    ("zero_q_tail", generate_causal_doc_mask_with_zero_q_tail),
    # Q padding
    ("q_padding", generate_doc_mask_with_q_padding),
    # Cross-batch with dead columns
    ("diff_batch_zero_q", generate_doc_mask_diff_batch_with_zero_q),
    # Padding between batches
    ("padding_between_batches", generate_doc_mask_with_padding_between_batches),
]


@pytest.mark.parametrize("dtype", [paddle.bfloat16])
@pytest.mark.parametrize("d, dv", [(192, 128), (256, 256)])
@pytest.mark.parametrize(
    "batch_size, seqlen_q, seqlen_k, nheads, nheads_kv",
    part_a_shape_cases,
)
@pytest.mark.parametrize("softmax_scale", [None])
@pytest.mark.parametrize(
    "mask_name, gen_mask",
    part_a_mask_generators,
    ids=[name for name, _ in part_a_mask_generators],
)
def test_use_varlen_comprehensive(
    batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, d, dv,
    dtype, softmax_scale, mask_name, gen_mask,
):
    """Comprehensive test for use_varlen covering all edge cases."""
    paddle.seed(2024)

    q = paddle.randn(shape=[batch_size, seqlen_q, nheads, d], dtype=dtype)
    k = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, d], dtype=dtype)
    v = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, dv], dtype=dtype)

    startend_row_indices, causal = gen_mask(batch_size, seqlen_q, seqlen_k, 1)

    label = f"{mask_name}_b{batch_size}_sq{seqlen_q}_d{d}"
    _run_forward_backward_check(
        q, k, v, startend_row_indices, causal, softmax_scale, label=label,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Part B: Dual-chunk CP tests (multi-call path)
# ─────────────────────────────────────────────────────────────────────────────

def generate_causal_document_sri_bound2(batch_size, seqlen, doc_seqlens):
    """Generate causal document startend_row_indices with bound_num=2."""
    total = sum(doc_seqlens)
    assert total <= seqlen
    padding = seqlen - total
    if padding > 0:
        doc_seqlens = list(doc_seqlens)
        doc_seqlens[-1] += padding

    seq_cusums = np.cumsum(doc_seqlens)
    lts = np.repeat(seq_cusums, doc_seqlens).astype(np.int32)

    ute = np.zeros(seqlen, dtype=np.int32)
    offset = 0
    for doc_len in doc_seqlens:
        for j in range(doc_len):
            ute[offset + j] = offset + j
        offset += doc_len

    lts_t = paddle.to_tensor(lts).reshape([1, 1, seqlen, 1])
    ute_t = paddle.to_tensor(ute).reshape([1, 1, seqlen, 1])
    sri = paddle.concat([lts_t, ute_t], axis=-1)
    sri = sri.expand([batch_size, 1, seqlen, 2])
    return sri


def simulate_dual_chunk_cp(
    batch_size, seqlen, cp_size, rank, nheads, nheads_kv, d, dv, doc_seqlens, dtype,
):
    """Simulate dual chunk CP for a given rank."""
    assert seqlen % (2 * cp_size) == 0
    seq_blocksize = seqlen // (2 * cp_size)
    local_seqlen = 2 * seq_blocksize

    q_full = paddle.randn([batch_size, seqlen, nheads, d], dtype=dtype)
    k_full = paddle.randn([batch_size, seqlen, nheads_kv, d], dtype=dtype)
    v_full = paddle.randn([batch_size, seqlen, nheads_kv, dv], dtype=dtype)

    chunk_id_first = rank
    chunk_id_second = 2 * cp_size - rank - 1

    q_first = q_full[:, chunk_id_first * seq_blocksize:(chunk_id_first + 1) * seq_blocksize, :, :]
    q_second = q_full[:, chunk_id_second * seq_blocksize:(chunk_id_second + 1) * seq_blocksize, :, :]
    q_local = paddle.concat([q_first, q_second], axis=1)

    sri_global = generate_causal_document_sri_bound2(batch_size, seqlen, doc_seqlens)

    sri_local = preprocess_index_dual_chunks(
        sri_global,
        chunk_id_first=chunk_id_first,
        chunk_id_second=chunk_id_second,
        seq_blocksize=seq_blocksize,
        max_seqlen_q=seq_blocksize,
    )

    return q_local, k_full, v_full, sri_local, q_full, sri_global


# Dual-chunk CP test cases:
# (seqlen, cp_size, doc_seqlens, nheads, nheads_kv, batch_size)
# Covers: many seqlens, many cp_sizes, single/multi doc, Q padding scenarios
dual_chunk_cases = [
    # Single document, basic CP sizes
    (128, 2, [128], 4, 2, 1),
    (256, 2, [256], 4, 2, 2),
    (512, 2, [512], 4, 2, 2),
    (512, 4, [512], 4, 2, 2),
    (1024, 2, [1024], 4, 2, 1),
    (1024, 4, [1024], 4, 2, 1),
    (1024, 8, [1024], 4, 2, 1),
    # Multiple documents (tests doc boundary handling in CP)
    (256, 2, [128, 128], 4, 2, 2),
    (512, 2, [256, 256], 4, 2, 2),
    (512, 2, [128, 256, 128], 4, 2, 1),
    (512, 4, [128, 128, 128, 128], 4, 2, 2),
    (1024, 2, [512, 512], 4, 1, 2),
    (1024, 4, [256, 256, 256, 256], 4, 1, 1),
    # Unequal document sizes (tests Q padding in dual-chunk)
    (256, 2, [96, 160], 4, 2, 2),
    (512, 2, [100, 200, 212], 4, 2, 2),
    (1024, 2, [300, 400, 324], 4, 1, 1),
    # Larger seqlens
    (2048, 2, [2048], 4, 1, 1),
    (2048, 4, [2048], 4, 1, 1),
    (2048, 2, [1024, 1024], 4, 1, 1),
    (4096, 2, [4096], 4, 1, 1),
    (4096, 4, [4096], 4, 1, 1),
    (4096, 2, [1500, 1200, 1396], 4, 1, 1),
    (8192, 2, [8192], 4, 1, 1),
    (8192, 2, [2538, 1742, 3213], 4, 1, 1),
    (8192, 4, [8192], 4, 1, 1),
    (8192, 4, [2538, 1742, 3213], 4, 1, 1),
    (8192, 8, [8192], 4, 1, 1),
]


@pytest.mark.parametrize("dtype", [paddle.bfloat16])
@pytest.mark.parametrize("d, dv", [(192, 128), (256, 256)])
@pytest.mark.parametrize(
    "seqlen, cp_size, doc_seqlens, nheads, nheads_kv, batch_size",
    dual_chunk_cases,
)
@pytest.mark.parametrize("softmax_scale", [None])
def test_dual_chunk_cp_comprehensive(
    seqlen, cp_size, doc_seqlens, nheads, nheads_kv, batch_size,
    d, dv, dtype, softmax_scale,
):
    """Comprehensive test for dual-chunk CP path (multi-call)."""
    paddle.seed(2024)

    for rank in range(cp_size):
        q_local, k_full, v_full, sri_local, q_full, sri_global = simulate_dual_chunk_cp(
            batch_size, seqlen, cp_size, rank, nheads, nheads_kv, d, dv, doc_seqlens, dtype,
        )

        local_seqlen = q_local.shape[1]
        label = f"cp_seq{seqlen}_cp{cp_size}_rank{rank}_docs{doc_seqlens}_d{d}_b{batch_size}"

        _run_forward_backward_check(
            q_local, k_full, v_full, sri_local,
            causal=False,
            softmax_scale=softmax_scale,
            label=label,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Part C: Focused edge-case tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", [paddle.bfloat16])
@pytest.mark.parametrize("d, dv", [(192, 128), (256, 256)])
def test_single_doc_causal_bound1(dtype, d, dv):
    """Single causal document, bound_num=1, various sizes - simplest path."""
    paddle.seed(2024)
    cases = [
        (1, 64, 4, 4),
        (2, 128, 4, 2),
        (1, 256, 8, 1),
        (2, 512, 4, 1),
        (1, 1024, 4, 1),
    ]
    for batch_size, seqlen, nheads, nheads_kv in cases:
        q = paddle.randn([batch_size, seqlen, nheads, d], dtype=dtype)
        k = paddle.randn([batch_size, seqlen, nheads_kv, d], dtype=dtype)
        v = paddle.randn([batch_size, seqlen, nheads_kv, dv], dtype=dtype)

        # Single doc covering full seqlen
        lts = paddle.full([batch_size, 1, seqlen, 1], seqlen, dtype=paddle.int32)
        causal = True

        _run_forward_backward_check(
            q, k, v, lts, causal, softmax_scale=None,
            label=f"single_doc_causal_b{batch_size}_s{seqlen}",
        )


@pytest.mark.parametrize("dtype", [paddle.bfloat16])
@pytest.mark.parametrize("d, dv", [(192, 128), (256, 256)])
def test_all_zero_q_docs(dtype, d, dv):
    """Edge case: all K columns are dead (all zero-Q docs).

    This tests the extreme where no Q row attends to any K column.
    """
    paddle.seed(2024)
    batch_size = 2
    seqlen = 256
    nheads = 4
    nheads_kv = 2

    q = paddle.randn([batch_size, seqlen, nheads, d], dtype=dtype)
    k = paddle.randn([batch_size, seqlen, nheads_kv, d], dtype=dtype)
    v = paddle.randn([batch_size, seqlen, nheads_kv, dv], dtype=dtype)

    # All dead: lts=0, ute=0 for all K columns
    lts = np.zeros(seqlen, dtype=np.int32)
    ute = np.zeros(seqlen, dtype=np.int32)

    lts_t = paddle.to_tensor(lts).reshape([1, 1, seqlen, 1])
    ute_t = paddle.to_tensor(ute).reshape([1, 1, seqlen, 1])
    sri = paddle.concat([lts_t, ute_t], axis=-1)
    sri = sri.expand([batch_size, 1, seqlen, 2])

    _run_forward_backward_check(
        q, k, v, sri, causal=False, softmax_scale=None,
        label="all_zero_q_docs",
    )


@pytest.mark.parametrize("dtype", [paddle.bfloat16])
@pytest.mark.parametrize("d, dv", [(192, 128), (256, 256)])
def test_many_small_docs(dtype, d, dv):
    """Many small documents (stress tests boundary detection)."""
    paddle.seed(2024)
    batch_size = 2
    seqlen = 1024
    nheads = 4
    nheads_kv = 2
    num_docs = 16
    doc_len = seqlen // num_docs

    q = paddle.randn([batch_size, seqlen, nheads, d], dtype=dtype)
    k = paddle.randn([batch_size, seqlen, nheads_kv, d], dtype=dtype)
    v = paddle.randn([batch_size, seqlen, nheads_kv, dv], dtype=dtype)

    # bound_num=2, causal within each doc
    lts = np.zeros(seqlen, dtype=np.int32)
    ute = np.zeros(seqlen, dtype=np.int32)

    offset = 0
    for doc_idx in range(num_docs):
        for j in range(doc_len):
            lts[offset + j] = offset + doc_len
            ute[offset + j] = offset + j
        offset += doc_len

    lts_t = paddle.to_tensor(lts).reshape([1, 1, seqlen, 1])
    ute_t = paddle.to_tensor(ute).reshape([1, 1, seqlen, 1])
    sri = paddle.concat([lts_t, ute_t], axis=-1)
    sri = sri.expand([batch_size, 1, seqlen, 2])

    _run_forward_backward_check(
        q, k, v, sri, causal=False, softmax_scale=None,
        label=f"many_small_docs_{num_docs}x{doc_len}",
    )


@pytest.mark.parametrize("dtype", [paddle.bfloat16])
@pytest.mark.parametrize("d, dv", [(192, 128), (256, 256)])
def test_alternating_dead_alive_docs(dtype, d, dv):
    """Alternating dead and alive documents (stress boundary detection)."""
    paddle.seed(2024)
    batch_size = 2
    seqlen = 512
    nheads = 4
    nheads_kv = 2

    q = paddle.randn([batch_size, seqlen, nheads, d], dtype=dtype)
    k = paddle.randn([batch_size, seqlen, nheads_kv, d], dtype=dtype)
    v = paddle.randn([batch_size, seqlen, nheads_kv, dv], dtype=dtype)

    # Pattern: alive(64) - dead(32) - alive(64) - dead(32) - ...
    alive_len = 64
    dead_len = 32
    segment_len = alive_len + dead_len

    lts = np.zeros(seqlen, dtype=np.int32)
    ute = np.zeros(seqlen, dtype=np.int32)

    q_offset = 0
    k_pos = 0
    while k_pos < seqlen:
        remaining = seqlen - k_pos
        # Alive segment
        cur_alive = min(alive_len, remaining)
        for j in range(cur_alive):
            lts[k_pos + j] = q_offset + cur_alive
            ute[k_pos + j] = q_offset + j
        q_offset += cur_alive
        k_pos += cur_alive

        if k_pos >= seqlen:
            break

        # Dead segment
        cur_dead = min(dead_len, seqlen - k_pos)
        for j in range(cur_dead):
            lts[k_pos + j] = q_offset
            ute[k_pos + j] = q_offset  # ute >= lts → dead
        k_pos += cur_dead

    lts = np.clip(lts, 0, seqlen)
    ute = np.clip(ute, 0, seqlen)

    lts_t = paddle.to_tensor(lts).reshape([1, 1, seqlen, 1])
    ute_t = paddle.to_tensor(ute).reshape([1, 1, seqlen, 1])
    sri = paddle.concat([lts_t, ute_t], axis=-1)
    sri = sri.expand([batch_size, 1, seqlen, 2])

    _run_forward_backward_check(
        q, k, v, sri, causal=False, softmax_scale=None,
        label="alternating_dead_alive",
    )


@pytest.mark.parametrize("dtype", [paddle.bfloat16])
@pytest.mark.parametrize("d, dv", [(192, 128), (256, 256)])
def test_large_batch_different_docs(dtype, d, dv):
    """Large batch size with very different doc patterns per batch item."""
    paddle.seed(2024)
    batch_size = 4
    seqlen = 512
    nheads = 4
    nheads_kv = 2

    q = paddle.randn([batch_size, seqlen, nheads, d], dtype=dtype)
    k = paddle.randn([batch_size, seqlen, nheads_kv, d], dtype=dtype)
    v = paddle.randn([batch_size, seqlen, nheads_kv, dv], dtype=dtype)

    # Different doc configs per batch
    doc_configs = [
        [512],                      # single doc
        [256, 256],                 # two equal docs
        [100, 200, 212],            # three unequal docs
        [64, 64, 64, 64, 64, 64, 64, 64],  # many small docs (sum=512)
    ]

    all_lts = []
    all_ute = []

    for bi in range(batch_size):
        doc_seqlens = doc_configs[bi % len(doc_configs)]
        total = sum(doc_seqlens)
        padding = seqlen - total

        lts_bi = np.zeros(seqlen, dtype=np.int32)
        ute_bi = np.zeros(seqlen, dtype=np.int32)

        offset = 0
        for doc_len in doc_seqlens:
            for j in range(doc_len):
                lts_bi[offset + j] = offset + doc_len
                ute_bi[offset + j] = offset + j
            offset += doc_len

        # Dead padding at end
        for j in range(padding):
            lts_bi[offset + j] = offset
            ute_bi[offset + j] = offset

        lts_bi = np.clip(lts_bi, 0, seqlen)
        ute_bi = np.clip(ute_bi, 0, seqlen)
        all_lts.append(lts_bi)
        all_ute.append(ute_bi)

    lts_np = np.stack(all_lts, axis=0)
    ute_np = np.stack(all_ute, axis=0)

    lts_t = paddle.to_tensor(lts_np).reshape([batch_size, 1, seqlen, 1])
    ute_t = paddle.to_tensor(ute_np).reshape([batch_size, 1, seqlen, 1])
    sri = paddle.concat([lts_t, ute_t], axis=-1)

    _run_forward_backward_check(
        q, k, v, sri, causal=False, softmax_scale=None,
        label=f"large_batch_diff_docs_d{d}",
    )


