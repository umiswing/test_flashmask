"""
Test: flashmask_attention(..., use_varlen=True) with dual chunk context parallel strategy.

Simulates the DualChunkSwap CP strategy where:
- Global sequence has seqlen tokens split across cp_size ranks
- Each rank holds Q of shape [batch, seqlen//cp_size, heads, dim] (two chunks from both ends)
- After all-gather, K/V have full shape [batch, seqlen, heads, dim]
- startend_row_indices are preprocessed via preprocess_index_dual_chunks

This creates asymmetric seqlen_q != seqlen_k which exercises convert_to_varlen's bound2 path.
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "flash-attention", "flashmask"))

import paddle

# from paddlefleet.ops.flash_mask import flashmask_attention
from flash_mask import flashmask_attention
from generate_startend_row_indices import startend_row_indices_to_attn_bias
from context_parallel_utils import preprocess_index_dual_chunks
from test_util import attention_ref


def generate_causal_document_sri(batch_size, seqlen, doc_seqlens):
    """Generate causal document startend_row_indices for the global sequence.

    Returns shape (batch, 1, seqlen, 1) with bound_num=1 (lts only, causal).
    """
    total = sum(doc_seqlens)
    assert total <= seqlen
    padding = seqlen - total
    if padding > 0:
        doc_seqlens = list(doc_seqlens)
        doc_seqlens[-1] += padding

    seq_cusums = np.cumsum(doc_seqlens)
    lts = np.repeat(seq_cusums, doc_seqlens)
    lts = paddle.to_tensor(lts, dtype=paddle.int32).reshape([1, 1, seqlen, 1])
    lts = lts.expand([batch_size, 1, seqlen, 1])
    return lts


def generate_causal_document_sri_bound2(batch_size, seqlen, doc_seqlens):
    """Generate causal document startend_row_indices with bound_num=2 (lts + ute).

    For a causal document mask with sq == sk per doc, the ute pattern is:
        ute[k_offset + j] = q_offset + max(0, j - (sk - sq))
    Since sq == sk within each doc: ute[k_offset + j] = q_offset + j

    Returns shape (batch, 1, seqlen, 2).
    """
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
    """Simulate dual chunk CP for a given rank.

    Returns:
        q_local: [batch, seqlen//cp_size, nheads, d] — the two chunks concatenated
        k_full:  [batch, seqlen, nheads_kv, d] — full K after all-gather
        v_full:  [batch, seqlen, nheads_kv, dv] — full V after all-gather
        sri_local: preprocessed startend_row_indices for this rank
        q_full:  [batch, seqlen, nheads, d] — full Q for reference computation
        sri_global: global startend_row_indices (bound_num=2) for reference
    """
    assert seqlen % (2 * cp_size) == 0
    seq_blocksize = seqlen // (2 * cp_size)
    local_seqlen = 2 * seq_blocksize  # seqlen // cp_size

    q_full = paddle.randn([batch_size, seqlen, nheads, d], dtype=dtype)
    k_full = paddle.randn([batch_size, seqlen, nheads_kv, d], dtype=dtype)
    v_full = paddle.randn([batch_size, seqlen, nheads_kv, dv], dtype=dtype)

    # Extract Q chunks for this rank (DualChunkSwap: first chunk from start, second from end)
    chunk_id_first = rank
    chunk_id_second = 2 * cp_size - rank - 1

    q_first = q_full[:, chunk_id_first * seq_blocksize:(chunk_id_first + 1) * seq_blocksize, :, :]
    q_second = q_full[:, chunk_id_second * seq_blocksize:(chunk_id_second + 1) * seq_blocksize, :, :]
    q_local = paddle.concat([q_first, q_second], axis=1)  # [batch, local_seqlen, nheads, d]

    # Generate global startend_row_indices (bound_num=2 for asymmetric support)
    sri_global = generate_causal_document_sri_bound2(batch_size, seqlen, doc_seqlens)

    # Preprocess indices for this rank's dual chunks
    sri_local = preprocess_index_dual_chunks(
        sri_global,
        chunk_id_first=chunk_id_first,
        chunk_id_second=chunk_id_second,
        seq_blocksize=seq_blocksize,
        max_seqlen_q=seq_blocksize,
    )

    return q_local, k_full, v_full, sri_local, q_full, sri_global


def compute_reference_dual_chunk(
    q_full, k_full, v_full, sri_local, seqlen, cp_size, rank, nheads, dtype, softmax_scale,
):
    """Compute reference attention for dual chunk CP rank using naive attention.

    Build attn_bias from the processed sri_local, then run attention_ref with
    the local Q and full K/V.
    """
    seq_blocksize = seqlen // (2 * cp_size)
    local_seqlen = 2 * seq_blocksize

    chunk_id_first = rank
    chunk_id_second = 2 * cp_size - rank - 1

    q_first = q_full[:, chunk_id_first * seq_blocksize:(chunk_id_first + 1) * seq_blocksize, :, :]
    q_second = q_full[:, chunk_id_second * seq_blocksize:(chunk_id_second + 1) * seq_blocksize, :, :]
    q_local = paddle.concat([q_first, q_second], axis=1)

    # sri_local has shape [batch, 1, seqlen_k, bound_num] after preprocess
    _, _, seqlen_k, bound_num = sri_local.shape

    attn_bias = startend_row_indices_to_attn_bias(
        sri_local, local_seqlen, nheads, dtype, causal=False
    )

    out_ref, _ = attention_ref(
        q_local, k_full, v_full,
        causal=False,
        attn_bias=attn_bias,
        softmax_scale=softmax_scale,
    )
    return out_ref


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

# (seqlen, cp_size, doc_seqlens, nheads, nheads_kv)
dual_chunk_cases = [
    # Single document, full sequence
    (256, 2, [256], 4, 2),
    (512, 2, [512], 4, 2),
    (512, 4, [512], 4, 2),
    # Multiple documents
    (256, 2, [128, 128], 4, 2),
    (512, 2, [256, 256], 4, 2),
    (512, 2, [128, 256, 128], 4, 2),
    (512, 4, [128, 128, 128, 128], 4, 2),
    # Unequal document sizes
    (256, 2, [96, 160], 4, 2),
    (512, 2, [100, 200, 212], 4, 2),
    # Large seqlen cases (exercises HD256 kernel path with d=256)
    (8192, 2, [8192], 4, 1),
    (8192, 2, [2538, 1742, 3213], 4, 1),
    (8192, 4, [8192], 4, 1),
    (8192, 4, [2538, 1742, 3213], 4, 1),
]


@pytest.mark.parametrize("dtype", [paddle.bfloat16])
@pytest.mark.parametrize("d, dv", [(128, 128), (256, 256)])
@pytest.mark.parametrize(
    "seqlen, cp_size, doc_seqlens, nheads, nheads_kv",
    dual_chunk_cases,
)
@pytest.mark.parametrize("softmax_scale", [None])
def test_dual_chunk_cp_use_varlen(
    seqlen, cp_size, doc_seqlens, nheads, nheads_kv, d, dv, dtype, softmax_scale,
):
    """Test flashmask_attention(use_varlen=True) with dual chunk CP asymmetric shapes."""
    paddle.seed(2024)
    batch_size = 2

    for rank in range(cp_size):
        q_local, k_full, v_full, sri_local, q_full, sri_global = simulate_dual_chunk_cp(
            batch_size, seqlen, cp_size, rank, nheads, nheads_kv, d, dv, doc_seqlens, dtype,
        )

        local_seqlen = q_local.shape[1]
        assert local_seqlen == seqlen // cp_size
        assert k_full.shape[1] == seqlen  # asymmetric: seqlen_q != seqlen_k

        attn_bias = startend_row_indices_to_attn_bias(
            sri_local, local_seqlen, nheads, dtype, causal=False
        )

        # Reference
        q_ref = q_local.detach().clone()
        k_ref = k_full.detach().clone()
        v_ref = v_full.detach().clone()
        q_ref.stop_gradient = False
        k_ref.stop_gradient = False
        v_ref.stop_gradient = False

        out_ref, _ = attention_ref(
            q_ref, k_ref, v_ref,
            causal=False,
            attn_bias=attn_bias,
            softmax_scale=softmax_scale,
        )

        # bf16 reference for tolerance
        q_bf16 = q_local.detach().clone()
        k_bf16 = k_full.detach().clone()
        v_bf16 = v_full.detach().clone()
        q_bf16.stop_gradient = False
        k_bf16.stop_gradient = False
        v_bf16.stop_gradient = False
        out_bf16, _ = attention_ref(
            q_bf16, k_bf16, v_bf16,
            causal=False,
            attn_bias=attn_bias,
            upcast=False,
            reorder_ops=True,
            softmax_scale=softmax_scale,
        )

        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        rtol = 2
        bf16_diff = (out_bf16 - out_ref).abs().max().item()

        # flashmask with use_varlen
        q_test = q_local.detach().clone()
        k_test = k_full.detach().clone()
        v_test = v_full.detach().clone()
        q_test.stop_gradient = False
        k_test.stop_gradient = False
        v_test.stop_gradient = False

        paddle.set_flags({"FLAGS_flash_attn_version": 4})
        out = flashmask_attention(
            q_test, k_test, v_test,
            startend_row_indices=sri_local,
            causal=False,
            return_softmax_lse=False,
            use_varlen=True,
            softmax_scale=softmax_scale,
        )

        max_diff = (out - out_ref).abs().max().item()
        print(f"  rank={rank}, seqlen={seqlen}, cp={cp_size}, "
              f"sq={local_seqlen}, sk={seqlen}, d={d}: "
              f"max_diff={max_diff:.6f}, bf16_diff={bf16_diff:.6f}, fwd_atol={fwd_atol:.6f}")

        assert max_diff <= rtol * bf16_diff + fwd_atol, (
            f"rank={rank}: max_diff={max_diff} > rtol*bf16_diff+atol={rtol * bf16_diff + fwd_atol}"
        )

        # Backward check
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

        assert dq_diff <= rtol * dq_bf16_diff + dq_atol, f"rank={rank}: dQ too large"
        assert dk_diff <= rtol * dk_bf16_diff + dk_atol, f"rank={rank}: dK too large"
        assert dv_diff <= rtol * dv_bf16_diff + dv_atol, f"rank={rank}: dV too large"


if __name__ == "__main__":
    print("=" * 70)
    print("Test: Dual Chunk CP with flashmask_attention use_varlen=True")
    print("=" * 70)

    for seqlen, cp_size, doc_seqlens, nheads, nheads_kv in dual_chunk_cases:
        for d, dv in [(128, 128), (256, 256)]:
            print(f"\n  seqlen={seqlen}, cp={cp_size}, docs={doc_seqlens}, d={d}")
            try:
                test_dual_chunk_cp_use_varlen(
                    seqlen=seqlen,
                    cp_size=cp_size,
                    doc_seqlens=doc_seqlens,
                    nheads=nheads,
                    nheads_kv=nheads_kv,
                    d=d,
                    dv=dv,
                    dtype=paddle.bfloat16,
                    softmax_scale=None,
                )
                print("  PASS")
            except AssertionError as e:
                print(f"  FAIL: {e}")

    print("\nAll tests done.")
