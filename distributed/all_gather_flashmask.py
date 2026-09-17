"""All-gather reference for context-parallel FlashMask (DualChunkSwap).

Accuracy baseline the overlap path is compared against: gather full K/V on every
rank, run the plain (no-overlap) FA-v4 SM100 kernel on the local query against the
full K/V, then reduce-scatter dK/dV back to the local dual-chunk share. Same
partition and same in-library kernel as the overlap path, so per-rank outputs are
directly comparable and differ only by floating-point reduction order.

Trimmed from dist_flashmask_dev/all_gather_flashmask.py to the dual_chunk
"all-gather" path only; dumps, profiling, the PHI fallback and the
contiguous/balance modes were dropped. Compute calls are unchanged.
"""

import paddle
from paddle.autograd.py_layer import PyLayer
from paddle.distributed import fleet

from context_parallel_utils import (
    all_gather_balance,
    reduce_scatter_any_axis_balance,
    preprocess_index_dual_chunks,
)

# FA-v4 SM100 in-library kernel (the test forces FLAGS_flash_attn_version=4 on
# Blackwell, so this import is expected to succeed).
from flash_mask.cute.interface import _flash_attn_fwd, _flash_attn_bwd


def _deterministic():
    return paddle.get_flags(["FLAGS_cudnn_deterministic"])["FLAGS_cudnn_deterministic"]


def _forward(query, key, value, startend_row_indices, group, causal):
    rank = group.rank
    cp_size = group.world_size

    key_gathered = all_gather_balance(key, group=group, axis=1)
    value_gathered = all_gather_balance(value, group=group, axis=1)

    seq_blocksize = query.shape[1] // 2
    startend_row_indices = preprocess_index_dual_chunks(
        startend_row_indices,
        chunk_id_first=rank,
        chunk_id_second=2 * cp_size - rank - 1,
        seq_blocksize=seq_blocksize,
        max_seqlen_q=seq_blocksize,
    )

    output, log_sum_exp = _flash_attn_fwd(
        query,
        key_gathered,
        value_gathered,
        startend_row_indices=startend_row_indices,
        causal=causal,
        return_lse=True,
        pack_gqa=False,
    )
    return output, log_sum_exp, startend_row_indices


def _backward(query, key, value, startend_row_indices, output, log_sum_exp, output_grad, group, causal):
    key_gathered = all_gather_balance(key, group=group, axis=1)
    value_gathered = all_gather_balance(value, group=group, axis=1)

    query_grad, key_grad_gathered, value_grad_gathered, _ = _flash_attn_bwd(
        query,
        key_gathered,
        value_gathered,
        output,
        output_grad,
        log_sum_exp,
        flashmask_info=startend_row_indices,
        causal=causal,
        deterministic=_deterministic(),
    )

    key_grad = reduce_scatter_any_axis_balance(key_grad_gathered, axis=1, group=group)
    value_grad = reduce_scatter_any_axis_balance(value_grad_gathered, axis=1, group=group)
    return query_grad, key_grad, value_grad


class FlashMaskContextParallel(PyLayer):
    """All-gather CP FlashMask reference (DualChunkSwap)."""

    @staticmethod
    def forward(ctx, query, key, value, startend_row_indices, causal=False, group=None):
        if causal:
            raise NotImplementedError("CP FlashMask reference does not support causal=True.")
        if group is None:
            group = fleet.get_hybrid_communicate_group().get_context_parallel_group()
        assert query.shape[1] % 2 == 0, (
            f"Query sequence length {query.shape[1]} must be divisible by 2 (DualChunkSwap)."
        )

        output, log_sum_exp, startend_row_indices = _forward(
            query, key, value, startend_row_indices, group, causal
        )
        ctx.save_for_backward(query, key, value, output, log_sum_exp, startend_row_indices)
        ctx.group = group
        ctx.causal = causal
        return output

    @staticmethod
    def backward(ctx, output_grad):
        query, key, value, output, log_sum_exp, startend_row_indices = ctx.saved_tensor()
        return _backward(
            query, key, value, startend_row_indices, output, log_sum_exp,
            output_grad, ctx.group, ctx.causal,
        )


def flashmask_attention_cp(query, key, value, startend_row_indices, causal=False, group=None, mode="all-gather"):
    """Public API: all-gather CP FlashMask attention (reference path)."""
    assert mode == "all-gather", f"reference only implements all-gather, got {mode}"
    return FlashMaskContextParallel.apply(query, key, value, startend_row_indices, causal, group)
