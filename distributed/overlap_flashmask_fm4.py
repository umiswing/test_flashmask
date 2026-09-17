"""FM-4 context-parallel FlashMask with distributed overlap (DualChunkSwap).

The path under test: local query + local K/V are handed to the FA-v4 SM100
in-library kernel together with the CP ``group``; the kernel gathers the remote
K/V internally (sparse all-gather runtime) while computing attention, overlapping
communication with compute, and reduce-scatters dK/dV back to the local share
(``USE_RS_OVERLAP``) so the PyLayer returns local-shaped gradients directly.

Trimmed from dist_flashmask_dev/overlap_flashmask_fm4.py to the dual_chunk overlap
path only; dumps, profiling and the PHI fallback were dropped. Compute calls are
unchanged.

Mask rolling asymmetry: forward consumes K/V in FORWARD-SR order (local chunk
last) so the mask is rolled by ``-(rank+1)*S_local``; the backward all-gather
lands K/V in BACKWARD-SR order (local chunk first) so it is re-rolled by
``-rank*S_local``. For CP that spans multiple nodes (hierarchical overlap) the
roll is replaced by an index-select permutation matching the Rail/LSA order.
"""

import os

import paddle
from paddle import distributed as dist
from paddle.autograd.py_layer import PyLayer
from paddle.distributed import fleet

from context_parallel_utils import preprocess_index_dual_chunks

from flash_mask.cute.interface import _flash_attn_fwd, _flash_attn_bwd

# dK/dV are reduced-scattered to the local share inside the overlap kernel, so the
# backward returns them as-is. Set False only for a build whose kernel emits
# gathered (full-sequence) dK/dV.
USE_RS_OVERLAP = True


# ── DualChunkSwap mask reorder + fallback reduce-scatter (from dev overlap_flashmask.py) ──
_REORDER_IDX_CACHE = {}


def _dual_chunk_reorder_idx(cp_size):
    idx = _REORDER_IDX_CACHE.get(cp_size)
    if idx is None:
        n_blocks = cp_size * 2
        order = []
        for i in range(cp_size):
            order.append(i)
            order.append(n_blocks - 1 - i)
        idx = paddle.to_tensor(order, dtype="int64")
        _REORDER_IDX_CACHE[cp_size] = idx
    return idx


def rearrange_blocks(input_tensor, cp_size):
    """Interleave the 2*cp mask blocks into per-rank dual-chunk order."""
    B, _, S, _ = input_tensor.shape
    n_blocks = cp_size * 2
    block_size = S // n_blocks
    blocks = input_tensor.reshape([B, -1, n_blocks, block_size, 2])
    reordered = blocks.index_select(index=_dual_chunk_reorder_idx(cp_size), axis=2)
    return reordered.reshape([B, -1, S, 2])


def reduce_scatter_any_axis_simpler(input_tensor, axis, group):
    """Alltoall reduce-scatter in local (PE) order; used only when the kernel does
    NOT reduce-scatter dK/dV itself (USE_RS_OVERLAP=False)."""
    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor
    rank = group.rank
    assert input_tensor.shape[axis] % parallelism == 0
    chunks = paddle.split(input_tensor, parallelism, axis=axis)
    ordered = chunks[-rank:] + chunks[:-rank] if rank else list(chunks)
    buffers = [paddle.empty(chunks[0].shape, dtype=input_tensor.dtype) for _ in range(parallelism)]
    dist.stream.alltoall(buffers, ordered, group=group, use_calc_stream=True)
    return paddle.stack(buffers, axis=0).sum(axis=0)


# Hierarchical (Rail + LSA) overlap for multi-node CP. Gate mirrors the runtime's
# own FLASHMASK_USE_HIERARCHICAL reading so mask ordering stays aligned with the
# C++ gather order; only engaged when cp_size spans more than one node.
_use_hierarchical_overlap = os.environ.get("FLASHMASK_USE_HIERARCHICAL", "0") in ("1", "false")
_hierarchical_gpus_per_node = int(os.environ.get("HIERARCHICAL_GPUS_PER_NODE", "8"))


def _hier_map_chunk(logical_pos, my_pe, total_n_pes, gpus_per_node):
    my_pe_node = my_pe % gpus_per_node
    my_node_id = my_pe // gpus_per_node
    num_nodes = total_n_pes // gpus_per_node
    if logical_pos < num_nodes:
        return my_pe_node + ((my_node_id + logical_pos) % num_nodes) * gpus_per_node
    adj_pos = logical_pos - num_nodes
    slot = adj_pos // num_nodes + 1
    sub = adj_pos % num_nodes
    base = (my_pe_node + slot) % gpus_per_node
    return base + ((my_node_id + sub) % num_nodes) * gpus_per_node


_hier_fwd_perm_cache = {}
_hier_bwd_perm_cache = {}


def _get_hier_fwd_perm(cp_size, rank, gpus_per_node):
    key = (cp_size, rank, gpus_per_node)
    if key not in _hier_fwd_perm_cache:
        perm = [_hier_map_chunk(cp_size - 1 - i, rank, cp_size, gpus_per_node) for i in range(cp_size)]
        _hier_fwd_perm_cache[key] = paddle.to_tensor(perm, dtype="int64")
    return _hier_fwd_perm_cache[key]


def _get_hier_bwd_perm(cp_size, rank, gpus_per_node):
    key = (cp_size, rank, gpus_per_node)
    if key not in _hier_bwd_perm_cache:
        perm = [_hier_map_chunk(i, rank, cp_size, gpus_per_node) for i in range(cp_size)]
        _hier_bwd_perm_cache[key] = paddle.to_tensor(perm, dtype="int64")
    return _hier_bwd_perm_cache[key]


def _hierarchical_active(cp_size):
    return _use_hierarchical_overlap and cp_size > _hierarchical_gpus_per_node


def _deterministic():
    return paddle.get_flags(["FLAGS_cudnn_deterministic"])["FLAGS_cudnn_deterministic"]


def _forward(query, key, value, startend_row_indices, group, causal):
    rank = group.rank
    cp_size = group.world_size

    seq_blocksize = query.shape[1] // 2
    startend_row_indices = preprocess_index_dual_chunks(
        startend_row_indices,
        chunk_id_first=rank,
        chunk_id_second=2 * cp_size - rank - 1,
        seq_blocksize=seq_blocksize,
        max_seqlen_q=seq_blocksize,
    )
    startend_row_indices = rearrange_blocks(startend_row_indices, cp_size)

    if _hierarchical_active(cp_size):
        s_local = query.shape[1]
        bsz, nheads, s_total, mask_dim = startend_row_indices.shape
        perm = _get_hier_fwd_perm(cp_size, rank, _hierarchical_gpus_per_node)
        processed_mask = (
            startend_row_indices.reshape([bsz, nheads, cp_size, s_local, mask_dim])
            .index_select(perm, axis=2)
            .reshape([bsz, nheads, s_total, mask_dim])
        )
    else:
        # cyclic left shift: align the mask with the circular-shift SR buffer order
        processed_mask = paddle._C_ops.roll(
            startend_row_indices, shifts=-query.shape[1] * (rank + 1), axis=2
        )

    output, log_sum_exp = _flash_attn_fwd(
        query,
        key,
        value,
        startend_row_indices=processed_mask,
        causal=causal,
        return_lse=True,
        pack_gqa=False,
        group=group,
    )
    # save the rearranged (pre-roll) mask; backward re-rolls it differently
    return output, log_sum_exp, startend_row_indices


def _backward(query, key, value, startend_row_indices, output, log_sum_exp, output_grad, group, causal):
    rank = group.rank
    cp_size = group.world_size

    if _hierarchical_active(cp_size):
        s_local = query.shape[1]
        bsz, nheads, s_total, mask_dim = startend_row_indices.shape
        perm = _get_hier_bwd_perm(cp_size, rank, _hierarchical_gpus_per_node)
        bwd_mask = (
            startend_row_indices.reshape([bsz, nheads, cp_size, s_local, mask_dim])
            .index_select(perm, axis=2)
            .reshape([bsz, nheads, s_total, mask_dim])
        )
    else:
        # re-roll the local chunk to the FRONT (backward SR order)
        bwd_mask = paddle._C_ops.roll(startend_row_indices, shifts=-query.shape[1] * rank, axis=2)

    query_grad, key_grad_gathered, value_grad_gathered, _ = _flash_attn_bwd(
        query,
        key,
        value,
        output,
        output_grad,
        log_sum_exp,
        flashmask_info=bwd_mask,
        causal=causal,
        group=group,
        deterministic=_deterministic(),
    )

    if USE_RS_OVERLAP or cp_size == 1:
        key_grad = key_grad_gathered
        value_grad = value_grad_gathered
    else:
        key_grad = reduce_scatter_any_axis_simpler(key_grad_gathered, axis=1, group=group)
        value_grad = reduce_scatter_any_axis_simpler(value_grad_gathered, axis=1, group=group)
    return query_grad, key_grad, value_grad


class OverlappedFlashMaskFM4(PyLayer):
    """FM-4 forward+backward overlap CP FlashMask (DualChunkSwap)."""

    @staticmethod
    def forward(ctx, query, key, value, startend_row_indices, causal=False, group=None):
        if causal:
            raise NotImplementedError("FM-4 overlap does not support causal=True.")
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


def overlap_flashmask_attention_fm4(query, key, value, startend_row_indices, causal=False, group=None, mode="overlap"):
    """Public API: FM-4 CP FlashMask attention with distributed overlap."""
    assert mode == "overlap", f"only overlap is implemented, got {mode}"
    return OverlappedFlashMaskFM4.apply(query, key, value, startend_row_indices, causal, group)
