"""Context-parallel comm primitives (DualChunkSwap), trimmed from
dist_flashmask_dev/cp_balance/context_parallel_utils.py.

Only the dual-chunk functions the CP-overlap test needs are kept; the kept code
is byte-identical to the dev originals (dual_chunk branches). The balanced-swap /
contiguous modes, PyLayer scatter/gather ops and the duplicate CP attention
PyLayer were dropped -- unused here.
"""

import paddle
from paddle import distributed as dist
from paddle.distributed import fleet


def scatter_balance(input_tensor, group=None, axis=0, mode="dual_chunk"):
    """Split ``input_tensor`` along ``axis`` into this rank's dual-chunk share
    (block ``r`` from the front, block ``2*cp-1-r`` from the back)."""
    assert mode == "dual_chunk", f"only dual_chunk is supported here, got {mode}"
    if group is None:
        group = fleet.get_hybrid_communicate_group().get_context_parallel_group()

    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor.clone()

    rank = group.rank
    seq_len = input_tensor.shape[axis]
    assert seq_len % (parallelism * 2) == 0, (
        f"Input sequence length {seq_len} can't be divided exactly by "
        f"sequence parallelism * 2 {parallelism * 2}"
    )

    interval = seq_len // parallelism // 2
    chunk_start = paddle.slice(input_tensor, axes=[axis], starts=[interval * rank], ends=[interval * (rank + 1)])
    chunk_end = paddle.slice(
        input_tensor, axes=[axis], starts=[seq_len - interval * (rank + 1)], ends=[seq_len - interval * rank]
    )
    result = paddle.concat([chunk_start, chunk_end], axis=axis)
    # assign copies out so the (large) source tensor can be freed (slice keeps a view)
    return paddle.assign(result)


def all_gather_balance(input_tensor, group=None, axis=0, mode="dual_chunk"):
    """Reconstruct the full sequence in ORIGINAL block order from dual-chunk shares."""
    assert mode == "dual_chunk", f"only dual_chunk is supported here, got {mode}"
    if group is None:
        group = fleet.get_hybrid_communicate_group().get_context_parallel_group()

    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor.clone()

    chunk_start, chunk_end = paddle.split(input_tensor, 2, axis=axis)
    gathered_start_list = [paddle.empty(chunk_start.shape, dtype=input_tensor.dtype) for _ in range(parallelism)]
    dist.stream.all_gather(gathered_start_list, chunk_start, group=group, use_calc_stream=True)
    gathered_end_list = [paddle.empty(chunk_end.shape, dtype=input_tensor.dtype) for _ in range(parallelism)]
    dist.stream.all_gather(gathered_end_list, chunk_end, group=group, use_calc_stream=True)
    gathered_end_list = gathered_end_list[::-1]
    return paddle.concat(gathered_start_list + gathered_end_list, axis=axis)


def reduce_scatter_any_axis_balance(input_tensor, axis, group=None):
    """Balanced reduce-scatter: adjoint of ``all_gather_balance`` (dual-chunk)."""
    if group is None:
        group = fleet.get_hybrid_communicate_group().get_context_parallel_group()

    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor.clone()

    assert input_tensor.shape[axis] % (parallelism * 2) == 0, (
        f"Input sequence length {input_tensor.shape[axis]} can't be "
        f"divided exactly by context parallelism * 2 {parallelism * 2}"
    )

    input_start, input_end = paddle.split(input_tensor, 2, axis=axis)
    chunks_start = paddle.split(input_start, parallelism, axis=axis)
    chunks_end = paddle.split(input_end, parallelism, axis=axis)
    chunks_end = chunks_end[::-1]
    combined_chunks = [
        paddle.concat([s, e], axis=axis) for s, e in zip(chunks_start, chunks_end)
    ]
    output_buffers = [paddle.empty(combined_chunks[0].shape, dtype=input_tensor.dtype) for _ in range(parallelism)]
    dist.stream.alltoall(output_buffers, combined_chunks, group=group, use_calc_stream=True)
    return paddle.stack(output_buffers, axis=0).sum(axis=0)


def preprocess_index_dual_chunks(startend_row_indices, chunk_id_first, chunk_id_second, seq_blocksize, max_seqlen_q):
    """Rebase FlashMask row indices for a rank owning two (front+back) chunks."""
    rows_min_first = chunk_id_first * seq_blocksize
    rows_min_second = chunk_id_second * seq_blocksize

    indices_first = paddle.clip(startend_row_indices - rows_min_first, min=0, max=max_seqlen_q)
    indices_second = paddle.clip(startend_row_indices - rows_min_second, min=0, max=max_seqlen_q)
    indices_second = paddle.where(indices_second != 0, indices_second + max_seqlen_q, indices_second)
    return paddle.maximum(indices_first, indices_second)
