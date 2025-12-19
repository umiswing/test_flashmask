import os
import math
import itertools
import pytest
import numpy as np
import time

import paddle
import paddle.nn.functional as F

from einops import rearrange, reduce, repeat
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
  generate_random_eviction_mask
)
from functools import partial
from test_util import flashmask_to_densemask

from paddlefleet._extensions.flashmask import (
    rr_attn_estimate_triton_op,
    rr_attention,
)

def run_ref_estimate_from_dense_mask(
    query_states: paddle.Tensor, 
    key_states: paddle.Tensor, 
    mask_dense: paddle.Tensor, 
    *, 
    block_size: int = 128, 
    stride: int = 8, 
    causal: bool = False, 
    chunk_size: int = 512,
    threshold: float = 1.0,
):
    """
    估算稀疏Attention Block分数与边界Mask
    """
    assert mask_dense is not None
    assert chunk_size % stride == 0, "chunk_size must be divisible by stride"
    assert block_size % stride == 0, "block_size must be divisible by stride"

    # [B, Q, H, D] -> [B, H, Q, D]
    query_states = query_states.transpose([0, 2, 1, 3])
    key_states = key_states.transpose([0, 2, 1, 3])

    batch_size, num_q_head, q_len, head_dim = query_states.shape
    _, num_kv_head, k_len, _ = key_states.shape

    # 1. 处理 GQA (Group Query Attention)
    if num_q_head != num_kv_head:
        assert num_q_head % num_kv_head == 0
        num_groups = num_q_head // num_kv_head
        # 在 Head 维度 (dim=1) 复制
        # key_states: [B, H_kv, K, D] -> [B, H_q, K, D]
        key_states = repeat(key_states, 'b h k d -> b (h g) k d', g=num_groups)

    nheads_dense_mask = mask_dense.shape[1]
    if num_q_head != nheads_dense_mask:
        assert num_q_head % nheads_dense_mask == 0
        num_groups = num_q_head // nheads_dense_mask 
        mask_dense = repeat(mask_dense, 'b h k d -> b (h g) k d', g=num_groups)

    # 2. Padding 计算与对齐
    # 统统 Pad 到 chunk_size 的倍数，这样处理最方便
    def get_pad_len(length, align):
        return (align - length % align) % align

    q_pad_len = get_pad_len(q_len, chunk_size)
    k_pad_len = get_pad_len(k_len, chunk_size)
    
    padded_q_len = q_len + q_pad_len
    padded_k_len = k_len + k_pad_len

    mask_dense = mask_dense.astype(paddle.float32)
    if q_pad_len > 0:
        query_states = F.pad(query_states, (0, 0, 0, q_pad_len), value=0)
        # Pad Mask Dense Height (Q dim)
        if mask_dense.shape[2] == q_len:
             mask_dense = F.pad(mask_dense, (0, 0, 0, q_pad_len), value=0)

    if k_pad_len > 0:
        key_states = F.pad(key_states, (0, 0, 0, k_pad_len), value=0)
        # Pad Mask Dense Width (K dim)
        if mask_dense.shape[3] == k_len:
            mask_dense = F.pad(mask_dense, (0, k_pad_len, 0, 0), value=0)

    # 3. 预计算 K 的 Padding Mask (用于 partial mask 判断)
    # [1, 1, 1, KC, S]
    num_k_strides = padded_k_len // stride
    k_global_indices = paddle.arange(padded_k_len, device=key_states.place)
    
    # 这里的 reshape 必须 match 后续 logits 广播的维度
    k_is_non_padding = (k_global_indices < k_len)
    k_is_non_padding = rearrange(k_is_non_padding, '(ks s) -> 1 1 1 ks s', s=stride)
    
    # 计算每个 Stride 内真正的有效长度 (分母)
    stride_valid_length = k_is_non_padding.astype('float32').sum(axis=-1) # -> [1, 1, 1, ks]

    # 4. Sampling Query (Round-Robin)
    # head_offsets: [1, H, 1, 1]
    head_offsets = rearrange(paddle.arange(num_q_head, device=query_states.place) % stride, 'h -> 1 h 1 1')
    
    num_q_strides = padded_q_len // stride
    # stride_starts: [1, 1, QC, 1]
    stride_starts = rearrange(paddle.arange(num_q_strides, device=query_states.place) * stride, 'qs -> 1 1 qs 1')
    
    gather_indices = head_offsets + stride_starts # [1, H, QC, 1]
    
    # Expand to D for gather
    gather_indices_expanded = repeat(gather_indices, '1 h qs 1 -> b h qs d', b=batch_size, d=head_dim)
    sampled_query = paddle.take_along_axis(query_states, gather_indices_expanded, axis=2)

    # 5. Sampling Mask Dense
    # Mask 通常是 [B, 1, Q, K] 或 [B, H, Q, K]
    # 我们需要在 Q 维度采样，在 K 维度保持完整
    mask_h_dim = mask_dense.shape[1]
    
    # 构造 mask 的 gather index
    mask_gather_idx = gather_indices # [1, H, QC, 1]
    if mask_h_dim == 1:
        # 如果 mask 是 shared head，取第一个 head 的 pattern 即可
        mask_gather_idx = mask_gather_idx[:, 0:1, :, :] # [1, 1, QC, 1]
    
    # Expand to K len
    mask_gather_idx = repeat(mask_gather_idx, '1 h qs 1 -> b h qs k', b=batch_size, k=padded_k_len)
    sampled_mask_dense = paddle.take_along_axis(mask_dense, mask_gather_idx, axis=2)

    # 6. Chunk 计算
    attn_sums_list = []
    boundary_masks_list = []
    
    scale = 1.0 / math.sqrt(head_dim) / stride 
    q_chunk_size = chunk_size // stride
    num_chunks = num_q_strides // q_chunk_size # 因为 pad 到了整倍数，可以直接整除

    for i in range(num_chunks):
        st = i * q_chunk_size
        ed = (i + 1) * q_chunk_size
        
        q_chunk = sampled_query[:, :, st:ed, :]       # [B, H, qc, D]
        mask_chunk = sampled_mask_dense[:, :, st:ed, :] # [B, H/1, qc, K]

        # Dot Product
        # [B, H, qc, D] @ [B, H, K, D]^T -> [B, H, qc, K]
        logits = paddle.matmul(q_chunk, key_states, transpose_y=True)
        logits = logits * scale 
        
        # Reshape to Stride view
        # [B, H, qc, (ks s)] -> [B, H, qc, ks, s]
        logits = rearrange(logits, 'b h qc (ks s) -> b h qc ks s', s=stride)
        mask_chunk = rearrange(mask_chunk, 'b h qc (ks s) -> b h qc ks s', s=stride)

        # Causal Logic
        logical_mask = mask_chunk 
        
        if causal:
            # global_row: [1, H, qc, 1, 1]
            q_idx_val = rearrange(paddle.arange(q_chunk_size, device=logits.place), 'qc -> 1 1 qc 1 1')
            global_q_stride_idx = st + q_idx_val
            h_idx_val = rearrange(paddle.arange(num_q_head, device=logits.place), 'h -> 1 h 1 1 1')
            real_row = global_q_stride_idx * stride + (h_idx_val % stride)
            
            # global_col: [1, 1, 1, ks, s]
            k_idx_val = rearrange(paddle.arange(num_k_strides, device=logits.place), 'ks -> 1 1 1 ks 1')
            s_idx_val = rearrange(paddle.arange(stride, device=logits.place), 's -> 1 1 1 1 s')
            real_col = k_idx_val * stride + s_idx_val
            
            shift = k_len - q_len
            
            # Causal Mask
            is_causal = (real_row + shift >= real_col).astype(logits.dtype)
            logical_mask = logical_mask * is_causal

        # --- 核心 Mask 融合逻辑 ---
        
        # final_effective_mask: [B, H, qc, ks, s]
        # 1. 逻辑允许 (Dense & Causal)
        # 2. 数据有效 (非 Padding)
        final_effective_mask = logical_mask * k_is_non_padding.astype(logits.dtype)
        
        # 应用 Mask (Zero out masked logits for sum reduction)
        logits = logits * final_effective_mask
        
        # 统计
        passed_counts = reduce(final_effective_mask, 'b h qc ks s -> b h qc ks', 'sum')
        total_valid_counts = stride_valid_length # Broadcasts automatically
        
        # 判断 Mask 类型
        is_fully_masked = (passed_counts == 0)
        is_partially_masked = (passed_counts > 0) & (passed_counts < total_valid_counts)

        # Reduce Stride -> Logits Sum (Mean estimate)
        logits_stride = reduce(logits, 'b h qc ks s -> b h qc ks', 'sum')
        
        # Fully masked 设为 -inf
        if is_fully_masked.any():
            neg_inf = paddle.to_tensor(float('-inf'), dtype=logits_stride.dtype, place=logits_stride.place)
            logits_stride = paddle.where(is_fully_masked, neg_inf, logits_stride)

        # Softmax
        scores_stride = F.softmax(logits_stride, axis=-1)
        # 简单的 NaN 处理
        scores_stride = paddle.nan_to_num(scores_stride, 0.0).astype(query_states.dtype)

        # Block Aggregation
        ratio = block_size // stride
        
        # Sum Reduce for Scores
        attn_sum_chunk = reduce(
            scores_stride, 
            'b h (qb r1) (kb r2) -> b h qb kb', 
            'sum', 
            r1=ratio, r2=ratio
        )
        attn_sums_list.append(attn_sum_chunk)

        # Max Reduce for Boundary Mask
        boundary_stride = is_partially_masked.astype('float32')
        boundary_mask_chunk = reduce(
            boundary_stride, 
            'b h (qb r1) (kb r2) -> b h qb kb', 
            'max', 
            r1=ratio, r2=ratio
        )
        boundary_masks_list.append(boundary_mask_chunk.astype('bool'))

    # 7. 合并与切片
    final_attn_sums = paddle.concat(attn_sums_list, axis=2)
    final_boundary_mask = paddle.concat(boundary_masks_list, axis=2)
    
    # 原始需要的 Block 数量
    valid_q_blocks = (q_len + block_size - 1) // block_size
    valid_k_blocks = (k_len + block_size - 1) // block_size
    
    return (
        final_attn_sums[:, :, :valid_q_blocks, :valid_k_blocks],
        final_boundary_mask[:, :, :valid_q_blocks, :valid_k_blocks]
    )

def find_blocks_chunked(
    input_tensor, threshold,
):
    """
        Finds and selects relevant blocks of attention for transformer-based models based on a 
        threshold or a predefined number of blocks.

        Parameters:
        - input_tensor (paddle.Tensor): The input tensor of shape (batch_size, head_num, chunk_num, block_num).
        - threshold (float or None): A threshold value used to determine the minimum attention weight sum.

        Returns:
        - paddle.Tensor: A boolean mask of shape (batch_size, head_num, chunk_num, block_num),
        indicating which blocks should be attended to.
    """
    assert threshold is not None

    x = input_tensor.astype("float32")
    B, H, C, N = x.shape
    total_sum = x.sum(axis=-1, keepdim=True)
    cutoff = total_sum * float(threshold)
    
    sorted_values, sorted_idx = paddle.compat.sort(x, dim=-1, descending=True)  # both [B,H,C,N]

    prefix = paddle.cumsum(sorted_values, axis=-1)  # [B,H,C,N]
    keep = (prefix - sorted_values) < cutoff  # [B,H,C,N], bool

    mask0 = paddle.zeros_like(x, dtype="int32")
    mask_int = paddle.put_along_axis(mask0, sorted_idx, keep.astype("int32"), axis=-1)
    mask = mask_int.astype("bool")

    mask = paddle.logical_and(mask, total_sum > 0)
    return mask

def verify_topp(
    out_kernel,
    topp_mask_kernel,
    top_p_value,
):
    out_tensor = out_kernel.astype('float32')
    
    mask_py = find_blocks_chunked(out_tensor, threshold=top_p_value)
    
    mask_ker_bool = topp_mask_kernel.astype('bool')
    values_kernel = paddle.masked_select(out_tensor, mask_ker_bool)
    
    mask_py_bool = mask_py.astype('bool')
    values_py = paddle.masked_select(out_tensor, mask_py_bool)
    
    count_ker = values_kernel.shape[0]
    count_py = values_py.shape[0]
    print(f"Selected Block Count - Kernel: {count_ker}, Python: {count_py}")
    
    if count_ker != count_py:
        print(f"Warning: Selection count mismatch! Diff: {abs(count_ker - count_py)}")
        assert False
    
    sum_ker = paddle.sum(values_kernel).item()
    sum_py = paddle.sum(values_py).item()
    sum_diff = abs(sum_ker - sum_py)
    
    print(f"Selected Mass Sum    - Kernel: {sum_ker:.6f}, Python: {sum_py:.6f}")
    print(f"Mass Diff: {sum_diff:.8f}")
    
    if count_ker == count_py:
        if sum_ker == 0:
            return
        val_k_sorted = paddle.sort(values_kernel, descending=True)
        val_p_sorted = paddle.sort(values_py, descending=True)
        
        max_val_diff = paddle.max(paddle.abs(val_k_sorted - val_p_sorted)).item()
        print(f"Max Diff in Sorted Values: {max_val_diff:.8f}")
        
        assert max_val_diff < 1e-3, f"Values mismatch! Max diff: {max_val_diff}"
        print("Value sets match perfectly (Sorted check passed).")
        
    else:
        print("Counts differ, skipping sorted element-wise check.")
        assert sum_diff < 1e-2, f"Mass diff too high: {sum_diff}"


shape_cases = (
    [
        (28, 128, 128, 16, 4),
        (4, 256, 256, 4, 1),
        # (2, 8192, 32768, 32, 4), # this will oom
        # (2, 8192, 8192, 32, 4), # this will oom
        (1, 8192, 8192, 1, 1),
        (2, 16384, 16384, 1, 1),
        (1, 128, 128, 1, 1),
        (1, 127, 128, 1, 1),
        (1, 16384, 16384, 1, 1),
        (2, 16384, 16383, 4, 1),
        # my case
    ]
    # tridao case
    + list(itertools.product(
        [1],                # batch_size
        [1, 64,  128, 256, 239, 799, 113, 113, 128, 113, 108, 256, 384, 640, 512, 1024, 1023, 1024,],       # seqlen_q
        [128, 192, 256,   203, 128, 217, 211, 256, 512, 256, 128, 256, 1024, 1024, 1023,],      # seqlen_k
        [1,2],                # nheads
        [1],          # nheads_kv
    ))
    + list(itertools.product(
        [2],                # batch_size
        [4096, 4224],       # seqlen_q
        [4096, 4224],      # seqlen_k
        [6],                # nheads
        [6, 2, 1],          # nheads_kv
    ))
)
# Generate all combinations for second param
def generate_shapes():
    for batch_size, seqlen_q, seqlen_k, nheads, nheads_kv in shape_cases:
        nheads_startend_row_indices_values = [1, nheads_kv]
        for nheads_startend_row_indices in nheads_startend_row_indices_values:
            yield (
                batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices
            )

@pytest.mark.parametrize("dtype", [paddle.bfloat16])
@pytest.mark.parametrize("stride", [8])
@pytest.mark.parametrize("dim", [128])
@pytest.mark.parametrize("threshold", [0.3, 0.8])
@pytest.mark.parametrize(
    "batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices",
    list(generate_shapes())
)
@pytest.mark.parametrize(
    "gen_startend_row_indices",
    [
        # partial(generate_none_mask, causal=False), # full
        # partial(generate_none_mask, causal=True), # causal
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
    ],
)
def test_rrattn_estimate(
    batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices, 
    dtype, gen_startend_row_indices, stride, dim, threshold,
):
    paddle.seed(2024)
    np.random.seed(2024)
    assert nheads % nheads_kv == 0
    
    q_ref_t = paddle.randn(shape=[batch_size, seqlen_q, nheads, dim], dtype='float32')
    k_ref_t = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, dim], dtype='float32')
    
    q_naive_t = q_ref_t.astype(dtype)
    k_naive_t = k_ref_t.astype(dtype)
    
    q_kernel_t = q_naive_t.detach().clone()
    k_kernel_t = k_naive_t.detach().clone()
    
    startend_row_indices, causal = gen_startend_row_indices(
        batch_size, seqlen_q, seqlen_k, nheads_startend_row_indices
    )

    # rr_attention(q_kernel_t, k_kernel_t, k_kernel_t, startend_row_indices, causal=causal, stride=stride, threshold=threshold)
    
    mask_dense = flashmask_to_densemask(
        startend_row_indices, seqlen_q, nheads_startend_row_indices, causal
    )

    print(f"Testing Config: B={batch_size}, Q={seqlen_q}, K={seqlen_k}, HQ={nheads}, H={nheads_kv}, Stride={stride}, Causal={causal}")

    out_ref, bound_ref = run_ref_estimate_from_dense_mask(
        q_ref_t,
        k_ref_t,
        mask_dense,
        block_size=128,
        stride=stride,
        causal=causal,
        chunk_size=2048,
    )
    
    out_naive, _ = run_ref_estimate_from_dense_mask(
        q_naive_t,
        k_naive_t,
        mask_dense,
        block_size=128,
        stride=stride,
        causal=causal,
        chunk_size=2048,
    )

    with paddle.compat.use_torch_proxy_guard():
        out_kernel, bound_kernel, topp_kernel = rr_attn_estimate_triton_op.rr_attn_estimate_triton_func(
            q=q_kernel_t,
            k=k_kernel_t,
            startend_row_indices=startend_row_indices,
            stride=stride,
            threshold=threshold,
            causal=causal
        )

    out_ref = out_ref.astype('float32')
    bound_ref = bound_ref.astype('int32')

    out_naive = out_naive.astype('float32')

    out_kernel = out_kernel.astype('float32')
    bound_kernel = bound_kernel.astype('int32')

    # -----------------------------------------------------------
    # Test 1: Boundary Check Mask (Exact Match)
    # -----------------------------------------------------------
    print("\n--- Testing Boundary Mask ---")
    assert bound_ref.shape == bound_kernel.shape, \
        f"Shape Mismatch! Ref: {bound_ref.shape}, Kernel: {bound_kernel.shape}"

    mask_diff_tensor = paddle.sum(paddle.abs(bound_ref - bound_kernel))
    mask_diff = mask_diff_tensor.item()
    total_elements = bound_ref.size

    print(f"Boundary Mask Mismatches: {mask_diff} / {total_elements} ({(mask_diff/total_elements)*100:.4f}%)")

    if mask_diff > 0:
        mismatch_indices = paddle.nonzero(bound_ref != bound_kernel)
        
        # 取前5个错误点
        top_indices = mismatch_indices[:5]
        print("First 5 mismatches (Indices):", top_indices.tolist())
        
        ref_vals = paddle.gather_nd(bound_ref, top_indices)
        kernel_vals = paddle.gather_nd(bound_kernel, top_indices)
        
        print("Ref Values:", ref_vals.tolist())
        print("Kernel Values:", kernel_vals.tolist())

    assert mask_diff == 0, "[FAIL] Boundary masks do not match exactly!"
    print("✅ Boundary Mask Matched Exactly.")

    # -----------------------------------------------------------
    # Test 2: Attention Score Estimation (Dynamic Tolerance)
    # -----------------------------------------------------------
    print("\n--- Testing Attention Scores ---")

    fwd_atol = 2 * paddle.max(paddle.abs(out_ref + 0.3 - 0.3 - out_ref)).item()
    rtol = 2

    # Baseline Error
    naive_diff = paddle.abs(out_naive - out_ref)
    naive_err = paddle.max(naive_diff).item()
    print(f"Naive float32 Output max diff (vs FP32): {naive_err:.6f}")

    # Kernel Error
    kernel_diff = paddle.abs(out_kernel - out_ref)
    kernel_err = paddle.max(kernel_diff).item()
    kernel_mean_err = paddle.mean(kernel_diff).item()

    print(f"Kernel Output max diff (vs FP32): {kernel_err:.6f}")
    print(f"Kernel Output mean diff (vs FP32): {kernel_mean_err:.6f}")

    allowed_error = rtol * naive_err + fwd_atol + 1e-4

    if kernel_err > allowed_error:
        print(f"[FAIL] Score error exceeds tolerance!")
        print(f"Max Diff: {kernel_err}")
        print(f"Allowed: {allowed_error}")
        
        flat_idx = paddle.argmax(kernel_diff).item()
        err_indices = []
        for dim in reversed(out_ref.shape):
            err_indices.append(flat_idx % dim)
            flat_idx //= dim
        err_indices = tuple(reversed(err_indices))
        
        ref_val = out_ref[err_indices].item()
        kernel_val = out_kernel[err_indices].item()
        print(f"Max Error at {err_indices}: Ref={ref_val}, Kernel={kernel_val}")

    assert kernel_err <= allowed_error, \
        f"Output max diff {kernel_err} > Allowed {allowed_error}"
    
    # -----------------------------------------------------------
    # Test 3: Top-p block selection
    # -----------------------------------------------------------

    verify_topp(out_kernel, topp_kernel, threshold)

    print("Attention Score Matched within tolerance.")
    print("All Tests Passed!")