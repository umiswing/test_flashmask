# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FlashMask Attention - PyTorch Implementation

This module provides a PyTorch implementation of FlashMask attention algorithm.
The core equation is:

    result = softmax(Q @ K^T / sqrt(d) + M) @ V

where M is the column-wise sparse mask introduced by FlashMask.

Note: This is a reference implementation using standard PyTorch operations.
For optimal performance on NVIDIA GPUs, consider using the CUDA-optimized version.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def flashmask_to_dense_mask(
    startend_row_indices: Tensor,
    seqlen_q: int,
    seqlen_k: int,
    dtype: torch.dtype,
    causal: bool = True,
) -> Tensor:
    """
    Convert FlashMask's startend_row_indices to dense attention mask.

    Args:
        startend_row_indices: Column-wise sparse attention mask row indices tensor.
            Shape: [batch_size, num_heads, seqlen_k, {1, 2, 4}]
        seqlen_q: Query sequence length
        seqlen_k: Key sequence length
        dtype: Data type for the mask (should match attention scores dtype)
        causal: Whether to use causal mask mode

    Returns:
        Dense attention mask of shape [batch_size, num_heads, seqlen_q, seqlen_k]
    """
    if startend_row_indices is None:
        return None

    bz, num_head, seq_len_k, bound_num = startend_row_indices.shape

    # Create mask tensor initialized to 0 (will be filled with -inf for masked positions)
    mask = torch.zeros((bz, num_head, seqlen_q, seqlen_k), dtype=dtype, device=startend_row_indices.device)

    has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)

    # Create index tensors for vectorized operations
    row_indices = torch.arange(seqlen_q, device=startend_row_indices.device, dtype=torch.int32)
    col_indices = torch.arange(seqlen_k, device=startend_row_indices.device, dtype=torch.int32)

    for bi in range(bz):
        for hi in range(num_head):
            for j in range(seq_len_k):
                # Lower triangular start
                downstart = startend_row_indices[bi, hi, j, 0].item()

                if has_end:
                    downend = startend_row_indices[bi, hi, j, 1].item()
                    # Mask lower triangular region [downstart:downend, j]
                    if downstart < downend:
                        start_row = max(0, downstart)
                        end_row = min(seqlen_q, downend)
                        mask[bi, hi, start_row:end_row, j] = float('-inf')

                    if causal:
                        # For causal mask, also mask upper triangle (future positions)
                        mask[bi, hi, :j, j] = float('-inf')
                    else:
                        # Upper triangular mask for bidirectional attention
                        upstart = startend_row_indices[bi, hi, j, 2].item()
                        upend = startend_row_indices[bi, hi, j, 3].item()
                        if upstart < upend:
                            start_row = max(0, upstart)
                            end_row = min(seqlen_q, upend)
                            mask[bi, hi, start_row:end_row, j] = float('-inf')
                else:
                    # Mask from downstart to end
                    if downstart < seqlen_q:
                        mask[bi, hi, downstart:, j] = float('-inf')

                    if causal:
                        mask[bi, hi, :j, j] = float('-inf')
                    else:
                        upend = startend_row_indices[bi, hi, j, 1].item()
                        if upend > 0:
                            mask[bi, hi, :upend, j] = float('-inf')

    return mask


def flashmask_to_dense_mask_vectorized(
    startend_row_indices: Tensor,
    seqlen_q: int,
    seqlen_k: int,
    dtype: torch.dtype,
    causal: bool = True,
) -> Tensor:
    """
    Vectorized version of flashmask_to_dense_mask for better performance.

    Args:
        startend_row_indices: Column-wise sparse attention mask row indices tensor.
            Shape: [batch_size, num_heads, seqlen_k, {1, 2, 4}]
        seqlen_q: Query sequence length
        seqlen_k: Key sequence length
        dtype: Data type for the mask
        causal: Whether to use causal mask mode

    Returns:
        Dense attention mask of shape [batch_size, num_heads, seqlen_q, seqlen_k]
    """
    if startend_row_indices is None:
        return None

    bz, num_head, seq_len_k, bound_num = startend_row_indices.shape
    device = startend_row_indices.device

    # Create row and column index tensors
    # row_idx: [seqlen_q, 1], col_idx: [1, seqlen_k]
    row_idx = torch.arange(seqlen_q, device=device, dtype=torch.int32).unsqueeze(1)
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.int32).unsqueeze(0)

    # Initialize mask to zeros
    mask = torch.zeros((bz, num_head, seqlen_q, seqlen_k), dtype=dtype, device=device)

    has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)

    if causal:
        # Causal mask: mask positions where row < col (upper triangle)
        causal_mask = (row_idx < col_idx).unsqueeze(0).unsqueeze(0)  # [1, 1, seqlen_q, seqlen_k]
        mask.masked_fill_(causal_mask.expand(bz, num_head, -1, -1), float('-inf'))

        if bound_num == 1:
            # Lower triangular start only
            # startend_row_indices[..., 0] gives the start row for each column
            # Mask rows >= downstart for each column
            downstart = startend_row_indices[..., 0]  # [bz, num_head, seq_len_k]
            # Expand for broadcasting: [bz, num_head, 1, seq_len_k]
            downstart_expanded = downstart.unsqueeze(2)
            # Mask where row >= downstart
            lower_mask = (row_idx.unsqueeze(0).unsqueeze(0) >= downstart_expanded)
            mask.masked_fill_(lower_mask, float('-inf'))

        elif bound_num == 2:
            # Lower triangular with start and end
            downstart = startend_row_indices[..., 0]  # [bz, num_head, seq_len_k]
            downend = startend_row_indices[..., 1]  # [bz, num_head, seq_len_k]

            downstart_expanded = downstart.unsqueeze(2)
            downend_expanded = downend.unsqueeze(2)

            # Mask where downstart <= row < downend
            lower_mask = (row_idx.unsqueeze(0).unsqueeze(0) >= downstart_expanded) & \
                        (row_idx.unsqueeze(0).unsqueeze(0) < downend_expanded)
            mask.masked_fill_(lower_mask, float('-inf'))
    else:
        # Bidirectional attention
        if bound_num == 2:
            # Lower triangular start + Upper triangular end
            downstart = startend_row_indices[..., 0]  # [bz, num_head, seq_len_k]
            upend = startend_row_indices[..., 1]  # [bz, num_head, seq_len_k]

            downstart_expanded = downstart.unsqueeze(2)
            upend_expanded = upend.unsqueeze(2)

            # Lower mask: row >= downstart
            lower_mask = (row_idx.unsqueeze(0).unsqueeze(0) >= downstart_expanded)
            mask.masked_fill_(lower_mask, float('-inf'))

            # Upper mask: row < upend
            upper_mask = (row_idx.unsqueeze(0).unsqueeze(0) < upend_expanded)
            mask.masked_fill_(upper_mask, float('-inf'))

        elif bound_num == 4:
            # Full bidirectional with start and end for both
            downstart = startend_row_indices[..., 0]
            downend = startend_row_indices[..., 1]
            upstart = startend_row_indices[..., 2]
            upend = startend_row_indices[..., 3]

            downstart_expanded = downstart.unsqueeze(2)
            downend_expanded = downend.unsqueeze(2)
            upstart_expanded = upstart.unsqueeze(2)
            upend_expanded = upend.unsqueeze(2)

            # Lower mask: downstart <= row < downend
            lower_mask = (row_idx.unsqueeze(0).unsqueeze(0) >= downstart_expanded) & \
                        (row_idx.unsqueeze(0).unsqueeze(0) < downend_expanded)
            mask.masked_fill_(lower_mask, float('-inf'))

            # Upper mask: upstart <= row < upend
            upper_mask = (row_idx.unsqueeze(0).unsqueeze(0) >= upstart_expanded) & \
                        (row_idx.unsqueeze(0).unsqueeze(0) < upend_expanded)
            mask.masked_fill_(upper_mask, float('-inf'))

    return mask


def flashmask_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    startend_row_indices: Optional[Tensor] = None,
    *,
    dropout: float = 0.0,
    causal: bool = False,
    window_size: Optional[Union[int, tuple]] = None,
    return_softmax_lse: bool = False,
    return_seed_offset: bool = False,
    fixed_seed_offset: Optional[Tensor] = None,
    rng_name: str = "",
    training: bool = True,
    name: Optional[str] = None,
    softmax_scale: Optional[float] = None,
    block_mask: Optional[Tensor] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    FlashMask: PyTorch Implementation

    This module provides the PyTorch implementation of the FlashMask algorithm.
    The core equation is:

        result = softmax(Q @ K^T / sqrt(d) + M) @ V

    where M is the column-wise sparse mask introduced by FlashMask.

    Args:
        query: The query tensor with shape [batch_size, q_seq_len, num_heads, head_dim].
            dtype can be float16 or bfloat16.
        key: The key tensor with shape [batch_size, k_seq_len, k_num_heads, head_dim].
            dtype can be float16 or bfloat16.
        value: The value tensor with shape [batch_size, k_seq_len, k_num_heads, head_dim].
            dtype can be float16 or bfloat16.
        startend_row_indices: Column-wise sparse attention mask row indices tensor.
            Shape: [batch_size, k_num_heads, k_seq_len, {1, 2, 4}]. dtype must be int32.

            - When `causal=True` and shape is [..., 1]: The value represents the starting
              row index of the lower triangular mask.
            - When `causal=True` and shape is [..., 2]: Values represent start and end
              row indices of the lower triangular mask.
            - When `causal=False` and shape is [..., 2]: Values represent lower triangular
              start and upper triangular end.
            - When `causal=False` and shape is [..., 4]: Values represent start/end for
              both lower and upper triangular masks.

        dropout: Dropout ratio. Default is 0.0.
        causal: Whether to enable causal mode. Default is False.
        window_size: Sliding window size for local attention. Default is None.
        return_softmax_lse: Whether to return log-sum-exp of softmax. Default is False.
        return_seed_offset: Whether to return random seed offset. Default is False.
        fixed_seed_offset: Fixed seed offset for dropout. Default is None.
        rng_name: Random number generator name. Default is "".
        training: Whether in training mode. Default is True.
        name: Operation name. Default is None.
        softmax_scale: Softmax scaling factor. Default is None (uses 1/sqrt(head_dim)).
        block_mask: Block-level mask tensor. Currently not supported in this implementation.

    Returns:
        Output tensor with shape [batch_size, q_seq_len, num_heads, head_dim].
        If return_softmax_lse is True, also returns the log-sum-exp tensor.

    Examples:
        >>> import torch
        >>> batch_size, seqlen, num_heads, head_dim = 1, 10, 2, 32
        >>> q = torch.rand(batch_size, seqlen, num_heads, head_dim, dtype=torch.bfloat16)
        >>> k = torch.rand(batch_size, seqlen, num_heads, head_dim, dtype=torch.bfloat16)
        >>> v = torch.rand(batch_size, seqlen, num_heads, head_dim, dtype=torch.bfloat16)
        >>> startend_row_indices = torch.tensor([8]*10, dtype=torch.int32).reshape(1, 1, 10, 1)
        >>> output = flashmask_attention(q, k, v, startend_row_indices, causal=True)
    """
    # Input validation
    assert query.dtype in [torch.float16, torch.bfloat16], \
        f"query dtype must be float16 or bfloat16, got {query.dtype}"
    assert query.dtype == key.dtype == value.dtype, \
        "query, key, value must have the same dtype"

    batch_size, seqlen_q, num_heads, head_dim = query.shape
    _, seqlen_k, num_heads_kv, _ = key.shape

    # Handle GQA (Grouped Query Attention)
    if num_heads != num_heads_kv:
        assert num_heads % num_heads_kv == 0, \
            f"num_heads ({num_heads}) must be divisible by num_heads_kv ({num_heads_kv})"
        # Repeat key and value to match query heads
        num_groups = num_heads // num_heads_kv
        key = key.repeat_interleave(num_groups, dim=2)
        value = value.repeat_interleave(num_groups, dim=2)

    # Handle window_size
    if window_size is not None:
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        assert startend_row_indices is None, \
            "Cannot use window_size with startend_row_indices"
        # Generate sliding window mask
        if causal:
            startend_row_indices = torch.arange(
                window_size[0] + 1, seqlen_q + window_size[0] + 1,
                dtype=torch.int32, device=query.device
            ).reshape(1, 1, seqlen_q, 1)
            startend_row_indices = torch.clamp(startend_row_indices, max=seqlen_q)
            startend_row_indices = startend_row_indices.repeat(batch_size, num_heads_kv, 1, 1)
        else:
            startend_row_indices = torch.empty((1, 1, seqlen_q, 2), dtype=torch.int32, device=query.device)
            startend_row_indices[0, 0, :, 0] = torch.arange(
                window_size[0] + 1, seqlen_q + window_size[0] + 1, dtype=torch.int32, device=query.device
            )
            startend_row_indices[0, 0, :, 1] = torch.arange(
                -window_size[1], seqlen_q - window_size[1], dtype=torch.int32, device=query.device
            )
            startend_row_indices = torch.clamp(startend_row_indices, min=0, max=seqlen_q)
            startend_row_indices = startend_row_indices.repeat(batch_size, num_heads_kv, 1, 1)

    # Validate startend_row_indices
    if startend_row_indices is not None:
        assert startend_row_indices.dtype == torch.int32, \
            f"startend_row_indices dtype must be int32, got {startend_row_indices.dtype}"
        assert len(startend_row_indices.shape) == 4, \
            f"startend_row_indices must be 4D, got {startend_row_indices.shape}"

        assert startend_row_indices.shape[0] == batch_size, \
            f"startend_row_indices batch size mismatch"
        assert startend_row_indices.shape[2] == seqlen_k, \
            f"startend_row_indices seq_len mismatch"

        # Handle head broadcasting
        if startend_row_indices.shape[1] == 1:
            startend_row_indices = startend_row_indices.expand(-1, num_heads, -1, -1)
        elif startend_row_indices.shape[1] != num_heads:
            if startend_row_indices.shape[1] == num_heads_kv:
                startend_row_indices = startend_row_indices.repeat_interleave(
                    num_heads // num_heads_kv, dim=1
                )

    # Set softmax scale
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # Convert to attention format [batch, heads, seq, dim] for scaled_dot_product_attention
    query_t = query.transpose(1, 2)  # [batch, num_heads, seqlen_q, head_dim]
    key_t = key.transpose(1, 2)  # [batch, num_heads, seqlen_k, head_dim]
    value_t = value.transpose(1, 2)  # [batch, num_heads, seqlen_k, head_dim]

    # Generate attention mask
    attn_mask = None
    if startend_row_indices is not None:
        attn_mask = flashmask_to_dense_mask_vectorized(
            startend_row_indices,
            seqlen_q,
            seqlen_k,
            dtype=query.dtype,
            causal=causal,
        )
    elif causal:
        # Standard causal mask
        attn_mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k, dtype=query.dtype, device=query.device) * float('-inf'),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

    # Compute attention using PyTorch's scaled_dot_product_attention
    # This will use Flash Attention v2 if available
    if attn_mask is not None:
        # scaled_dot_product_attention expects attn_mask where masked positions have -inf
        attn_output = F.scaled_dot_product_attention(
            query_t, key_t, value_t,
            attn_mask=attn_mask,
            dropout_p=dropout if training else 0.0,
            scale=softmax_scale,
        )
    else:
        attn_output = F.scaled_dot_product_attention(
            query_t, key_t, value_t,
            dropout_p=dropout if training else 0.0,
            is_causal=causal,
            scale=softmax_scale,
        )

    # Convert back to [batch, seq, heads, dim]
    output = attn_output.transpose(1, 2)

    if return_softmax_lse:
        # Compute log-sum-exp manually
        # scores = (query_t @ key_t.transpose(-2, -1)) * softmax_scale
        # if attn_mask is not None:
        #     scores = scores + attn_mask
        # lse = torch.logsumexp(scores, dim=-1)
        # lse = lse.transpose(1, 2)  # [batch, seq, heads]

        # For efficiency, compute LSE in half precision
        with torch.no_grad():
            scores = torch.matmul(query_t, key_t.transpose(-2, -1)) * softmax_scale
            if attn_mask is not None:
                scores = scores + attn_mask
            lse = torch.logsumexp(scores.float(), dim=-1)
            lse = lse.transpose(1, 2)  # [batch, seq, heads]

        return output, lse

    if return_seed_offset:
        # Not fully supported in PyTorch version
        return output, None

    return output


class FlashMaskAttention(torch.nn.Module):
    """
    FlashMask Attention Module for use in neural networks.

    This module wraps the flashmask_attention function for convenient use
    in transformer architectures.

    Args:
        causal: Whether to use causal attention. Default is False.
        softmax_scale: Softmax scaling factor. Default is None (auto-computed).
        dropout: Dropout probability. Default is 0.0.

    Examples:
        >>> attn = FlashMaskAttention(causal=True)
        >>> q = torch.rand(1, 10, 2, 32, dtype=torch.bfloat16)
        >>> k = torch.rand(1, 10, 2, 32, dtype=torch.bfloat16)
        >>> v = torch.rand(1, 10, 2, 32, dtype=torch.bfloat16)
        >>> mask = torch.tensor([8]*10, dtype=torch.int32).reshape(1, 1, 10, 1)
        >>> output = attn(q, k, v, mask)
    """

    def __init__(
        self,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = dropout

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        startend_row_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of FlashMask attention.

        Args:
            query: Query tensor [batch, seq_q, heads, dim]
            key: Key tensor [batch, seq_k, heads_kv, dim]
            value: Value tensor [batch, seq_k, heads_kv, dim]
            startend_row_indices: Optional mask indices [batch, heads_kv, seq_k, bounds]

        Returns:
            Output tensor [batch, seq_q, heads, dim]
        """
        return flashmask_attention(
            query, key, value,
            startend_row_indices=startend_row_indices,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
            dropout=self.dropout,
            training=self.training,
        )


# Utility functions for creating common mask patterns

def create_causal_mask(seqlen: int, batch_size: int = 1, num_heads: int = 1) -> Tensor:
    """Create a standard causal mask's startend_row_indices."""
    startend_row_indices = torch.full(
        (batch_size, num_heads, seqlen, 1),
        seqlen, dtype=torch.int32
    )
    return startend_row_indices


def create_sliding_window_mask(
    seqlen: int,
    window_size: int,
    batch_size: int = 1,
    num_heads: int = 1,
) -> Tensor:
    """Create a sliding window mask's startend_row_indices for causal attention."""
    startend_row_indices = torch.arange(
        window_size + 1, seqlen + window_size + 1, dtype=torch.int32
    ).reshape(1, 1, seqlen, 1)
    startend_row_indices = torch.clamp(startend_row_indices, max=seqlen)
    return startend_row_indices.repeat(batch_size, num_heads, 1, 1)


def create_document_mask(
    seqlen: int,
    doc_boundaries: list,
    batch_size: int = 1,
    num_heads: int = 1,
    causal: bool = True,
) -> Tensor:
    """
    Create a document mask's startend_row_indices.

    Args:
        seqlen: Total sequence length
        doc_boundaries: List of document end positions (e.g., [4, 7, 10] for docs of lengths 4, 3, 3)
        batch_size: Batch size
        num_heads: Number of heads
        causal: Whether to create causal document mask

    Returns:
        startend_row_indices tensor
    """
    startend = torch.zeros(seqlen, dtype=torch.int32)
    doc_start = 0
    for doc_end in doc_boundaries:
        startend[doc_start:doc_end] = doc_end
        doc_start = doc_end

    startend_row_indices = startend.reshape(1, 1, seqlen, 1)
    return startend_row_indices.repeat(batch_size, num_heads, 1, 1)
