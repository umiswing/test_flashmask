"""Document / prefix-LM FlashMask generators + the one document layout used.

Trimmed from dist_flashmask_dev/utils/mask_utils.py to just the two non-causal
generators the CP-overlap test uses; the two kept functions are byte-identical to
the dev originals. A FlashMask mask is a ``(B, 1, S, 2)`` int32
``startend_row_indices`` tensor (one mask head, broadcast over query heads).

``DOC_PACKING`` is a single packing of (prefix_len, doc_len) pairs whose doc lengths
sum to exactly 32768, so it fits an S_total of 32768 (= s_local * cp) with no padding
or scaling. The plain document mask uses doc_len only; the prefix-LM variant uses both.
"""

import paddle
import numpy as np

DOC_PACKING = [(360, 1478), (581, 3207), (225, 839), (266, 1721), (742, 3329), (666, 3018),
               (276, 2302), (424, 2872), (381, 2381), (455, 2022), (312, 1995), (597, 2786),
               (406, 2373), (310, 2445)]


def generate_document_mask(batch_size, seqlen_q, seqlen_k, h, doc_seqlens):
    """Block-diagonal document mask. Ported verbatim from the dev harness."""
    if isinstance(doc_seqlens[0], tuple):
        doc_seqlens = [seqs[1] for seqs in doc_seqlens]
    total_seqlen = np.sum(doc_seqlens)
    assert total_seqlen <= seqlen_k
    assert len(doc_seqlens) >= 3
    padding = seqlen_k - np.sum(doc_seqlens)

    down_left_row_indices = []
    up_right_row_indices = []

    cur_len_so_far = doc_seqlens[0]
    for i in range(len(doc_seqlens)):
        down_left_row_indices.extend([cur_len_so_far] * doc_seqlens[i])
        if i < len(doc_seqlens) - 1:
            cur_len_so_far += doc_seqlens[i + 1]
    if padding > 0:
        down_left_row_indices.extend([cur_len_so_far] * padding)

    cur_len_so_far = 0
    for i in range(len(doc_seqlens)):
        up_right_row_indices.extend([cur_len_so_far] * doc_seqlens[i])
        if i < len(doc_seqlens) - 1:
            cur_len_so_far += doc_seqlens[i + 1]
    if padding > 0:
        up_right_row_indices.extend([cur_len_so_far] * padding)

    down_left_row_indices = paddle.to_tensor(down_left_row_indices, dtype=paddle.int32).reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
    startend_row_indices = paddle.clip(startend_row_indices, max=seqlen_q)

    causal = False
    return startend_row_indices, causal


def generate_prefix_lm_document_mask(batch_size, seqlen_q, seqlen_k, h, doc_seqlens):
    """Prefix-LM document mask (bidirectional prefix then causal). Ported verbatim."""
    assert len(doc_seqlens) >= 2
    total_seqlen = 0
    for prefix_length, seq_length in doc_seqlens:
        total_seqlen += seq_length
    assert total_seqlen <= seqlen_k
    padding = seqlen_k - total_seqlen

    down_left_row_indices = []
    cur_len_so_far = doc_seqlens[0][1]
    for i in range(len(doc_seqlens)):
        down_left_row_indices.extend([cur_len_so_far] * doc_seqlens[i][1])
        if i < len(doc_seqlens) - 1:
            cur_len_so_far += doc_seqlens[i + 1][1]
    if padding > 0:
        down_left_row_indices.extend([cur_len_so_far] * padding)
    down_left_row_indices = paddle.to_tensor(down_left_row_indices, dtype=paddle.int32).reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)

    up_right_row_indices = []
    cur_len_so_far = 0
    for prefix_length, seq_length in doc_seqlens:
        up_right_row_indices.extend([cur_len_so_far] * prefix_length + list(range(cur_len_so_far + prefix_length, cur_len_so_far + seq_length)))
        cur_len_so_far += seq_length
    if padding > 0:
        up_right_row_indices.extend([total_seqlen] * padding)
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)

    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
    startend_row_indices = paddle.clip(startend_row_indices, max=seqlen_q)

    causal = False
    return startend_row_indices, causal
