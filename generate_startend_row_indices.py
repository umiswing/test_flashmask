import paddle
import numpy as np

def _rotate_list(values, shift: int):
    if not values:
        return values
    shift = shift % len(values)
    return values[shift:] + values[:shift]

def _stack_per_head_startend_row_indices(
    batch_size,
    seqlen_q,
    seqlen_k,
    h,
    per_head_fn,
):
    """
    Build [b, h, seqlen_k, bound_num] by calling a per-head generator that returns
    [b, 1, seqlen_k, bound_num] for each head.
    """
    if h == 1:
        return per_head_fn(0)
    startend_list = []
    causal = None
    for head_idx in range(h):
        s, c = per_head_fn(head_idx)
        if causal is None:
            causal = c
        else:
            assert c == causal
        assert s.shape[0] == batch_size and s.shape[1] == 1 and s.shape[2] == seqlen_k
        startend_list.append(s)
    return paddle.concat(startend_list, axis=1), causal

def startend_row_indices_to_attn_bias(startend_row_indices, seqlen_q, nheads, dtype, causal=True):
    if startend_row_indices is None:
        return None
    bz, num_head, seqlen_k, bound_num = startend_row_indices.shape
    assert nheads % num_head == 0
    m = paddle.zeros((bz, num_head, seqlen_q, seqlen_k), dtype=dtype)
    has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)
    for bi in range(bz):
        for hi in range(num_head):
            for j in range(seqlen_k):
                downstart = startend_row_indices[bi, hi, j, 0]
                if has_end:
                    downend = startend_row_indices[bi, hi, j, 1]
                    m[bi, hi, downstart:downend, j] = -np.inf
                else:
                    m[bi, hi, downstart:, j] = -np.inf
                if causal:
                    # from flash-attention 2.1 and in flash-attention 3, If seqlen_q != seqlen_k and causal=True,
                    # the causal mask is aligned to the bottom right corner of the attention matrix,
                    # instead of the top-left corner.
                    # See: https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#21-change-behavior-of-causal-flag
                    m[bi, hi, :max(0, j - (seqlen_k - seqlen_q)), j] = -np.inf
                else:
                    if has_end:
                        upstart = startend_row_indices[bi, hi, j, 2]
                        upend = startend_row_indices[bi, hi, j, 3]
                        m[bi, hi, upstart:upend, j] = -np.inf
                    else:
                        upend = startend_row_indices[bi, hi, j, 1]
                        m[bi, hi, :upend, j] = -np.inf
    m = paddle.repeat_interleave(x=m, repeats=nheads // num_head, axis=1)
    return m

def generate_none_mask(batch_size, seqlen_q, seqlen_k, h, causal=True):
    return None, causal

def generate_sliding_window_mask(batch_size, seqlen_q, seqlen_k, h, window_size=None):
    if window_size == None:
        window_size = 1024
        if seqlen_k != 8192:
            window_size = int(window_size * (seqlen_k / 8192))
            print(f"{seqlen_k=}, auto setting window_size to {window_size}")

    startend_row_indices = paddle.arange(
        window_size, seqlen_k + window_size, dtype="int32"
    ).reshape((1, 1, seqlen_k, 1))
    startend_row_indices = paddle.clip(
        startend_row_indices, max=seqlen_q
    ).repeat_interleave(batch_size, 0)

    causal=True
    return startend_row_indices, causal

def generate_sliding_window_mask_distinct_heads(batch_size, seqlen_q, seqlen_k, h, window_size=None):
    """
    Per-head distinct sliding window: each head uses a different window_size.
    Output shape: [b, h, seqlen_k, 1]
    """
    if window_size is None:
        window_size = 1024
        if seqlen_k != 8192:
            window_size = int(window_size * (seqlen_k / 8192))
            print(f"{seqlen_k=}, auto setting window_size to {window_size}")

    # Choose window sizes spanning roughly [window_size/2, window_size], clipped to [1, seqlen_q].
    max_window = max(1, int(window_size))
    head_ids = paddle.arange(h, dtype=paddle.int32)
    denom = max(1, h - 1)
    # linear ramp: largest window at head 0, smallest at head h-1
    win = max_window - (head_ids * (max_window // 2) // denom)
    win = paddle.clip(win, min=1, max=seqlen_q).reshape((1, h, 1, 1))

    col = paddle.arange(seqlen_k, dtype=paddle.int32).reshape((1, 1, seqlen_k, 1))
    startend_row_indices = paddle.clip(col + win, max=seqlen_q).repeat_interleave(batch_size, 0)
    causal = True
    return startend_row_indices, causal

def generate_causal_document_mask(batch_size, seqlen_q, seqlen_k, h, doc_seqlens=None):
    # TODO: this seems buggy, to be fixed
    if doc_seqlens == None:
        doc_seqlens = [2538, 1742, 3213]
        if seqlen_k != 8192:
            doc_seqlens = [int(doc_seqlen * (seqlen_k / 8192)) for doc_seqlen in doc_seqlens]
            print(f"{seqlen_k=}, auto setting doc_seqlens to {doc_seqlens}")
    total_seqlen = np.sum(doc_seqlens)
    assert total_seqlen <= seqlen_k
    assert len(doc_seqlens) >= 3
    padding = seqlen_k - np.sum(doc_seqlens)
    doc_seqlens[-1] += padding
    seq_cusums = np.cumsum(doc_seqlens)

    startend_row_indices = np.repeat(seq_cusums, doc_seqlens)
    startend_row_indices = paddle.to_tensor(startend_row_indices, dtype=paddle.int32).reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)
    startend_row_indices = paddle.clip(startend_row_indices, max=seqlen_q)
    
    causal = True
    return startend_row_indices, causal

def generate_causal_document_mask_distinct_heads(batch_size, seqlen_q, seqlen_k, h, doc_seqlens=None):
    """
    Per-head distinct causal document mask via rotating document segments per head.
    Output shape: [b, h, seqlen_k, 1]
    """
    def _per_head(head_idx: int):
        if doc_seqlens is None:
            return generate_causal_document_mask(batch_size, seqlen_q, seqlen_k, 1, None)
        return generate_causal_document_mask(
            batch_size, seqlen_q, seqlen_k, 1, _rotate_list(list(doc_seqlens), head_idx)
        )

    return _stack_per_head_startend_row_indices(batch_size, seqlen_q, seqlen_k, h, _per_head)

def generate_document_mask(batch_size, seqlen_q, seqlen_k, h, doc_seqlens=None):
    # TODO: this seems buggy, to be fixed
    if doc_seqlens == None:
        doc_seqlens = [2538, 1742, 3213]
        if seqlen_k != 8192:
            doc_seqlens = [int(doc_seqlen * (seqlen_k / 8192)) for doc_seqlen in doc_seqlens]
            print(f"{seqlen_k=}, auto setting doc_seqlens to {doc_seqlens}")
    total_seqlen = np.sum(doc_seqlens)
    assert total_seqlen <= seqlen_k
    assert len(doc_seqlens) >= 3
    padding = seqlen_k - np.sum(doc_seqlens)

    down_left_row_indices = []
    up_right_row_indices = []

    cur_len_so_far = doc_seqlens[0]
    for i in range(len(doc_seqlens)):
        down_left_row_indices.extend([cur_len_so_far] * doc_seqlens[i])
        if i < len(doc_seqlens) -1:
            cur_len_so_far += doc_seqlens[i+1]
    if padding > 0:
        down_left_row_indices.extend([cur_len_so_far] * padding)

    cur_len_so_far = 0
    for i in range(len(doc_seqlens)):
        up_right_row_indices.extend([cur_len_so_far] * doc_seqlens[i])
        if i < len(doc_seqlens) -1:
            cur_len_so_far += doc_seqlens[i+1]
    if padding > 0:
        up_right_row_indices.extend([cur_len_so_far] * padding)
    
    down_left_row_indices = paddle.to_tensor(down_left_row_indices, dtype=paddle.int32).reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
    startend_row_indices = paddle.clip(startend_row_indices, max=seqlen_q)
    
    causal = False
    return startend_row_indices, causal

def generate_document_mask_distinct_heads(batch_size, seqlen_q, seqlen_k, h, doc_seqlens=None):
    """
    Per-head distinct bidirectional document mask via rotating document segments per head.
    Output shape: [b, h, seqlen_k, 2]
    """
    def _per_head(head_idx: int):
        if doc_seqlens is None:
            return generate_document_mask(batch_size, seqlen_q, seqlen_k, 1, None)
        return generate_document_mask(
            batch_size, seqlen_q, seqlen_k, 1, _rotate_list(list(doc_seqlens), head_idx)
        )

    return _stack_per_head_startend_row_indices(batch_size, seqlen_q, seqlen_k, h, _per_head)

def generate_share_question_mask(batch_size, seqlen_q, seqlen_k, h, doc_seqlens=None):
    if doc_seqlens == None:
        doc_seqlens = [2538, 1742, 3213]
        if seqlen_k != 8192:
            doc_seqlens = [int(doc_seqlen * (seqlen_k / 8192)) for doc_seqlen in doc_seqlens]
            print(f"{seqlen_k=}, auto setting doc_seqlens to {doc_seqlens}")

    seq_cusums = np.cumsum(doc_seqlens)
    seq_cusums = np.append(seq_cusums, 128)

    total_seqlen = np.sum(doc_seqlens)
    assert total_seqlen <= seqlen_k
    assert len(doc_seqlens) >= 3
    padding = seqlen_k - total_seqlen

    #startend_row_indices = [S] * doc_seq_lens[0]
    startend_row_indices = [total_seqlen] * doc_seqlens[0]

    cur_len_so_far = doc_seqlens[0]
    for idx in range(1, len(doc_seqlens)):
        cur_len_so_far += doc_seqlens[idx]
        startend_row_indices.extend([cur_len_so_far] * doc_seqlens[idx])

    if padding > 0:
        startend_row_indices.extend([cur_len_so_far] * padding)
        
    startend_row_indices = paddle.to_tensor(startend_row_indices, dtype=paddle.int32).reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)
    startend_row_indices = paddle.clip(startend_row_indices, max=seqlen_q)
    
    causal = True
    return startend_row_indices, causal

def generate_share_question_mask_distinct_heads(batch_size, seqlen_q, seqlen_k, h, doc_seqlens=None):
    """
    Per-head distinct share-question mask via rotating document segments per head.
    Output shape: [b, h, seqlen_k, 1]
    """
    def _per_head(head_idx: int):
        if doc_seqlens is None:
            return generate_share_question_mask(batch_size, seqlen_q, seqlen_k, 1, None)
        return generate_share_question_mask(
            batch_size, seqlen_q, seqlen_k, 1, _rotate_list(list(doc_seqlens), head_idx)
        )

    return _stack_per_head_startend_row_indices(batch_size, seqlen_q, seqlen_k, h, _per_head)

def generate_global_sliding_window_mask(batch_size, seqlen_q, seqlen_k, h, global_token=16, window_size=None):
    if window_size == None:
        window_size = (512, 512)
        if seqlen_k != 8192:
            window_size = tuple(int(ws * (seqlen_k / 8192)) for ws in window_size)
            print(f"{seqlen_k=}, auto setting window_size to {window_size}")
    assert len(window_size) == 2
    left_window_size, right_window_size = window_size

    down_left_start_row_indices = []
    down_left_end_row_indices = []
    up_right_start_row_indices = []
    up_right_end_row_indices = []

    down_left_start_row_indices = paddle.arange(
        left_window_size + 1, seqlen_k + left_window_size + 1, dtype="int32"
    ).clip(max=seqlen_q)
    down_left_start_row_indices[:global_token] = seqlen_q
    down_left_start_row_indices =  down_left_start_row_indices.reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)

    down_left_end_row_indices = paddle.full([seqlen_k], seqlen_q, dtype="int32").reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)

    up_right_start_row_indices = paddle.full([seqlen_k], global_token, dtype="int32")
    up_right_start_row_indices[:global_token+right_window_size+1] = 0
    up_right_start_row_indices = up_right_start_row_indices.reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)

    up_right_end_row_indices = paddle.arange(
        -right_window_size, seqlen_k - right_window_size, dtype="int32"
    )
    up_right_end_row_indices[:global_token+right_window_size+1] = 0
    up_right_end_row_indices = up_right_end_row_indices.reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)

    startend_row_indices = paddle.concat([down_left_start_row_indices, down_left_end_row_indices, up_right_start_row_indices, up_right_end_row_indices], axis=-1)
    startend_row_indices = paddle.clip(startend_row_indices, max=seqlen_q)

    causal = False
    return startend_row_indices, causal

def generate_global_sliding_window_mask_distinct_heads(batch_size, seqlen_q, seqlen_k, h, global_token=16, window_size=None):
    """
    Per-head distinct global sliding window mask by varying (global_token, left/right window) per head.
    Output shape: [b, h, seqlen_k, 4]
    """
    if window_size is None:
        window_size = (512, 512)
        if seqlen_k != 8192:
            window_size = tuple(int(ws * (seqlen_k / 8192)) for ws in window_size)
            print(f"{seqlen_k=}, auto setting window_size to {window_size}")
    left_base, right_base = window_size

    head_ids = paddle.arange(h, dtype=paddle.int32)
    denom = max(1, h - 1)

    # Vary global token count modestly across heads.
    gt_base = int(global_token)
    gt = gt_base + (head_ids - (h // 2))  # centered offsets
    gt = paddle.clip(gt, min=0, max=seqlen_k).reshape((1, h, 1, 1))

    # Vary left/right windows across heads.
    left = int(left_base) - (head_ids * max(1, int(left_base) // 4) // denom)
    right = int(right_base) - (head_ids * max(1, int(right_base) // 4) // denom)
    left = paddle.clip(left, min=0, max=seqlen_q).reshape((1, h, 1, 1))
    right = paddle.clip(right, min=0, max=seqlen_q).reshape((1, h, 1, 1))

    col = paddle.arange(seqlen_k, dtype=paddle.int32).reshape((1, 1, seqlen_k, 1))

    down_left_start = paddle.clip(col + left + 1, max=seqlen_q)
    down_left_start = paddle.where(col < gt, paddle.full_like(down_left_start, seqlen_q), down_left_start)
    down_left_end = paddle.full_like(down_left_start, seqlen_q)

    up_right_start = paddle.full_like(down_left_start, 0)
    up_right_start = paddle.where(col >= (gt + right + 1), gt, up_right_start)

    up_right_end = col - right
    up_right_end = paddle.where(col < (gt + right + 1), paddle.zeros_like(up_right_end), up_right_end)

    startend_row_indices = paddle.concat(
        [down_left_start, down_left_end, up_right_start, up_right_end], axis=-1
    ).repeat_interleave(batch_size, 0)
    startend_row_indices = paddle.clip(startend_row_indices, max=seqlen_q)
    causal = False
    return startend_row_indices, causal

def generate_causal_blockwise_mask(batch_size, seqlen_q, seqlen_k, h, doc_seqlens=None):
    # TODO: this seems buggy, to be fixed
    if doc_seqlens == None:
        doc_seqlens = [2538, 1742, 3213]
        if seqlen_k != 8192:
            doc_seqlens = [int(doc_seqlen * (seqlen_k / 8192)) for doc_seqlen in doc_seqlens]
            print(f"{seqlen_k=}, auto setting doc_seqlens to {doc_seqlens}")
    total_seqlen = np.sum(doc_seqlens)
    assert total_seqlen <= seqlen_k
    assert len(doc_seqlens) >= 3
    padding = seqlen_k - np.sum(doc_seqlens)

    start_row_indices = []
    cur_len_so_far = doc_seqlens[0]
    for i in range(len(doc_seqlens)):
        start_row_indices.extend([cur_len_so_far] * doc_seqlens[i])
        if i < len(doc_seqlens) - 1:
            cur_len_so_far += doc_seqlens[i+1]
    if padding > 0:
        start_row_indices.extend([cur_len_so_far] * padding)
    start_row_indices = paddle.to_tensor(start_row_indices, dtype=paddle.int32).reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)

    seq_cusums = np.cumsum(doc_seqlens)
    end_row_indices = [seq_cusums[-2]] * seq_cusums[-2] + [seq_cusums[-1]] * doc_seqlens[-1] + [seqlen_k] * padding
    end_row_indices = paddle.to_tensor(end_row_indices, dtype=paddle.int32).reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)

    startend_row_indices = paddle.concat([start_row_indices, end_row_indices], axis=-1)
    startend_row_indices = paddle.clip(startend_row_indices, max=seqlen_q)

    causal = True
    return startend_row_indices, causal

def generate_causal_blockwise_mask_distinct_heads(batch_size, seqlen_q, seqlen_k, h, doc_seqlens=None):
    """
    Per-head distinct causal blockwise mask via rotating document segments per head.
    Output shape: [b, h, seqlen_k, 2]
    """
    def _per_head(head_idx: int):
        if doc_seqlens is None:
            return generate_causal_blockwise_mask(batch_size, seqlen_q, seqlen_k, 1, None)
        return generate_causal_blockwise_mask(
            batch_size, seqlen_q, seqlen_k, 1, _rotate_list(list(doc_seqlens), head_idx)
        )

    return _stack_per_head_startend_row_indices(batch_size, seqlen_q, seqlen_k, h, _per_head)

def generate_prefix_lm_document_mask(batch_size, seqlen_q, seqlen_k, h, doc_seqlens=None):
    """
    tuple(prefix_length, seq_length)
    """
    if doc_seqlens == None:
        doc_seqlens=[(1024, 2538), (1742, 1742), (512, 3213)]
        if seqlen_k != 8192:
            scale = seqlen_k / 8192
            doc_seqlens = [tuple(int(v * scale) for v in pair) for pair in doc_seqlens]
            print(f"{seqlen_k=}, auto setting doc_seqlens to {doc_seqlens}")

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
            cur_len_so_far += doc_seqlens[i+1][1]
    if padding > 0:
        down_left_row_indices.extend([cur_len_so_far] * padding)
    down_left_row_indices = paddle.to_tensor(down_left_row_indices, dtype=paddle.int32).reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)

    up_right_row_indices = []
    cur_len_so_far = 0
    for prefix_length, seq_length in doc_seqlens:
        up_right_row_indices.extend([cur_len_so_far] * prefix_length + list(range(cur_len_so_far+prefix_length, cur_len_so_far+seq_length)))
        cur_len_so_far += seq_length
    if padding > 0:
        up_right_row_indices.extend([total_seqlen] * padding)
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)

    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)

    startend_row_indices = paddle.clip(startend_row_indices, max=seqlen_q)

    causal = False
    return startend_row_indices, causal

def generate_prefix_lm_document_mask_distinct_heads(batch_size, seqlen_q, seqlen_k, h, doc_seqlens=None):
    """
    Per-head distinct prefix-LM document mask via rotating (prefix_length, seq_length) segments per head.
    Output shape: [b, h, seqlen_k, 2]
    """
    def _per_head(head_idx: int):
        if doc_seqlens is None:
            return generate_prefix_lm_document_mask(batch_size, seqlen_q, seqlen_k, 1, None)
        return generate_prefix_lm_document_mask(
            batch_size, seqlen_q, seqlen_k, 1, _rotate_list(list(doc_seqlens), head_idx)
        )

    return _stack_per_head_startend_row_indices(batch_size, seqlen_q, seqlen_k, h, _per_head)

def generate_prefix_lm_causal_mask(batch_size, seqlen_q, seqlen_k, h, prefix_length=None):
    """
    tuple(prefix_length, seq_length)
    """
    if prefix_length == None:
        prefix_length = 1024
        if seqlen_k != 8192:
            prefix_length = int(prefix_length * (seqlen_k / 8192))
            print(f"{seqlen_k=}, auto setting doc_seqlens to {prefix_length}")
    assert prefix_length <= seqlen_k
    down_left_row_indices = paddle.full([seqlen_k], seqlen_k, dtype=paddle.int32).reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)
    up_right_row_indices = paddle.to_tensor([0] * prefix_length + list(range(prefix_length, seqlen_k)), dtype=paddle.int32).reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
    startend_row_indices = paddle.clip(startend_row_indices, max=seqlen_q)

    causal = False
    return startend_row_indices, causal

def generate_prefix_lm_causal_mask_distinct_heads(batch_size, seqlen_q, seqlen_k, h, prefix_length=None):
    """
    Per-head distinct prefix-LM causal mask by varying prefix_length per head.
    Output shape: [b, h, seqlen_k, 2]
    """
    if prefix_length is None:
        prefix_length = 1024
        if seqlen_k != 8192:
            prefix_length = int(prefix_length * (seqlen_k / 8192))
            print(f"{seqlen_k=}, auto setting doc_seqlens to {prefix_length}")
    base = int(prefix_length)
    head_ids = paddle.arange(h, dtype=paddle.int32)
    denom = max(1, h - 1)
    # Spread prefix lengths roughly in [base/2, base], clipped.
    pref = base - (head_ids * (base // 2) // denom)
    pref = paddle.clip(pref, min=0, max=seqlen_k).reshape((1, h, 1, 1))

    col = paddle.arange(seqlen_k, dtype=paddle.int32).reshape((1, 1, seqlen_k, 1))
    down_left = paddle.full((1, h, seqlen_k, 1), seqlen_k, dtype=paddle.int32)
    up_right = paddle.where(col < pref, paddle.zeros_like(col), col)

    startend_row_indices = paddle.concat([down_left, up_right], axis=-1).repeat_interleave(batch_size, 0)
    startend_row_indices = paddle.clip(startend_row_indices, max=seqlen_q)
    causal = False
    return startend_row_indices, causal

def generate_qk_sparse_mask(batch_size, seqlen_q, seqlen_k, h, maskout_pair=None):
    """
    tuple(offset, maskout_len)
    """
    if maskout_pair == None:
        maskout_pair=[(1024, 538), (2358, 1700)]
        if seqlen_k != 8192:
            scale = seqlen_k / 8192
            maskout_pair = [tuple(int(v * scale) for v in pair) for pair in maskout_pair]
            print(f"{seqlen_k=}, auto setting maskout_pair to {maskout_pair}")
    start_row_indices = []
    end_row_indices  = []
    last_offset = 0
    for offset, maskout_len in maskout_pair:
        assert offset > last_offset
        start_row_indices.extend([seqlen_k]*(offset-last_offset))
        end_row_indices.extend([seqlen_k]*(offset-last_offset))

        start_row_indices.extend(list(range(offset, offset+maskout_len)))
        end_row_indices.extend([offset+maskout_len]*(maskout_len))

        last_offset = offset + maskout_len

    last_offset <= seqlen_k
    start_row_indices.extend([seqlen_k]*(seqlen_k-last_offset))
    end_row_indices.extend([seqlen_k]*(seqlen_k-last_offset))

    start_row_indices = paddle.to_tensor(start_row_indices, dtype=paddle.int32).reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)
    end_row_indices = paddle.to_tensor(end_row_indices, dtype=paddle.int32).reshape((1, 1, seqlen_k, 1)).repeat_interleave(batch_size, 0)
    startend_row_indices = paddle.concat([start_row_indices, end_row_indices], axis=-1)
    startend_row_indices = paddle.clip(startend_row_indices, max=seqlen_q)

    causal = True
    return startend_row_indices, causal

def generate_qk_sparse_mask_distinct_heads(batch_size, seqlen_q, seqlen_k, h, maskout_pair=None):
    """
    Per-head distinct qk-sparse mask by shifting maskout offsets per head.
    Output shape: [b, h, seqlen_k, 2]
    """
    if maskout_pair is None:
        maskout_pair = [(1024, 538), (2358, 1700)]
        if seqlen_k != 8192:
            scale = seqlen_k / 8192
            maskout_pair = [tuple(int(v * scale) for v in pair) for pair in maskout_pair]
            print(f"{seqlen_k=}, auto setting maskout_pair to {maskout_pair}")

    base_pairs = list(maskout_pair)
    shift_stride = max(1, seqlen_k // max(8, h))

    def _per_head(head_idx: int):
        shift = head_idx * shift_stride
        pairs = []
        for offset, length in base_pairs:
            o = min(max(0, offset + shift), seqlen_k)
            l = min(max(0, length), max(0, seqlen_k - o))
            pairs.append((o, l))
        # Ensure offsets are strictly increasing for the generator.
        pairs.sort(key=lambda x: x[0])
        dedup = []
        last_offset = -1
        for o, l in pairs:
            o = max(o, last_offset + 1)
            o = min(o, seqlen_k)
            l = min(l, max(0, seqlen_k - o))
            dedup.append((o, l))
            last_offset = o
        return generate_qk_sparse_mask(batch_size, seqlen_q, seqlen_k, 1, dedup)

    return _stack_per_head_startend_row_indices(batch_size, seqlen_q, seqlen_k, h, _per_head)

def generate_random_eviction_mask(batch_size, seqlen_q, seqlen_k, h, start_row=None):
    # np.random.seed(0)
    if start_row == None:
        start_row = 4096
        if seqlen_k != 8192:
            start_row = int(start_row * (seqlen_k / 8192))
            print(f"{seqlen_k=}, auto setting start_row to {start_row}")
    start_rows_list = []
    for bz_idx in range(batch_size):
        for head_idx in range(h):
            start_rows = np.array([seqlen_k+1] * seqlen_k)
            mask_pos = np.random.choice(seqlen_k-1, seqlen_k - start_row, replace=False)
            index = np.arange(start_row, seqlen_k)
            mask_pos = np.concatenate([mask_pos[mask_pos < index - 1], mask_pos[mask_pos >= index - 1]])
            start_rows[mask_pos] = index
            start_rows_list.append(start_rows)
    startend_row_indices = paddle.to_tensor(start_rows_list, dtype=paddle.int32).reshape((batch_size, h, seqlen_k, 1))
    startend_row_indices = paddle.clip(startend_row_indices, max=seqlen_q)
    causal = True
    return startend_row_indices, causal
