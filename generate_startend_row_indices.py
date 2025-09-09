import paddle
import numpy as np

def startend_row_indices_to_attn_bias(startend_row_indices, dtype, causal=True):
    if startend_row_indices is None:
        return None
    bz, num_head, seq_len, bound_num = startend_row_indices.shape
    m = paddle.zeros((bz, num_head, seq_len, seq_len), dtype=dtype)
    has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)
    for bi in range(bz):
        for hi in range(num_head):
            for j in range(seq_len):
                downstart = startend_row_indices[bi, hi, j, 0]
                if has_end:
                    downend = startend_row_indices[bi, hi, j, 1]
                    m[bi, hi, downstart:downend, j] = -np.inf
                else:
                    m[bi, hi, downstart:, j] = -np.inf
                if causal:
                    m[bi, hi, :j, j] = -np.inf
                else:
                    if has_end:
                        upstart = startend_row_indices[bi, hi, j, 2]
                        upend = startend_row_indices[bi, hi, j, 3]
                        m[bi, hi, upstart:upend, j] = -np.inf
                    else:
                        upend = startend_row_indices[bi, hi, j, 1]
                        m[bi, hi, :upend, j] = -np.inf
    return m

def generate_none_mask(batch_size, seqlen_q, seqlen_k, h, causal=True):
    return None, causal

def generate_sliding_window_mask(batch_size, seqlen_q, seqlen_k, h, window_size=1024):
    startend_row_indices = paddle.arange(
        window_size, seqlen_k + window_size, dtype="int32"
    ).reshape((1, 1, seqlen_k, 1))
    startend_row_indices = paddle.clip(
        startend_row_indices, max=seqlen_q
    ).repeat_interleave(batch_size, 0)

    causal=True
    return startend_row_indices, causal

def generate_causal_document_mask(B, S, H, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 3
    padding = S - np.sum(doc_seq_lens)
    doc_seq_lens[-1] += padding
    seq_cusums = np.cumsum(doc_seq_lens)

    startend_row_indices = np.repeat(seq_cusums, doc_seq_lens)
    startend_row_indices = paddle.to_tensor(startend_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    
    causal = True
    return startend_row_indices, causal

def generate_document_mask(B, S, H, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 3
    padding = S - np.sum(doc_seq_lens)

    down_left_row_indices = []
    up_right_row_indices = []

    cur_len_so_far = doc_seq_lens[0]
    for i in range(len(doc_seq_lens)):
        down_left_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) -1:
            cur_len_so_far += doc_seq_lens[i+1]
    if padding > 0:
        down_left_row_indices.extend([cur_len_so_far] * padding)

    cur_len_so_far = 0
    for i in range(len(doc_seq_lens)):
        up_right_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) -1:
            cur_len_so_far += doc_seq_lens[i+1]
    if padding > 0:
        up_right_row_indices.extend([cur_len_so_far] * padding)
    
    down_left_row_indices = paddle.to_tensor(down_left_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
    
    causal = False
    return startend_row_indices, causal

def generate_share_question_mask(B, S, H, doc_seq_lens=[2538, 1742, 3213]):
    seq_cusums = np.cumsum(doc_seq_lens)
    seq_cusums = np.append(seq_cusums, 128)

    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 3
    padding = S - total_seq_len

    #startend_row_indices = [S] * doc_seq_lens[0]
    startend_row_indices = [total_seq_len] * doc_seq_lens[0]

    cur_len_so_far = doc_seq_lens[0]
    for idx in range(1, len(doc_seq_lens)):
        cur_len_so_far += doc_seq_lens[idx]
        startend_row_indices.extend([cur_len_so_far] * doc_seq_lens[idx])

    if padding > 0:
        startend_row_indices.extend([cur_len_so_far] * padding)
        
    startend_row_indices = paddle.to_tensor(startend_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    
    causal = True
    return startend_row_indices, causal

def generate_global_sliding_window_mask(B, S, H, global_token=16, window_size=(512, 512)):
    assert len(window_size) == 2
    left_window_size, right_window_size = window_size

    down_left_start_row_indices = []
    down_left_end_row_indices = []
    up_right_start_row_indices = []
    up_right_end_row_indices = []

    down_left_start_row_indices = paddle.arange(
        left_window_size + 1, S + left_window_size + 1, dtype="int32"
    ).clip(max=S)
    down_left_start_row_indices[:global_token] = S
    down_left_start_row_indices =  down_left_start_row_indices.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    down_left_end_row_indices = paddle.full([S], S, dtype="int32").reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    up_right_start_row_indices = paddle.full([S], global_token, dtype="int32")
    up_right_start_row_indices[:global_token+right_window_size+1] = 0
    up_right_start_row_indices = up_right_start_row_indices.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    up_right_end_row_indices = paddle.arange(
        -right_window_size, S - right_window_size, dtype="int32"
    )
    up_right_end_row_indices[:global_token+right_window_size+1] = 0
    up_right_end_row_indices = up_right_end_row_indices.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    startend_row_indices = paddle.concat([down_left_start_row_indices, down_left_end_row_indices, up_right_start_row_indices, up_right_end_row_indices], axis=-1)

    causal = False
    return startend_row_indices, causal

def generate_causal_blockwise_mask(B, S, H, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 3
    padding = S - np.sum(doc_seq_lens)

    start_row_indices = []
    cur_len_so_far = doc_seq_lens[0]
    for i in range(len(doc_seq_lens)):
        start_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) - 1:
            cur_len_so_far += doc_seq_lens[i+1]
    if padding > 0:
        start_row_indices.extend([cur_len_so_far] * padding)
    start_row_indices = paddle.to_tensor(start_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    seq_cusums = np.cumsum(doc_seq_lens)
    end_row_indices = [seq_cusums[-2]] * seq_cusums[-2] + [seq_cusums[-1]] * doc_seq_lens[-1] + [S] * padding
    end_row_indices = paddle.to_tensor(end_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    startend_row_indices = paddle.concat([start_row_indices, end_row_indices], axis=-1)

    causal = True
    return startend_row_indices, causal

def generate_prefix_lm_document_mask(B, S, H, doc_seq_lens=[(1024, 2538), (1742, 1742), (512, 3213)]):
    """
    tuple(prefix_length, seq_length)
    """
    assert len(doc_seq_lens) >= 2
    total_seq_len = 0
    for prefix_length, seq_length in doc_seq_lens:
        total_seq_len += seq_length
    assert total_seq_len <= S
    padding = S - total_seq_len

    down_left_row_indices = []
    cur_len_so_far = doc_seq_lens[0][1]
    for i in range(len(doc_seq_lens)):
        down_left_row_indices.extend([cur_len_so_far] * doc_seq_lens[i][1])
        if i < len(doc_seq_lens) - 1:
            cur_len_so_far += doc_seq_lens[i+1][1]
    if padding > 0:
        down_left_row_indices.extend([cur_len_so_far] * padding)
    down_left_row_indices = paddle.to_tensor(down_left_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    up_right_row_indices = []
    cur_len_so_far = 0
    for prefix_length, seq_length in doc_seq_lens:
        up_right_row_indices.extend([cur_len_so_far] * prefix_length + list(range(cur_len_so_far+prefix_length, cur_len_so_far+seq_length)))
        cur_len_so_far += seq_length
    if padding > 0:
        up_right_row_indices.extend([total_seq_len] * padding)
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)

    causal = False
    return startend_row_indices, causal

def generate_prefix_lm_causal_mask(B, S, H, prefix_length=1024):
    """
    tuple(prefix_length, seq_length)
    """
    assert prefix_length <= S
    down_left_row_indices = paddle.full([S], S, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    up_right_row_indices = paddle.to_tensor([0] * prefix_length + list(range(prefix_length, S)), dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)

    causal = False
    return startend_row_indices, causal

def generate_qk_sparse_mask(B, S, H, maskout_pair=[(1024, 538), (2358, 1700)]):
    """
    tuple(offset, maskout_len)
    """
    start_row_indices = []
    end_row_indices  = []
    last_offset = 0
    for offset, maskout_len in maskout_pair:
        assert offset > last_offset
        start_row_indices.extend([S]*(offset-last_offset))
        end_row_indices.extend([S]*(offset-last_offset))

        start_row_indices.extend(list(range(offset, offset+maskout_len)))
        end_row_indices.extend([offset+maskout_len]*(maskout_len))

        last_offset = offset + maskout_len

    last_offset <= S
    start_row_indices.extend([S]*(S-last_offset))
    end_row_indices.extend([S]*(S-last_offset))

    start_row_indices = paddle.to_tensor(start_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    end_row_indices = paddle.to_tensor(end_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([start_row_indices, end_row_indices], axis=-1)

    causal = True
    return startend_row_indices, causal

def generate_random_eviction_mask(B, S, H, start_row=4096):
    # np.random.seed(0)
    start_rows_list = []
    for bz_idx in range(B):
        for head_idx in range(H):
            start_rows = np.array([S+1] * S)
            mask_pos = np.random.choice(S-1, S - start_row, replace=False)
            index = np.arange(start_row, S)
            mask_pos = np.concatenate([mask_pos[mask_pos < index - 1], mask_pos[mask_pos >= index - 1]])
            start_rows[mask_pos] = index
            start_rows_list.append(start_rows)
    startend_row_indices = paddle.to_tensor(start_rows_list, dtype=paddle.int32).reshape((B, H, S, 1))
    causal = True
    return startend_row_indices, causal
