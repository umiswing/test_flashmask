import paddle
def convert_to_varlen(
    q,
    k,
    v,
    causal,
    startend_row_indices,
):
    b, sq, hq, d = q.shape
    _, skv, hkv, dv = v.shape
    assert sq == skv
    q_varlen = q.reshape([b * sq, hq, d])
    k_varlen = k.reshape([b * skv, hkv, d])
    v_varlen = v.reshape([b * skv, hkv, dv])

    cu_seqlens_q = paddle.to_tensor([0, b * sq], dtype=paddle.int32)
    cu_seqlens_k = paddle.to_tensor([0, b * skv], dtype=paddle.int32)

    max_seqlen_q = b * sq
    max_seqlen_k = b * skv

    return {
        "q": q_varlen,
        "k": k_varlen,
        "v": v_varlen,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "causal": causal,
    }
