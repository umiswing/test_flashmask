import paddle
import torch


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

    # ── Extract document boundaries from startend_row_indices ───────────
    # startend_row_indices shape: (batch, nheads_sri, seqlen_k, bound_num)
    # Column 0 encodes the "end of document" boundary for each key position.
    # Tokens within the same document share the same value.
    # Document boundaries are where this value changes.
    s = startend_row_indices[0, 0, :, 0]  # (seqlen_k,)

    # Find positions where values change -> document start positions
    diff = paddle.not_equal(s[1:], s[:-1])  # (seqlen_k - 1,)
    change_idx = paddle.nonzero(diff).flatten().cast(paddle.int32) + 1

    # The real end of documents = max value in column 0.
    # For causal: equals seqlen_k (padding absorbed into last doc).
    # For non-causal: may be < seqlen_k (padding rows attend to nothing).
    real_end = int(s.max().item())

    # Always use seqlen_k as last boundary so padding tokens are included
    # in the last document (their KV is visible to the last doc's rows).
    boundaries = paddle.concat([
        paddle.zeros([1], dtype=paddle.int32),
        change_idx,
        paddle.to_tensor([skv], dtype=paddle.int32),
    ])  # (num_docs + 1,)

    # ── Flatten q, k, v: (batch, seqlen, heads, dim) -> (total, heads, dim)
    q_varlen = q.reshape([b * sq, hq, d])
    k_varlen = k.reshape([b * skv, hkv, d])
    v_varlen = v.reshape([b * skv, hkv, dv])

    # ── Build cu_seqlens for all batch items ────────────────────────────
    batch_offsets = (paddle.arange(b, dtype=paddle.int32) * skv).unsqueeze(1)
    per_batch_starts = boundaries[:-1].unsqueeze(0) + batch_offsets  # (b, num_docs)
    cu_seqlens = paddle.concat([
        per_batch_starts.reshape([-1]),
        paddle.to_tensor([b * skv], dtype=paddle.int32),
    ])

    # ── Max sequence length (max document length) ───────────────────────
    doc_lengths = boundaries[1:] - boundaries[:-1]
    max_seqlen = int(doc_lengths.max().item())

    result = {
        "q": q_varlen,
        "k": k_varlen,
        "v": v_varlen,
        "cu_seqlens_q": cu_seqlens,
        "cu_seqlens_k": cu_seqlens,
        "max_seqlen_q": max_seqlen,
        "max_seqlen_k": max_seqlen,
        "causal": causal,
    }

    # For non-causal masks with trailing padding: padding rows attend to
    # nothing in flashmask (zero output), but varlen computes non-zero
    # output for them. Zero out padding rows to match flashmask.
    if real_end < skv:
        _b, _sq, _real_end = b, sq, real_end

        def output_to_padded(out_varlen_pt):
            out_padded = out_varlen_pt.reshape(_b, _sq, -1, out_varlen_pt.shape[-1])
            out_padded[:, _real_end:] = 0
            return out_padded

        result["output_to_padded"] = output_to_padded

    return result
