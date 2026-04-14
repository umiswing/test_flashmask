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

    # ── Extract document boundaries PER BATCH from startend_row_indices ──
    # startend_row_indices shape: (batch, nheads_sri, seqlen_k, bound_num)
    # Column 0 encodes the "end of document" boundary for each key position.
    # Tokens within the same document share the same value; boundaries are
    # where this value changes.  Different batch items may have different
    # document layouts, so we extract boundaries for each batch independently.
    s = startend_row_indices[:, 0, :, 0]  # (batch, seqlen_k)

    cu_seqlens_parts = []
    max_doc_len = 0
    needs_padding_fixup = False
    real_ends = []

    for bi in range(b):
        s_bi = s[bi]  # (seqlen_k,)

        # Find change positions -> document boundaries
        diff_bi = paddle.not_equal(s_bi[1:], s_bi[:-1])
        change_idx_bi = paddle.nonzero(diff_bi).flatten().cast(paddle.int32) + 1

        # Real end of documents (max value in column 0)
        real_end_bi = int(s_bi.max().item())
        real_ends.append(real_end_bi)
        if real_end_bi < skv:
            needs_padding_fixup = True

        # Boundaries: [0, change_1, ..., seqlen_k]
        # Always use seqlen_k as last boundary so padding KV (visible to
        # the last doc's rows in flashmask) is included in the last doc.
        boundaries_bi = paddle.concat([
            paddle.zeros([1], dtype=paddle.int32),
            change_idx_bi,
            paddle.to_tensor([skv], dtype=paddle.int32),
        ])

        # Track max document length across all batches
        doc_lens_bi = boundaries_bi[1:] - boundaries_bi[:-1]
        max_doc_len = max(max_doc_len, int(doc_lens_bi.max().item()))

        # Collect document start positions with batch offset
        cu_seqlens_parts.append(boundaries_bi[:-1] + bi * skv)

    # Build cu_seqlens: concat per-batch starts + final endpoint
    cu_seqlens = paddle.concat(
        cu_seqlens_parts + [paddle.to_tensor([b * skv], dtype=paddle.int32)]
    )

    # ── Flatten q, k, v: (batch, seqlen, heads, dim) -> (total, heads, dim)
    q_varlen = q.reshape([b * sq, hq, d])
    k_varlen = k.reshape([b * skv, hkv, d])
    v_varlen = v.reshape([b * skv, hkv, dv])

    result = {
        "q": q_varlen,
        "k": k_varlen,
        "v": v_varlen,
        "cu_seqlens_q": cu_seqlens,
        "cu_seqlens_k": cu_seqlens,
        "max_seqlen_q": max_doc_len,
        "max_seqlen_k": max_doc_len,
        "causal": causal,
    }

    # For non-causal masks with trailing padding: padding rows attend to
    # nothing in flashmask (zero output), but varlen computes non-zero
    # output for them.  Zero out padding rows per batch to match flashmask.
    # Note: real_end can differ across batch items.
    if needs_padding_fixup:
        _b, _sq = b, sq
        _real_ends = real_ends

        def output_to_padded(out_varlen_pt):
            nh = out_varlen_pt.shape[1]
            dv_out = out_varlen_pt.shape[2]
            out_padded = out_varlen_pt.reshape(_b, _sq, nh, dv_out)
            # Vectorised per-batch zeroing
            row_idx = torch.arange(_sq, device=out_padded.device)
            real_end_t = torch.tensor(
                _real_ends, device=out_padded.device, dtype=torch.int64,
            ).unsqueeze(1)
            padding_mask = row_idx.unsqueeze(0) >= real_end_t  # (b, sq)
            out_padded[padding_mask] = 0
            return out_padded

        result["output_to_padded"] = output_to_padded

    return result
