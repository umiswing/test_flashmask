import paddle
from functools import partial
from generate_startend_row_indices import generate_global_sliding_window_mask
from paddle.nn.functional.flash_attention import flashmask_attention

batch_size = 2
seqlen_q = 131072
seqlen_k = 131072
nheads = 32
nheads_kv = 4
nheads_startend_row_indices = 1
d = 128
dv = 128
dtype = paddle.bfloat16
warmup_times = 50
gen_startend_row_indices = partial(generate_global_sliding_window_mask)

q = paddle.randn(shape=[batch_size, seqlen_q, nheads, d], dtype=dtype)
k = paddle.randn(shape=[batch_size, seqlen_k, nheads, d], dtype=dtype)
v = paddle.randn(shape=[batch_size, seqlen_k, nheads, dv], dtype=dtype)
startend_row_indices, causal = gen_startend_row_indices(batch_size, seqlen_q, seqlen_k, nheads_startend_row_indices)

paddle.set_flags({'FLAGS_flash_attn_version': 3})
paddle.set_flags({'FLAGS_cudnn_deterministic': 0})

for i in range(warmup_times):
    out, lse = flashmask_attention(
        q,
        k,
        v,
        startend_row_indices=startend_row_indices,
        causal=causal,
        return_softmax_lse=True
    )

paddle.base.core.nvprof_nvtx_push("flashmask")
out, lse = flashmask_attention(
    q,
    k,
    v,
    startend_row_indices=startend_row_indices,
    causal=causal,
    return_softmax_lse=True
)
paddle.base.core.nvprof_nvtx_pop()
