import paddle
import tqdm
import configargparse
from functools import partial
from generate_startend_row_indices import generate_global_sliding_window_mask
from paddle.nn.functional.flash_attention import flashmask_attention

def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--config", is_config_file=True, help="Config file path")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--seqlen_q", type=int, default=32 * 1024, help="Q sequence length")
    parser.add_argument("--seqlen_k", type=int, default=32 * 1024, help="K sequence length")
    parser.add_argument("--nheads", type=int, default=32, help="Number of heads")
    parser.add_argument("--nheads_kv", type=int, default=4, help="Number of heads (KV)")
    parser.add_argument("--nheads_startend_row_indices", type=int, default=1, help="Start end row indices")
    parser.add_argument("--d", type=int, default=128, help="Latent dim d")
    parser.add_argument("--dv", type=int, default=128, help="Latent dim dv")
    parser.add_argument("--warmup_times", type=int, default=50, help="Number of times for warmup")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Dtype for attention calculation")
    parser.add_argument("-b", "--backward_prof", default=False, action="store_true", help="Whether to profile backward")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Whether to print runtime config")
    parser.add_argument("--causal", default=False, action="store_true", help="Whether to use causal mask")
    return parser.parse_args()

if __name__ == "__main__":
    opts = parse_args()
    batch_size = opts.batch_size
    seqlen_q = opts.seqlen_q
    seqlen_k = opts.seqlen_k
    nheads = opts.nheads
    nheads_kv = opts.nheads_kv
    nheads_startend_row_indices = opts.nheads_startend_row_indices
    d = opts.d
    dv = opts.dv
    dtype = opts.dtype
    warmup_times = opts.warmup_times
    gen_startend_row_indices = partial(generate_global_sliding_window_mask)

    q = paddle.randn(shape=[batch_size, seqlen_q, nheads, d], dtype=dtype)
    k = paddle.randn(shape=[batch_size, seqlen_k, nheads, d], dtype=dtype)
    v = paddle.randn(shape=[batch_size, seqlen_k, nheads, dv], dtype=dtype)

    if opts.verbose:
        print("FlashAttn profiling configuration:")
        print(opts)

    NO_BACKWARD = not opts.backward_prof
    if NO_BACKWARD:
        print("Backward profiling is disabled.")
    else:
        print("Backward profiling is enabled.")

    q.stop_gradient = NO_BACKWARD
    k.stop_gradient = NO_BACKWARD
    v.stop_gradient = NO_BACKWARD

    # startend_row_indices, causal = gen_startend_row_indices(batch_size, seqlen_q, seqlen_k, nheads_startend_row_indices)
    startend_row_indices, causal = (None, opts.causal)

    paddle.set_flags({'FLAGS_flash_attn_version': 3})
    paddle.set_flags({'FLAGS_cudnn_deterministic': 0})

    out, lse = flashmask_attention(
        q,
        k,
        v,
        startend_row_indices=startend_row_indices,
        causal=causal,
        return_softmax_lse=True
    )

    g = paddle.randn(shape=out.shape, dtype=out.dtype)
    if not NO_BACKWARD:
        out.backward(g)

    print(f"Warming up run for {warmup_times} time(s)...")
    for i in tqdm.tqdm(range(warmup_times)):
        out, lse = flashmask_attention(
            q,
            k,
            v,
            startend_row_indices=startend_row_indices,
            causal=causal,
            return_softmax_lse=True
        )
        if not NO_BACKWARD:
            out.backward(g)

    if NO_BACKWARD:
        paddle.base.core.nvprof_nvtx_push("flashmask")
    out, lse = flashmask_attention(
        q,
        k,
        v,
        startend_row_indices=startend_row_indices,
        causal=causal,
        return_softmax_lse=True
    )
    if not NO_BACKWARD:
        out.backward(g)
    paddle.base.core.nvprof_nvtx_pop()
