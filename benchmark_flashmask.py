import json
import numpy as np
from functools import partial
from typing import Optional, List
from tabulate import tabulate
import time
import paddle
import random
import os
import gc
from datetime import datetime

np.random.seed(0)
random.seed(0)

def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = paddle.quantile(times, paddle.to_tensor(quantiles, dtype=paddle.float32)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times.tolist()
    return getattr(paddle, return_mode)(times).item()

def cal_flops(B, H, Sq, Sk, D, DV, mode='fwd'):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    if mode == "fwd":
        f = 2 * B * Sq * Sk * H * (D + DV)
    elif mode == "bwd":
        f = 2 * B * Sq * Sk * H * (3 * D + 2 * DV)
    else:
        f = 2 * B * Sq * Sk * H * (4 * D + 3 * DV)
    return f

def cal_tflops(flops, time_ms):
    return  flops * (1e3 / time_ms) / 1e12

# KV shared: k and v are ONE buffer (the MLA convention, v = k[..., :dv]) and the
# SM100 big-headdim backward merges dK and dV into a single accumulator. Only these
# (D, DV) pairs are implemented
KV_SHARED_D_DV = ((512, 512), (576, 512))


def kv_shared_detected(k, v):
    """Whether the backward will take its kv-shared path for these two tensors.
    """
    try:
        same_storage = k.data_ptr() == v.data_ptr()
    except (AttributeError, RuntimeError):
        return False
    return (
        same_storage
        and k.dtype == v.dtype
        and list(k.shape[:-1]) == list(v.shape[:-1])
        and v.shape[-1] <= k.shape[-1]
        and tuple(k.strides[:-1]) == tuple(v.strides[:-1])
    )


def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean"):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param fast_flush: Use faster kernel to flush L2 cache between measurements
    :type fast_flush: bool, default is True
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all" Default is "mean".    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    fn()

    paddle.device.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2 cache
    # doesn't contain any input data before the run
    cache_size = 256 * 1024 * 1024
    if fast_flush:
        cache = paddle.empty([int(cache_size // 4)], dtype=paddle.int32)
    else:
        cache = paddle.empty([int(cache_size)], dtype=paddle.int8)

    # Estimate the runtime of the function
    start_event = paddle.device.Event(enable_timing=True)
    end_event = paddle.device.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    paddle.device.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    n_warmup = 10
    n_repeat = 50
    start_event = [paddle.device.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [paddle.device.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark

    gc.collect()
    gc.disable()
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        #cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    gc.enable()
    # Record clocks
    paddle.device.synchronize()
    times = paddle.to_tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=paddle.float32)
    return _summarize_statistics(times, quantiles, return_mode)

def test_mask(
    generate_mask_fn,
    B,
    S,
    SKV,
    H,
    HKV,
    D,
    DV,
    dtype = 'bf16',
    use_sink = False,
    backend = 'cutedsl',
    kv_mode = 'split',
):

    if dtype == 'bf16':
        data_type = paddle.bfloat16
    else:
        data_type = paddle.float16

    query = paddle.randn([B, S, H, D], dtype=data_type)
    key = paddle.randn([B, SKV, HKV, D], dtype=data_type)
    query.stop_gradient = False
    key.stop_gradient = False
    if kv_mode == 'shared':
        # The MLA call convention: one D-wide latent buffer is K, and V is its
        # leading DV columns as a stride-1 view. That aliasing is what the
        # backward detects to merge dK and dV into one accumulator. The view is
        # taken after stop_gradient is cleared so K stays the only leaf.
        value = key[..., :DV]
    else:
        value = paddle.randn([B, SKV, HKV, DV], dtype=data_type)
        value.stop_gradient = False
    assert kv_shared_detected(key, value) == (kv_mode == 'shared'), (
        f"kv_mode={kv_mode!r} but the kernel would detect "
        f"kv_shared={kv_shared_detected(key, value)}"
    )
    gradOut = paddle.randn([B, S, H, DV], dtype=data_type)

    # startend_row_indices is a given (the layer already has it) and flashmask
    # consumes it as-is, so there is no mask-conversion step to report here.
    # Building it below is test-data setup, not a cost the kernel imposes. The
    # competitor benchmarks time exactly the step this path does not have:
    # startend_row_indices -> the mask their kernel needs.
    def build_mask():
        if generate_mask_fn is None:
            return None, True
        return generate_mask_fn(B, SKV, HKV, D)

    startend_row_indices, causal = build_mask()

    sparsity = flashmask_block_sparsity(causal, startend_row_indices, B, H, HKV, S, SKV)
    density = 1.0 - sparsity

    sink = None
    if use_sink:
        sink = paddle.randn(shape=[H], dtype=data_type)
        sink.stop_gradient = False

    flashmask = lambda: flashmask_attention(query, key, value, startend_row_indices=startend_row_indices, causal=causal, return_softmax_lse=True)

    use_cutedsl = (backend == 'cutedsl')

    if use_cutedsl:
        from flash_mask.cute.interface import flashmask_attention
    else:
        from paddle.nn.functional.flash_attention import flashmask_attention

    if use_cutedsl:
        query.stop_gradient = True
        key.stop_gradient = True
        value.stop_gradient = True
        if sink is not None:
            # Same leak, and this one hits kv_mode='split' too.
            sink.stop_gradient = True
        def flashmask_fwd():
            from flash_mask.cute.interface import _flash_attn_fwd
            out, lse = _flash_attn_fwd(
                query,
                key,
                value,
                causal=causal,
                softmax_scale=None,
                return_lse=True,
                startend_row_indices=startend_row_indices,
                pack_gqa=False,
                learnable_sink=sink,
            )
        # paddle.base.core.nvprof_nvtx_push("flashmask")
        fwd_time_ms = do_bench(flashmask_fwd)
        # paddle.base.core.nvprof_nvtx_pop()
    else:
        fwd_time_ms = do_bench(flashmask)

    flashmask_out, lse = flashmask()

    if use_cutedsl:
        def flashmask_bwd():
            from flash_mask.cute import flashmask_utils as fm
            from flash_mask.cute.interface import _flash_attn_bwd
            flashmask_info = None
            if startend_row_indices is not None:
                flashmask_info = fm.FlashMaskInfoPaddle(
                    startend_row_indices=startend_row_indices,
                    is_causal=causal,
                )
            fm4_query_grad, fm4_key_grad, fm4_value_grad, _ = _flash_attn_bwd(
                query,
                key,
                value,
                flashmask_out,
                gradOut,
                lse,
                flashmask_info,
                causal=causal,
                learnable_sink=sink,
            )

        # paddle.base.core.nvprof_nvtx_push("flashmask")
        bwd_time_ms = do_bench(flashmask_bwd)
        # paddle.base.core.nvprof_nvtx_pop()
    else:
        bwd_time_ms = do_bench(lambda: flashmask_out.backward(gradOut, retain_graph=True))

    total_time_ms = fwd_time_ms + bwd_time_ms

    fwd_flops = density * cal_flops(B, H, S, SKV, D, DV, mode='fwd')
    bwd_flops = density * cal_flops(B, H, S, SKV, D, DV, mode='bwd')
    total_flops = density * cal_flops(B, H, S, SKV, D, DV, mode='fwd_bwd')

    fwd_tflops = cal_tflops(fwd_flops, fwd_time_ms)
    bwd_tflops = cal_tflops(bwd_flops, bwd_time_ms)
    total_tflops = cal_tflops(total_flops, total_time_ms)

    return fwd_time_ms, bwd_time_ms, total_time_ms, fwd_flops, bwd_flops, total_flops, fwd_tflops, bwd_tflops, total_tflops, sparsity

def flashmask_block_sparsity(
    causal,
    flashmask,
    B=None,
    H=None,
    HKV=None,
    S=None,
    SKV=None,
    KV_BLOCK_SIZE=128,
    Q_BLOCK_SIZE=128,
    ):

    if flashmask is None and not causal:
        return 0.0
    elif flashmask is None and causal:
        assert S == SKV
        Br = Q_BLOCK_SIZE
        Bc = KV_BLOCK_SIZE
        Tr = S // Br
        Tc = SKV // Bc
        total_size = B * H * S * SKV
        num_sparse_blocks = Tr * (Tc - 1) // 2 * B * H
        sparsity = ((num_sparse_blocks * Bc * Br) / total_size)
        return sparsity

    LTS = None
    LTE = None
    UTS = None
    UTE = None
    if flashmask.shape[-1] == 4:
        LTS, LTE, UTS, UTE = flashmask.split(4, axis=-1)
        LTS = LTS.squeeze(-1)
        LTE = LTE.squeeze(-1)
        UTS = UTS.squeeze(-1)
        UTE = UTE.squeeze(-1)
    elif flashmask.shape[-1] == 2 and causal:
        LTS, LTE = flashmask.split(2, axis=-1)
        LTS = LTS.squeeze(-1)
        LTE = LTE.squeeze(-1)
    elif flashmask.shape[-1] == 2 and not causal:
        LTS, UTE = flashmask.split(2, axis=-1)
        LTS = LTS.squeeze(-1)
        UTE = UTE.squeeze(-1)
    else:
        LTS = flashmask.squeeze(-1)

    Br = Q_BLOCK_SIZE
    Bc = KV_BLOCK_SIZE

    # Note(umiswing): hack block size to seqlen when seqlen < block size, so the calculation code can reuse.
    Br = min(Br, S)
    Bc = min(Bc, SKV)
    
    if LTS is not None:
        B, H_mask, S = LTS.shape
    if LTE is not None:
        B, H_mask, S = LTE.shape
    if UTS is not None:
        B, H_mask, S = UTS.shape
    if UTE is not None:
        B, H_mask, S = UTE.shape
    
    Tr = S // Br
    Tc = SKV // Bc

    if LTS is not None:
        LTS = LTS.cpu().detach().numpy()
    else:
        LTS = np.full((B, H_mask, SKV), S, dtype=np.int32)
    LTStartMax = np.array(LTS).reshape([B, H_mask, -1, Bc]).max(axis=-1)
    LTStartMin = np.array(LTS).reshape([B, H_mask, -1, Bc]).min(axis=-1)

    if LTE is not None:
        LTE = LTE.cpu().detach().numpy()
    else:
        LTE = np.full((B, H_mask, SKV), S, dtype=np.int32)
    LTEndMax = np.array(LTE).reshape([B, H_mask, -1, Bc]).max(-1)
    LTEndMin = np.array(LTE).reshape([B, H_mask, -1, Bc]).min(-1)
    
    if UTS is not None:
        UTS = UTS.cpu().detach().numpy()
    else:
        UTS = np.full((B, H_mask, SKV,), 0, dtype=np.int32)
    UTStartMax = np.array(UTS).reshape([B, H_mask, -1, Bc]).max(-1)
    UTStartMin = np.array(UTS).reshape([B, H_mask, -1, Bc]).min(-1)

    if UTE is not None:
        UTE = UTE.cpu().detach().numpy()
    else:
        assert S == SKV
        UTE = np.tile(np.arange(S, dtype=np.int32).reshape(1, 1, S), (B, H_mask, 1))
    UTEndMax = np.array(UTE).reshape([B, H_mask, -1, Bc]).max(-1)
    UTEndMin = np.array(UTE).reshape([B, H_mask, -1, Bc]).min(-1)

    
    num_dense_blocks = 0
    for bsz in range(B):
        for q_head in range(H):
            head = q_head // (H // H_mask)
            for i in range(Tr):
                for j in range(Tc):
                    if causal and j > i:
                        #print('S', end="")
                        continue
                    if i * Br >= LTStartMax[bsz, head, j] and (i+1) * Br <= LTEndMin[bsz, head, j]:
                        #print('S', end="")
                        continue
                    if i * Br >= UTStartMax[bsz, head, j] and (i+1) * Br <= UTEndMin[bsz, head, j]:
                        #print('S', end="")
                        continue
            
                    if (i+1) * Br > LTStartMin[bsz, head, j] and i * Br < LTEndMax[bsz, head, j]:
                        #print('A', end="")
                        num_dense_blocks += 1
                        continue
                    if (i+1) * Br > UTStartMin[bsz, head, j] and i * Br < UTEndMax[bsz, head, j]:
                        #print('A', end="")
                        num_dense_blocks += 1
                        continue
            
                    #print('C', end="")
                    num_dense_blocks += 1
                #print()

    num_sparse_blocks = B * H * Tc * Tr - num_dense_blocks
    total_size = B * H * S * SKV
    sparsity = ((num_sparse_blocks * Bc * Br) / total_size)
    return sparsity


def generate_none_mask(B, S, H, D, causal=True):
    return None, causal

def generate_sliding_window_mask(B, S, H, D, window_size=1024):
    startend_row_indices = paddle.arange(
        window_size, S + window_size, dtype="int32"
    ).reshape((1, 1, S, 1))
    startend_row_indices = paddle.clip(
        startend_row_indices, max=S
    ).repeat_interleave(B, 0)

    causal=True
    return startend_row_indices, causal

def generate_causal_document_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 1
    padding = S - np.sum(doc_seq_lens)
    doc_seq_lens[-1] += padding
    seq_cusums = np.cumsum(doc_seq_lens)

    startend_row_indices = np.repeat(seq_cusums, doc_seq_lens)
    startend_row_indices = paddle.to_tensor(startend_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    
    causal = True
    return startend_row_indices, causal


def generate_document_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 1
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
            cur_len_so_far += doc_seq_lens[i]
    if padding > 0:
        up_right_row_indices.extend([cur_len_so_far] * padding)
    
    down_left_row_indices = paddle.to_tensor(down_left_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
    
    causal = False
    return startend_row_indices, causal

def generate_share_question_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):

    total_seq_len = sum([sum(doc) for doc in doc_seq_lens])
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 1
    padding = S - total_seq_len
    if padding > 0:
        doc_seq_lens.append([padding])

    startend_row_indices = []
    seqlen_so_far = 0
    for doc in doc_seq_lens:
        assert len(doc) >= 1
        doc_len = sum(doc)
        for idx, seqlen in enumerate(doc):
            if idx == 0:
                startend_row_indices.extend([seqlen_so_far + doc_len] * doc[idx])
            else:
                startend_row_indices.extend([seqlen_so_far + seqlen] * doc[idx])
            seqlen_so_far += seqlen

    assert seqlen_so_far == S

    startend_row_indices = paddle.to_tensor(startend_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    
    causal = True
    return startend_row_indices, causal

def generate_global_sliding_window_mask(B, S, H, D, global_token=16, window_size=(512, 512)):
    assert len(window_size) == 2
    left_window_size, right_window_size = window_size

    down_left_start_row_indices = []
    down_left_end_row_indices = []
    up_right_start_row_indices = []
    up_right_end_row_indices = []

    down_left_start_row_indices = paddle.arange(
        left_window_size + 1, S + left_window_size + 1, dtype="int32"
    ).clip(max=S)
    down_left_start_row_indices[:global_token] = 0
    down_left_start_row_indices = down_left_start_row_indices.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    down_left_end_row_indices = paddle.full([S], S, dtype="int32")
    down_left_end_row_indices[:global_token] = 0
    down_left_end_row_indices = down_left_end_row_indices.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

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

def generate_causal_blockwise_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
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

def generate_prefix_lm_document_mask(B, S, H, D, doc_seq_lens=[(1024, 2538), (1742, 1742), (512, 3213)]):
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

def generate_prefix_lm_causal_mask(B, S, H, D, prefix_length=1024):
    """
    tuple(prefix_length, seq_length)
    """
    assert prefix_length <= S
    down_left_row_indices = paddle.full([S], S, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    up_right_row_indices = paddle.to_tensor([0] * prefix_length + list(range(prefix_length, S)), dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)

    causal = False
    return startend_row_indices, causal

def generate_qk_sparse_mask(B, S, H, D, maskout_pair=[(1024, 538), (2358, 1700)]):
    """
    tuple(offset, maskout_len)
    """
    start_row_indices = []
    end_row_indices  = []
    last_offset = 0
    for offset, maskout_len in maskout_pair:
        assert offset >= last_offset
        start_row_indices.extend(list(range(last_offset, offset)))
        end_row_indices.extend(list(range(last_offset, offset)))

        start_row_indices.extend(list(range(offset, offset+maskout_len)))
        end_row_indices.extend([offset+maskout_len]*(maskout_len))

        last_offset = offset + maskout_len

    last_offset <= S
    start_row_indices.extend(list(range(last_offset, S)))
    end_row_indices.extend(list(range(last_offset, S)))

    start_row_indices = paddle.to_tensor(start_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    end_row_indices = paddle.to_tensor(end_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([start_row_indices, end_row_indices], axis=-1)

    causal = True
    return startend_row_indices, causal

def generate_random_eviction_mask(B, S, H, D, start_row=4096):
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

def generate_hybrid_swa_causal_mask(batch_size, seqlen, hkv, d, window_size=512, ratio=3):
    assert hkv % (ratio + 1) == 0
    hswa = hkv // (ratio + 1) * ratio
    hcausal = hkv // (ratio + 1)

    swa_startend_row_indices = paddle.arange(
        window_size, seqlen + window_size, dtype="int32"
    ).reshape((1, 1, seqlen, 1))

    swa_startend_row_indices = paddle.clip(
        swa_startend_row_indices, max=seqlen,
    ).repeat_interleave(batch_size, 0).repeat_interleave(hswa, 1)

    causal_startend_row_indices = paddle.arange(0, seqlen, dtype="int32"
    ).reshape((1, 1, seqlen, 1)).repeat_interleave(batch_size, 0).repeat_interleave(hcausal, 1)

    startend_row_indices = paddle.concat(x=[swa_startend_row_indices, causal_startend_row_indices], axis=1)
    return startend_row_indices, True

def generate_hybrid_swa_prefix_lm_document_mask(batch_size, seqlen, hkv, d, doc_seq_lens, window_size=512, ratio=3):
    assert hkv % (ratio + 1) == 0
    hswa = hkv // (ratio + 1) * ratio
    hprefix = hkv // (ratio + 1)

    # Note(umiswing): its so silly that this gen func dont do anything for num head
    prefix_lm_document_mask, _ = generate_prefix_lm_document_mask(batch_size, seqlen, hkv, d, doc_seq_lens)
    prefix_lm_document_mask = paddle.repeat_interleave(prefix_lm_document_mask, hkv, 1)

    swa_prefix_lm_document_mask = prefix_lm_document_mask[:,:hswa,:,:]
    pure_prefix_lm_document_mask = prefix_lm_document_mask[:,hswa:,:,:]

    lts = swa_prefix_lm_document_mask[..., 0].unsqueeze(axis=-1)
    ute = swa_prefix_lm_document_mask[..., 1].unsqueeze(axis=-1)

    swa_startend_row_indices = paddle.arange(
        window_size, seqlen + window_size, dtype="int32"
    ).reshape((1, 1, seqlen, 1))

    swa_startend_row_indices = paddle.clip(
        swa_startend_row_indices, max=seqlen,
    ).repeat_interleave(batch_size, 0).repeat_interleave(hswa, 1)

    hybrid_lts = paddle.minimum(swa_startend_row_indices, lts)
    swa_prefix_lm_document_mask = paddle.concat(x=[hybrid_lts, ute], axis=3)

    hybrid_mask = paddle.concat(x=[swa_prefix_lm_document_mask, pure_prefix_lm_document_mask], axis=1)
    return hybrid_mask, False

def hybrid_swa(batch_size, seqlen, hkv, causal, startend_row_indices, window_size, swa_ratio):
    assert not causal
    assert startend_row_indices.shape[-1] == 2
    assert startend_row_indices.shape[1] <= hkv

    if startend_row_indices.shape[1] != hkv:
        startend_row_indices = paddle.repeat_interleave(startend_row_indices, hkv, 1)

    h_hybrid = int(hkv * swa_ratio)

    hybrid_part = startend_row_indices[:, :h_hybrid, :, :]
    non_hybrid_part = startend_row_indices[:, h_hybrid:, :, :]

    hybrid_lts = hybrid_part[..., 0].unsqueeze(axis=-1)
    hybrid_ute = hybrid_part[..., 1].unsqueeze(axis=-1)

    swa_startend_row_indices = paddle.arange(
        window_size, seqlen + window_size, dtype="int32"
    ).reshape((1, 1, seqlen, 1))

    swa_startend_row_indices = paddle.clip(
        swa_startend_row_indices, max=seqlen,
    ).repeat_interleave(batch_size, 0).repeat_interleave(h_hybrid, 1)

    hybrid_lts = paddle.minimum(swa_startend_row_indices, hybrid_lts)
    hybrid_part = paddle.concat(x=[hybrid_lts, hybrid_ute], axis=3)

    startend_row_indices = paddle.concat(x=[hybrid_part, non_hybrid_part], axis=1)
    return startend_row_indices

def preprocess_index_dual_chunks(startend_row_indices, chunk_id_first, chunk_id_second, seq_blocksize, max_seqlen_q):
    """ 
    Preprocess row indices for dual chunks (DualChunkSwap strategy).

    This function handles the index preprocessing for the balanced dual-chunk
    strategy where each rank processes chunks from both ends of the sequence.

    Args:
        startend_row_indices (paddle.Tensor): Original row indices
        chunk_id_first (int): ID of the first chunk
        chunk_id_second (int): ID of the second chunk
        seq_blocksize (int): Size of each sequence block
        max_seqlen_q (int): Maximum sequence length for queries

    Returns:
        paddle.Tensor: Preprocessed row indices for dual chunks
    """
    # Calculate starting positions for both chunks
    rows_min_first = chunk_id_first * seq_blocksize
    rows_min_second = chunk_id_second * seq_blocksize

    # Process first chunk indices
    indices_first = startend_row_indices - rows_min_first
    indices_first = paddle.clip(indices_first, min=0, max=max_seqlen_q)

    # Process second chunk indices
    indices_second = startend_row_indices - rows_min_second
    indices_second = paddle.clip(indices_second, min=0, max=max_seqlen_q)

    # Offset second chunk indices to avoid overlap
    indices_second = paddle.where(indices_second != 0, indices_second + max_seqlen_q, indices_second)

    # Combine indices from both chunks
    combined_indices = paddle.maximum(indices_first, indices_second)
    return combined_indices

def load_mask(batch_size, seqlen, hkv, head_dim, path, causal, hybrid_mask_fn=None, cp_size=1, cp_rank=0):
    startend_row_indices = paddle.load(path)
    if hybrid_mask_fn is not None:
        startend_row_indices = hybrid_mask_fn(batch_size, seqlen, hkv, causal, startend_row_indices)

    if cp_size > 1:
        startend_row_indices = preprocess_index_dual_chunks(
            startend_row_indices,
            chunk_id_first=cp_rank,
            chunk_id_second=2 * cp_size - cp_rank - 1,
            seq_blocksize=seqlen // 2,
            max_seqlen_q=seqlen // 2,
        )

    mask_np = startend_row_indices.numpy()
    return startend_row_indices, causal

def split_sequence(sequence_length):
    if sequence_length < 3:
        raise ValueError("序列长度必须至少为 3，以保证能够分配给一个 Question 和两个 Answer。")
    
    # 确定 Answer 的数量
    num_answers = random.randint(2, 6)
    
    # 初始化分配的长度
    lengths = [1] * (num_answers + 1)  # 至少给每个部分分配一个长度，确保为正整数
    
    # 剩余的长度需要分配
    remaining_length = sequence_length - sum(lengths)
    
    # 随机分配剩余的长度
    for _ in range(remaining_length):
        # 随机选择一个位置增加长度
        index = random.randint(0, num_answers)
        lengths[index] += 1

    return lengths

def method_name(fm_version, kv_mode):
    """CSV / plot_radar method prefix for a (version, kv_mode) pair.

    'shared' gets a suffix WITHOUT a separating underscore because plot_radar.py
    globs '{method}_*': 'flashmaskv4_kvshared_...' would also match method
    'flashmaskv4' and silently average the two modes together.
    """
    return f"flashmaskv{fm_version}" + ("kvshared" if kv_mode == "shared" else "")


def main(examples: List[str] = ["all"], dtype='bf16', fm_version=1, suffix="", overwrite=True, head_dim=None, current_time=None, backend='cutedsl', kv_mode=None, use_sink=False, vs_sparse_attn=False, dedup_static_masks=False):
    """Run the benchmark with the given examples.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
        kv_mode: 'split' keeps K and V in separate buffers (the default, and the
            only option below head_dim 512). 'shared' passes V as a stride-1 view
            onto K, which is what makes the SM100 big-headdim backward merge dK
            and dV into one accumulator. 'sweep' measures both. 'shared' is
            silently dropped for every (D, DV) outside KV_SHARED_D_DV: without
            the merge it is the split kernel with an aliased V, so it would burn
            a second full run to reproduce the 'split' number.
        use_sink: add the learnable per-head attention sink. The online latent-MQA
            layers carry one when ``add_full_attention_sink_bias`` is set and run
            sinkless otherwise, so both are worth measuring.
        vs_sparse_attn: restrict the run to what benchmark_flashmla_sparse_attn.py
            can be compared against. It only emits the ``Causal`` and ``Causal
            Document Mask`` operations, and plot_radar drops any row not shared by
            every method, so the other masks would be measured and then thrown
            away. It also pins the shapes to that kernel's constraints
            (H=64, HKV=1, d_v=512, kv_mode='shared'); an explicit --head_dim or
            --kv_mode still wins.
        dedup_static_masks: skip the masks that only depend on S (S_ONLY_EXAMPLES)
            on every sample but the first of each ``Total length`` block. Those
            masks are identical across the 5 samples, so the extra 4 runs measure
            the same kernel again. Applies under vs_sparse_attn as well, where it
            leaves ``Causal`` on the first sample and ``Causal Document Mask`` on
            all five.

    Sequence lengths and document layouts come from kernel_test_seq_info.txt --
    every ``Total length`` block, every sample. Add a case by adding a line
    there; nothing here is parameterised by seqlen.
    """
    # Operations benchmark_flashmla_sparse_attn.py also produces (its `layouts`
    # dict). Keep in sync with that file.
    VS_SPARSE_EXAMPLES = ("Causal", "Causal Document Mask")

    # These masks are a function of S alone (Full/Causal, or a window / prefix /
    # start row derived from S), so the 5 samples inside one "Total length" block
    # of kernel_test_seq_info.txt all produce the identical mask. With
    # --dedup_static_masks they are measured on the first sample of each block only.
    # This applies under --vs_sparse_attn too: Causal is S-only there as well, so
    # only Causal Document Mask is left on the later samples. plot_radar averages
    # the samples of one seqlen by Operation name, so a row present in one sample
    # file and absent from the others is handled.
    S_ONLY_EXAMPLES = ("Full", "Causal", "Sliding Window",
                       "Prefix LM Causal Mask", "Random Eviction Mask")

    if kv_mode is None:
        # The competitor's K and V are one single-head latent buffer, so 'shared'
        # is the layout that lines up with it.
        kv_mode = 'shared' if vs_sparse_attn else 'split'
    if kv_mode not in ('split', 'shared', 'sweep'):
        raise ValueError(f"kv_mode must be 'split', 'shared' or 'sweep', but got {kv_mode}")
    kv_modes = ['split', 'shared'] if kv_mode == 'sweep' else [kv_mode]

    if current_time is None:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    if backend not in ('cpp', 'cutedsl'):
        raise ValueError(f"backend must be 'cpp' or 'cutedsl', but got {backend}")
    if backend == 'cpp' and fm_version == 4:
        raise ValueError(
            f"backend 'cpp' is not supported for fa4 (fm_version=4), "
            f"please use 'cutedsl' instead"
        )
    if backend == 'cutedsl' and fm_version not in (3, 4):
        raise ValueError(
            f"backend switching to 'cutedsl' is only allowed for fa3/fa4 "
            f"(fm_version=3 or 4), but got fm_version={fm_version}"
        )

    if fm_version == 1:
        paddle.set_flags({'FLAGS_flash_attn_version': 2})
    elif fm_version == 3:
        paddle.set_flags({'FLAGS_flash_attn_version': 3})
    elif fm_version == 4:
        paddle.set_flags({'FLAGS_flash_attn_version': 4})
    else:
        raise ArgumentError(f"fm_version must be 1 or 3 or 4, but got {fm_version}")

    d_list = [128, 192, 256, 512] if fm_version == 4 else [64, 128, 256]
    if head_dim is not None:
        d_list = [head_dim]

    if vs_sparse_attn:
        if head_dim is None:
            d_list = [576]
        elif head_dim not in (512, 576):
            # Mirror benchmark_flashmla_sparse_attn.py's unsupported_reason(): that
            # kernel only runs (D, DV) in ((512, 512), (576, 512)), so nothing else
            # can be put next to it -- and those are also the only pairs with a
            # kv-shared backward.
            raise ValueError(
                f"--vs_sparse_attn cannot measure head_dim={head_dim}: "
                f"benchmark_flashmla_sparse_attn.py only runs {KV_SHARED_D_DV} "
                "(see its unsupported_reason). Drop --head_dim or pass 512 / 576."
            )

    total_length = 0
    doc_seq_lens_list = []
    with open('kernel_test_seq_info.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if 'Total length' in line:
                total_length = int(line.split(":")[1].split(',')[0].strip())
            else:
                doc_list = eval(line.split(":")[-1].split("#")[0].strip())
                qksparse_mask = eval(line.split(":")[-1].split("#")[1].strip())
                doc_seq_lens_list.append((total_length, doc_list, qksparse_mask))
        #doc_seq_lens_list = doc_seq_lens_list[::-1]
        # The first sample of each "Total length" block: the one entry per S that
        # still measures the S_ONLY_EXAMPLES under --dedup_static_masks.
        first_idx_of_S = {}
        for i, (s, _, _) in enumerate(doc_seq_lens_list):
            first_idx_of_S.setdefault(s, i)
        for D in d_list:
            if D == 192:
                DV = 128
                H = 16
            elif D == 576:
                DV = 512
                H = 8
            else:
                DV = D
                H = 4096 // D

            if vs_sparse_attn:
                H = 64
                HKV = 1
            else:    
                HKV = H

            # kv_mode='shared' only means something where the backward actually
            # merges dK and dV; anywhere else it is the split kernel with an
            # aliased V, i.e. a duplicate of the 'split' number for twice the
            # wall clock. Decide once per D instead of re-testing per sample.
            if (D, DV) in KV_SHARED_D_DV:
                d_kv_modes = kv_modes
            else:
                d_kv_modes = [m for m in kv_modes if m != 'shared']
                if 'shared' in kv_modes:
                    print(f"D={D} DV={DV}: skipping kv_mode=shared, the dK/dV "
                          f"merge is only implemented for {KV_SHARED_D_DV} so it "
                          f"would just repeat the 'split' measurement.")
                if not d_kv_modes:
                    continue

            for idx, (S, prefix_doc_seq_lens, qksparse_mask) in enumerate(doc_seq_lens_list):
                if vs_sparse_attn:
                    B = 1
                else:
                    B = 128 * 1024 // S

                SQ = S
                SKV = S
                doc_seq_lens = [x[1] for x in prefix_doc_seq_lens]
                maskout_pair = []
                offset = 0
                print(f"{B}_{S}_{H}_{HKV}_{D}_{DV}_{idx}_{dtype}")
                if not overwrite:
                    done = [
                        os.path.exists(f"{dtype}{suffix}/{method_name(fm_version, m)}_{B}_{S}_{H}_{D}_{DV}_{idx}.csv")
                        for m in d_kv_modes
                    ]
                    if all(done):
                        print(f"{dtype}{suffix}/*_{B}_{S}_{H}_{D}_{DV}_{idx}.csv already exists for kv_mode={d_kv_modes}, skipping. To enable overwrite, use: --overwrite (True by default).")
                        continue
                if sum(qksparse_mask) == 0:
                    maskout_pair = [(1024, 538), (2358, 1700)]
                else:
                    for is_maskout, doc_seq in zip(qksparse_mask, doc_seq_lens):
                        if is_maskout:
                            maskout_pair.append((offset, doc_seq))
                        offset += doc_seq

                share_qa_docs = [split_sequence(doc_seq) for doc_seq in doc_seq_lens]
                # Every entry takes kv_mode ('split' | 'shared') so the same mask
                # can be measured with K and V separate or aliased into one buffer.
                available_examples = {
                    "Full": lambda kv_mode: test_mask(generate_mask_fn=partial(generate_none_mask, causal=False), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend, kv_mode=kv_mode, use_sink=use_sink),
                    "Causal": lambda kv_mode: test_mask(generate_mask_fn=partial(generate_none_mask, causal=True), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend, kv_mode=kv_mode, use_sink=use_sink),
                    "Sliding Window": lambda kv_mode: test_mask(generate_mask_fn=partial(generate_sliding_window_mask, window_size=int(S*0.0625)), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend, kv_mode=kv_mode, use_sink=use_sink),
                    "Causal Document Mask": lambda kv_mode: test_mask(generate_mask_fn=partial(generate_causal_document_mask, doc_seq_lens=doc_seq_lens), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend, kv_mode=kv_mode, use_sink=use_sink),
                    "Document Mask": lambda kv_mode: test_mask(generate_mask_fn=partial(generate_document_mask, doc_seq_lens=doc_seq_lens), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend, kv_mode=kv_mode, use_sink=use_sink),
                    "Share Question Mask": lambda kv_mode: test_mask(generate_mask_fn=partial(generate_share_question_mask, doc_seq_lens=share_qa_docs), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend, kv_mode=kv_mode, use_sink=use_sink),
                    "Causal Blockwise Mask": lambda kv_mode: test_mask(generate_mask_fn=partial(generate_causal_blockwise_mask, doc_seq_lens=doc_seq_lens), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend, kv_mode=kv_mode, use_sink=use_sink),
                    "Prefix LM Document Mask": lambda kv_mode: test_mask(generate_mask_fn=partial(generate_prefix_lm_document_mask, doc_seq_lens=prefix_doc_seq_lens), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend, kv_mode=kv_mode, use_sink=use_sink),
                    "Prefix LM Causal Mask": lambda kv_mode: test_mask(generate_mask_fn=partial(generate_prefix_lm_causal_mask, prefix_length=int(S*0.5)), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend, kv_mode=kv_mode, use_sink=use_sink),
                    "QK-sparse Mask": lambda kv_mode: test_mask(generate_mask_fn=partial(generate_qk_sparse_mask, maskout_pair=maskout_pair), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend, kv_mode=kv_mode, use_sink=use_sink),
                    "Random Eviction Mask": lambda kv_mode: test_mask(generate_mask_fn=partial(generate_random_eviction_mask, start_row=S//2), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend, kv_mode=kv_mode, use_sink=use_sink),
                    # "Hybrid SWA Prefix LM Doc": lambda kv_mode: test_mask(generate_mask_fn=partial(generate_hybrid_swa_prefix_lm_document_mask, doc_seq_lens=prefix_doc_seq_lens), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend, kv_mode=kv_mode, use_sink=use_sink),
                    # Note(umiswing): support load mask and hybrid mask like this, and also, support simulate cp benchmark
                    # "Dumped Mask": lambda kv_mode: test_mask(generate_mask_fn=partial(load_mask, path=mask_path, causal=False, cp_size=cp_size, cp_rank=cp_rank), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend, kv_mode=kv_mode, use_sink=use_sink),
                    # "Hybrid SWA": lambda kv_mode: test_mask(generate_mask_fn=partial(load_mask, path=mask_path, causal=False, cp_size=cp_size, cp_rank=cp_rank, hybrid_mask_fn=partial(hybrid_swa, window_size=512, swa_ratio=0.75)), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend, kv_mode=kv_mode, use_sink=use_sink),
                }

                # Global Sliding Window is enabled for fa3, but disabled for fa4.
                if fm_version == 3:
                    available_examples["Global Sliding Window"] = lambda kv_mode: test_mask(generate_mask_fn=partial(generate_global_sliding_window_mask, global_token=16, window_size=(int(S*0.0625), int(S*0.0625))), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend, kv_mode=kv_mode)


                if "all" in examples:
                    ex_to_run = list(available_examples.keys())
                else:
                    ex_to_run = examples

                if dedup_static_masks and idx != first_idx_of_S[S]:
                    ex_to_run = [ex for ex in ex_to_run if ex not in S_ONLY_EXAMPLES]

                if vs_sparse_attn:
                    dropped = [ex for ex in ex_to_run if ex not in VS_SPARSE_EXAMPLES]
                    if dropped:
                        print(f"--vs_sparse_attn: dropping {dropped}, "
                              f"benchmark_flashmla_sparse_attn.py only emits "
                              f"{list(VS_SPARSE_EXAMPLES)} and plot_radar keeps "
                              f"only the operations every method shares.")
                    ex_to_run = [ex for ex in ex_to_run if ex in VS_SPARSE_EXAMPLES]

                if not ex_to_run:
                    print("Nothing left to measure for this sample after the "
                          "--dedup_static_masks / --vs_sparse_attn filters, skipping.")
                    continue

                for mode in d_kv_modes:
                    results = []
                    for ex in ex_to_run:
                        if ex in available_examples:
                            print(f"{ex} [kv_mode={mode}]")
                            fw_time, bw_time, total_time, fw_flops, bw_flops, total_flops, fw_tflops, bw_tflops, total_tflops, sparsity = available_examples[ex](mode)
                            results.append([ex, f"{fw_time:.4f}", f"{bw_time:.4f}", f"{total_time:.4f}", f"{fw_flops:.4f}", f"{bw_flops:.4f}", f"{total_flops:.4f}", f"{fw_tflops:.4f}", f"{bw_tflops:.4f}", f"{total_tflops:4f}", f"{sparsity:.4f}"])
                        else:
                            print(f"Warning: Unknown example key '{ex}'. Skipping.")

                    # Usage in your results formatting:
                    headers = [
                        "Operation",
                        "FW Time (ms)",
                        "BW Time (ms)",
                        "TOTAL Time (ms)",
                        "FW FLOPs",
                        "BW FLOPs",
                        "TOTAL FLOPs",
                        "FW TFLOPs/s",
                        "BW TFLOPs/s",
                        "TOTAL TFLOPs/s",
                        "Sparsity",
                    ]
                    print(
                        tabulate(
                            results,
                            headers=headers,
                            tablefmt="grid",
                        )
                    )
                    content2=tabulate(results, headers=headers, tablefmt="tsv")
                    os.makedirs(f"{dtype}{suffix}", exist_ok=True)
                    # Note(umiswing): this file name is better, but i need to keep the old name for fig plotting
                    # Note: no underscore before "kvshared" on purpose -- plot_radar.py
                    # globs "{method}_*", so "flashmaskv4_kvshared_..." would also be
                    # picked up as method "flashmaskv4" and mix the two modes.
                    method = method_name(fm_version, mode)
                    text_file = open(f"{dtype}{suffix}/{method}_{current_time}_{B}_{S}_{H}_{HKV}_{D}_{DV}_{idx}.csv","w")
                    text_file.write(content2)
                    text_file.close()

if __name__ == "__main__":
    from jsonargparse import ArgumentParser
    parser = ArgumentParser(description="Run specific examples or all examples.")
    parser.add_argument(
        "--examples",
        type=str,
        nargs="+",
        default=["all"],
        help="List of examples to run. Use space to separate multiple examples. "
        "Available options: causal, alibi, sliding_window, prefix_lm, "
        "document, softcap, softcap_approx, or 'all' to run all examples.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16"
    )
    parser.add_argument(
        "--fm_version",
        type=int,
        default=1
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=""
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=None
    )
    parser.add_argument(
        "--current_time",
        type=str,
        default=None
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cutedsl",
        choices=["cpp", "cutedsl"],
        help="Kernel backend to benchmark. Only switchable for fa3 (fm_version=3): "
        "'cpp' uses the paddle C++ flashmask kernel, 'cutedsl' uses the cutedsl kernel.",
    )

    parser.add_argument(
        "--kv_mode",
        type=str,
        default=None,
        choices=["split", "shared", "sweep"],
        help="KV layout to benchmark. 'split': K and V are separate buffers "
        "(the default). 'shared': V is a stride-1 view onto K (the MLA "
        "convention), which lets the SM100 big-headdim backward merge dK and dV "
        "-- only implemented for (D, DV) in ((512,512),(576,512)). 'sweep': both. "
        "Left unset, --vs_sparse_attn picks 'shared' and everything else 'split'.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--overwrite", action="store_true", default=True)
    group.add_argument("--no-overwrite", action="store_false", dest="overwrite")

    sink_group = parser.add_mutually_exclusive_group()
    sink_group.add_argument(
        "--use_sink",
        action="store_true",
        default=False,
        help="Add the learnable per-head attention sink (the online latent-MQA "
        "layers carry one when add_full_attention_sink_bias is set).",
    )
    sink_group.add_argument("--no-use_sink", action="store_false", dest="use_sink")

    parser.add_argument(
        "--vs_sparse_attn",
        action="store_true",
        default=False,
        help="Only measure what benchmark_flashmla_sparse_attn.py can be "
        "compared against: the Causal and Causal Document Mask operations, at "
        "that kernel's shapes (head_dim 576, 64 query heads, 1 KV head, "
        "kv_mode shared). An explicit --head_dim or --kv_mode still wins. "
        "Without this flag every mask is measured, as before.",
    )

    parser.add_argument(
        "--dedup_static_masks",
        action="store_true",
        default=False,
        help="Skip the masks that only depend on the sequence length (Full, "
        "Causal, Sliding Window, Prefix LM Causal Mask, Random Eviction Mask) on "
        "every sample but the first of each 'Total length' block of "
        "kernel_test_seq_info.txt. Those samples only differ in their document "
        "layout, so for these masks the extra runs re-measure the same kernel. "
        "Applies under --vs_sparse_attn too, where only Causal Document Mask is "
        "then left on the later samples.",
    )

    args = parser.parse_args()
    main(**vars(args))