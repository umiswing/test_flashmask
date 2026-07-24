import json
import numpy as np
from functools import partial
from typing import Optional, List
from tabulate import tabulate
import time
import paddle
try:
    from flash_mask.cute.interface import flashmask_attention
except (ImportError, ModuleNotFoundError):
    from paddle.nn.functional.flash_attention import flashmask_attention
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
    backend = 'cpp',
):

    if dtype == 'bf16':
        data_type = paddle.bfloat16
    else:
        data_type = paddle.float16

    query = paddle.randn([B, S, H, D], dtype=data_type)
    key = paddle.randn([B, SKV, HKV, D], dtype=data_type)
    value = paddle.randn([B, SKV, HKV, DV], dtype=data_type)
    gradOut = paddle.randn([B, S, H, DV], dtype=data_type)

    query.stop_gradient = False
    key.stop_gradient = False
    value.stop_gradient = False

    startend_row_indices, causal = None, True
    if generate_mask_fn is not None:
        startend_row_indices, causal = generate_mask_fn(B, SKV, HKV, D)

    sparsity = flashmask_block_sparsity(causal, startend_row_indices, B, H, HKV, S, SKV)
    density = 1.0 - sparsity

    sink = None
    if use_sink:
        sink = paddle.randn(shape=[H], dtype=data_type)
        sink.stop_gradient = False

    flashmask = lambda: flashmask_attention(query, key, value, startend_row_indices=startend_row_indices, causal=causal, return_softmax_lse=True)

    fa_version = paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])["FLAGS_flash_attn_version"]

    # fa4 always runs on the cutedsl path; for fa3 the backend is switchable
    # between the paddle C++ kernel ("cpp") and the cutedsl kernel ("cutedsl").
    use_cutedsl = (fa_version == 4) or (backend == 'cutedsl')

    if use_cutedsl:
        query.stop_gradient = True
        key.stop_gradient = True
        value.stop_gradient = True
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

def main(examples: List[str] = ["all"], dtype='bf16', fm_version=1, suffix="_base", overwrite=True, head_dim=None, current_time=None, backend='cpp'):
    """Run the benchmark with the given examples.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
    """
    if current_time is None:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    if backend not in ('cpp', 'cutedsl'):
        raise ValueError(f"backend must be 'cpp' or 'cutedsl', but got {backend}")
    # The cpp/cutedsl switch is only meaningful for fa3. fa1/fa2 only have the
    # cpp kernel, and fa4 always runs on cutedsl.
    if backend == 'cutedsl' and fm_version != 3:
        raise ValueError(f"backend switching to 'cutedsl' is only allowed for fa3 (fm_version=3), but got fm_version={fm_version}")

    if fm_version == 1:
        paddle.set_flags({'FLAGS_flash_attn_version': 2})
    elif fm_version == 3:
        paddle.set_flags({'FLAGS_flash_attn_version': 3})
    elif fm_version == 4:
        paddle.set_flags({'FLAGS_flash_attn_version': 4})
    else:
        raise ArgumentError(f"fm_version must be 1 or 3 or 4, but got {fm_version}")

    d_list = [128, 192, 256] if fm_version == 4 else [64, 128, 256]
    if head_dim is not None:
        d_list = [head_dim]

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
        for D in d_list:
            if D == 192:
                DV = 128
                H = 16
            else:
                DV = D
                H = 4096 // D
            HKV = H

            for idx, (S, prefix_doc_seq_lens, qksparse_mask) in enumerate(doc_seq_lens_list):
                B = 128 * 1024 // S

                SQ = S
                SKV = S
                doc_seq_lens = [x[1] for x in prefix_doc_seq_lens]
                maskout_pair = []
                offset = 0
                print(f"{B}_{S}_{H}_{HKV}_{D}_{DV}_{idx}_{dtype}")
                if not overwrite:
                    if os.path.exists(f"{dtype}{suffix}/flashmaskv{fm_version}_{B}_{S}_{H}_{D}_{DV}_{idx}.csv"):
                        print(f"{dtype}{suffix}/flashmaskv{fm_version}_{B}_{S}_{H}_{D}_{DV}_{idx}.csv already exists, skipping. To enable overwrite, use: --overwrite (True by default).")
                        continue
                if sum(qksparse_mask) == 0:
                    maskout_pair = [(1024, 538), (2358, 1700)]
                else:
                    for is_maskout, doc_seq in zip(qksparse_mask, doc_seq_lens):
                        if is_maskout:
                            maskout_pair.append((offset, doc_seq))
                        offset += doc_seq

                share_qa_docs = [split_sequence(doc_seq) for doc_seq in doc_seq_lens]
                available_examples = {
                    "Full": lambda: test_mask(generate_mask_fn=partial(generate_none_mask, causal=False), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend),
                    "Causal": lambda: test_mask(generate_mask_fn=partial(generate_none_mask, causal=True), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend),
                    "Sliding Window": lambda: test_mask(generate_mask_fn=partial(generate_sliding_window_mask, window_size=int(S*0.0625)), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend),
                    "Causal Document Mask": lambda: test_mask(generate_mask_fn=partial(generate_causal_document_mask, doc_seq_lens=doc_seq_lens), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend),
                    "Document Mask": lambda: test_mask(generate_mask_fn=partial(generate_document_mask, doc_seq_lens=doc_seq_lens), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend),
                    "Share Question Mask": lambda: test_mask(generate_mask_fn=partial(generate_share_question_mask, doc_seq_lens=share_qa_docs), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend),
                    "Causal Blockwise Mask": lambda: test_mask(generate_mask_fn=partial(generate_causal_blockwise_mask, doc_seq_lens=doc_seq_lens), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend),
                    "Prefix LM Document Mask": lambda: test_mask(generate_mask_fn=partial(generate_prefix_lm_document_mask, doc_seq_lens=prefix_doc_seq_lens), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend),
                    "Prefix LM Causal Mask": lambda: test_mask(generate_mask_fn=partial(generate_prefix_lm_causal_mask, prefix_length=int(S*0.5)), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend),
                    "QK-sparse Mask": lambda: test_mask(generate_mask_fn=partial(generate_qk_sparse_mask, maskout_pair=maskout_pair), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend),
                    "Random Eviction Mask": lambda: test_mask(generate_mask_fn=partial(generate_random_eviction_mask, start_row=S//2), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend),
                    # "Hybrid SWA Prefix LM Doc": lambda: test_mask(generate_mask_fn=partial(generate_hybrid_swa_prefix_lm_document_mask, doc_seq_lens=prefix_doc_seq_lens), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend),
                    # Note(umiswing): support load mask and hybrid mask like this, and also, support simulate cp benchmark
                    # "Dumped Mask": lambda: test_mask(generate_mask_fn=partial(load_mask, path=mask_path, causal=False, cp_size=cp_size, cp_rank=cp_rank), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend),
                    # "Hybrid SWA": lambda: test_mask(generate_mask_fn=partial(load_mask, path=mask_path, causal=False, cp_size=cp_size, cp_rank=cp_rank, hybrid_mask_fn=partial(hybrid_swa, window_size=512, swa_ratio=0.75)), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend),
                }

                # Global Sliding Window is enabled for fa3, but disabled for fa4.
                if fm_version == 3:
                    available_examples["Global Sliding Window"] = lambda: test_mask(generate_mask_fn=partial(generate_global_sliding_window_mask, global_token=16, window_size=(int(S*0.0625), int(S*0.0625))), B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype, backend=backend)


                if "all" in examples:
                    ex_to_run = list(available_examples.keys())
                else:
                    ex_to_run = examples

                results = []
                for ex in ex_to_run:
                    if ex in available_examples:
                        print(ex)
                        fw_time, bw_time, total_time, fw_flops, bw_flops, total_flops, fw_tflops, bw_tflops, total_tflops, sparsity = available_examples[ex]()
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
                text_file = open(f"{dtype}{suffix}/flashmaskv{fm_version}_{current_time}_{B}_{S}_{H}_{HKV}_{D}_{DV}_{idx}.csv","w")
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
        default="cpp",
        choices=["cpp", "cutedsl"],
        help="Kernel backend to benchmark. Only switchable for fa3 (fm_version=3): "
        "'cpp' uses the paddle C++ flashmask kernel, 'cutedsl' uses the cutedsl kernel.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--overwrite", action="store_true", default=True)
    group.add_argument("--no-overwrite", action="store_false", dest="overwrite")

    args = parser.parse_args()
    main(**vars(args))