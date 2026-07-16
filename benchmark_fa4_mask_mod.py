import os
import numpy as np
from functools import lru_cache
from typing import Optional, List
import random
import gc

import torch

from tabulate import tabulate
from torch.nn.attention.flex_attention import (
    create_block_mask,
    and_masks,
)

from attn_gym.masks import causal_mask

from flash_attn.cute.compute_block_sparsity import compute_block_sparsity

from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd

from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from flash_attn.cute import utils
from flash_attn.cute.cute_dsl_utils import to_cute_tensor

from datetime import datetime

torch.set_default_device("cuda")
torch.manual_seed(0)

np.random.seed(0)
random.seed(0)

torch._dynamo.config.cache_size_limit = 1000

def compute_block_sparse_tensors(
    flex_mask_mod,
    batch_size,
    block_mask_nheads,
    seqlen_q,
    seqlen_k,
    sparse_tile_m,
    tile_n,
):
    if flex_mask_mod is not None:
        bm = create_block_mask(
            flex_mask_mod,
            batch_size,
            block_mask_nheads,
            seqlen_q,
            seqlen_k,
            device="cuda",
            _compile=True,
            BLOCK_SIZE=(sparse_tile_m, tile_n),
        )
        (
            _seq_q,
            _seq_k,
            kv_mask_cnt,
            kv_mask_idx,
            full_kv_cnt,
            full_kv_idx,
            q_mask_cnt,
            q_mask_idx,
            full_q_cnt,
            full_q_idx,
            *_,
        ) = bm.as_tuple()
        block_sparse_tensors_fwd = BlockSparseTensorsTorch(
            mask_block_cnt=kv_mask_cnt,
            mask_block_idx=kv_mask_idx,
            full_block_cnt=full_kv_cnt,
            full_block_idx=full_kv_idx,
            block_size=(sparse_tile_m, tile_n),
        )

        sparse_tile_m_bwd =sparse_tile_m
        block_sparse_tensors_bwd = BlockSparseTensorsTorch(
            mask_block_cnt=q_mask_cnt,
            mask_block_idx=q_mask_idx,
            full_block_cnt=full_q_cnt,
            full_block_idx=full_q_idx,
            block_size=(sparse_tile_m_bwd, tile_n),
        )
    else:
        block_sparse_tensors_fwd = None
        block_sparse_tensors_bwd = None
    return block_sparse_tensors_fwd, block_sparse_tensors_bwd

def compute_density_sparsity(flex_mask_mod, causal, B, H, M, N, tile_m, tile_n, device="cuda"):
    if flex_mask_mod is not None:
        block_mask = create_block_mask(flex_mask_mod, B, H, M, N, device=device, _compile=True, BLOCK_SIZE=[tile_m, tile_n])
        density = (100 - block_mask.sparsity()) / 100
    elif causal:
        # Note(umiswing): if seqlen_q != seqlen_k, the density of causal is not 0.5
        assert M == N
        density = 0.5
    else:
        density = 1.0

    sparsity = 1.0 - density
    return density, sparsity

def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    return multiplier * flops * (1e3 / time_ms) / 1e12

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

def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float32)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times.tolist()
    return getattr(torch, return_mode)(times).item()

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

    torch.cuda.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2 cache
    # doesn't contain any input data before the run
    cache_size = 256 * 1024 * 1024
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if fast_flush:
        cache = torch.empty(int(cache_size // 4), dtype=torch.int32, device=device)
    else:
        cache = torch.empty(int(cache_size), dtype=torch.int8, device=device)

    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    n_warmup = 10
    n_repeat = 50
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    gc.collect()
    gc.disable()
    for i in range(n_repeat):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        #cache.zero_()
        start_event[i].record()
        fn()
        end_event[i].record()
    gc.enable()
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float32)
    return _summarize_statistics(times, quantiles, return_mode)

def print_header(text):
    width = 91
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {text.center(width - 4)} ║")
    print("╚" + "═" * (width - 2) + "╝")

def test_mask(
    mask_info,
    B,
    H,
    HKV,
    S,
    D,
    DV,
    dtype,
    skip_correctness: bool = False,
    print_mask: bool = True,
    device: str = "cuda",
):
    # Note(umiswing): see https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py#L708-L711
    # btw, we actually should not set block_mask_nheads to 1, it should always be qhead, otherwise the mask info is wrong
    # original fa cute test so stupid...
    pack_gqa = False
    cute_mask_mod = mask_info["cute_mask_mod"]
    flex_mask_mod = mask_info["flex_mask_mod"]
    aux_tensors = mask_info["aux_tensors"]
    if aux_tensors is not None:
        aux_tensors_cute = [to_cute_tensor(t, assumed_align=4) for t in aux_tensors]
    else:
        aux_tensors_cute = None
    causal = mask_info["causal"]

    tile_m = 128
    tile_n = 128

    sparse_tile_m_fwd = 2 * tile_m

    if dtype == 'bf16':
        data_type = torch.bfloat16
    else:
        data_type = torch.float16

    block_sparse_tensors_fwd, block_sparse_tensors_bwd = compute_block_sparse_tensors(
        flex_mask_mod=flex_mask_mod,
        batch_size=B,
        block_mask_nheads=1 if pack_gqa else H, # TODO(umiswing): try pack_gqa
        seqlen_q=S,
        seqlen_k=S,
        sparse_tile_m=sparse_tile_m_fwd,
        tile_n=tile_n,
    )

    density, sparsity = compute_density_sparsity(
        flex_mask_mod=flex_mask_mod,
        causal=causal,
        B=B,
        H=H,
        M=S,
        N=S,
        tile_m=tile_m,
        tile_n=tile_n
    )

    q = torch.randn(B, S, H, D, device=device, dtype=data_type, requires_grad=True)
    k = torch.randn(B, S, HKV, D, device=device, dtype=data_type, requires_grad=True)
    v = torch.randn(B, S, HKV, DV, device=device, dtype=data_type, requires_grad=True)

    out, gradOut = [
        torch.randn(B, S, H, DV, device=device, dtype=data_type, requires_grad=True)
        for _ in range(2)
    ]

    lse = torch.empty(B, H, S, device=device, dtype=torch.float32)

    fa4_mask_mod_call = lambda: _flash_attn_fwd(
        q=q, k=k, v=v,
        softmax_scale=None, causal=causal,
        tile_mn=(tile_m, tile_n),
        mask_mod=cute_mask_mod, block_sparse_tensors=block_sparse_tensors_fwd,
        return_lse=True,
        pack_gqa=pack_gqa,
        out=out,
        lse=lse,
        aux_tensors=aux_tensors,
    )

    results = []

    # Forward pass
    # torch.cuda.nvtx.range_push("fa4")
    fwd_time_ms = do_bench(fa4_mask_mod_call)
    # torch.cuda.nvtx.range_pop()
    torch._functorch.config.donated_buffer=False
    # Backward pass
    out_cute, lse_cute, _, _ = fa4_mask_mod_call()

    fa4_mask_mod_bwd_call = lambda: _flash_attn_bwd(
        q=q,
        k=k,
        v=v,
        out=out_cute,
        dout=gradOut,
        lse=lse_cute,
        causal=causal,
        m_block_size=tile_m,
        n_block_size=tile_n,
        mask_mod=cute_mask_mod,
        # Note(umiswing): idk how to pass aux tensors in bm of bwd
        block_sparse_tensors=block_sparse_tensors_bwd,
        # block_sparse_tensors=None,
        aux_tensors=aux_tensors,
        pack_gqa=pack_gqa,
    )

    # torch.cuda.nvtx.range_push("fa4")
    bwd_time_ms = do_bench(fa4_mask_mod_bwd_call)
    # torch.cuda.nvtx.range_pop()

    total_time_ms = fwd_time_ms + bwd_time_ms

    fwd_flops = density * cal_flops(B, H, S, S, D, DV, mode='fwd')
    bwd_flops = density * cal_flops(B, H, S, S, D, DV,  mode='bwd')
    total_flops = density * cal_flops(B, H, S, S, D, DV,  mode='fwd_bwd')

    fwd_tflops = cal_tflops(fwd_flops, fwd_time_ms)
    bwd_tflops = cal_tflops(bwd_flops, bwd_time_ms)
    total_tflops = cal_tflops(total_flops, total_time_ms)

    return fwd_time_ms, bwd_time_ms, total_time_ms, fwd_flops, bwd_flops, total_flops, fwd_tflops, bwd_tflops, total_tflops, sparsity

def generate_sliding_window(window_size: int):

    def sliding_window(b, h, q_idx, kv_idx):
        return q_idx - kv_idx <= window_size

    sliding_window_mask = and_masks(sliding_window, causal_mask)
    sliding_window_mask.__name__ = f"sliding_window_{window_size}"

    @cute.jit
    def cute_sliding_window(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ):
        window_size_cute = utils.scalar_to_ssa(window_size, cutlass.Int32)
        offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        offset_ssa = utils.scalar_to_ssa(offset, cutlass.Int32)
        return (m_idx - n_idx <= window_size_cute) & (n_idx <= (m_idx + offset_ssa))

    return {
        "cute_mask_mod": cute_sliding_window,
        "flex_mask_mod": sliding_window_mask,
        "aux_tensors": None,
        "causal": True,
    }

def generate_causal_document_mask(B=16, S=8192, doc_seq_lens=[2538, 1742, 3213], device="cuda"):

    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 3
    padding = S - np.sum(doc_seq_lens)
    doc_seq_lens[-1] += padding
    seq_cusums = np.cumsum(doc_seq_lens)

    startend_row_indices = np.repeat(seq_cusums, doc_seq_lens).reshape([1, -1])
    startend_row_indices = np.repeat(startend_row_indices, B, axis=0)
    startend_row_indices = torch.tensor(startend_row_indices, device=device, dtype=torch.int32)
     
    def causal_document(b, h, q_idx, kv_idx):
        return q_idx < startend_row_indices[b, kv_idx] 

    causal_document_mask = and_masks(causal_document, causal_mask)
    causal_document_mask.__name__ = f"causal_document_mask"

    @cute.jit
    def cute_causal_document_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ):
        startend_row_indices = aux_tensors[0]
        offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        offset_ssa = utils.scalar_to_ssa(offset, cutlass.Int32)
        return (m_idx < startend_row_indices[batch[0], n_idx[0]]) & (n_idx <= (m_idx + offset_ssa))

    return {
        "cute_mask_mod": cute_causal_document_mask,
        "flex_mask_mod": causal_document_mask,
        "aux_tensors": [startend_row_indices],
        "causal": True,
    }

def generate_document_mask(B=16, S=8192, doc_seq_lens=[2538, 1742, 3213], device="cuda"):

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
            cur_len_so_far += doc_seq_lens[i]
    if padding > 0:
        up_right_row_indices.extend([cur_len_so_far] * padding)

    down_left_row_indices = torch.tensor(down_left_row_indices,  device=device, dtype=torch.int32).reshape((1, -1)).repeat_interleave(B, 0)
    up_right_row_indices= torch.tensor(up_right_row_indices,  device=device, dtype=torch.int32).reshape((1, -1)).repeat_interleave(B, 0)

    def document_mask(b, h, q_idx, kv_idx):
        return (q_idx < down_left_row_indices[b, kv_idx]) & (q_idx >= up_right_row_indices[b, kv_idx])

    @cute.jit
    def cute_document_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        down_left_row_indices = aux_tensors[0]
        up_right_row_indices = aux_tensors[1]
        return (m_idx < down_left_row_indices[batch[0], n_idx[0]]) & (m_idx >= up_right_row_indices[batch[0], n_idx[0]])

    return {
        "cute_mask_mod": cute_document_mask,
        "flex_mask_mod": document_mask,
        "aux_tensors": [down_left_row_indices, up_right_row_indices],
        "causal": False,
    }

def generate_share_question_mask(B=16, S=8192, doc_seq_lens=[2538, 1742, 3213], device="cuda"):

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

    startend_row_indices = torch.tensor(startend_row_indices, device=device, dtype=torch.int32).reshape((1, -1)).repeat_interleave(B, 0)
    
    def share_question_mask(b, h, q_idx, kv_idx):
        return q_idx < startend_row_indices[b, kv_idx] 

    causal_share_question_mask = and_masks(share_question_mask, causal_mask)
    causal_share_question_mask.__name__ = f"causal_share_question_mask"

    @cute.jit
    def cute_share_question_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        startend_row_indices = aux_tensors[0]
        offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        offset_ssa = utils.scalar_to_ssa(offset, cutlass.Int32)
        return (m_idx < startend_row_indices[batch[0], n_idx[0]]) & (n_idx <= (m_idx + offset_ssa))

    return {
        "cute_mask_mod": cute_share_question_mask,
        "flex_mask_mod": causal_share_question_mask,
        "aux_tensors": [startend_row_indices],
        "causal": True,
    }

def generate_global_sliding_window_mask(B=16, S=8192, global_token=16, window_size=(512, 512), device="cuda"):
    assert len(window_size) == 2
    left_window_size, right_window_size = window_size

    def global_sliding_window(b, h, q_idx, kv_idx):
        return ((q_idx >= kv_idx) & ((q_idx - kv_idx <= (left_window_size)) | (kv_idx < global_token))) | ((kv_idx >= q_idx) & ((kv_idx - q_idx <= (right_window_size)) | (q_idx < global_token)))

    @cute.jit
    def cute_global_sliding_window_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        left_window_size_cute = utils.scalar_to_ssa(left_window_size, cutlass.Int32)
        right_window_size_cute = utils.scalar_to_ssa(right_window_size, cutlass.Int32)
        global_token_cute = utils.scalar_to_ssa(global_token, cutlass.Int32)
        return ((m_idx >= n_idx) & ((m_idx - n_idx <= (left_window_size_cute)) | (n_idx < global_token_cute))) | ((n_idx >= m_idx) & ((n_idx - m_idx <= (right_window_size_cute)) | (m_idx < global_token_cute)))

    return {
        "cute_mask_mod": cute_global_sliding_window_mask,
        "flex_mask_mod": global_sliding_window,
        "aux_tensors": None,
        "causal": False,
    }

def generate_causal_blockwise_mask(B=16, S=8192, doc_seq_lens=[2538, 1742, 3213], device="cuda"):
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
    start_row_indices = torch.tensor(start_row_indices, device=device, dtype=torch.int32).reshape((1, -1)).repeat_interleave(B, 0)

    seq_cusums = np.cumsum(doc_seq_lens)
    end_row_indices = [seq_cusums[-2]] * seq_cusums[-2] + [seq_cusums[-1]] * doc_seq_lens[-1] + [S] * padding
    end_row_indices = torch.tensor(end_row_indices, device=device, dtype=torch.int32).reshape((1, -1)).repeat_interleave(B, 0)

    def causal_blockwise(b, h, q_idx, kv_idx):
        return (q_idx < start_row_indices[b, kv_idx]) | (q_idx >= end_row_indices[b, kv_idx])

    causal_blockwise_mask = and_masks(causal_blockwise, causal_mask)
    causal_blockwise_mask.__name__ = f"causal_blockwise_mask"

    @cute.jit
    def cute_causal_blockwise_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        start_row_indices = aux_tensors[0]
        end_row_indices = aux_tensors[1]
        offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        offset_ssa = utils.scalar_to_ssa(offset, cutlass.Int32)
        return ((m_idx < start_row_indices[batch[0], n_idx[0]]) | (m_idx >= end_row_indices[batch[0], n_idx[0]])) & (n_idx <= (m_idx + offset_ssa))

    return {
        "cute_mask_mod": cute_causal_blockwise_mask,
        "flex_mask_mod": causal_blockwise_mask,
        "aux_tensors": [start_row_indices, end_row_indices],
        "causal": True,
    }

def generate_prefix_lm_document_mask(B=16, S=8192, doc_seq_lens=[(1024, 2538), (1742, 1742), (512, 3213)], device="cuda"):
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

    up_right_row_indices = []
    cur_len_so_far = 0
    for prefix_length, seq_length in doc_seq_lens:
        up_right_row_indices.extend([cur_len_so_far] * prefix_length + list(range(cur_len_so_far+prefix_length, cur_len_so_far+seq_length)))
        cur_len_so_far += seq_length
    if padding > 0:
        up_right_row_indices.extend([total_seq_len] * padding)

    down_left_row_indices = torch.tensor(down_left_row_indices,  device=device, dtype=torch.int32).reshape((1, -1)).repeat_interleave(B, 0)
    up_right_row_indices= torch.tensor(up_right_row_indices,  device=device, dtype=torch.int32).reshape((1, -1)).repeat_interleave(B, 0)

    def prefix_lm_document_mask(b, h, q_idx, kv_idx):
        return (q_idx < down_left_row_indices[b, kv_idx]) & (q_idx >= up_right_row_indices[b, kv_idx])

    @cute.jit
    def cute_prefix_lm_document_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        down_left_row_indices = aux_tensors[0]
        up_right_row_indices = aux_tensors[1]
        return (m_idx < down_left_row_indices[batch[0], n_idx[0]]) & (m_idx >= up_right_row_indices[batch[0], n_idx[0]])

    return {
        "cute_mask_mod": cute_prefix_lm_document_mask,
        "flex_mask_mod": prefix_lm_document_mask,
        "aux_tensors": [down_left_row_indices, up_right_row_indices],
        "causal": False,
    }

def generate_prefix_lm_causal_mask(B=16, S=8192, prefix_length=1024, device="cuda"):
    """
    tuple(prefix_length, seq_length)
    """
    assert prefix_length <= S

    up_right_row_indices = torch.tensor([0] * prefix_length + list(range(prefix_length, S)),  device=device, dtype=torch.int32).reshape((1, -1)).repeat_interleave(B, 0)

    def prefix_lm_causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= up_right_row_indices[b, kv_idx]

    @cute.jit
    def cute_prefix_lm_causal_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        up_right_row_indices = aux_tensors[0]
        return m_idx >= up_right_row_indices[batch[0], n_idx[0]]

    return {
        "cute_mask_mod": cute_prefix_lm_causal_mask,
        "flex_mask_mod": prefix_lm_causal_mask,
        "aux_tensors": [up_right_row_indices],
        "causal": True, # Note(umiswing): should this be true?
    }

def generate_qk_sparse_mask(B=16, S=8192, maskout_pair=[(1024, 538), (2358, 1700)], device="cuda"):

    """
    tuple(offset, maskout_len)
    """
    row_indices  = []
    last_offset = 0
    for offset, maskout_len in maskout_pair:
        assert offset >= last_offset
        row_indices.extend(list(range(last_offset, offset)))
        row_indices.extend([offset+maskout_len]*(maskout_len))

        last_offset = offset + maskout_len

    last_offset <= S
    row_indices.extend(list(range(last_offset, S)))

    assert len(row_indices) == S, len(row_indices)
    row_indices = torch.tensor(row_indices, device=device, dtype=torch.int32).reshape((1, -1)).repeat_interleave(B, 0)

    def qk_sparse_mask(b, h, q_idx, kv_idx):
        return q_idx >= row_indices[b, kv_idx]

    @cute.jit
    def cute_qk_sparse_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        row_indices = aux_tensors[0]
        return m_idx >= row_indices[batch[0], n_idx[0]]

    return {
        "cute_mask_mod": cute_qk_sparse_mask,
        "flex_mask_mod": qk_sparse_mask,
        "aux_tensors": [row_indices],
        "causal": True, # Note(umiswing): should this be true?
    }


def generate_random_eviction_mask(B=16, H=16, HKV=16, S=8192, start_row=4096, device="cuda"):
    assert H == HKV
    start_rows_list = []
    for bz_idx in range(B):
        for head_idx in range(H):
            start_rows = np.array([S+1] * S)
            mask_pos = np.random.choice(S-1, S - start_row, replace=False)
            index = np.arange(start_row, S)
            mask_pos = np.concatenate([mask_pos[mask_pos < index - 1], mask_pos[mask_pos >= index - 1]])
            start_rows[mask_pos] = index
            start_rows_list.append(start_rows)
    startend_row_indices = torch.tensor(start_rows_list, device=device, dtype=torch.int32).reshape((B, H, S))

    def random_eviction_mask(b, h, q_idx, kv_idx):
        return q_idx < startend_row_indices[b, h, kv_idx]
    causal_random_eviction_mask = and_masks(random_eviction_mask, causal_mask)

    @cute.jit
    def cute_random_eviction_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        startend_row_indices = aux_tensors[0]
        offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        offset_ssa = utils.scalar_to_ssa(offset, cutlass.Int32)
        return (m_idx < startend_row_indices[batch[0], head[0], n_idx[0]]) & (n_idx <= (m_idx + offset_ssa))

    return {
        "cute_mask_mod": cute_random_eviction_mask,
        "flex_mask_mod": causal_random_eviction_mask,
        "aux_tensors": [startend_row_indices],
        "causal": True,
    }

def generate_hybrid_swa_prefix_lm_document_mask(batch_size, seqlen, hkv, d, doc_seq_lens, window_size=512, ratio=3, device="cuda"):
    assert hkv % (ratio + 1) == 0
    hswa = hkv // (ratio + 1) * ratio

    # Note(umiswing): i have no idea how to reuse this code snippet
    assert len(doc_seq_lens) >= 2
    total_seq_len = 0
    for prefix_length, seq_length in doc_seq_lens:
        total_seq_len += seq_length
    assert total_seq_len <= seqlen
    padding = seqlen - total_seq_len

    prefix_lts = []
    cur_len_so_far = doc_seq_lens[0][1]
    for i in range(len(doc_seq_lens)):
        prefix_lts.extend([cur_len_so_far] * doc_seq_lens[i][1])
        if i < len(doc_seq_lens) - 1:
            cur_len_so_far += doc_seq_lens[i+1][1]
    if padding > 0:
        prefix_lts.extend([cur_len_so_far] * padding)

    ute = []
    cur_len_so_far = 0
    for prefix_length, seq_length in doc_seq_lens:
        ute.extend([cur_len_so_far] * prefix_length + list(range(cur_len_so_far+prefix_length, cur_len_so_far+seq_length)))
        cur_len_so_far += seq_length
    if padding > 0:
        ute.extend([total_seq_len] * padding)

    prefix_lts = torch.tensor(prefix_lts,  device=device, dtype=torch.int32).reshape((1, 1, -1)).repeat_interleave(batch_size, 0).repeat_interleave(hkv, 1)
    ute= torch.tensor(ute,  device=device, dtype=torch.int32).reshape((1, 1, -1)).repeat_interleave(batch_size, 0).repeat_interleave(hkv, 1)

    swa_lts = torch.arange(
        start=window_size, end=seqlen + window_size, dtype=torch.int32,
    ).reshape([1, 1, seqlen])
    swa_lts = torch.clip(
        swa_lts, max=seqlen,
    ).repeat_interleave(batch_size, 0).repeat_interleave(hswa, 1)

    lts = torch.concat((torch.minimum(swa_lts, prefix_lts[:, :hswa, :]), prefix_lts[:, hswa:, :]), dim=1)

    def hybrid_swa_prefix_lm_document_mask(b, h, q_idx, kv_idx):
        return (q_idx < lts[b, h, kv_idx]) & (q_idx >= ute[b, h, kv_idx])

    @cute.jit
    def cute_hybrid_swa_prefix_lm_document_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        down_left_row_indices = aux_tensors[0]
        up_right_row_indices = aux_tensors[1]
        return (m_idx < down_left_row_indices[batch[0], head[0], n_idx[0]]) & (m_idx >= up_right_row_indices[batch[0], head[0], n_idx[0]])

    return {
        "cute_mask_mod": cute_hybrid_swa_prefix_lm_document_mask,
        "flex_mask_mod": hybrid_swa_prefix_lm_document_mask,
        "aux_tensors": [lts, ute],
        "causal": False,
    }

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

def main(examples: List[str] = ["all"], dtype='bf16'):
    """Run the benchmark with the given examples.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
    """
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
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
        # Note(umiswing): fa4 does not support d 256
        for D in [128]:
            if D == 192:
                DV = 128
                H = 16
            else:
                DV = D
                H = 4096 // D
            HKV = H
    
            for idx, (S, prefix_doc_seq_lens, qksparse_mask) in enumerate(doc_seq_lens_list):
                B = 128 * 1024 // S

                doc_seq_lens = [x[1] for x in prefix_doc_seq_lens]
                maskout_pair = []
                offset = 0
                print(f"{B}_{S}_{H}_{D}_{DV}_{idx}_{dtype}")
                if sum(qksparse_mask) == 0:
                    maskout_pair = [(1024, 538), (2358, 1700)]
                else:
                    for is_maskout, doc_seq in zip(qksparse_mask, doc_seq_lens):
                        if is_maskout:
                            maskout_pair.append((offset, doc_seq))
                        offset += doc_seq

                share_qa_docs = [split_sequence(doc_seq) for doc_seq in doc_seq_lens]

                available_examples = {
                    "Full": lambda: test_mask(mask_info={"cute_mask_mod": None, "flex_mask_mod": None, "aux_tensors": None, "causal": False}, B=B, S=S, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype),
                    "Causal": lambda: test_mask(mask_info={"cute_mask_mod": None, "flex_mask_mod": None, "aux_tensors": None, "causal": True}, B=B, S=S, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype),
                    "Sliding Window": lambda: test_mask(mask_info=generate_sliding_window(window_size=int(S*0.0625)), B=B, S=S, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype),
                    "Causal Document Mask": lambda: test_mask(mask_info=generate_causal_document_mask(doc_seq_lens=doc_seq_lens, B=B, S=S), B=B, S=S, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype),
                    # Note(umiswing): FA4 mask_mod will hang in Document Mask, and idk why
                    # "Document Mask": lambda: test_mask(mask_info=generate_document_mask(doc_seq_lens=doc_seq_lens, B=B, S=S), B=B, S=S, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype),
                    "Share Question Mask": lambda: test_mask(mask_info=generate_share_question_mask(doc_seq_lens=share_qa_docs, B=B, S=S), B=B, S=S, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype),
                    "Global Sliding Window": lambda: test_mask(mask_info=generate_global_sliding_window_mask(global_token=16, B=B, S=S, window_size=(int(S*0.0625), int(S*0.0625))), B=B, S=S, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype),
                    "Causal Blockwise Mask": lambda: test_mask(mask_info=generate_causal_blockwise_mask(doc_seq_lens=doc_seq_lens, B=B, S=S), B=B, S=S, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype),
                    "Prefix LM Document Mask": lambda: test_mask(mask_info=generate_prefix_lm_document_mask(doc_seq_lens=prefix_doc_seq_lens, B=B, S=S), B=B, S=S, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype),
                    "Prefix LM Causal Mask": lambda: test_mask(mask_info=generate_prefix_lm_causal_mask(prefix_length=int(S*0.5), B=B, S=S), B=B, S=S, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype),
                    "QK-sparse Mask": lambda: test_mask(mask_info=generate_qk_sparse_mask(maskout_pair=maskout_pair, B=B, S=S), B=B, S=S, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype),
                    "Random Eviction Mask": lambda: test_mask(mask_info=generate_random_eviction_mask(start_row=S//2, B=B, S=S, H=H, HKV=HKV), B=B, S=S, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype),
                    # "Hybrid SWA Prefix LM Doc": lambda: test_mask(mask_info=generate_hybrid_swa_prefix_lm_document_mask(batch_size=B, seqlen=S, hkv=H, d=D, doc_seq_lens=prefix_doc_seq_lens), B=B, S=S, H=H, HKV=HKV, D=D, DV=DV, dtype=dtype),
                }

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
                os.makedirs(f"{dtype}", exist_ok=True)
                text_file = open(f"{dtype}/fa4_mask_mod_{current_time}_{B}_{S}_{H}_{HKV}_{D}_{DV}_{idx}.csv","w")
                text_file.write(content2)
                text_file.close()

if __name__ == "__main__":
    try:
        from jsonargparse import ArgumentParser
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
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

    args = parser.parse_args()
    main(**vars(args))

