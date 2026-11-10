


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

import torch.nn.functional as F

from cudnn import create_mask_plan
# kernel-only entries (no autograd), like FA4's _flash_attn_fwd / _flash_attn_bwd
from cudnn.flex_attention.execution import _flex_attention_forward, _flex_attention_backward

# only kept so the generate_* mask builders below stay unchanged (their cute_mask_mod is unused here)
import cutlass
import cutlass.cute as cute
from flash_attn.cute import utils

from datetime import datetime

torch.set_default_device("cuda")
torch.manual_seed(0)

import bench_common

torch._dynamo.config.cache_size_limit = 1000
# dynamic-shape recompiles of create_block_mask return wrong mask_block_cnt; keep them static
torch._dynamo.config.automatic_dynamic_shapes = False

# row-wise endpoints can explode (e.g. Random Eviction needs thousands of intervals per row); skip such masks
MASK_FUNC_MAX_BYTES = 4 * 1024**3

def visible_to_endpoints(vis):
    """vis [Hm, R, Sk] bool -> endpoints [Hm, R, nfunc] int32, visible = [0,F0) U [F1,F2) U ..."""
    seqlen_k = vis.shape[-1]
    padded = F.pad(vis.to(torch.int8), (1, 1))
    change = padded[..., 1:] != padded[..., :-1]  # [Hm, R, Sk+1], alternating start / end
    m = int(change.sum(-1).max().item())
    pos = torch.arange(seqlen_k + 1, device=vis.device, dtype=torch.int32).expand_as(change)
    slot = torch.where(change, change.to(torch.int32).cumsum(-1) - 1, m)  # slot m is a dump slot
    ends = torch.full((*change.shape[:-1], m + 1), seqlen_k, device=vis.device, dtype=torch.int32)
    ends.scatter_(-1, slot.long(), pos)
    # F0 = 0 (empty first interval), then start/end pairs
    return torch.cat([torch.zeros_like(ends[..., :1]), ends[..., :m]], dim=-1)

def flex_mask_mod_to_mask_func(flex_mask_mod, causal, per_head, batch_size, nheads, seqlen, row_chunk=1024):
    """Build cuDNN Flex mask_func [Hmask, nfunc, B*S] from a flex mask_mod, chunked over rows (no dense B*H*S*S)."""
    if flex_mask_mod is None:
        q = torch.arange(seqlen, dtype=torch.int32)
        f0 = q + 1 if causal else torch.full_like(q, seqlen)
        return f0.repeat(batch_size).view(1, 1, -1).contiguous()

    hmask = nheads if per_head else 1
    h_idx = torch.arange(hmask).view(hmask, 1, 1)
    kv_idx = torch.arange(seqlen).view(1, 1, seqlen)
    chunks = []
    nfunc = 1
    for b in range(batch_size):
        b_idx = torch.tensor(b)
        for r0 in range(0, seqlen, row_chunk):
            q_idx = torch.arange(r0, min(r0 + row_chunk, seqlen)).view(1, -1, 1)
            vis = torch.broadcast_to(flex_mask_mod(b_idx, h_idx, q_idx, kv_idx), (hmask, q_idx.shape[1], seqlen))
            ep = visible_to_endpoints(vis)
            nfunc = max(nfunc, ep.shape[-1])
            nbytes = hmask * nfunc * batch_size * seqlen * 4
            if nbytes > MASK_FUNC_MAX_BYTES:
                print(f"Warning: row-wise mask_func needs nfunc>={nfunc} ({nbytes / 1024**3:.1f} GB), skipping.")
                return None
            chunks.append(ep)
    # pad every chunk to the global nfunc with [Sk, Sk) empty intervals
    chunks = [F.pad(ep, (0, nfunc - ep.shape[-1]), value=seqlen) for ep in chunks]
    mask_func = torch.cat(chunks, dim=1).permute(0, 2, 1).contiguous()  # [Hmask, nfunc, B*S]
    print(f"mask_func {tuple(mask_func.shape)} nfunc={nfunc} {mask_func.numel() * 4 / 1024**2:.1f} MB")
    return mask_func



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


def do_bench(fn, S):
    return bench_common.do_bench(fn, S, torch.cuda.synchronize, torch.cuda.Event)

def print_header(text):
    width = 91
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {text.center(width - 4)} ║")
    print("╚" + "═" * (width - 2) + "╝")

def test_mask(
    mask_info,
    spec,
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
    inputs=None,
    outputs_only: bool = False,
):
    """Time the fwd / bwd of one mask, or with outputs_only run them once on inputs
    (numpy q, k, v, dout from bench_common.acc_inputs) and return the outputs."""
    flex_mask_mod = mask_info["flex_mask_mod"]
    aux_tensors = mask_info["aux_tensors"]
    causal = mask_info["causal"]
    # [B, H, S] aux tensors (Random Eviction, Hybrid SWA) make the mask head-dependent -> Hmask = H
    per_head = aux_tensors is not None and any(t.dim() == 3 for t in aux_tensors)

    if dtype == 'bf16':
        data_type = torch.bfloat16
    else:
        data_type = torch.float16

    mask_func = flex_mask_mod_to_mask_func(flex_mask_mod, causal, per_head, B, H, S)
    if mask_func is None:
        return None

    # Element-level sparsity of the shared flashmask layout (bench_common), the same for
    # every backend; --check_masks verifies this backend's mask is that layout.
    sparsity = bench_common.element_sparsity(causal, spec.startend, S, S)
    density = 1.0 - sparsity

    q = torch.randn(B, S, H, D, device=device, dtype=data_type, requires_grad=True)
    k = torch.randn(B, S, HKV, D, device=device, dtype=data_type, requires_grad=True)
    v = torch.randn(B, S, HKV, DV, device=device, dtype=data_type, requires_grad=True)

    gradOut = torch.randn(B, S, H, DV, device=device, dtype=data_type)
    if inputs is not None:
        q, k, v, gradOut = [torch.from_numpy(x).to(device, data_type) for x in inputs]
        for t in (q, k, v):
            t.requires_grad_()

    # Built once outside the timed fwd / bwd calls, like the FA4 block-sparse tensors (kernel-only timing).
    plan = create_mask_plan(mask_func, q, k, v, build_backward=True)

    cudnn_flex_call = lambda: _flex_attention_forward(q, k, v, mask_plan=plan, return_lse=True)

    if outputs_only:
        # The timed fwd / bwd calls, once, with every float buffer they allocate NaN.
        with bench_common.torch_poison_empty():
            out_cudnn, lse_cudnn, _ = cudnn_flex_call()
            # The result is a cudnn TupleDict: integer indexing works, slicing does not.
            bwd = _flex_attention_backward(q, k, v, out_cudnn, gradOut, lse_cudnn, mask_plan=plan)
            dq, dk, dv = bwd[0], bwd[1], bwd[2]
        return bench_common.torch_outputs(out_cudnn, dq, dk, dv)

    results = []

    # Forward pass
    fwd_time_ms = do_bench(cudnn_flex_call, S)
    # Backward pass
    out_cudnn, lse_cudnn, _ = cudnn_flex_call()

    cudnn_flex_bwd_call = lambda: _flex_attention_backward(q, k, v, out_cudnn, gradOut, lse_cudnn, mask_plan=plan)

    bwd_time_ms = do_bench(cudnn_flex_bwd_call, S)

    total_time_ms = fwd_time_ms + bwd_time_ms

    fwd_flops = density * cal_flops(B, H, S, S, D, DV, mode='fwd')
    bwd_flops = density * cal_flops(B, H, S, S, D, DV,  mode='bwd')
    total_flops = density * cal_flops(B, H, S, S, D, DV,  mode='fwd_bwd')

    fwd_tflops = cal_tflops(fwd_flops, fwd_time_ms)
    bwd_tflops = cal_tflops(bwd_flops, bwd_time_ms)
    total_tflops = cal_tflops(total_flops, total_time_ms)

    return fwd_time_ms, bwd_time_ms, total_time_ms, fwd_flops, bwd_flops, total_flops, fwd_tflops, bwd_tflops, total_tflops, sparsity

def generate_sliding_window(window_size: int, B=16, S=8192, device="cuda"):

    def sliding_window(b, h, q_idx, kv_idx):
        return q_idx - kv_idx < window_size

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
        return (m_idx - n_idx < window_size_cute) & (n_idx <= (m_idx + offset_ssa))

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
    doc_seq_lens = list(doc_seq_lens)
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
        doc_seq_lens = list(doc_seq_lens) + [[padding]]

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
        "causal": False, # the prefix is bidirectional, same as benchmark_flashmask
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


def generate_random_eviction_mask(B=16, H=16, HKV=16, S=8192, start_row=4096, rows=None, device="cuda"):
    assert H == HKV
    # rows: bench_common.eviction_rows, [B, H, S], the draw every backend shares.
    startend_row_indices = torch.as_tensor(rows, device=device, dtype=torch.int32).reshape((B, H, S))

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



def _mask_func_fingerprint(mask_info, spec, B, H, S):
    """Fingerprint of the mask_func cuDNN actually receives (None: too large, not run)."""
    aux = mask_info["aux_tensors"]
    per_head = aux is not None and any(t.dim() == 3 for t in aux)
    mask_func = flex_mask_mod_to_mask_func(mask_info["flex_mask_mod"], spec.causal, per_head, B, H, S)
    if mask_func is None:
        return None
    return bench_common.fingerprint(*bench_common.torch_mask_func_counts(mask_func.cuda(), B, S))


def make_mask_info(spec, case, device="cuda"):
    """This backend's mask_mod for one bench_common mask, built from the same parameters."""
    B, S, H, HKV = case.B, case.sample.S, case.H, case.HKV
    p = spec.params
    name = spec.name
    if name in ("Full", "Causal"):
        return {"cute_mask_mod": None, "flex_mask_mod": None, "aux_tensors": None, "causal": spec.causal}
    if name == "Sliding Window":
        return generate_sliding_window(window_size=p["window_size"], B=B, S=S, device=device)
    if name == "Global Sliding Window":
        return generate_global_sliding_window_mask(B=B, S=S, global_token=p["global_token"], window_size=p["window_size"], device=device)
    if name == "Causal Document Mask":
        return generate_causal_document_mask(B=B, S=S, doc_seq_lens=p["doc_seq_lens"], device=device)
    if name == "Document Mask":
        return generate_document_mask(B=B, S=S, doc_seq_lens=p["doc_seq_lens"], device=device)
    if name == "Share Question Mask":
        return generate_share_question_mask(B=B, S=S, doc_seq_lens=p["doc_seq_lens"], device=device)
    if name == "Causal Blockwise Mask":
        return generate_causal_blockwise_mask(B=B, S=S, doc_seq_lens=p["doc_seq_lens"], device=device)
    if name == "Prefix LM Document Mask":
        return generate_prefix_lm_document_mask(B=B, S=S, doc_seq_lens=p["doc_seq_lens"], device=device)
    if name == "Prefix LM Causal Mask":
        return generate_prefix_lm_causal_mask(B=B, S=S, prefix_length=p["prefix_length"], device=device)
    if name == "QK-sparse Mask":
        return generate_qk_sparse_mask(B=B, S=S, maskout_pair=p["maskout_pair"], device=device)
    if name == "Random Eviction Mask":
        return generate_random_eviction_mask(B=B, H=H, HKV=HKV, S=S, start_row=p["start_row"], rows=p["rows"], device=device)
    raise ValueError(f"unknown mask {name!r}")


def main(examples: List[str] = ["all"], dtype='bf16', head_dim=None, current_time=None,
         dedup_static_masks=False, first_sample_only=False, check_masks=None, seqlens=None,
         series=None, results_dir=bench_common.RESULTS_DIR, overwrite=True, interfere_grace=60.0,
         interfere_max_wait=1800.0, allow_handoff=False):
    """Run the benchmark with the given examples.

    Cases, masks, sparsity and iteration counts come from bench_common, shared with
    benchmark_flashmask.py and the other competitor benchmark. --check_masks writes
    one fingerprint per (case, mask) of what this backend hands its kernel, instead
    of timing (see check_bench_consistency.py).
    """
    if current_time is None:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    if check_masks:
        open(check_masks, "w").close()
    series = series or "cudnn" + ("" if dtype == "bf16" else f"_{dtype}")

    head_dims = [128, 192, 256] if head_dim is None else [head_dim]
    for case in bench_common.iter_cases(head_dims, examples, dedup_static_masks, first_sample_only, seqlens):
        B, H, HKV, D, DV = case.B, case.H, case.HKV, case.D, case.DV
        S, idx = case.sample.S, case.sample.idx
        print(f"{B}_{S}_{H}_{D}_{DV}_{idx}_{dtype}")
        out = bench_common.csv_path(results_dir, current_time, series, case)
        if not check_masks:
            bench_common.write_meta(results_dir, current_time, series, "cudnn", packages=["cudnn", "torch", "cutlass"],
                                    dtype=dtype, gpu=torch.cuda.get_device_name())

            def run_mask(ex):
                print(ex)
                spec = bench_common.build_mask(ex, case)
                ret = test_mask(mask_info=make_mask_info(spec, case), spec=spec, B=B, S=S, H=H, HKV=HKV,
                                D=D, DV=DV, dtype=dtype)
                if ret is None:
                    return None
                fw_time, bw_time, total_time, fw_flops, bw_flops, total_flops, fw_tflops, bw_tflops, total_tflops, sparsity = ret
                return [ex, f"{fw_time:.4f}", f"{bw_time:.4f}", f"{total_time:.4f}", f"{fw_flops:.4f}", f"{bw_flops:.4f}", f"{total_flops:.4f}", f"{fw_tflops:.4f}", f"{bw_tflops:.4f}", f"{total_tflops:4f}", f"{sparsity:.4f}"]

            # Same --current_time + --no-overwrite: a rerun (or a handed-off task) picks
            # up where the last one stopped. Disturbed measurements are never written.
            results = bench_common.run_masks_to_csv(out, case.masks, run_mask, overwrite,
                                                    own_mem=torch.cuda.memory_reserved,
                                                    grace=interfere_grace, max_wait=interfere_max_wait,
                                                    allow_handoff=allow_handoff)
            if results is None:
                continue
            print(tabulate(results, headers=bench_common.CSV_HEADERS, tablefmt="grid"))
            print(f"wrote {out}")
            continue

        for ex in case.masks:
            spec = bench_common.build_mask(ex, case)
            mask_info = make_mask_info(spec, case)
            if check_masks:
                fp = _mask_func_fingerprint(mask_info, spec, B, H, S)
                acc = None
                if fp is not None:
                    acc = bench_common.run_accuracy(case, ex, lambda c, s, inputs: test_mask(
                        mask_info=make_mask_info(s, c), spec=s, B=c.B, S=S, H=c.H, HKV=c.HKV, D=D, DV=DV,
                        dtype=dtype, inputs=inputs, outputs_only=True), check_masks)
                    print(f"[acc] S={S} D={D} {ex}: {'ok' if acc is None or acc['ok'] else 'FAIL'}")
                bench_common.append_jsonl(check_masks, bench_common.check_record("cudnn", case, spec, fp, acc=acc))
                continue



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
        help="Masks to run, or 'all'. Names as in bench_common.MASKS (the CSV "
        "Operation column), e.g. 'Causal Document Mask'.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16"
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=None,
        help="Run only this head_dim (128 / 192 / 256), like benchmark_flashmask.py --head_dim. "
        "Default: all three in one process (the random masks of 192 / 256 then differ from flashmask's).",
    )
    parser.add_argument(
        "--current_time",
        type=str,
        default=None,
        help="Timestamp used in the CSV file names (default: now), like benchmark_flashmask.py --current_time.",
    )

    parser.add_argument(
        "--dedup_static_masks",
        action="store_true",
        default=False,
        help="Measure the masks that only depend on S (bench_common.S_ONLY_MASKS) on "
        "Sample 1 of each 'Total length' block only, like benchmark_flashmask.py.",
    )
    parser.add_argument(
        "--series",
        type=str,
        default=None,
        help="Result series name (default cudnn). CSVs go to <results_dir>/<current_time>/<series>/.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=bench_common.RESULTS_DIR,
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Skip the samples whose CSV for this --current_time already exists "
        "(resume a killed run by passing its --current_time again).",
    )
    parser.add_argument(
        "--interfere_grace",
        type=float,
        default=60.0,
        help="Seconds another process may hold the GPU before giving up on it: with "
        "--allow_handoff exit 75 (run_perf.sh moves the task to another aq GPU), else "
        "keep waiting up to --interfere_max_wait. Disturbed measurements are discarded.",
    )
    parser.add_argument(
        "--interfere_max_wait",
        type=float,
        default=1800.0,
        help="Without --allow_handoff, fail after waiting this long for the GPU.",
    )
    parser.add_argument(
        "--allow_handoff",
        action="store_true",
        help="Exit with code 75 when another process keeps the GPU past --interfere_grace.",
    )
    parser.add_argument(
        "--seqlens",
        type=int,
        nargs="+",
        default=None,
        help="Only these 'Total length' blocks of kernel_test_seq_info.txt, e.g. --seqlens 4096.",
    )
    parser.add_argument(
        "--first_sample_only",
        action="store_true",
        default=False,
        help="Run only Sample 1 of each 'Total length' block of kernel_test_seq_info.txt.",
    )
    parser.add_argument(
        "--check_masks",
        type=str,
        default=None,
        help="Write the mask fingerprint of every (case, mask) to this jsonl file instead "
        "of timing; compare backends with check_bench_consistency.py.",
    )

    args = parser.parse_args()
    bench_common.run_main(main, **vars(args))

