import os
import math
import itertools
import pytest
from einops import rearrange, repeat
# import paddle

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
try:
    from flashmask_interface import flashmask_attention
except Exception as e:
    import traceback
    traceback.print_exc()
    raise

import torch
from generate_startend_row_indices import (
  startend_row_indices_to_attn_bias,
  generate_none_mask,
  generate_sliding_window_mask,
  generate_causal_document_mask,
  generate_document_mask,
  generate_share_question_mask,
  generate_global_sliding_window_mask,
  generate_causal_blockwise_mask,
  generate_prefix_lm_document_mask,
  generate_prefix_lm_causal_mask,
  generate_qk_sparse_mask,
  generate_random_eviction_mask
)
from functools import partial
from test_util import attention_ref

# batch_size, seqlen_q, seqlen_k, nheads, nheads_kv
shape_cases = (
    [
        (2840, 32, 32, 16, 4),
        (1, 300, 300, 16, 16),
        # (2, 8192, 32768, 32, 4), # this will oom
        # (2, 8192, 8192, 32, 4), # this will oom
        (2, 8192, 8192, 14, 1),
        (2, 16384, 16384, 4, 1),
        (1, 128, 127, 1, 1),
        (1, 127, 128, 1, 1),
        (2, 16383, 16384, 4, 1),
        (2, 16384, 16383, 4, 1),
        # my case
    ]
    # tridao case
    + list(itertools.product(
        [9],                # batch_size
        [1, 64,  128, 256, 239, 799, 113, 113, 128, 113, 108, 256, 384, 640, 512, 1024, 1023, 1024,],       # seqlen_q
        [128, 192, 256,   203, 128, 217, 211, 256, 512, 256, 128, 256, 1024, 1024, 1023,],      # seqlen_k
        [6],                # nheads
        [6, 2, 1],          # nheads_kv
    ))
    + list(itertools.product(
        [2],                # batch_size
        [4096, 4224],       # seqlen_q
        [4096, 4224],      # seqlen_k
        [6],                # nheads
        [6, 2, 1],          # nheads_kv
    ))
)

# Generate all combinations for second param
def generate_shapes():
    for batch_size, seqlen_q, seqlen_k, nheads, nheads_kv in shape_cases:
        nheads_startend_row_indices_values = [1, nheads_kv]
        for nheads_startend_row_indices in nheads_startend_row_indices_values:
            yield (
                batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices
            )

# @pytest.mark.parametrize("dtype", [paddle.bfloat16])
@pytest.mark.parametrize("dtype", [torch.bfloat16]) # 使用 torch.bfloat16
@pytest.mark.parametrize("fa_version", [3])
@pytest.mark.parametrize("d, dv",
    [
        (64, 64),
        (80, 80),
        (128, 128),
        (192, 192),
        (256, 256),
    ])
@pytest.mark.parametrize(
    "batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices",
    list(generate_shapes())
)
@pytest.mark.parametrize(
    "gen_startend_row_indices",
    [
        # partial(generate_none_mask, causal=False), # full
        # partial(generate_none_mask, causal=True), # causal
        partial(generate_sliding_window_mask), # sliding window
        partial(generate_causal_document_mask), # causal document mask
        partial(generate_document_mask), # document mask
        partial(generate_share_question_mask), # share question mask
        partial(generate_global_sliding_window_mask), # global sliding window
        partial(generate_causal_blockwise_mask), # causal blockwise mask
        partial(generate_prefix_lm_document_mask), # prefix lm document mask
        partial(generate_prefix_lm_causal_mask), # prefix lm causal mask
        partial(generate_qk_sparse_mask), # qk-sparse mask
        partial(generate_random_eviction_mask), # random eviction mask
    ],
)
def test_flashmask(
    batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, d, dv, nheads_startend_row_indices, fa_version, dtype, gen_startend_row_indices, softcap=0.0
):
    torch.manual_seed(2024)
    # paddle.seed(2024)
    assert nheads % nheads_kv == 0

    q_ref = torch.randn(batch_size, seqlen_q, nheads, d, dtype=dtype, device='cuda', requires_grad=True)
    k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, dtype=dtype, device='cuda', requires_grad=True)
    v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, dv, dtype=dtype, device='cuda', requires_grad=True)


    q_bf16 = q_ref.detach().clone().requires_grad_(True)
    k_bf16 = k_ref.detach().clone().requires_grad_(True)
    v_bf16 = v_ref.detach().clone().requires_grad_(True)

    q = q_ref.detach().clone().requires_grad_(True)
    k = k_ref.detach().clone().requires_grad_(True)
    v = v_ref.detach().clone().requires_grad_(True)

    startend_row_indices, causal = gen_startend_row_indices(batch_size, seqlen_q, seqlen_k, nheads_startend_row_indices)

    if startend_row_indices is None:
        pytest.skip("Skipping because startend_row_indices is None")

    if startend_row_indices is not None:
        if not isinstance(startend_row_indices, torch.Tensor):
            # 如果是 numpy 或 paddle tensor (先转numpy)
            if hasattr(startend_row_indices, 'numpy'): 
                startend_row_indices = torch.tensor(startend_row_indices.numpy(), device='cuda', dtype=torch.int32)
            else:
                startend_row_indices = torch.tensor(startend_row_indices, device='cuda', dtype=torch.int32)
        else:
            startend_row_indices = startend_row_indices.to('cuda', dtype=torch.int32)

    if startend_row_indices is None and causal and d in (80, 192):
      pytest.skip(f"Skipping because running headdim {d} with flash_attn in causal mask")

    attn_bias = startend_row_indices_to_attn_bias(startend_row_indices, seqlen_q, nheads, dtype, causal)

    out_ref, attn_ref = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        causal=causal,
        attn_bias=attn_bias
    )

    out_bf16, attn_bf16 = attention_ref(
        q_bf16,
        k_bf16,
        v_bf16,
        causal=causal,
        attn_bias=attn_bias,
        upcast=False,
        reorder_ops=True
    )

    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    assert softcap == 0.0
    rtol = 2 if softcap == 0.0 else 3

    print(f"Torch naive bf16 Output max diff: {(out_bf16 - out_ref).abs().max().item()}")
    print(f"Torch naive bf16 Output mean diff: {(out_bf16 - out_ref).abs().mean().item()}")

    # 确保 startend_row_indices 在 CUDA 上且为 int32
    if isinstance(startend_row_indices, torch.Tensor):
        startend_row_indices = startend_row_indices.to('cuda', dtype=torch.int32)

    out, lse = flashmask_attention(
        q,
        k,
        v,
        startend_row_indices=startend_row_indices,
        causal=causal,
        return_softmax_lse=True
    )
    print(f"flashmask Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"flashmask Output mean diff: {(out - out_ref).abs().mean().item()}")

    assert (out - out_ref).abs().max().item() <= rtol * (out_bf16 - out_ref).abs().max().item() + fwd_atol

    # #Backward Check
    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g)
    out_bf16.backward(g)

    print(f"flashmask dQ max diff: {(q.grad - q_ref.grad).abs().max().item()}")
    print(f"flashmask dK max diff: {(k.grad - k_ref.grad).abs().max().item()}")
    print(f"flashmask dV max diff: {(v.grad - v_ref.grad).abs().max().item()}")
    print(f"flashmask dQ mean diff: {(q.grad - q_ref.grad).abs().mean().item()}")
    print(f"flashmask dK mean diff: {(k.grad - k_ref.grad).abs().mean().item()}")
    print(f"flashmask dV mean diff: {(v.grad - v_ref.grad).abs().mean().item()}")

    print(f"Torch naive bf16 dQ max diff: {(q_bf16.grad - q_ref.grad).abs().max().item()}")
    print(f"Torch naive bf16 dK max diff: {(k_bf16.grad - k_ref.grad).abs().max().item()}")
    print(f"Torch naive bf16 dV max diff: {(v_bf16.grad - v_ref.grad).abs().max().item()}")
    print(f"Torch naive bf16 dQ mean diff: {(q_bf16.grad - q_ref.grad).abs().mean().item()}")
    print(f"Torch naive bf16 dK mean diff: {(k_bf16.grad - k_ref.grad).abs().mean().item()}")
    print(f"Torch naive bf16 dV mean diff: {(v_bf16.grad - v_ref.grad).abs().mean().item()}")

    dq_atol = 2 * (q_ref.grad + 0.3 - 0.3 - q_ref.grad).abs().max().item() + (0 if softcap == 0 else 3e-4)
    assert (q.grad - q_ref.grad).abs().max().item() <= rtol * (q_bf16.grad - q_ref.grad).abs().max().item() + dq_atol
    dk_atol = 2 * (k_ref.grad + 0.3 - 0.3 - k_ref.grad).abs().max().item() + (0 if softcap == 0 else 3e-4)
    assert (k.grad - k_ref.grad).abs().max().item() <= rtol * (k_bf16.grad - k_ref.grad).abs().max().item() + dk_atol
    dv_atol = 2 * (v_ref.grad + 0.3 - 0.3 - v_ref.grad).abs().max().item() + (0 if softcap == 0 else 3e-4)
    assert (v.grad - v_ref.grad).abs().max().item() <= rtol * (v_bf16.grad - v_ref.grad).abs().max().item() + dv_atol
