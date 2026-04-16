"""
Cross-framework Flash Attention varlen test.

Reference : PyTorch F.scaled_dot_product_attention  (torch_flash_ref)
Under test: Paddle  flash_attn_varlen_func          (flash_mask)

Data is generated via NumPy so that both frameworks receive bit-identical
inputs and upstream gradients.
"""

# 为什么 ref 使用 PyTorch 而非 Paddle：
# 1. Paddle 的 scaled_dot_product_attention(is_causal=True) 走 memory_efficient_attention (CUTLASS) 路径，
#    不支持 head_dim=256 + GQA/MQA（Q/K 头数不同）的组合，会报 kernel_launched=false。
# 2. 手写的 matmul+softmax ref 无法精确复现 Flash Attention kernel 的混合精度反向计算
#    （online softmax、分块累加、P→bf16 截断等），在 head_dim=256 时约有 1% 的 case 精度不达标。
# 3. PyTorch 的 F.scaled_dot_product_attention 支持此类场景, ref 结果与 FA4 varlen 一致。
# 待 Paddle SDPA 支持上述场景后可切换回来。

import math
from typing import Optional

import numpy as np
import pytest

import torch
import torch.nn.functional as F_torch

import paddle
from flash_mask import flash_attn_varlen_func


# ---------------------------------------------------------------------------
# Test configuration  (edit here to control pytest parametrisation)
# ---------------------------------------------------------------------------

TEST_B            = [1, 7, 20]
TEST_H            = [1, 4, 6]
TEST_D            = [64, 128, 192, 256]
TEST_MIN_SEQ_LEN  = [1, 32, 128]
TEST_MAX_SEQ_LEN  = [8, 64, 2048]
TEST_CAUSAL       = [True, False]
TEST_SOFTMAX_SCALE = [None, 0.1]
TEST_DTYPE        = ["bfloat16", "float16"]
TEST_MHA_TYPE     = ["mha", "mqa", "gqa"]


# ---------------------------------------------------------------------------
# NumPy data generation (framework-agnostic, bit-reproducible)
# ---------------------------------------------------------------------------

def generate_varlen_data(
    batch_size: int = 8,
    n_heads: int = 16,
    d_head: int = 128,
    min_len: int = 32,
    max_len: int = 64,
    mha_type: str = "mha",
    dtype_str: str = "bfloat16",
    seed: int = 0,
):
    """Return NumPy arrays for Q, K, V, cu_seqlens, and a random grad_out.

    All floating-point arrays are stored as float32 (bf16 values are first
    generated in float32, then rounded to bf16 via a torch round-trip so the
    bit patterns are exact bf16 representable values).
    """
    rng = np.random.RandomState(seed)

    assert mha_type in ("mha", "mqa", "gqa")

    # --- sequence lengths (identical for Q and K) ---
    lens = rng.randint(min_len, max_len + 1, size=(batch_size,)).astype(np.int64)
    cu_seqlens = np.zeros(batch_size + 1, dtype=np.int32)
    cu_seqlens[1:] = np.cumsum(lens)
    total = int(cu_seqlens[-1])

    # --- head counts ---
    if mha_type == "gqa":
        H, H_kv = 3 * n_heads, n_heads
    elif mha_type == "mha":
        H = H_kv = n_heads
    else:  # mqa
        H, H_kv = n_heads, 1

    d_head_v = 128 if d_head == 192 else d_head

    # --- generate random data in float32, then quantise to bf16 ---
    def _randn_bf16(*shape):
        x = rng.standard_normal(shape).astype(np.float32)
        # round-trip through torch bf16 to get exact bf16 bit patterns
        t = torch.from_numpy(x).to(torch.bfloat16 if dtype_str == "bfloat16" else torch.float16)
        return t.float().numpy()

    q_np = _randn_bf16(total, H, d_head)
    k_np = _randn_bf16(total, H_kv, d_head)
    v_np = _randn_bf16(total, H_kv, d_head_v)

    # grad_out shares the shape of the output: (total_q, H, d_head_v)
    grad_np = _randn_bf16(total, H, d_head_v)

    return q_np, k_np, v_np, cu_seqlens, total, grad_np


# ---------------------------------------------------------------------------
# Framework conversion helpers
# ---------------------------------------------------------------------------

_TORCH_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16}
_PADDLE_DTYPE = {"bfloat16": paddle.bfloat16, "float16": paddle.float16}


def _to_torch(arr: np.ndarray, dtype_str: str, requires_grad: bool = True):
    t = torch.from_numpy(arr.copy()).cuda()
    t = t.to(_TORCH_DTYPE[dtype_str])
    t.requires_grad_(requires_grad)
    return t


def _to_paddle(arr: np.ndarray, dtype_str: str, stop_gradient: bool = False):
    t = paddle.to_tensor(arr.copy(), place=paddle.CUDAPlace(0))
    t = t.cast(_PADDLE_DTYPE[dtype_str])
    t.stop_gradient = stop_gradient
    return t


# ---------------------------------------------------------------------------
# Torch reference  (F.scaled_dot_product_attention, per-batch loop)
# ---------------------------------------------------------------------------

def torch_flash_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    total_q: int,
    total_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
):
    H = q.shape[1]
    H_kv = k.shape[1]
    B = cu_seqlens_q.shape[0] - 1

    hcseq_q = cu_seqlens_q.cpu()
    hcseq_k = cu_seqlens_k.cpu()

    outs = []
    for b in range(B):
        qs, qe = int(hcseq_q[b]), int(hcseq_q[b + 1])
        ks, ke = int(hcseq_k[b]), int(hcseq_k[b + 1])

        qb = q[qs:qe].permute(1, 0, 2).unsqueeze(0)   # (1, H,  Sq, d)
        kb = k[ks:ke].permute(1, 0, 2).unsqueeze(0)    # (1, Hkv, Sk, d)
        vb = v[ks:ke].permute(1, 0, 2).unsqueeze(0)    # (1, Hkv, Sk, dv)

        ob = F_torch.scaled_dot_product_attention(
            qb, kb, vb,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
            enable_gqa=(H_kv != H),
        )
        ob = ob.squeeze(0).permute(1, 0, 2).contiguous()
        outs.append(ob)

    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def check_cross_framework(
    q_np, k_np, v_np, cu_seqlens_np, total, grad_np,
    dtype_str: str = "bfloat16",
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    atol: float = 3e-2,
    rtol: float = 3e-2,
):
    # ---- Torch side ----
    q_t = _to_torch(q_np, dtype_str)
    k_t = _to_torch(k_np, dtype_str)
    v_t = _to_torch(v_np, dtype_str)
    cu_t = torch.from_numpy(cu_seqlens_np.copy()).cuda().to(torch.int32)
    grad_t = _to_torch(grad_np, dtype_str, requires_grad=False)

    out_ref = torch_flash_ref(
        q_t, k_t, v_t,
        cu_seqlens_q=cu_t, cu_seqlens_k=cu_t,
        total_q=total, total_k=total,
        softmax_scale=softmax_scale,
        causal=causal,
    )

    out_ref.backward(grad_t)
    dq_ref = q_t.grad.float().cpu().numpy()
    dk_ref = k_t.grad.float().cpu().numpy()
    dv_ref = v_t.grad.float().cpu().numpy()
    out_ref_np = out_ref.float().detach().cpu().numpy()

    # ---- Paddle side ----
    q_p = _to_paddle(q_np, dtype_str, stop_gradient=False)
    k_p = _to_paddle(k_np, dtype_str, stop_gradient=False)
    v_p = _to_paddle(v_np, dtype_str, stop_gradient=False)
    cu_p = paddle.to_tensor(cu_seqlens_np.copy(), place=paddle.CUDAPlace(0)).cast(paddle.int32)
    grad_p = _to_paddle(grad_np, dtype_str, stop_gradient=True)

    scale = (1.0 / q_np.shape[-1] ** 0.5) if softmax_scale is None else softmax_scale

    out_fa, _ = flash_attn_varlen_func(
        q_p, k_p, v_p,
        cu_seqlens_q=cu_p,
        cu_seqlens_k=cu_p,
        softmax_scale=scale,
        causal=causal,
        window_size=(None, None),
        softcap=0.0,
        return_lse=False,
    )

    out_fa.backward(grad_p)
    dq_fa = q_p.grad.cast(paddle.float32).numpy()
    dk_fa = k_p.grad.cast(paddle.float32).numpy()
    dv_fa = v_p.grad.cast(paddle.float32).numpy()
    out_fa_np = out_fa.cast(paddle.float32).numpy()

    # ---- Compare ----
    def _check(name, a, b):
        diff = np.abs(a - b)
        max_diff = diff.max()
        ok = np.allclose(a, b, atol=atol, rtol=rtol)
        if not ok:
            idx = np.unravel_index(diff.argmax(), diff.shape)
            print(f"  {name} FAIL: max_diff={max_diff:.4e}  "
                  f"idx={idx}  fa={a[idx]:.6f}  ref={b[idx]:.6f}")
        else:
            print(f"  {name} OK:   max_diff={max_diff:.4e}")
        return ok

    print("---- Forward ----")
    ok_fwd = _check("Out", out_fa_np, out_ref_np)

    print("---- Backward ----")
    ok_dq = _check("dQ", dq_fa, dq_ref)
    ok_dk = _check("dK", dk_fa, dk_ref)
    ok_dv = _check("dV", dv_fa, dv_ref)

    return ok_fwd and ok_dq and ok_dk and ok_dv


# ---------------------------------------------------------------------------
# Pytest parametrised test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B", TEST_B)
@pytest.mark.parametrize("H", TEST_H)
@pytest.mark.parametrize("D", TEST_D)
@pytest.mark.parametrize("min_seq_len", TEST_MIN_SEQ_LEN)
@pytest.mark.parametrize("max_seq_len", TEST_MAX_SEQ_LEN)
@pytest.mark.parametrize("causal", TEST_CAUSAL)
@pytest.mark.parametrize("softmax_scale", TEST_SOFTMAX_SCALE)
@pytest.mark.parametrize("dtype_str", TEST_DTYPE)
@pytest.mark.parametrize("mha_type", TEST_MHA_TYPE)
def test_varlen_cross(
    B, H, D, min_seq_len, max_seq_len,
    causal, softmax_scale, dtype_str, mha_type,
):
    if min_seq_len > max_seq_len:
        pytest.skip("min_seq_len > max_seq_len")

    q_np, k_np, v_np, cu_seqlens_np, total, grad_np = generate_varlen_data(
        batch_size=B, n_heads=H, d_head=D,
        min_len=min_seq_len, max_len=max_seq_len,
        mha_type=mha_type, dtype_str=dtype_str,
    )

    ok = check_cross_framework(
        q_np, k_np, v_np, cu_seqlens_np, total, grad_np,
        dtype_str=dtype_str,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    assert ok


# ---------------------------------------------------------------------------
# Quick standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Smoke test: B=7, H=6, D=256, mqa, causal, bf16 ===\n")
    q_np, k_np, v_np, cu, total, grad_np = generate_varlen_data(
        batch_size=7, n_heads=6, d_head=256,
        min_len=1, max_len=2048,
        mha_type="mqa", dtype_str="bfloat16",
    )
    ok = check_cross_framework(
        q_np, k_np, v_np, cu, total, grad_np,
        dtype_str="bfloat16",
        softmax_scale=None,
        causal=True,
    )
    print(f"\nResult: {'PASS' if ok else 'FAIL'}")
