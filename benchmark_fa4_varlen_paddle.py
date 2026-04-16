"""
Performance benchmark for flash_attn_varlen_func (Paddle varlen).

Causal Document Mask (FA mask_mod) and flash_attn_varlen are equivalent:
  - varlen: packs multiple independent sequences into one flat tensor,
    using cu_seqlens to delimit boundaries; causal=True is applied per-sequence.
  - Causal Document Mask: a single padded batch where a mask prevents
    cross-document attention and enforces causal ordering.

This script benchmarks paddle flash_attn_varlen_func (FLAGS_flash_attn_version=3)
across a range of configs that mirror the shape / doc-distribution used in
benchmark_flashmask.py, so numbers can be compared directly.

Usage:
    python benchmark_fa4_varlen.py
    python benchmark_fa4_varlen.py --dtype fp16
    python benchmark_fa4_varlen.py --seqlens 2048 4096 8192
    python benchmark_fa4_varlen.py --from_file kernel_test_seq_info.txt
"""

import os
import random
import argparse
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import paddle
from tabulate import tabulate

from flash_mask.flash_attn_v4.paddle.interface import _flash_attn_fwd, _flash_attn_bwd

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
paddle.seed(0)
np.random.seed(0)
random.seed(0)
paddle.set_device('gpu')

# ---------------------------------------------------------------------------
# Benchmarking helpers  (mirror benchmark_flashmask.py)
# ---------------------------------------------------------------------------

def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = paddle.quantile(times, paddle.to_tensor(quantiles, dtype=paddle.float32)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times.tolist()
    return getattr(paddle, return_mode)(times).item()


def do_bench(fn, warmup=10, rep=50, grad_to_none=None,
             quantiles=None, return_mode="mean"):
    assert return_mode in ["min", "max", "mean", "median", "all"]

    fn()
    paddle.device.synchronize()

    # L2 cache flush buffer (256 MB)
    cache = paddle.empty([int(256 * 1024 * 1024 // 4)], dtype=paddle.int32)

    start_event = [paddle.device.Event(enable_timing=True) for _ in range(rep)]
    end_event   = [paddle.device.Event(enable_timing=True) for _ in range(rep)]

    for _ in range(warmup):
        fn()

    for i in range(rep):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        cache.zero_()
        start_event[i].record()
        fn()
        end_event[i].record()

    paddle.device.synchronize()
    times = paddle.to_tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=paddle.float32,
    )
    return _summarize_statistics(times, quantiles, return_mode)


def cal_flops(B_eff, H, Sq_total, Sk_per_seq, D, mode="fwd"):
    """
    Approximate FLOPs for a varlen batch.

    B_eff      – effective number of sequences
    H          – number of query heads
    Sq_total   – total query tokens
    Sk_per_seq – list of per-sequence key lengths (same as query lens for self-attn)
    D          – head dimension
    """
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    # Each sequence i contributes 4 * seqlen_i^2 * H * D for causal (density=0.5)
    # or 4 * seqlen_i^2 * H * D for full attention.
    flops = sum(4 * s * s * H * D for s in Sk_per_seq)
    mult = {"fwd": 1.0, "bwd": 2.5, "fwd_bwd": 3.5}[mode]
    return mult * flops


def cal_tflops(flops, time_ms):
    return flops * (1e3 / time_ms) / 1e12


def print_header(text):
    width = 91
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {text.center(width - 4)} ║")
    print("╚" + "═" * (width - 2) + "╝")


# ---------------------------------------------------------------------------
# varlen tensor builder
# ---------------------------------------------------------------------------

def build_varlen_tensors(
    doc_seq_lens: List[int],
    H: int,
    HKV: int,
    D: int,
    DV: int,
    dtype,
):
    """
    Pack multiple sequences of lengths doc_seq_lens into flat varlen tensors.

    Returns q, k, v  (total_tokens, nheads, D/DV)
            cu_seqlens_q, cu_seqlens_k  (B+1,) int32
            max_seqlen
    """
    total = sum(doc_seq_lens)
    q = paddle.randn([total, H,   D],  dtype=dtype)
    k = paddle.randn([total, HKV, D],  dtype=dtype)
    v = paddle.randn([total, HKV, DV], dtype=dtype)

    lens_t = paddle.to_tensor(doc_seq_lens, dtype=paddle.int32)
    cu_seqlens = paddle.concat([
        paddle.zeros([1], dtype=paddle.int32),
        lens_t.cumsum(0).cast(paddle.int32),
    ])

    max_seqlen = max(doc_seq_lens)
    return q, k, v, cu_seqlens, max_seqlen


# ---------------------------------------------------------------------------
# Causal Document Mask → varlen equivalence conversion
# ---------------------------------------------------------------------------

def causal_doc_mask_to_varlen(
    doc_seq_lens: List[int],
    S: int,
) -> Tuple[List[int], int]:
    """
    A Causal Document Mask over a padded sequence of length S with
    sub-sequence lengths doc_seq_lens is *equivalent* to running
    flash_attn_varlen with causal=True over those same doc_seq_lens.

    This function just returns the canonical (doc_seq_lens, max_seqlen)
    tuple that varlen needs, stripping any padding from the last doc.
    """
    # doc_seq_lens already sums to S (including padding absorbed into last doc)
    assert sum(doc_seq_lens) == S, (
        f"sum(doc_seq_lens)={sum(doc_seq_lens)} != S={S}"
    )
    max_seqlen = max(doc_seq_lens)
    return doc_seq_lens, max_seqlen


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------

def benchmark_varlen(
    doc_seq_lens: List[int],
    S: int,
    H: int,
    HKV: int,
    D: int,
    DV: int,
    dtype,
    causal: bool = True,
):
    """
    Benchmark flash_attn_varlen_func fwd + bwd for the given doc distribution.
    Equivalent to the Causal Document Mask benchmark in benchmark_flashmask.py.
    """
    seq_lens, max_seqlen = causal_doc_mask_to_varlen(doc_seq_lens, S)

    q, k, v, cu_seqlens, max_seqlen = build_varlen_tensors(
        seq_lens, H, HKV, D, DV, dtype
    )

    softmax_scale = D ** -0.5

    # ---------- forward ----------
    # Call _flash_attn_fwd directly — no autograd graph, pure kernel timing.
    fwd_fn = lambda: _flash_attn_fwd(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        return_lse=True,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    fwd_time_ms = do_bench(fwd_fn)

    # ---------- backward ----------
    # Run fwd once to get out and lse, then benchmark bwd kernel directly.
    # _flash_attn_bwd is a pure function: same inputs → same gradients,
    # no autograd state, no graph, no PyLayer overhead.
    out, lse = fwd_fn()
    grad_out = paddle.randn_like(out)

    bwd_fn = lambda: _flash_attn_bwd(
        q, k, v,
        out=out,
        dout=grad_out,
        lse=lse,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    bwd_time_ms = do_bench(bwd_fn)

    total_time_ms = fwd_time_ms + bwd_time_ms

    # ---------- FLOPs  (causal ⇒ density ≈ 0.5 per-seq) ----------
    # FLOPs formula: 4 * seqlen^2 * H * D  (D here is head_dim of QK, i.e. D)
    # matches benchmark_flashmask.py cal_flops convention
    density = 0.5 if causal else 1.0
    fwd_flops   = density * cal_flops(len(seq_lens), H, sum(seq_lens), seq_lens, D, "fwd")
    bwd_flops   = density * cal_flops(len(seq_lens), H, sum(seq_lens), seq_lens, D, "bwd")
    total_flops = density * cal_flops(len(seq_lens), H, sum(seq_lens), seq_lens, D, "fwd_bwd")

    fwd_tflops   = cal_tflops(fwd_flops,   fwd_time_ms)
    bwd_tflops   = cal_tflops(bwd_flops,   bwd_time_ms)
    total_tflops = cal_tflops(total_flops, total_time_ms)
    sparsity     = 1.0 - density

    return (fwd_time_ms, bwd_time_ms, total_time_ms,
            fwd_flops, bwd_flops, total_flops,
            fwd_tflops, bwd_tflops, total_tflops,
            sparsity)


# ---------------------------------------------------------------------------
# Doc-len generators  (mirrors benchmark_flashmask.py helpers)
# ---------------------------------------------------------------------------

def make_uniform_docs(S: int, n_docs: int) -> List[int]:
    """Divide S tokens into n_docs equal-ish chunks."""
    base = S // n_docs
    lens = [base] * n_docs
    lens[-1] += S - sum(lens)
    return lens


def make_random_docs(S: int, n_docs: int, seed: int = 42) -> List[int]:
    """Random split of S into n_docs positive chunks."""
    rng = np.random.default_rng(seed)
    cuts = np.sort(rng.choice(S - 1, n_docs - 1, replace=False)) + 1
    cuts = np.concatenate([[0], cuts, [S]])
    lens = [int(cuts[i+1] - cuts[i]) for i in range(n_docs)]
    return lens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    seqlens: List[int] = None,
    D: int = 192,
    DV: int = 128,
    dtype: str = "bf16",
    from_file: Optional[str] = None,
    n_docs: int = 4,
    causal: bool = True,
):
    if seqlens is None:
        seqlens = [2048, 4096, 8192]

    paddle.set_flags({'FLAGS_flash_attn_version': 3})

    dtype_map = {"bf16": paddle.bfloat16, "fp16": paddle.float16}
    data_type = dtype_map[dtype]

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(dtype, exist_ok=True)

    # ----------------------------------------------------------------
    # Build list of (S, doc_seq_lens) configs
    # ----------------------------------------------------------------
    configs: List[Tuple[int, List[int]]] = []

    if from_file is not None:
        # Same format as benchmark_flashmask.py reads kernel_test_seq_info.txt
        total_length = 0
        with open(from_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "Total length" in line:
                    total_length = int(line.split(":")[1].split(",")[0].strip())
                else:
                    doc_list = eval(line.split(":")[-1].split("#")[0].strip())
                    doc_seq_lens = [x[1] for x in doc_list]
                    configs.append((total_length, doc_seq_lens))
    else:
        for S in seqlens:
            configs.append((S, make_random_docs(S, n_docs)))

    # ----------------------------------------------------------------
    # Run benchmarks
    # ----------------------------------------------------------------
    for D in [128, 192, 256]:
        if D == 192:
            DV = 128
            H = 16
        else:
            DV: int = D
            H = 4096 // D

        HKV = H

        for idx, (S, doc_seq_lens) in enumerate(configs):
            B = 128 * 1024 // S  # match benchmark_flashmask convention

            print(f"{B}_{S}_{H}_{HKV}_{D}_{DV}_{idx}_{dtype}")
            # print(f"  doc_seq_lens = {doc_seq_lens}")

            # Replicate doc_seq_lens B times so total tokens = B*S,
            # matching benchmark_flashmask which processes B padded samples.
            batched_doc_seq_lens = doc_seq_lens * B

            results = []

            # ---- varlen ----
            label = "varlen causal"
            print(label)
            try:
                (fwd_t, bwd_t, tot_t,
                fwd_f, bwd_f, tot_f,
                fwd_tf, bwd_tf, tot_tf,
                sparsity) = benchmark_varlen(
                    doc_seq_lens=batched_doc_seq_lens,
                    S=S * B, H=H, HKV=HKV, D=D, DV=DV,
                    dtype=data_type, causal=causal,
                )
                results.append([
                    label,
                    f"{fwd_t:.4f}",
                    f"{bwd_t:.4f}",
                    f"{tot_t:.4f}",
                    f"{fwd_f:.4f}",
                    f"{bwd_f:.4f}",
                    f"{tot_f:.4f}",
                    f"{fwd_tf:.4f}",
                    f"{bwd_tf:.4f}",
                    f"{tot_tf:.4f}",
                    f"{sparsity:.4f}",
                ])
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append([label] + ["ERROR"] * 10)

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
            content2 = tabulate(results, headers=headers, tablefmt="tsv")
            os.makedirs(dtype, exist_ok=True)
            text_file = open(f"{dtype}/fa4_varlen_paddle_{current_time}_{B}_{S}_{H}_{HKV}_{D}_{DV}_{idx}.csv", "w")
            text_file.write(content2)
            text_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark paddle flash_attn_varlen_func (Paddle varlen).\n"
                    "Causal Document Mask and varlen are equivalent — see module docstring."
    )
    parser.add_argument(
        "--seqlens", type=int, nargs="+", default=[2048, 4096, 8192],
        help="List of total sequence lengths to benchmark.",
    )
    parser.add_argument(
        "--D", type=int, default=128,
        help="Query/Key head dimension.",
    )
    parser.add_argument(
        "--DV", type=int, default=128,
        help="Value head dimension.",
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp16"],
        help="Data type.",
    )
    parser.add_argument(
        "--from_file", type=str, default="kernel_test_seq_info.txt",
        help="Read doc-seq-len configs from a kernel_test_seq_info.txt file "
             "(same format as benchmark_flashmask.py).",
    )
    parser.add_argument(
        "--n_docs", type=int, default=4,
        help="Number of documents to split each sequence into (random split).",
    )
    parser.add_argument(
        "--no_causal", action="store_true",
        help="Disable causal masking (runs full attention instead).",
    )
    args = parser.parse_args()

    main(
        seqlens=args.seqlens,
        D=args.D,
        DV=args.DV,
        dtype=args.dtype,
        from_file=args.from_file,
        n_docs=args.n_docs,
        causal=not args.no_causal,
    )
