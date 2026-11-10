import contextlib
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

import bench_common


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
# SM100 big-headdim backward merges dK and dV into a single accumulator. The pair
# list and the detection predicate live in test_util.py so that the benchmark and the
# tests cannot drift apart (the predicate mirrors flashmask/cute/interface.py).
from test_util import KV_SHARED_D_DV, kv_shared_detected


def do_bench(fn, S):
    return bench_common.do_bench(fn, S, paddle.device.synchronize, paddle.device.Event)


def flashmask_bwd_call(query, key, value, out, gradOut, lse, startend_row_indices, causal, sink):
    """The cutedsl backward that is timed; returns (dq, dk, dv, dsink)."""
    from flashmask.cute import flashmask_utils as fm
    from flashmask.cute.interface import _flash_attn_bwd
    flashmask_info = None
    if startend_row_indices is not None:
        flashmask_info = fm.FlashMaskInfoPaddle(
            startend_row_indices=startend_row_indices,
            is_causal=causal,
        )
    return _flash_attn_bwd(
        query,
        key,
        value,
        out,
        gradOut,
        lse,
        flashmask_info,
        causal=causal,
        learnable_sink=sink,
    )


@contextlib.contextmanager
def _poison_empty():
    """FlashMask empty / empty_like float buffers as NaN (see test_flashmask.py), so an
    output row the kernels never write shows up in the --check_masks output check.
    Older flashmask builds without flashmask.backend run unpoisoned (a warning says so)."""
    try:
        import flashmask.backend._paddle as fm_paddle_backend
    except ImportError:
        print("[acc] flashmask.backend._paddle not found: outputs are not NaN-poisoned")
        yield
        return
    backend = fm_paddle_backend.PaddleBackend
    orig_empty, orig_empty_like = backend.empty, backend.empty_like

    def poisoned(f):
        def g(self, *args, **kwargs):
            t = f(self, *args, **kwargs)
            if t.dtype in (paddle.float16, paddle.bfloat16, paddle.float32):
                t = paddle.full_like(t, float("nan"))
            return t
        return g

    backend.empty, backend.empty_like = poisoned(orig_empty), poisoned(orig_empty_like)
    try:
        yield
    finally:
        backend.empty, backend.empty_like = orig_empty, orig_empty_like


def test_mask(
    spec,
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
    inputs = None,
    outputs_only = False,
):
    """Time the fwd / bwd of one mask, or with outputs_only run them once on inputs
    (numpy q, k, v, dout from bench_common.acc_inputs) and return the outputs."""

    if dtype == 'bf16':
        data_type = paddle.bfloat16
    else:
        data_type = paddle.float16

    if inputs is not None:
        query, key, value, gradOut = [paddle.to_tensor(x).astype(data_type) for x in inputs]
        key.stop_gradient = False
        query.stop_gradient = False
    else:
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
    elif inputs is None:
        value = paddle.randn([B, SKV, HKV, DV], dtype=data_type)
        value.stop_gradient = False
    else:
        value.stop_gradient = False
    assert kv_shared_detected(key, value) == (kv_mode == 'shared'), (
        f"kv_mode={kv_mode!r} but the kernel would detect "
        f"kv_shared={kv_shared_detected(key, value)}"
    )
    if inputs is None:
        gradOut = paddle.randn([B, S, H, DV], dtype=data_type)

    # startend_row_indices is a given (the layer already has it) and flashmask
    # consumes it as-is, so there is no mask-conversion step to report here.
    # Building it below is test-data setup, not a cost the kernel imposes. The
    # competitor benchmarks time exactly the step this path does not have:
    # startend_row_indices -> the mask their kernel needs.
    causal = spec.causal
    startend_row_indices = (
        None if spec.startend is None else paddle.to_tensor(spec.startend, dtype=paddle.int32)
    )

    sparsity = bench_common.element_sparsity(causal, spec.startend, S, SKV)
    density = 1.0 - sparsity

    sink = None
    if use_sink:
        sink = paddle.randn(shape=[H], dtype=data_type)
        sink.stop_gradient = False

    flashmask = lambda: flashmask_attention(query, key, value, startend_row_indices=startend_row_indices, causal=causal, return_softmax_lse=True)

    use_cutedsl = (backend == 'cutedsl')

    if use_cutedsl:
        from flashmask.cute.interface import flashmask_attention
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
            from flashmask.cute.interface import _flash_attn_fwd
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
            return out, lse
        if not outputs_only:
            # paddle.base.core.nvprof_nvtx_push("flashmask")
            fwd_time_ms = do_bench(flashmask_fwd, S)
            # paddle.base.core.nvprof_nvtx_pop()
    else:
        assert not outputs_only, "outputs_only runs the cutedsl fwd / bwd that is timed"
        fwd_time_ms = do_bench(flashmask, S)

    if outputs_only:
        # The timed fwd / bwd calls, once, with every float buffer they allocate NaN.
        with _poison_empty():
            flashmask_out, lse = flashmask_fwd()
            dq, dk, dv, _ = flashmask_bwd_call(query, key, value, flashmask_out, gradOut, lse,
                                               startend_row_indices, causal, sink)
        return {n: t.astype(paddle.float32).numpy()
                for n, t in zip(bench_common.ACC_TENSORS, (flashmask_out, dq, dk, dv))}

    flashmask_out, lse = flashmask()

    if use_cutedsl:
        def flashmask_bwd():
            flashmask_bwd_call(query, key, value, flashmask_out, gradOut, lse,
                               startend_row_indices, causal, sink)

        # paddle.base.core.nvprof_nvtx_push("flashmask")
        bwd_time_ms = do_bench(flashmask_bwd, S)
        # paddle.base.core.nvprof_nvtx_pop()
    else:
        bwd_time_ms = do_bench(lambda: flashmask_out.backward(gradOut, retain_graph=True), S)

    total_time_ms = fwd_time_ms + bwd_time_ms

    fwd_flops = density * cal_flops(B, H, S, SKV, D, DV, mode='fwd')
    bwd_flops = density * cal_flops(B, H, S, SKV, D, DV, mode='bwd')
    total_flops = density * cal_flops(B, H, S, SKV, D, DV, mode='fwd_bwd')

    fwd_tflops = cal_tflops(fwd_flops, fwd_time_ms)
    bwd_tflops = cal_tflops(bwd_flops, bwd_time_ms)
    total_tflops = cal_tflops(total_flops, total_time_ms)

    return fwd_time_ms, bwd_time_ms, total_time_ms, fwd_flops, bwd_flops, total_flops, fwd_tflops, bwd_tflops, total_tflops, sparsity















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
    prefix_lm_document_mask = paddle.to_tensor(
        bench_common.prefix_lm_document_startend(doc_seq_lens, seqlen, batch_size), dtype=paddle.int32
    )
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


def main(examples: List[str] = ["all"], dtype='bf16', fm_version=1, series=None, results_dir=bench_common.RESULTS_DIR, overwrite=True, head_dim=None, current_time=None, backend='cutedsl', kv_mode=None, use_sink=False, vs_sparse_attn=False, dedup_static_masks=False, first_sample_only=False, check_masks=None, seqlens=None, interfere_grace=60.0,
         interfere_max_wait=1800.0, allow_handoff=False):
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
        dedup_static_masks: skip the masks that only depend on S
            (bench_common.S_ONLY_MASKS) on every sample but the first of each
            ``Total length`` block. Applies under vs_sparse_attn as well.
        first_sample_only: run only Sample 1 of each ``Total length`` block.
        check_masks: write one fingerprint per (case, mask, kv_mode) to this jsonl
            file instead of timing anything (see check_bench_consistency.py), plus the
            output check of the timed fwd / bwd (bench_common.run_accuracy; kv_mode
            split, cutedsl, no sink).

    Cases, masks, sparsity and iteration counts come from bench_common, shared with
    benchmark_fa4_mask_mod.py and benchmark_cudnn_flex_mask_mod.py.
    """
    # Operations benchmark_flashmla_sparse_attn.py also produces (its `layouts`
    # dict). Keep in sync with that file.
    VS_SPARSE_EXAMPLES = ("Causal", "Causal Document Mask")

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

    if check_masks:
        open(check_masks, "w").close()

    # Result series: <series> for kv_mode split, <series>_kvshared for shared.
    base_series = series or f"fm{fm_version}" + ("" if dtype == "bf16" else f"_{dtype}")
    series_of = lambda mode: base_series + ("_kvshared" if mode == "shared" else "")

    last_D = None
    for case in bench_common.iter_cases(d_list, examples, dedup_static_masks, first_sample_only, seqlens):
        if vs_sparse_attn:
            case = case._replace(B=1, H=64, HKV=1)
        sample, B, H, HKV, D, DV = case.sample, case.B, case.H, case.HKV, case.D, case.DV
        S, idx = sample.S, sample.idx
        SQ = SKV = S

        # kv_mode='shared' only means something where the backward actually
        # merges dK and dV; anywhere else it is the split kernel with an
        # aliased V, i.e. a duplicate of the 'split' number for twice the
        # wall clock.
        if (D, DV) in KV_SHARED_D_DV:
            d_kv_modes = kv_modes
        else:
            d_kv_modes = [m for m in kv_modes if m != 'shared']
            if 'shared' in kv_modes and D != last_D:
                print(f"D={D} DV={DV}: skipping kv_mode=shared, the dK/dV "
                      f"merge is only implemented for {KV_SHARED_D_DV} so it "
                      f"would just repeat the 'split' measurement.")
        last_D = D
        if not d_kv_modes:
            continue

        ex_to_run = list(case.masks)
        if vs_sparse_attn:
            dropped = [ex for ex in ex_to_run if ex not in VS_SPARSE_EXAMPLES]
            if dropped:
                print(f"--vs_sparse_attn: dropping {dropped}, "
                      f"benchmark_flashmla_sparse_attn.py only emits "
                      f"{list(VS_SPARSE_EXAMPLES)} and plot_radar keeps "
                      f"only the operations every method shares.")
            ex_to_run = [ex for ex in ex_to_run if ex in VS_SPARSE_EXAMPLES]
        if not ex_to_run:
            continue

        print(f"{B}_{S}_{H}_{HKV}_{D}_{DV}_{idx}_{dtype}")
        specs = {ex: bench_common.build_mask(ex, case) for ex in ex_to_run}

        if check_masks:
            for ex, spec in specs.items():
                fp = bench_common.fingerprint(*bench_common.visible_counts(spec.causal, spec.startend, B, S, SKV))
                for mode in d_kv_modes:
                    # Outputs are checked on kv_mode split only (the reference has no shared V).
                    acc = None
                    if mode == 'split' and backend == 'cutedsl' and not use_sink:
                        acc = bench_common.run_accuracy(case, ex, lambda c, s, inputs: test_mask(
                            spec=s, B=c.B, S=S, SKV=SKV, H=c.H, HKV=c.HKV, D=D, DV=DV, dtype=dtype,
                            backend=backend, kv_mode=mode, inputs=inputs, outputs_only=True), check_masks)
                        print(f"[acc] S={S} D={D} {ex}: {'ok' if acc is None or acc['ok'] else 'FAIL'}")
                    bench_common.append_jsonl(check_masks, bench_common.check_record("flashmask", case, spec, fp, mode, acc))
            continue

        for mode in d_kv_modes:
            out = bench_common.csv_path(results_dir, current_time, series_of(mode), case)
            bench_common.write_meta(results_dir, current_time, series_of(mode), "flashmask",
                                    packages=["flashmask", "paddle"], fm_version=fm_version,
                                    kv_mode=mode, dtype=dtype, backend_kind=backend, use_sink=use_sink,
                                    gpu=paddle.device.cuda.get_device_name())

            def run_mask(ex):
                print(f"{ex} [kv_mode={mode}]")
                t_mask = time.time()
                fw_time, bw_time, total_time, fw_flops, bw_flops, total_flops, fw_tflops, bw_tflops, total_tflops, sparsity = test_mask(
                    spec=specs[ex], B=B, S=SQ, SKV=SKV, H=H, HKV=HKV, D=D, DV=DV,
                    dtype=dtype, backend=backend, kv_mode=mode, use_sink=use_sink,
                )
                # Wall clock of the whole mask (compile + timing runs).
                print(f"[time] S={S} D={D} mask={ex} kv_mode={mode} wall={time.time() - t_mask:.1f}s "
                      f"(fwd {fw_time:.2f} ms, bwd {bw_time:.2f} ms)")
                return [ex, f"{fw_time:.4f}", f"{bw_time:.4f}", f"{total_time:.4f}", f"{fw_flops:.4f}", f"{bw_flops:.4f}", f"{total_flops:.4f}", f"{fw_tflops:.4f}", f"{bw_tflops:.4f}", f"{total_tflops:4f}", f"{sparsity:.4f}"]

            # Same --current_time + --no-overwrite: a rerun (or a handed-off task) picks
            # up where the last one stopped. Disturbed measurements are never written.
            results = bench_common.run_masks_to_csv(out, ex_to_run, run_mask, overwrite,
                                                    own_mem=paddle.device.cuda.memory_reserved,
                                                    grace=interfere_grace, max_wait=interfere_max_wait,
                                                    allow_handoff=allow_handoff)
            if results is None:
                continue

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
            print(f"wrote {out}")

if __name__ == "__main__":
    from jsonargparse import ArgumentParser
    parser = ArgumentParser(description="Run specific examples or all examples.")
    parser.add_argument(
        "--examples",
        type=str,
        nargs="+",
        default=["all"],
        help="Masks to run, or 'all'. Names as in the CSV Operation column: "
        "'Full', 'Causal', 'Sliding Window', 'Global Sliding Window', "
        "'Causal Document Mask', 'Document Mask', 'Share Question Mask', "
        "'Causal Blockwise Mask', 'Prefix LM Document Mask', 'Prefix LM Causal Mask', "
        "'QK-sparse Mask', 'Random Eviction Mask'.",
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
        "--series",
        type=str,
        default=None,
        help="Result series name (default fm<fm_version>; kv_mode shared adds _kvshared). "
        "CSVs go to <results_dir>/<current_time>/<series>/.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=bench_common.RESULTS_DIR,
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

    parser.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Skip the samples whose CSVs for this --current_time already exist "
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
        "--use_sink",
        action="store_true",
        default=False,
        help="Add the learnable per-head attention sink (the online latent-MQA "
        "layers carry one when add_full_attention_sink_bias is set).",
    )

    parser.add_argument(
        "--vs_sparse_attn",
        action="store_true",
        default=False,
        help="Only measure Causal and Causal Document Mask at the shapes of "
        "benchmark_flashmla_sparse_attn.py (head_dim 576, H=64, HKV=1, kv_mode shared).",
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

    parser.add_argument(
        "--check_masks",
        type=str,
        default=None,
        help="Write the mask fingerprint of every (case, mask, kv_mode) and the output check "
        "of the timed fwd / bwd against a shared reference to this jsonl file instead of "
        "timing; compare backends with check_bench_consistency.py.",
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

    args = parser.parse_args()
    bench_common.run_main(main, **vars(args))