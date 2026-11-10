# Bitwise determinism ("aadiff") of flashmask forward out + backward dq / dk / dv.
#
#   bash run_aadiff.sh                    # compare with flashmask_gt_fa{N}.json, one aq task per shard
#   REGEN_GT=1 bash run_aadiff.sh         # precision changed on purpose: record a new GT, then compare
#
# By hand:
#   python -m pytest -v test_md5sum.py                                  # all cases vs the default GT
#   AADIFF_SHARD=512,512,shared AADIFF_GT=x.json python -m pytest test_md5sum.py
#   AADIFF_SHARD=512,512,shared python test_md5sum.py record --out x.json
#
# A case missing from the GT fails (it does not skip): a stale or partial GT must not pass.
import argparse
import datetime
import json
import os
import sys
from functools import partial

import numpy as np
import paddle
import pytest
from generate_startend_row_indices import (
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
from test_util import detect_fa_versions, kv_shared_detected, KV_SHARED_D_DV

# The package was renamed flash_mask -> flashmask; paddle's own kernel is the fa2 / fa3
# fallback. IMPL is stored in the GT, so a GT is never checked against another implementation.
try:
    from flashmask.cute.interface import flashmask_attention
    IMPL = "flashmask.cute"
except (ImportError, ModuleNotFoundError):
    try:
        from flash_mask.cute.interface import flashmask_attention
        IMPL = "flash_mask.cute"
    except (ImportError, ModuleNotFoundError):
        from paddle.nn.functional.flash_attention import flashmask_attention
        IMPL = "paddle"

GEN_FUNCTIONS_DICT = {
    "full": partial(generate_none_mask, causal=False),
    "causal": partial(generate_none_mask, causal=True),
    "sliding_window": partial(generate_sliding_window_mask),
    "causal_document": partial(generate_causal_document_mask),
    "document": partial(generate_document_mask),
    "share_question": partial(generate_share_question_mask),
    "global_sliding_window": partial(generate_global_sliding_window_mask),
    "causal_blockwise": partial(generate_causal_blockwise_mask),
    "prefix_lm_document_mask": partial(generate_prefix_lm_document_mask),
    "prefix_lm_causal": partial(generate_prefix_lm_causal_mask),
    "qk_sparse": partial(generate_qk_sparse_mask),
    "random_eviction": partial(generate_random_eviction_mask),
}

fa_versions = detect_fa_versions()
d_dv_combinations = [
    (64, 64),
    (80, 80),
    (128, 128),
    (192, 128),
    (256, 256),
    (512, 512),
    (576, 512),
]

# "split": k and v are two independent buffers, so dq / dk / dv all come back
# separately. "shared": v is a stride-1 view onto k (the MLA convention,
# v = k[..., :dv]); k is then the only leaf, the kernel merges dK and dV into
# one accumulator and k.grad carries dK + dV, so there is no dv to record.
kv_modes = ["split", "shared"]

# AADIFF_SHARD="d,dv,kv_mode" (run_aadiff.sh: one aq task per shard) narrows the cases.
SHARD = os.environ.get("AADIFF_SHARD")
if SHARD:
    _d, _dv, _kv = SHARD.split(",")
    d_dv_combinations = [(int(_d), int(_dv))]
    kv_modes = [_kv]

# The flags the GT is recorded and checked under (run_aadiff.sh exports them for both).
# FLAGS_alloc_fill_value=255 makes a never-written bf16 element NaN, FLAGS_use_system_allocator
# makes every buffer a fresh, filled allocation, FLAGS_check_cuda_error pins a fault to its op.
AADIFF_FLAGS = ("FLAGS_cudnn_deterministic", "FLAGS_alloc_fill_value", "FLAGS_use_system_allocator",
                "FLAGS_check_cuda_error")


def get_gt_filename(base="flashmask_gt"):
    # Tag the ground truth file with the fa version(s) so fa3 and fa4 ground
    # truth can coexist, e.g. flashmask_gt_fa3.json / flashmask_gt_fa4.json.
    version_tag = "_".join(str(v) for v in fa_versions)
    return f"{base}_fa{version_tag}.json"


GT_FILE = os.environ.get("AADIFF_GT") or get_gt_filename()
META_KEY = "gt_meta"
# Top-level keys of older GT files that are not cases.
NON_CASE_KEYS = (META_KEY, "gt_commit_id", "gt_commit_msg")


def _is_oom(exc):
    # Paddle maps a GPU BadAlloc to MemoryError, but the same condition also
    # surfaces as an EnforceNotMet carrying ResourceExhausted, so match on the
    # message too. Anything else must NOT be swallowed while recording.
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc)
    return "ResourceExhausted" in msg or "Out of memory" in msg or "out of memory" in msg


def unsupported_reason(fa_version, d, dv, kv_mode, **_):
    """Why a case is not run at all (skipped before the GT is consulted), or None."""
    # Only the fa4 cutedsl kernels handle head_dim > 256 (the big-headdim split path);
    # fa2/fa3 have no such kernel and would fall back / fail.
    if fa_version != 4 and max(d, dv) > 256:
        return f"d{d}-dv{dv} on fa{fa_version}: head_dim > 256 is fa4 only"
    if kv_mode == "shared" and (d, dv) not in KV_SHARED_D_DV:
        return f"kv shared for d{d}-dv{dv}: merge not implemented"
    if kv_mode == "shared" and fa_version != 4:
        return "kv shared on fa2/fa3: no big-headdim cutedsl backward"
    return None


def run_flashmask(batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, d, dv,
                  nheads_startend_row_indices, fa_version, dtype, mask_type,
                  gen_startend_row_indices, kv_mode="split", softcap=0.0):
    """One forward + backward: ({out, dq, dk, dv: md5}, {name: NaN/inf count}).

    The backward needs the forward anyway, so both directions are checked from a
    single run: the forward md5 costs nothing on top of the backward.
    """
    paddle.seed(2024)
    np.random.seed(2024)
    assert nheads % nheads_kv == 0
    kv_shared = kv_mode == "shared"

    q = paddle.randn(shape=[batch_size, seqlen_q, nheads, d], dtype=dtype)
    k = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, d], dtype=dtype)

    q.stop_gradient = False
    k.stop_gradient = False

    # The view is taken AFTER stop_gradient is cleared, so k is the leaf and
    # autograd folds dV back into k.grad the same way the kernel's merged
    # accumulator does.
    if kv_shared:
        v = k[..., :dv]
        assert kv_shared_detected(k, v), (
            "v is not a stride-1 view onto k -- paddle materialised the slice, "
            "so the kernel's kv-shared path would not be exercised"
        )
    else:
        v = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, dv], dtype=dtype)
        v.stop_gradient = False

    startend_row_indices, causal = gen_startend_row_indices(
        batch_size, seqlen_q, seqlen_k, nheads_startend_row_indices
    )

    if fa_version not in (2, 3, 4):
        raise ValueError(f"Invalid flash attention version: {fa_version}")
    paddle.set_flags({'FLAGS_flash_attn_version': fa_version})
    paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
    out, lse = flashmask_attention(
        q, k, v,
        startend_row_indices=startend_row_indices,
        causal=causal,
        return_softmax_lse=True
    )

    g = paddle.randn(shape=[batch_size, seqlen_q, nheads, dv], dtype=dtype)
    out.backward(g)

    # In shared mode v is a view, not a leaf: dV is already merged into k.grad.
    tensors = {"out": out, "dq": q.grad, "dk": k.grad, "dv": None if kv_shared else v.grad}
    md5 = {n: (None if t is None else t._md5sum()) for n, t in tensors.items()}
    # With FLAGS_alloc_fill_value=255 an element the kernel never wrote is NaN.
    nonfinite = {n: int((~paddle.isfinite(t.astype("float32"))).sum().item())
                 for n, t in tensors.items() if t is not None}
    return md5, {n: c for n, c in nonfinite.items() if c}


# 形状组合
shape_cases = [
    (1, 8192, 32768+1024, 2, 1),
    (2840, 32, 32, 16, 4),
    (1, 300, 300, 16, 16),
    (1, 128, 127, 1, 1),
    (2, 16384, 16383, 4, 1),
]


def generate_shapes():
    seen = set()
    for batch_size, seqlen_q, seqlen_k, nheads, nheads_kv in shape_cases:
        # nheads_kv == 1 makes both values 1: one case, not the same case twice.
        for nheads_startend_row_indices in dict.fromkeys([1, nheads_kv]):
            shape = (batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices)
            if shape not in seen:
                seen.add(shape)
                yield shape


def generate_all_param_combinations():
    combinations = []
    for batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices in generate_shapes():
        for fa_version in fa_versions:
            for d, dv in d_dv_combinations:
                for kv_mode in kv_modes:
                    for mask_type, gen_func in GEN_FUNCTIONS_DICT.items():
                        combinations.append({
                            'batch_size': batch_size,
                            'seqlen_q': seqlen_q,
                            'seqlen_k': seqlen_k,
                            'nheads': nheads,
                            'nheads_kv': nheads_kv,
                            'd': d,
                            'dv': dv,
                            'nheads_startend_row_indices': nheads_startend_row_indices,
                            'fa_version': fa_version,
                            'dtype': paddle.bfloat16,
                            'mask_type': mask_type,
                            'gen_startend_row_indices': gen_func,
                            'kv_mode': kv_mode,
                            'softcap': 0.0
                        })
    return combinations


def generate_param_key(params):
    nheads_startend = params['nheads_startend_row_indices']
    dtype_index = get_dtype_index(params['dtype'])

    if isinstance(nheads_startend, (list, tuple)):
        nheads_startend_str = '_'.join(map(str, nheads_startend))
    else:
        nheads_startend_str = str(nheads_startend)

    # Only the shared mode is suffixed, so the split keys stay byte-identical to
    # the ones the separate fwd / bwd ground truth files used.
    kv_suffix = "-kvshared" if params.get('kv_mode', 'split') == 'shared' else ""

    return (f"{params['mask_type']}-"
            f"{params['batch_size']}-{params['seqlen_q']}-{params['seqlen_k']}-"
            f"{params['nheads']}-{params['nheads_kv']}-{nheads_startend_str}-"
            f"{params['d']}-{params['dv']}-{params['fa_version']}-dtype{dtype_index}"
            f"{kv_suffix}")


def get_dtype_index(dtype):
    dtype_list = [paddle.bfloat16]
    for i, dt in enumerate(dtype_list):
        if dtype == dt:
            return i
    return -1


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def gt_meta():
    """What a GT was recorded with: the implementation, versions / commits, GPU and flags."""
    meta = {"impl": IMPL, "fa_versions": fa_versions, "shard": SHARD,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "flags": {f: os.environ.get(f) for f in AADIFF_FLAGS}}
    try:
        meta["gpu"] = paddle.device.cuda.get_device_name()
    except Exception:  # noqa: BLE001 - informational
        meta["gpu"] = None
    try:
        import bench_common
        meta["packages"] = [bench_common.package_info(p) for p in ("flashmask", "flash_mask", "paddle", "cutlass")]
    except Exception:  # noqa: BLE001 - informational
        meta["packages"] = []
    return meta


def record_gt(output_file):
    """Record the md5 of every case of this shard (non-interactive).

    Refuses to write anything if an output holds NaN / inf: with the run_aadiff.sh flags that
    is an element the kernel never wrote, and it must not become the ground truth.
    A case that does not fit in memory is stored as {"skip": ...}.
    """
    records, bad = {}, []
    combos = generate_all_param_combinations()
    print(f"Recording {len(combos)} cases (shard {SHARD or 'all'}) -> {output_file}")
    for i, params in enumerate(combos):
        key = generate_param_key(params)
        if unsupported_reason(**params):
            continue
        try:
            md5, nonfinite = run_flashmask(**params)
        except Exception as e:  # noqa: BLE001 - only out-of-memory is tolerated
            if not _is_oom(e):
                raise
            print(f"out of memory, recorded as skipped: {key}")
            records[key] = {"skip": "out of memory"}
            paddle.device.cuda.empty_cache()
            continue
        if nonfinite:
            bad.append((key, nonfinite))
        records[key] = md5
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(combos)} recorded")
    if bad:
        for key, nonfinite in bad[:20]:
            print(f"NaN/inf (never written?) in {key}: {nonfinite}")
        sys.exit(f"{len(bad)} case(s) have NaN/inf outputs; ground truth NOT written")
    records[META_KEY] = gt_meta()
    tmp = f"{output_file}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(records, f, indent=2)
    os.replace(tmp, output_file)
    print(f"Ground truth saved to '{output_file}', {len(records) - 1} cases.")


def load_gt(path):
    try:
        with open(path) as f:
            return json.load(f), None
    except FileNotFoundError:
        return {}, f"ground truth file {path} not found (REGEN_GT=1 bash run_aadiff.sh records one)"
    except ValueError as e:
        return {}, f"ground truth file {path} is not valid JSON: {e}"


gt_records, GT_ERROR = load_gt(GT_FILE)
_gt_impl = gt_records.get(META_KEY, {}).get("impl")
if _gt_impl and _gt_impl != IMPL and not GT_ERROR:
    GT_ERROR = f"{GT_FILE} was recorded with {_gt_impl}, this run imports {IMPL}"


@pytest.mark.parametrize("dtype", [paddle.bfloat16])
@pytest.mark.parametrize("fa_version", fa_versions)
@pytest.mark.parametrize("kv_mode", kv_modes)
@pytest.mark.parametrize("d, dv", d_dv_combinations)
@pytest.mark.parametrize(
    "batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices",
    list(generate_shapes())
)
@pytest.mark.parametrize(
    "mask_type, gen_startend_row_indices",
    list(GEN_FUNCTIONS_DICT.items()),
)
def test_flashmask_md5(
    batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, d, dv,
    nheads_startend_row_indices, kv_mode, fa_version, dtype, mask_type, gen_startend_row_indices, softcap=0.0
):
    params = {
        'batch_size': batch_size,
        'seqlen_q': seqlen_q,
        'seqlen_k': seqlen_k,
        'nheads': nheads,
        'nheads_kv': nheads_kv,
        'd': d,
        'dv': dv,
        'nheads_startend_row_indices': nheads_startend_row_indices,
        'fa_version': fa_version,
        'dtype': dtype,
        'mask_type': mask_type,
        'gen_startend_row_indices': gen_startend_row_indices,
        'kv_mode': kv_mode,
        'softcap': softcap
    }
    reason = unsupported_reason(**params)
    if reason:
        pytest.skip(reason)
    if GT_ERROR:
        pytest.fail(GT_ERROR)
    param_key = generate_param_key(params)
    expected = gt_records.get(param_key)
    if expected is None:
        pytest.fail(f"No ground truth for {param_key} in {GT_FILE}. If the change is intended, "
                    "record a new one: REGEN_GT=1 bash run_aadiff.sh")
    if "skip" in expected:
        pytest.skip(f"recorded as skipped: {expected['skip']}")

    md5, nonfinite = run_flashmask(**params)

    # Report every tensor that differs, not just the first one.
    diff = [f"  {n}: expected {expected.get(n)}, got {md5[n]}" for n in ("out", "dq", "dk", "dv")
            if md5[n] != expected.get(n)]
    if diff or nonfinite:
        hint = f"\n  NaN/inf elements (never written?): {nonfinite}" if nonfinite else ""
        pytest.fail(f"MD5 mismatch for {param_key} ({GT_FILE}):\n" + "\n".join(diff) + hint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record the md5 ground truth (pytest runs the check).")
    parser.add_argument("cmd", choices=["record"])
    parser.add_argument("--out", required=True, help="GT JSON to write (AADIFF_SHARD narrows the cases)")
    args = parser.parse_args()
    record_gt(args.out)
