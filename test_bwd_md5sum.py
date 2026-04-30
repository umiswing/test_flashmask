# run this to check aadiff:
# python -m pytest -v test_bwd_md5sum.py
#
# run this to record the bwd md5sum (only necessary when you want to update ground truth):
# python test_bwd_md5sum.py
import os
import json
import itertools
import unittest
import paddle
from functools import partial
import numpy as np
import pytest
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
from test_util import attention_ref
try:
    from flash_mask.cute.interface import flashmask_attention
except (ImportError, ModuleNotFoundError):
    from paddle.nn.functional.flash_attention import flashmask_attention

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

fa_versions = [4]
d_dv_combinations = [
    (64, 64),
    (80, 80),
    (128, 128),
    (192, 128),
]

def record_gt(output_file="flashmask_bwd_gt.json"):
    gt_records = {}

    param_combinations = generate_all_param_combinations()

    print(f"Start recording test cases, {len(param_combinations)} test cases in total.")

    for i, params in enumerate(param_combinations):
        try:
            dq_md5, dk_md5, dv_md5 = run_flashmask_backward(**params)
            param_key = generate_param_key(params)

            gt_records[param_key] = {
                "dq": dq_md5,
                "dk": dk_md5,
                "dv": dv_md5,
            }
            if (i + 1) % 10 == 0:
                print(f"{i+1}/{len(param_combinations)} test cases recorded")

        except pytest.skip.Exception as e:
            print(f"Skipping test case due to exception: {params}: {e}")
            continue
    gt_records["gt_commit_id"] = input("Please input the commit ID of bwd GT md5sum: ")
    gt_records["gt_commit_msg"] = input("Please input the commit msg of bwd GT md5sum: ")
    with open(output_file, 'w') as f:
        json.dump(gt_records, f, indent=2)

    print(f"Ground truth saved to '{output_file}', {len(gt_records)} test cases recorded.")
    return gt_records


def run_flashmask_backward(batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, d, dv,
                           nheads_startend_row_indices, fa_version, dtype, mask_type,
                           gen_startend_row_indices, softcap=0.0):
    paddle.seed(2024)
    np.random.seed(2024)
    assert nheads % nheads_kv == 0

    q = paddle.randn(shape=[batch_size, seqlen_q, nheads, d], dtype=dtype)
    k = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, d], dtype=dtype)
    v = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, dv], dtype=dtype)

    q.stop_gradient = False
    k.stop_gradient = False
    v.stop_gradient = False

    startend_row_indices, causal = gen_startend_row_indices(
        batch_size, seqlen_q, seqlen_k, nheads_startend_row_indices
    )

    if fa_version == 4 and mask_type == "global_sliding_window":
        pytest.skip(f"Skipping because running fa4 in global_sliding_window")

    if fa_version == 2:
        paddle.set_flags({'FLAGS_flash_attn_version': 2})
    elif fa_version == 3:
        paddle.set_flags({'FLAGS_flash_attn_version': 3})
    elif fa_version == 4:
        paddle.set_flags({'FLAGS_flash_attn_version': 4})
    else:
        raise ValueError(f"Invalid flash attention version: {fa_version}")

    paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
    out, lse = flashmask_attention(
        q, k, v,
        startend_row_indices=startend_row_indices,
        causal=causal,
        return_softmax_lse=True
    )

    g = paddle.randn(shape=[batch_size, seqlen_q, nheads, dv], dtype=dtype)
    out.backward(g)

    dq_md5 = q.grad._md5sum()
    dk_md5 = k.grad._md5sum()
    dv_md5 = v.grad._md5sum()

    return dq_md5, dk_md5, dv_md5


# 形状组合
shape_cases = [
    (1, 8192, 32768+1024, 2, 1),
    (2840, 32, 32, 16, 4),
    (1, 300, 300, 16, 16),
    (1, 128, 127, 1, 1),
    (2, 16384, 16383, 4, 1),
]

def generate_shapes():
    for batch_size, seqlen_q, seqlen_k, nheads, nheads_kv in shape_cases:
        nheads_startend_row_indices_values = [1, nheads_kv]
        for nheads_startend_row_indices in nheads_startend_row_indices_values:
            yield (
                batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices
            )


def generate_all_param_combinations():
    combinations = []

    dtypes = [paddle.bfloat16]

    for batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices in generate_shapes():
        for dtype in dtypes:
            for fa_version in fa_versions:
                for d, dv in d_dv_combinations:
                    for mask_type, gen_func in GEN_FUNCTIONS_DICT.items():
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
                            'gen_startend_row_indices': gen_func,
                            'softcap': 0.0
                        }
                        combinations.append(params)

    return combinations


def generate_param_key(params):
    nheads_startend = params['nheads_startend_row_indices']
    dtype_index = get_dtype_index(params['dtype'])

    if isinstance(nheads_startend, (list, tuple)):
        nheads_startend_str = '_'.join(map(str, nheads_startend))
    else:
        nheads_startend_str = str(nheads_startend)

    return (f"{params['mask_type']}-"
            f"{params['batch_size']}-{params['seqlen_q']}-{params['seqlen_k']}-"
            f"{params['nheads']}-{params['nheads_kv']}-{nheads_startend_str}-"
            f"{params['d']}-{params['dv']}-{params['fa_version']}-dtype{dtype_index}")

def get_dtype_index(dtype):
    dtype_list = [paddle.bfloat16]
    for i, dt in enumerate(dtype_list):
        if dtype == dt:
            return i
    return -1


gt_records = {}
try:
    with open("flashmask_bwd_gt.json", 'r') as f:
        gt_records = json.load(f)
except FileNotFoundError:
    pass


@pytest.mark.parametrize("dtype", [paddle.bfloat16])
@pytest.mark.parametrize("fa_version", fa_versions)
@pytest.mark.parametrize("d, dv", d_dv_combinations)
@pytest.mark.parametrize(
    "batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices",
    list(generate_shapes())
)
@pytest.mark.parametrize(
    "mask_type, gen_startend_row_indices",
    list(GEN_FUNCTIONS_DICT.items()),
)
def test_flashmask_bwd_md5(
    batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, d, dv,
    nheads_startend_row_indices, fa_version, dtype, mask_type, gen_startend_row_indices, softcap=0.0
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
        'softcap': softcap
    }

    param_key = generate_param_key(params)

    if param_key not in gt_records:
        pytest.skip(f"No ground truth record for {param_key}")

    dq_md5, dk_md5, dv_md5 = run_flashmask_backward(**params)

    expected = gt_records[param_key]

    assert dq_md5 == expected["dq"], f"dq MD5 mismatch for {param_key}\nExpected: {expected['dq']}\nGot: {dq_md5}"
    assert dk_md5 == expected["dk"], f"dk MD5 mismatch for {param_key}\nExpected: {expected['dk']}\nGot: {dk_md5}"
    assert dv_md5 == expected["dv"], f"dv MD5 mismatch for {param_key}\nExpected: {expected['dv']}\nGot: {dv_md5}"


if __name__ == "__main__":
    if not os.path.exists("flashmask_bwd_gt.json"):
        print("Start recording ground truth...")
        record_gt()
    else:
        print("Ground truth file exists, run pytest to execute tests")
