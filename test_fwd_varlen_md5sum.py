import os
import json
import math
import unittest
import paddle
from functools import partial
import numpy as np
import pytest
from generate_startend_row_indices import (
    generate_causal_document_mask,
    generate_document_mask,
    generate_causal_document_mask_diff_batch,
    generate_document_mask_diff_batch,
    generate_document_mask_simu,
    generate_document_mask_diff_batch_simu,
)

from flash_mask import flashmask_attention

# Only mask types that are compatible with varlen (causal-style masks).
GEN_FUNCTIONS_DICT = {
    "document": partial(generate_document_mask),
    "causal_document": partial(generate_causal_document_mask),
    "document_diff_batch": partial(generate_document_mask_diff_batch),
    "causal_document_diff_batch": partial(generate_causal_document_mask_diff_batch),
    "document_simu": partial(generate_document_mask_simu),
    "document_diff_batch_simu": partial(generate_document_mask_diff_batch_simu),
}

fa_versions = [4]
# Align head_dim set with test_flashmask_use_varlen.py.
d_dv_combinations = [
    (64, 64),
    (128, 128), 
    (192, 128),
    (256, 256),
]

softmax_scale_cases = [None, 1.0 / math.sqrt(64)]


def run_flashmask_varlen_forward(
    batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, d, dv,
    nheads_startend_row_indices, fa_version, dtype, mask_type,
    gen_startend_row_indices, softmax_scale, softcap=0.0,
):
    paddle.seed(2024)
    np.random.seed(2024)

    assert nheads % nheads_kv == 0

    deterministic = paddle.get_flags(["FLAGS_cudnn_deterministic"])[
        "FLAGS_cudnn_deterministic"
    ]
    if deterministic and d == 256:
        pytest.skip(
            "flash_attn varlen forward does not support head_dim=256 "
            "when FLAGS_cudnn_deterministic=1."
        )

    q = paddle.randn(shape=[batch_size, seqlen_q, nheads, d], dtype=dtype)
    k = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, d], dtype=dtype)
    v = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, dv], dtype=dtype)

    startend_row_indices, causal = gen_startend_row_indices(
        batch_size, seqlen_q, seqlen_k, nheads_startend_row_indices
    )

    if fa_version == 2:
        paddle.set_flags({'FLAGS_flash_attn_version': 2})
    elif fa_version == 3:
        paddle.set_flags({'FLAGS_flash_attn_version': 3})
    elif fa_version == 4:
        paddle.set_flags({'FLAGS_flash_attn_version': 4})
    else:
        raise ValueError(f"Invalid flash attention version: {fa_version}")

    paddle.set_flags({'FLAGS_cudnn_deterministic': 1})

    out = flashmask_attention(
        q, k, v,
        startend_row_indices=startend_row_indices,
        causal=causal,
        return_softmax_lse=False,
        use_varlen=True,
        softmax_scale=softmax_scale,
    )

    return out._md5sum()


# Varlen shape cases (kept aligned with test_flashmask_use_varlen.py).
shape_cases = [
    (2840, 32, 32, 16, 4),
    (1, 300, 300, 16, 16),
    (1, 256, 256, 4, 4),
    (2, 512, 512, 8, 2),
    (1, 1024, 1024, 4, 1),
    (2, 300, 300, 6, 2),
    (1, 128, 128, 1, 1),
    (2, 1000, 1000, 4, 1),
    (2, 8192, 8192, 4, 1),
    (2, 8192, 8192, 14, 1),
    (2, 16384, 16384, 4, 1),
    (2, 2000, 2000, 4, 1),
    (2, 3000, 3000, 4, 1),
    (1, 4000, 4000, 1, 1),
    (2, 7600, 7600, 32, 8),
]


def generate_shapes():
    for batch_size, seqlen_q, seqlen_k, nheads, nheads_kv in shape_cases:
        nheads_startend_row_indices = 1
        yield (batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices)


def softmax_scale_index(scale):
    for i, s in enumerate(softmax_scale_cases):
        if s is None and scale is None:
            return i
        if s is not None and scale is not None and math.isclose(s, scale):
            return i
    return -1


def generate_all_param_combinations():
    combinations = []

    dtypes = [paddle.bfloat16]

    for batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices in generate_shapes():
        for dtype in dtypes:
            for fa_version in fa_versions:
                for d, dv in d_dv_combinations:
                    for softmax_scale in softmax_scale_cases:
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
                                'softmax_scale': softmax_scale,
                                'softcap': 0.0,
                            }
                            combinations.append(params)

    return combinations


def generate_param_key(params):
    nheads_startend = params['nheads_startend_row_indices']
    dtype_index = get_dtype_index(params['dtype'])
    scale_index = softmax_scale_index(params['softmax_scale'])

    if isinstance(nheads_startend, (list, tuple)):
        nheads_startend_str = '_'.join(map(str, nheads_startend))
    else:
        nheads_startend_str = str(nheads_startend)

    return (f"{params['mask_type']}-"
            f"{params['batch_size']}-{params['seqlen_q']}-{params['seqlen_k']}-"
            f"{params['nheads']}-{params['nheads_kv']}-{nheads_startend_str}-"
            f"{params['d']}-{params['dv']}-{params['fa_version']}-"
            f"scale{scale_index}-dtype{dtype_index}")


def get_dtype_index(dtype):
    dtype_list = [paddle.bfloat16]
    for i, dt in enumerate(dtype_list):
        if dtype == dt:
            return i
    return -1


def record_gt(output_file="flashmask_varlen_fwd_gt.json"):
    gt_records = {}

    param_combinations = generate_all_param_combinations()

    print(f"Start recording test cases, {len(param_combinations)} test cases in total.")

    for i, params in enumerate(param_combinations):
        try:
            md5sum = run_flashmask_varlen_forward(**params)
            param_key = generate_param_key(params)

            gt_records[param_key] = md5sum
            if (i + 1) % 10 == 0:
                print(f"{i+1}/{len(param_combinations)} test cases recorded")

        except pytest.skip.Exception as e:
            print(f"Skipping test case due to exception: {params}: {e}")
            continue
    gt_records["gt_commit_id"] = input("Please input the commit ID of varlen fwd GT md5sum: ")
    gt_records["gt_commit_msg"] = input("Please input the commit msg of varlen fwd GT md5sum: ")
    with open(output_file, 'w') as f:
        json.dump(gt_records, f, indent=2)

    print(f"Ground truth saved to '{output_file}', {len(gt_records)} test cases recorded.")
    return gt_records


gt_records = {}
try:
    with open("flashmask_varlen_fwd_gt.json", 'r') as f:
        gt_records = json.load(f)
except FileNotFoundError:
    pass


@pytest.mark.parametrize("dtype", [paddle.bfloat16])
@pytest.mark.parametrize("fa_version", fa_versions)
@pytest.mark.parametrize("d, dv", d_dv_combinations)
@pytest.mark.parametrize("softmax_scale", softmax_scale_cases)
@pytest.mark.parametrize(
    "batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices",
    list(generate_shapes())
)
@pytest.mark.parametrize(
    "mask_type, gen_startend_row_indices",
    list(GEN_FUNCTIONS_DICT.items()),
)
def test_flashmask_varlen_fwd_md5(
    batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, d, dv,
    nheads_startend_row_indices, fa_version, dtype, mask_type,
    gen_startend_row_indices, softmax_scale, softcap=0.0,
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
        'softmax_scale': softmax_scale,
        'softcap': softcap,
    }

    param_key = generate_param_key(params)

    if param_key not in gt_records:
        pytest.skip(f"No ground truth record for {param_key}")

    actual_md5 = run_flashmask_varlen_forward(**params)
    expected_md5 = gt_records[param_key]

    assert actual_md5 == expected_md5, (
        f"MD5 mismatch for {param_key}\nExpected: {expected_md5}\nGot: {actual_md5}"
    )


if __name__ == "__main__":
    if not os.path.exists("flashmask_varlen_fwd_gt.json"):
        print("Start recording ground truth...")
        record_gt()
    else:
        print("Ground truth file exists, run pytest to execute tests")
