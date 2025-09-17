import os
import numpy as np
from functools import lru_cache
from typing import Optional, List
import random

import torch
import torch.nn.functional as F

from tabulate import tabulate
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    and_masks,
    flex_attention,
    _score_mod_signature,
    _mask_mod_signature,
)

from triton.testing import do_bench

from attn_gym.masks.document_mask import length_to_offsets
from attn_gym.masks import (
    causal_mask,
    generate_sliding_window,
    generate_prefix_lm_mask,
    generate_doc_mask_mod,
)
from attn_gym.mods import generate_alibi_bias, generate_tanh_softcap


torch.set_default_device("cuda")
torch.manual_seed(0)

np.random.seed(0)
random.seed(0)

torch._dynamo.config.cache_size_limit = 1000

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)

# For better performance, you can use:
# flex_attention = torch.compile(_flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


# The kernels will utilize block sparsity to increase performance
print(f"Using the default sparsity block size: {_DEFAULT_SPARSE_BLOCK_SIZE}")


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device, _compile=True, BLOCK_SIZE=[128, 128])
    return block_mask


def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    return multiplier * flops * (1e3 / time_ms) / 1e12

def cal_flops(B, H, Sq, Sk, D, mode='fwd'):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * B * Sq * Sk * H * D
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def cal_tflops(flops, time_ms):
    return  flops * (1e3 / time_ms) / 1e12

def print_header(text):
    width = 91
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {text.center(width - 4)} ║")
    print("╚" + "═" * (width - 2) + "╝")


def test_mask(
    score_mod: Optional[_score_mod_signature] = None,
    mask_mod: Optional[_mask_mod_signature] = None,
    B: int = 16,
    H: int = 16,
    S: int = 8192,
    D: int = 64,
    dtype = 'bf16',
    skip_correctness: bool = False,
    print_mask: bool = True,
    device: str = "cuda",
):
    if dtype == 'bf16':
        data_type = torch.bfloat16
    else:
        data_type = torch.float16

    #assert score_mod is not None or mask_mod is not None, "Must provide a score_mod or mask_mod"
    if mask_mod is not None:
        block_mask = create_block_mask_cached(mask_mod, 1, 1, S, S, device=device)
    else:
        block_mask = None

    qkv = [
        torch.randn(B, H, S, D, device=device, dtype=data_type, requires_grad=True)
        for _ in range(3)
    ]
    gradOut = torch.randn(B, H, S, D, device=device, dtype=data_type)

    flex_attention_call = lambda: flex_attention(*qkv, score_mod=score_mod, block_mask=block_mask)

    results = []
    if block_mask is not None:
        density = (100 - block_mask.sparsity()) / 100
    else:
        density = 1.0

    sparsity = 1.0 - density

    # Forward pass
    fwd_time_ms = do_bench(flex_attention_call)
    torch._functorch.config.donated_buffer=False
    # Backward pass
    flex_out = flex_attention_call()
    bwd_time_ms = do_bench(lambda: flex_out.backward(gradOut, retain_graph=True))
    total_time_ms = fwd_time_ms + bwd_time_ms

    fwd_flops = density * cal_flops(B, H, S, S, D, mode='fwd')
    bwd_flops = density * cal_flops(B, H, S, S, D, mode='bwd')
    total_flops = density * cal_flops(B, H, S, S, D, mode='fwd_bwd')

    fwd_tflops = cal_tflops(fwd_flops, fwd_time_ms)
    bwd_tflops = cal_tflops(bwd_flops, bwd_time_ms)
    total_tflops = cal_tflops(total_flops, total_time_ms)

    return fwd_time_ms, bwd_time_ms, total_time_ms, fwd_flops, bwd_flops, total_flops, fwd_tflops, bwd_tflops, total_tflops, sparsity



def run_document_masking(max_seq_len: int, num_docs: int):
    import random

    random.seed(0)

    def generate_random_lengths(total_length, num_documents):
        # Initialize all lengths to 1 to ensure each document has at least one token
        lengths = [1] * num_documents
        remaining_length = total_length - num_documents

        # Randomly distribute the remaining length
        for _ in range(remaining_length):
            index = random.randint(0, num_documents - 1)
            lengths[index] += 1

        return lengths

    lengths = generate_random_lengths(max_seq_len, num_docs)
    offsets = length_to_offsets(lengths, "cuda")
    document_causal_mask = generate_doc_mask_mod(causal_mask, offsets)
    test_mask(mask_mod=document_causal_mask, S=32768)

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
    return causal_document_mask

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
            cur_len_so_far += doc_seq_lens[i+1]
    if padding > 0:
        up_right_row_indices.extend([cur_len_so_far] * padding)

    down_left_row_indices = torch.tensor(down_left_row_indices,  device=device, dtype=torch.int32).reshape((1, -1)).repeat_interleave(B, 0)
    up_right_row_indices= torch.tensor(up_right_row_indices,  device=device, dtype=torch.int32).reshape((1, -1)).repeat_interleave(B, 0)

    def document_mask(b, h, q_idx, kv_idx):
        return (q_idx < down_left_row_indices[b, kv_idx]) & (q_idx >= up_right_row_indices[b, kv_idx])

    return document_mask

#def generate_share_question_mask(B=16, S=8192, doc_seq_lens=[2538, 1742, 3213], device="cuda"):
#    total_seq_len = np.sum(doc_seq_lens)
#    assert total_seq_len <= S
#    assert len(doc_seq_lens) >= 3
#    padding = S - total_seq_len
#
#    startend_row_indices = [S] * doc_seq_lens[0]
#
#    cur_len_so_far = doc_seq_lens[0]
#    for idx in range(1, len(doc_seq_lens)):
#        cur_len_so_far += doc_seq_lens[idx]
#        startend_row_indices.extend([cur_len_so_far] * doc_seq_lens[idx])
#
#    if padding > 0:
#        startend_row_indices.extend([cur_len_so_far] * padding)
#        
#    startend_row_indices = torch.tensor(startend_row_indices, device=device, dtype=torch.int32).reshape((1, -1)).repeat_interleave(B, 0)
#    
#    def share_question_mask(b, h, q_idx, kv_idx):
#        return q_idx < startend_row_indices[b, kv_idx] 
#
#    causal_share_question_mask = and_masks(share_question_mask, causal_mask)
#    causal_share_question_mask.__name__ = f"causal_share_question_mask"
#    return causal_share_question_mask

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
    return causal_share_question_mask

def generate_global_sliding_window_mask(B=16, S=8192, global_token=16, window_size=(512, 512), device="cuda"):
    assert len(window_size) == 2
    left_window_size, right_window_size = window_size

    def global_sliding_window(b, h, q_idx, kv_idx):
        return ((q_idx >= kv_idx) & ((q_idx - kv_idx <= (left_window_size)) | (kv_idx < global_token))) | ((kv_idx >= q_idx) & ((kv_idx - q_idx <= (right_window_size)) | (q_idx < global_token)))

    return global_sliding_window

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
    return causal_blockwise_mask

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

    return prefix_lm_document_mask

def generate_prefix_lm_causal_mask(B=16, S=8192, prefix_length=1024, device="cuda"):
    """
    tuple(prefix_length, seq_length)
    """
    assert prefix_length <= S

    up_right_row_indices = torch.tensor([0] * prefix_length + list(range(prefix_length, S)),  device=device, dtype=torch.int32).reshape((1, -1)).repeat_interleave(B, 0)

    def prefix_lm_causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= up_right_row_indices[b, kv_idx]

    return prefix_lm_causal_mask

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

    return qk_sparse_mask


def generate_random_eviction_mask(B=16, H=16, S=8192, start_row=4096, device="cuda"):
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
    return causal_random_eviction_mask

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
        for D in [128, 64]:
            H = 4096 // D
            for idx, (S, prefix_doc_seq_lens, qksparse_mask) in enumerate(doc_seq_lens_list):
                B = 128 * 1024 // S

                doc_seq_lens = [x[1] for x in prefix_doc_seq_lens]
                maskout_pair = []
                offset = 0
                print(f"{B}_{S}_{H}_{D}_{idx}_{dtype}")
                if sum(qksparse_mask) == 0:
                    maskout_pair = [(1024, 538), (2358, 1700)]
                else:
                    for is_maskout, doc_seq in zip(qksparse_mask, doc_seq_lens):
                        if is_maskout:
                            maskout_pair.append((offset, doc_seq))
                        offset += doc_seq

                share_qa_docs = [split_sequence(doc_seq) for doc_seq in doc_seq_lens]

                available_examples = {
                    "Full": lambda: test_mask(mask_mod=None, B=B, S=S, H=H, D=D, dtype=dtype),
                    "Causal": lambda: test_mask(mask_mod=causal_mask, B=B, S=S, H=H, D=D, dtype=dtype),
                    "Sliding Window": lambda: test_mask(mask_mod=generate_sliding_window(window_size=int(S*0.0625)), B=B, S=S, H=H, D=D, dtype=dtype),
                    "Causal Document Mask": lambda: test_mask(mask_mod=generate_causal_document_mask(doc_seq_lens=doc_seq_lens, B=B, S=S), B=B, S=S, H=H, D=D, dtype=dtype),
                    "Document Mask": lambda: test_mask(mask_mod=generate_document_mask(doc_seq_lens=doc_seq_lens, B=B, S=S), B=B, S=S, H=H, D=D, dtype=dtype),
                    "Share Question Mask": lambda: test_mask(mask_mod=generate_share_question_mask(doc_seq_lens=share_qa_docs, B=B, S=S), B=B, S=S, H=H, D=D, dtype=dtype),
                    "Global Sliding Window": lambda: test_mask(mask_mod=generate_global_sliding_window_mask(global_token=16, B=B, S=S, window_size=(int(S*0.0625), int(S*0.0625))), B=B, S=S, H=H, D=D, dtype=dtype),
                    "Causal Blockwise Mask": lambda: test_mask(mask_mod=generate_causal_blockwise_mask(doc_seq_lens=doc_seq_lens, B=B, S=S), B=B, S=S, H=H, D=D, dtype=dtype),
                    "Prefix LM Document Mask": lambda: test_mask(mask_mod=generate_prefix_lm_document_mask(doc_seq_lens=prefix_doc_seq_lens, B=B, S=S), B=B, S=S, H=H, D=D, dtype=dtype),
                    "Prefix LM Causal Mask": lambda: test_mask(mask_mod=generate_prefix_lm_causal_mask(prefix_length=int(S*0.5), B=B, S=S), B=B, S=S, H=H, D=D, dtype=dtype),
                    "QK-sparse Mask": lambda: test_mask(mask_mod=generate_qk_sparse_mask(maskout_pair=maskout_pair, B=B, S=S), B=B, S=S, H=H, D=D, dtype=dtype),
                    "Random Eviction Mask": lambda: test_mask(mask_mod=generate_random_eviction_mask(start_row=S//2, B=B, S=S, H=H), B=B, S=S, H=H, D=D, dtype=dtype),
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
                text_file = open(f"{dtype}/flexattention_{B}_{S}_{H}_{D}_{idx}.csv","w")
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

