import numpy as np
import paddle
import itertools
import random
import os
import argparse
import glob

# 强制使用 GPU
paddle.set_device('gpu')

# 设置 Flash Attention 版本标记
paddle.set_flags({'FLAGS_flash_attn_version': 2})
# 开启确定性
paddle.set_flags({'FLAGS_cudnn_deterministic': 1})

# 数据保存目录
DATA_DIR = "./paddle_pure_comparison_data"

def tonp(x):
    """将 Paddle Tensor 转换为 Numpy array，特殊处理 bfloat16"""
    if isinstance(x, paddle.Tensor):
        if x.dtype == paddle.bfloat16:
            # bfloat16 在 numpy 中没有原生类型，转存为 uint16 以保持二进制一致性
            return x.view('uint16').numpy()
        elif x.dtype in [paddle.float32, paddle.float16, paddle.int32, paddle.int64]:
            return x.numpy()
        else:
            assert False, f'Unsupported dtype for saving: {x.dtype}'
    elif isinstance(x, np.ndarray):
        return x
    else:
        assert False, f'wrong type: {type(x)}'

def from_numpy(x_np, dtype_str, place):
    """从 Numpy array 恢复 Paddle Tensor，特殊处理 bfloat16"""
    tensor = paddle.to_tensor(x_np, place=place)
    
    if dtype_str == 'paddle.bfloat16':
        # 从 uint16 视图恢复为 bfloat16
        return tensor.view(paddle.bfloat16)
    elif dtype_str == 'paddle.float16':
        return tensor.cast(paddle.float16)
    elif dtype_str == 'paddle.float32':
        return tensor.cast(paddle.float32)
    elif dtype_str == 'paddle.int32':
        return tensor.cast(paddle.int32)
    else:
        # 尝试直接转换
        return tensor

def cmp(x_actual, x_ref_np, msg, array_equal=True, atol=0, rtol=0):
    """对比函数：x_actual 是 Paddle Tensor，x_ref_np 是加载的 Numpy 数据"""
    x = tonp(x_actual) # 如果是 bf16，这里会变成 uint16 numpy array
    
    if array_equal:
        diff = np.abs(x - x_ref_np)
        # 设定一个阈值，查看超过阈值的具体值
        bad_mask = diff > atol + rtol * np.abs(x_ref_np)
        if np.any(bad_mask):
            print(f"--- Debug Fail {msg} ---")
            print(f"Max Diff: {np.max(diff)}")
            indices = np.where(bad_mask)
            # 打印前 5 个错误点
            for i in range(min(5, len(indices[0]))):
                idx = tuple(ind[i] for ind in indices)
                print(f"Index {idx}: Act={x[idx]}, Ref={x_ref_np[idx]}, Diff={diff[idx]}")

        np.testing.assert_array_equal(x, x_ref_np, err_msg=f'{msg} mismatch', strict=True)
    else:
        # 数值对比模式
        # 准备实际值
        if x_actual.dtype == paddle.bfloat16:
            val_act = x_actual.cast(paddle.float32).numpy()
        else:
            val_act = x_actual.numpy()
            
        # 准备参考值 (处理 ref 是 uint16 (bf16) 的情况)
        if x_ref_np.dtype == np.uint16 and x_actual.dtype == paddle.bfloat16:
            # 将 ref 的 uint16 还原回 paddle bf16 再转 float32 供 numpy 对比
            tmp_tensor = paddle.to_tensor(x_ref_np).view(paddle.bfloat16)
            val_ref = tmp_tensor.cast(paddle.float32).numpy()
        else:
            val_ref = x_ref_np

        np.testing.assert_allclose(actual=val_act, desired=val_ref, rtol=rtol, atol=atol, equal_nan=False, err_msg=msg)

def random_cu_seqlens_paddle(total_tokens, batch_size):
    """完全使用 Paddle 生成 cu_seqlens"""
    if batch_size == 1:
        cu_seqlens = paddle.to_tensor([0, total_tokens], dtype='int32')
        return cu_seqlens.cuda()

    # 生成切分点
    # randperm 返回 0 到 n-1，我们需要 1 到 total_tokens-1 之间的切点
    random_points = paddle.randperm(total_tokens - 1)[:batch_size - 1] + 1
    random_points = paddle.sort(random_points)
    
    # 拼接 [0, ..., total]
    zeros = paddle.to_tensor([0], dtype='int64')
    total = paddle.to_tensor([total_tokens], dtype='int64')
    
    cu_seqlens = paddle.concat([zeros, random_points, total])
    return cu_seqlens.cast('int32').cuda()

# ==========================================
# Mode 1: 生成数据并保存 (基准环境运行)
# ==========================================
def run_save(case_name, batch_size, total_q, total_k, nheads, nheads_k, headdim, headdim_v, softmax_scale, causal, dtype):
    print(f'SAVE: {case_name}, {batch_size=}, {total_q=}, {total_k=}, {nheads=}, {nheads_k=}, {headdim=}, {headdim_v=} {softmax_scale=} {causal=} {dtype=}')
    
    # 1. 使用 Paddle 生成输入
    q = paddle.randn([total_q, nheads, headdim], dtype=dtype)
    k = paddle.randn([total_k, nheads_k, headdim], dtype=dtype)
    v = paddle.randn([total_k, nheads_k, headdim_v], dtype=dtype)
    
    q.stop_gradient = False
    k.stop_gradient = False
    v.stop_gradient = False

    cu_seqlens_q = random_cu_seqlens_paddle(total_q, batch_size)
    cu_seqlens_k = random_cu_seqlens_paddle(total_k, batch_size)

    max_seqlen_q = paddle.max(cu_seqlens_q[1:] - cu_seqlens_q[:-1]).item()
    max_seqlen_k = paddle.max(cu_seqlens_k[1:] - cu_seqlens_k[:-1]).item()

    # 2. 运行 Forward
    out, softmax_lse = paddle.nn.functional.flash_attention.flash_attn_unpadded(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 
        scale=softmax_scale, dropout=0.0, causal=causal
    )

    # 3. 运行 Backward
    # 生成随机梯度
    out_grad = paddle.randn_like(out)
    
    # 只有当维度满足条件时才运行 backward
    # Paddle FA3 支持 backward 的条件：headdim==headdim_v 或者 (headdim!=headdim_v 且 非GQA/MQA)
    # 这里保持逻辑简单，尽可能跑 backward
    if headdim == headdim_v or (nheads == nheads_k):
        out.backward(out_grad)

    # 4. 保存所有数据到 npz
    save_path = os.path.join(DATA_DIR, f"{case_name}.npz")
    
    data_dict = {
        "config": np.array([max_seqlen_q, max_seqlen_k], dtype=np.int32),
        "params": np.array([softmax_scale if softmax_scale is not None else -1.0], dtype=np.float32),
        "meta": np.array([1 if causal else 0, 1 if softmax_scale is None else 0], dtype=np.int32),
        "dtype_str": str(q.dtype),
        
        # 输入 (Tensor -> Numpy)
        "q": tonp(q),
        "k": tonp(k),
        "v": tonp(v),
        "cu_seqlens_q": tonp(cu_seqlens_q),
        "cu_seqlens_k": tonp(cu_seqlens_k),
        "out_grad": tonp(out_grad),
        
        # 期望输出 (Reference)
        "ref_out": tonp(out),
    }

    if q.grad is not None: data_dict["ref_dq"] = tonp(q.grad)
    if k.grad is not None: data_dict["ref_dk"] = tonp(k.grad)
    if v.grad is not None: data_dict["ref_dv"] = tonp(v.grad)

    np.savez(save_path, **data_dict)

# ==========================================
# Mode 2: 加载数据并验证 (测试环境运行)
# ==========================================
def run_verify(file_path):
    print(f'\nVERIFYING: {os.path.basename(file_path)}')
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    # 1. 恢复配置参数
    max_seqlen_q = int(data["config"][0])
    max_seqlen_k = int(data["config"][1])
    softmax_scale = float(data["params"][0])
    is_causal = bool(data["meta"][0])
    is_scale_none = bool(data["meta"][1])
    if is_scale_none: softmax_scale = None
    dtype_str = str(data["dtype_str"])

    # 2. 恢复 Tensor
    place = paddle.CUDAPlace(0)
    q = from_numpy(data["q"], dtype_str, place)
    k = from_numpy(data["k"], dtype_str, place)
    v = from_numpy(data["v"], dtype_str, place)
    cu_seqlens_q = from_numpy(data["cu_seqlens_q"], 'paddle.int32', place)
    cu_seqlens_k = from_numpy(data["cu_seqlens_k"], 'paddle.int32', place)
    out_grad = from_numpy(data["out_grad"], dtype_str, place)

    q.stop_gradient = False
    k.stop_gradient = False
    v.stop_gradient = False
    
    # 获取维度信息用于逻辑判断
    headdim = q.shape[2]
    headdim_v = v.shape[2]
    nheads = q.shape[1]
    nheads_k = k.shape[1]

    # 3. 运行当前环境的 Forward
    out, softmax_lse = paddle.nn.functional.flash_attention.flash_attn_unpadded(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 
        scale=softmax_scale, dropout=0.0, causal=is_causal
    )

    # 4. 对比 Forward
    cmp(out, data["ref_out"], 'out', array_equal=True)
    print('>> fwd out pass')

    # 5. 运行当前环境的 Backward
    if headdim == headdim_v or (nheads == nheads_k):
        out.backward(out_grad)
    
    if headdim != headdim_v and nheads == nheads_k:
         paddle.device.synchronize()
         assert q.grad is not None, "q.grad is None in MLA case"
         assert k.grad is not None, "k.grad is None in MLA case"
         assert v.grad is not None, "v.grad is None in MLA case"
         print(">> mla paddle can run bwd")

    # 6. 对比 Backward
    if "ref_dq" in data:
        try:
            cmp(q.grad, data["ref_dq"], 'q.grad', array_equal=True)
            print('>> dq pass')
        except AssertionError as e:
            print(f"!! dq mismatch: {e}")

    if "ref_dk" in data:
        try:
            cmp(k.grad, data["ref_dk"], 'k.grad', array_equal=True)
            print('>> dk pass')
        except AssertionError as e:
            print(f"!! dk mismatch: {e}")

    if "ref_dv" in data:
        try:
            cmp(v.grad, data["ref_dv"], 'v.grad', array_equal=True)
            print('>> dv pass')
        except AssertionError as e:
            print(f"!! dv mismatch: {e}")

def main_gen_loops():
    """生成测试用例的循环逻辑"""
    counter = 0
    dtype_options = [paddle.bfloat16, paddle.float16]
    causal_options = [True, False]

    # ================= Case 7 =================
    print('\nGenerating Case 7 (Random shapes, headdim=headdim_v)...')
    for causal, dtype in itertools.product(causal_options, dtype_options):
        # 跑几次随机
        for _ in range(2): 
            headdim = random.randrange(1, 33) * 8
            headdim_v = headdim
            total_q = random.randrange(100, 2048) 
            total_k = random.randrange(100, 2048)
            batch_size = random.randint(1, min(total_q, total_k, 32)) # 限制 batch_size 避免过大

            nheads_k = random.randrange(1, 17)
            group = random.randrange(1, 9)
            nheads = group * nheads_k

            softmax_scale = random.uniform(0.000001, 1.0)

            run_save(f"case7_{counter}", batch_size, total_q, total_k, nheads, nheads_k, headdim, headdim_v, softmax_scale, causal, dtype)
            counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["save", "verify"], 
                        help="'save': 在旧环境中生成并保存基准数据; 'verify': 在新环境中加载数据并对比")
    args = parser.parse_args()

    if args.mode == "save":
        if os.path.exists(DATA_DIR):
            print(f"Cleaning old data in {DATA_DIR}...")
            import shutil
            shutil.rmtree(DATA_DIR)
        os.makedirs(DATA_DIR)
        
        print(f"Running generation mode using Paddle {paddle.__version__}")
        main_gen_loops()
        print(f"Done. Data saved to {DATA_DIR}")
    
    elif args.mode == "verify":
        if not os.path.exists(DATA_DIR):
            print(f"Error: Directory {DATA_DIR} does not exist. Run --mode save first.")
            exit(1)
            
        print(f"Running verification mode using Paddle {paddle.__version__}")
        files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
        if not files:
            print("No data files found.")
            exit(1)
        
        for f in files:
            run_verify(f)
        print("Done verification.")