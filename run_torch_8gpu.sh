#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  Multi-GPU pytest runner with persistent split + resume
#
#  特性:
#    1) split 只生成一次
#    2) 如果 flashmask_split_ 存在则直接复用
#    3) resume 通过统计 log 已完成数量
# ============================================================

NUM_GPUS=${NUM_GPUS:-8}
TEST_FILE="test_flashmask_torch.py"
LOG_DIR="logs"
SPLIT_DIR="./flashmask_split_"

mkdir -p "${LOG_DIR}"
mkdir -p "${SPLIT_DIR}"

echo ""
echo "========================================"
echo " Multi-GPU Pytest Runner"
echo " GPUs      : ${NUM_GPUS}"
echo " Test file : ${TEST_FILE}"
echo " Split dir : ${SPLIT_DIR}"
echo "========================================"
echo ""

# ──────────────────────────────────────
# Step 1: 如果没有 split 文件，则生成
# ──────────────────────────────────────

need_generate=false

for (( gpu=0; gpu<NUM_GPUS; gpu++ )); do
    if [ ! -f "${SPLIT_DIR}/gpu_${gpu}.txt" ]; then
        need_generate=true
        break
    fi
done

if [ "${need_generate}" = true ]; then
    echo "[INFO] No existing split found. Generating..."

    python3 -m pytest --collect-only -q "${TEST_FILE}" 2>/dev/null \
        | grep '::' \
        > "${SPLIT_DIR}/all_tests.txt" || true

    TOTAL=$(wc -l < "${SPLIT_DIR}/all_tests.txt")
    echo "[INFO] Total test cases: ${TOTAL}"

    if [ "${TOTAL}" -eq 0 ]; then
        echo "[ERROR] No test cases collected."
        exit 1
    fi

    # 初始化 GPU 文件
    for (( gpu=0; gpu<NUM_GPUS; gpu++ )); do
        > "${SPLIT_DIR}/gpu_${gpu}.txt"
    done

    idx=0
    while IFS= read -r line; do
        gpu=$(( idx % NUM_GPUS ))
        echo "${line}" >> "${SPLIT_DIR}/gpu_${gpu}.txt"
        idx=$(( idx + 1 ))
    done < "${SPLIT_DIR}/all_tests.txt"

    echo "[INFO] Split completed:"
    for (( gpu=0; gpu<NUM_GPUS; gpu++ )); do
        cnt=$(wc -l < "${SPLIT_DIR}/gpu_${gpu}.txt")
        echo "  GPU ${gpu}: ${cnt} test cases"
    done
else
    echo "[INFO] Existing split detected. Reusing."
fi

# ──────────────────────────────────────
# Step 2: 并行执行 + resume
# ──────────────────────────────────────

pids=()

for (( gpu=0; gpu<NUM_GPUS; gpu++ )); do

    orig_list="${SPLIT_DIR}/gpu_${gpu}.txt"
    log_file="${LOG_DIR}/torch_gpu_${gpu}.log"

    if [ ! -f "${orig_list}" ]; then
        echo "[INFO] GPU ${gpu}: no split file"
        continue
    fi

    total_count=$(wc -l < "${orig_list}")
    if [ "${total_count}" -eq 0 ]; then
        echo "[INFO] GPU ${gpu}: empty split"
        continue
    fi

    done_count=0
    if [ -f "${log_file}" ]; then
        done_count=$(grep -E "^[^ ]+::.* (PASSED|FAILED|ERROR|SKIPPED)" "${log_file}" | wc -l || echo 0)
    fi

    if [ "${done_count}" -ge "${total_count}" ]; then
        echo "[INFO] GPU ${gpu}: all tests already done"
        continue
    fi

    echo "[INFO] GPU ${gpu}: resume from $((done_count + 1)) / ${total_count}"

    remain_list="${SPLIT_DIR}/gpu_${gpu}_remain.txt"
    tail -n +$((done_count + 1)) "${orig_list}" > "${remain_list}"

    test_args=$(tr '\n' ' ' < "${remain_list}")

    (
        export CUDA_VISIBLE_DEVICES="${gpu}"

        if [ -f "${log_file}" ]; then
            echo "===== RESUME $(date) =====" >> "${log_file}"
        else
            echo "===== START $(date) =====" > "${log_file}"
        fi

        python -m pytest -v --tb=short ${test_args} >> "${log_file}" 2>&1 || true

        echo "" >> "${log_file}"
        echo "===== GPU ${gpu}: done =====" >> "${log_file}"
    ) &

    pids+=($!)
    echo "[INFO] GPU ${gpu}: started (PID $!), log -> ${log_file}"
done

# ──────────────────────────────────────
# Step 3: 等待
# ──────────────────────────────────────

echo ""
echo "[INFO] Waiting for all GPU processes..."

for pid in "${pids[@]}"; do
    wait "${pid}"
done

# ──────────────────────────────────────
# Step 4: 汇总
# ──────────────────────────────────────

echo ""
echo "========================================"
echo "  All GPU processes finished."
echo "========================================"

for (( gpu=0; gpu<NUM_GPUS; gpu++ )); do
    log_file="${LOG_DIR}/torch_gpu_${gpu}.log"
    if [ -f "${log_file}" ]; then
        p=$(grep -c ' PASSED' "${log_file}" 2>/dev/null || echo 0)
        f=$(grep -c ' FAILED' "${log_file}" 2>/dev/null || echo 0)
        s=$(grep -c ' SKIPPED' "${log_file}" 2>/dev/null || echo 0)
        e=$(grep -c ' ERROR' "${log_file}" 2>/dev/null || echo 0)
        printf "  GPU %d: %4d passed, %4d failed, %4d skipped, %4d errors\n" \
            "${gpu}" "${p}" "${f}" "${s}" "${e}"
    fi
done

echo ""
echo "[INFO] Done."
