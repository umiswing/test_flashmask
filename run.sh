#!/bin/bash
# =======================================================
# 1. 单卡跑（默认模式）
#    默认 GPU6 跑
#       bash run.sh
#       bash run.sh single
# 2. 根据 d/dv拆分测试case，一张卡跑一组d/dv，多卡并行跑
#    现在有7组d/dv，默认 GPU1-7 跑
#       bash run.sh parallel
#       GPUS="1 2 3 4 5 6 7" bash run.sh parallel
# =======================================================

# =========================
# 公共环境变量
# =========================
export FLAGS_alloc_fill_value=255
export FLAGS_use_system_allocator=1
export FLAGS_check_cuda_error=1

# =========================
# 默认：单卡整文件测试
# =========================
if [ "$1" != "parallel" ]; then
    echo "Running ALL tests on single GPU"
    # 默认用卡6跑
    export CUDA_VISIBLE_DEVICES=6
    python -m pytest -v test_flashmask.py 2>&1 | tee test.log
    exit 0
fi

# =========================
# 并行模式
# =========================
echo "Running in PARALLEL mode"

# 独立日志目录
LOG_DIR="parallel_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "LOG_DIR: $LOG_DIR"

cases=(
  "d32-dv32"
  "d64-dv64"
  "d80-dv80"
  "d128-dv128"
  "d192-dv128"
  "d192-dv192"
  "d256-dv256"
)

# 默认 GPU（也可以外部覆盖）
if [ -z "$GPUS" ]; then
    gpus=(1 2 3 4 5 6 7)
else
    read -ra gpus <<< "$GPUS"
fi

echo "GPUS: ${gpus[@]} (count=${#gpus[@]})"

if [ ${#cases[@]} -ne ${#gpus[@]} ]; then
    echo "Error: cases 数量 (${#cases[@]}) != gpus 数量 (${#gpus[@]})"
    echo "cases: ${cases[@]}"
    echo "gpus : ${gpus[@]}"
    exit 1
fi

for i in ${!cases[@]}; do
    case=${cases[$i]}
    gpu=${gpus[$i]}

    echo "Launching case $case on GPU $gpu -> $LOG_DIR/test_${case}.log"

    (
        CUDA_VISIBLE_DEVICES=$gpu \
        python -m pytest -v test_flashmask.py \
        -k "$case" \
        > "$LOG_DIR/test_${case}.log" 2>&1
    ) &
done


wait

# =========================
# Summary
# =========================
summary_file="$LOG_DIR/summary.log"

# 解析单个 case 日志，把状态/各项数字写到标准输出
# 输出格式（单行，字段以 \t 分隔）:
#   status  passed  failed  skipped  errors  duration  summary_line
parse_case_log() {
    local log_path="$1"
    local status passed failed skipped errors duration summary_line

    if [ ! -f "$log_path" ]; then
        printf 'MISSING\t0\t0\t0\t0\t-\t(log missing)\n'
        return
    fi
    if [ ! -s "$log_path" ]; then
        printf 'EMPTY\t0\t0\t0\t0\t-\t(log empty; pytest probably failed to start)\n'
        return
    fi

    summary_line=$(grep -E '^=+ .*(passed|failed|error|no tests ran).* =+$' "$log_path" | tail -n 1)
    if [ -z "$summary_line" ]; then
        # 没有正常的 pytest summary 行，可能 collection error 等
        if grep -qiE 'ERROR|Traceback' "$log_path"; then
            printf 'ERROR\t0\t0\t0\t0\t-\t(no pytest summary; error detected)\n'
        else
            printf 'UNKNOWN\t0\t0\t0\t0\t-\t(no pytest summary found)\n'
        fi
        return
    fi

    extract() {
        local kw="$1"
        echo "$summary_line" | grep -oE "[0-9]+ ${kw}" | grep -oE '[0-9]+' | head -n 1
    }
    passed=$(extract passed);  [ -z "$passed" ]  && passed=0
    failed=$(extract failed);  [ -z "$failed" ]  && failed=0
    skipped=$(extract skipped);[ -z "$skipped" ] && skipped=0
    errors=$(extract error);   [ -z "$errors" ]  && errors=0

    duration=$(echo "$summary_line" | grep -oE 'in [0-9.]+s' | head -n 1 | awk '{print $2}')
    [ -z "$duration" ] && duration="-"

    if [ "$failed" -gt 0 ] || [ "$errors" -gt 0 ]; then
        status=FAIL
    elif [ "$passed" -eq 0 ] && [ "$skipped" -eq 0 ]; then
        status=NO_TEST
    else
        status=OK
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$status" "$passed" "$failed" "$skipped" "$errors" "$duration" "$summary_line"
}

# 先收集每个 case 的解析结果到数组
declare -a rows=()
total_passed=0
total_failed=0
total_skipped=0
total_errors=0
count_ok=0
count_fail=0
count_error=0
count_empty=0
count_missing=0
count_unknown=0
count_notest=0

for i in ${!cases[@]}; do
    case_name=${cases[$i]}
    gpu=${gpus[$i]}
    log_path="$LOG_DIR/test_${case_name}.log"

    parsed=$(parse_case_log "$log_path")
    status=$(echo "$parsed" | cut -f1)
    p=$(echo "$parsed" | cut -f2)
    f=$(echo "$parsed" | cut -f3)
    s=$(echo "$parsed" | cut -f4)
    e=$(echo "$parsed" | cut -f5)
    dur=$(echo "$parsed" | cut -f6)
    sline=$(echo "$parsed" | cut -f7-)

    rows+=("$case_name"$'\t'"$gpu"$'\t'"$status"$'\t'"$p"$'\t'"$f"$'\t'"$s"$'\t'"$e"$'\t'"$dur"$'\t'"$log_path"$'\t'"$sline")

    case "$status" in
        OK)      count_ok=$((count_ok+1)) ;;
        FAIL)    count_fail=$((count_fail+1)) ;;
        ERROR)   count_error=$((count_error+1)) ;;
        EMPTY)   count_empty=$((count_empty+1)) ;;
        MISSING) count_missing=$((count_missing+1)) ;;
        NO_TEST) count_notest=$((count_notest+1)) ;;
        *)       count_unknown=$((count_unknown+1)) ;;
    esac
    total_passed=$((total_passed + p))
    total_failed=$((total_failed + f))
    total_skipped=$((total_skipped + s))
    total_errors=$((total_errors + e))
done

# 生成 summary 文本（同时输出到终端和 summary.log）
{
    echo "==== Parallel run summary ===="
    echo "LOG_DIR: $LOG_DIR"
    echo ""
    printf '%-14s %-4s %-7s %-8s %-8s %-8s %-7s %-10s %s\n' \
        case gpu status passed failed skipped errors duration log
    printf '%-14s %-4s %-7s %-8s %-8s %-8s %-7s %-10s %s\n' \
        -------------- ---- ------- -------- -------- -------- ------- ---------- ---
    for row in "${rows[@]}"; do
        IFS=$'\t' read -r c g st p f s e dur lp sline <<< "$row"
        printf '%-14s %-4s %-7s %-8s %-8s %-8s %-7s %-10s %s\n' \
            "$c" "$g" "$st" "$p" "$f" "$s" "$e" "$dur" "$lp"
    done
    echo ""
    echo "Per-case pytest summary line:"
    for row in "${rows[@]}"; do
        IFS=$'\t' read -r c g st p f s e dur lp sline <<< "$row"
        echo "  [$c] $sline"
    done
    echo ""
    printf 'TOTAL: %d cases | OK=%d FAIL=%d ERROR=%d EMPTY=%d MISSING=%d NO_TEST=%d UNKNOWN=%d\n' \
        "${#cases[@]}" "$count_ok" "$count_fail" "$count_error" "$count_empty" "$count_missing" "$count_notest" "$count_unknown"
    printf '       passed=%d failed=%d skipped=%d errors=%d\n' \
        "$total_passed" "$total_failed" "$total_skipped" "$total_errors"

    abnormal=()
    for row in "${rows[@]}"; do
        IFS=$'\t' read -r c g st p f s e dur lp sline <<< "$row"
        case "$st" in
            OK) ;;
            *) abnormal+=("$c ($st) -> $lp") ;;
        esac
    done
    if [ ${#abnormal[@]} -gt 0 ]; then
        echo ""
        echo "Abnormal cases:"
        for line in "${abnormal[@]}"; do
            echo "  $line"
        done
    fi

    echo ""
    echo "All jobs finished"
} | tee "$summary_file"

echo ""
echo "Summary saved to: $summary_file"