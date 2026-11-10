#!/bin/bash
# =======================================================
# Check that the three competitor benchmarks measure the same masks.
# Each backend runs in its own python (different cutedsl versions):
#
#   PY_FM=/path/to/paddle/python PY_FA4=/path/to/fa4/python PY_CUDNN=/path/to/cudnn/python \
#       bash check_bench_consistency.sh
#   (PY_FM_BASE adds the baseline flashmask; FM_VERSION=3 for SM90.)
#
#   SEQLENS="4096 8192" HEAD_DIMS="128" bash check_bench_consistency.sh   # other cases
#
# Default: the first sample of S=4096 (32_4096_*_0, all 12 masks) for D = 128, 192, 256.
# The masks are built by the same code at every S and sample (only the layouts in
# kernel_test_seq_info.txt differ, and those are checked at load time), so one case per D
# covers the mask logic.
# Besides the mask fingerprints, every backend runs its timed fwd / bwd once (B=2, H=2,
# NaN-poisoned outputs) and is compared with one numpy reference (S <= 8192 only),
# computed by the first backend on the CPU and cached in $OUT/acc_ref/ for the others.
# bash run_perf.sh check runs the same check as one aq task per (backend, head_dim).
# Extra args go to all benchmarks. PY_FM is required (flashmask is the reference);
# an unset PY_FM_BASE / PY_FA4 / PY_CUDNN skips that backend.
# =======================================================
cd "$(dirname "$0")"
SEQLENS="${SEQLENS:-4096}"
HEAD_DIMS="${HEAD_DIMS:-128 192 256}"
OUT="${OUT:-check_masks/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ -z "$PY_FM" ]; then
    echo "PY_FM is required (flashmask is the reference)"
    exit 1
fi

# backend name, python, script, script-specific args
run_backend() {
    local name=$1 py=$2 script=$3; shift 3
    [ -z "$py" ] && { echo "skip $name (no python given)"; return 0; }
    : > "$OUT/$name.jsonl"
    for D in $HEAD_DIMS; do
        echo "$name D=$D S=[$SEQLENS] ..."
        "$py" "$script" "$@" --head_dim "$D" --seqlens $SEQLENS \
            --first_sample_only --check_masks "$OUT/${name}_D$D.jsonl" "${EXTRA[@]}" > "$OUT/${name}_D$D.log" 2>&1 \
            || { echo "$name D=$D failed, see $OUT/${name}_D$D.log"; exit 1; }
        cat "$OUT/${name}_D$D.jsonl" >> "$OUT/$name.jsonl"
    done
    files+=("$OUT/$name.jsonl")
}

EXTRA=("$@")
files=()
run_backend flashmask "$PY_FM" benchmark_flashmask.py --fm_version "${FM_VERSION:-4}" --kv_mode split
run_backend flashmask_base "$PY_FM_BASE" benchmark_flashmask.py --fm_version "${FM_VERSION:-4}" --kv_mode split
run_backend fa4 "$PY_FA4" benchmark_fa4_mask_mod.py
run_backend cudnn "$PY_CUDNN" benchmark_cudnn_flex_mask_mod.py
"$PY_FM" check_bench_consistency.py "${files[@]}"
