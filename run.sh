#!/bin/bash
# =======================================================
# test_flashmask.py on all cards via pytest-xdist.
#
#   bash run.sh                        # new run in parallel_logs/<time>
#   bash run.sh -k d256-dv256          # extra args go to pytest
#   GPUS="6" bash run.sh               # single card
#   bash run.sh resume [args]          # continue the latest run after a kill
#   bash run.sh rerun-failed [args]    # run the latest run's failed / error / crashed tests again
#   RUN_DIR=parallel_logs/<time> bash run.sh resume   # continue an older run
#
# Every run appends finished tests to $RUN_DIR/done.tsv (see conftest.py); only
# resume / rerun-failed read it back. Pass the same -k to resume as to the
# original run. Do not resume across edits to the parametrize lists: the node
# ids (e.g. sink_dtypeN) shift with them.
# =======================================================
cd "$(dirname "$0")"

# =========================
# 公共环境变量
# =========================
# export FLAGS_alloc_fill_value=255
# export FLAGS_use_system_allocator=1
# export FLAGS_check_cuda_error=1
# export FLAGS_cudnn_deterministic=1

export GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

mode=new
case "$1" in
    resume)       mode=resume; shift; export FM_TEST_RESUME=1 ;;
    rerun-failed) mode=rerun-failed; shift; export FM_TEST_RERUN_FAILED=1 ;;
esac

if [ "$mode" = new ]; then
    RUN_DIR="${RUN_DIR:-parallel_logs/$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "$RUN_DIR"
    ln -sfn "$(basename "$RUN_DIR")" parallel_logs/latest
else
    RUN_DIR="${RUN_DIR:-parallel_logs/latest}"
    if [ \! -f "$RUN_DIR/done.tsv" ]; then
        echo "Error: $RUN_DIR/done.tsv not found, nothing to $mode"
        exit 1
    fi
fi
export FM_TEST_LEDGER="$RUN_DIR/done.tsv"

read -ra gpus <<< "$GPUS"
num_workers=${#gpus[@]}
log="$RUN_DIR/test_$(date +%Y%m%d_%H%M%S).log"

done_count=0
[ -f "$FM_TEST_LEDGER" ] && done_count=$(wc -l < "$FM_TEST_LEDGER")
echo "mode: $mode  GPUS: ${gpus[*]}  workers: $num_workers  ledger lines: $done_count"
echo "ledger: $FM_TEST_LEDGER  log: $log"

# conftest.py replaces the scheduler: tests sharing a compile key run on one worker,
# so each kernel compiles once instead of once per worker; largest groups go first.
# --max-worker-restart: a worker killed by a CUDA fault is restarted, the run goes on.
python -m pytest -v test_flashmask.py \
    -n "$num_workers" --dist loadscope --max-worker-restart 1000 \
    "$@" 2>&1 | tee "$log"
exit "${PIPESTATUS[0]}"
