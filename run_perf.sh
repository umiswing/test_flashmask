#!/bin/bash
# =======================================================
# Submit the benchmarks to aq, one 1-GPU task per (series, head_dim, kv_mode).
#
#   bash run_perf.sh [series ...] [-- benchmark args]   # series: any of fm4 fm4_base fa4 cudnn (default fm4)
#   bash run_perf.sh status <run>                       # aq state + finished/expected CSVs per task
#   bash run_perf.sh resume <run>                       # resubmit the unfinished tasks of a run
#   bash run_perf.sh cancel <run>                       # cancel this run's queued / running tasks
#   bash run_perf.sh check [-- benchmark args]          # consistency check: one aq task per (backend, head_dim)
#   bash run_perf.sh check_result <id>                  # state of a check; compares once all its tasks are done
#
#   PY_FM / PY_FM_BASE / PY_FA4 / PY_CUDNN   python of each series, set below (env overrides)
#                                             fm4 / fm4_base / fa4 / cudnn
#   RUN         run id (default: now). Reuse one to add series to an existing run.
#   AQ_ARGS     extra aq submit args, e.g. "--any", "--at 02:00", "--for 8h"
#   HEAD_DIMS   only these head dims, e.g. "128 256"
#   FM_VERSION  4 (SM100, default) or 3 (SM90: fm3 / FA4 SM90 paths)
#   CHECK_HEAD_DIMS / CHECK_SEQLENS   check cases (default "128 192 256" / "4096"; SM90: "128 256"),
#                                     first sample of each seqlen only, all 12 masks
#   CHECK_BACKENDS   only these of flashmask flashmask_base fa4 cudnn (default: all with a python)
#   CHECK_ID         rerun into an existing check (replaces those backends' results there)
#   INTERFERE_GRACE  seconds another process may hold our GPU before the task moves (default 60)
#   HANDOFF_MAX      moves per task before it just waits for its GPU (default 3)
#
# A measurement another GPU process disturbed is discarded and redone once the GPU is
# ours again. If the process stays past INTERFERE_GRACE, the task hands off: from inside
# its aq task (still holding this GPU) it submits a successor, which aq places on
# another GPU or queues, and exits; the successor resumes with --no-overwrite.
#
# Every task runs all seqlens and samples of its head_dim in one process, so each
# kernel compiles once. Results: results/<run>/<series>/*.csv (+ meta.json); tasks are
# logged in results/<run>/jobs.tsv. Plot with plot_radar.py --series <dir> ...
# =======================================================
set -o pipefail
cd "$(dirname "$0")" || exit 1
REPO=$(pwd)
RESULTS="${RESULTS:-$REPO/results}"
FM_VERSION="${FM_VERSION:-4}"
INTERFERE_GRACE="${INTERFERE_GRACE:-60}"
HANDOFF_MAX="${HANDOFF_MAX:-3}"

# ---- python of each series: edit for your machine (an env var of the same name overrides) ----
PY_FM="${PY_FM:-python3}"                 # latest flashmask (paddle)
PY_FM_BASE="${PY_FM_BASE:-}"              # baseline flashmask (paddle), e.g. /path/fm_base/bin/python
PY_FA4="${PY_FA4:-}"                      # FA4 (flash_attn.cute, torch), e.g. /path/fa4/bin/python
PY_CUDNN="${PY_CUDNN:-}"                  # cuDNN Flex (torch), e.g. /path/cudnn/bin/python

SHARED=/root/paddlejob/share-storage/

die() { echo "error: $*" >&2; exit 1; }

# ---- task list: lines "name|python|script args" for one series ----
fm_configs() {
    if [ "$FM_VERSION" = "3" ]; then
        echo "256 split"; echo "128 split"; echo "64 split"
    else
        echo "576 split"; echo "576 shared"; echo "512 split"; echo "512 shared"
        echo "256 split"; echo "192 split"; echo "128 split"
    fi
}
competitor_dims() {
    if [ "$FM_VERSION" = "3" ]; then echo "256 128"; else echo "256 192 128"; fi
}
want_dim() { [ -z "$HEAD_DIMS" ] || [[ " $HEAD_DIMS " == *" $1 "* ]]; }

tasks_for() {
    local series=$1 py base D kv
    case "$series" in
        fm4|fm4_base)
            if [ "$series" = fm4 ]; then py=$PY_FM; base="fm$FM_VERSION"; else py=$PY_FM_BASE; base="fm${FM_VERSION}_base"; fi
            [ -n "$py" ] || die "series $series needs PY_FM_BASE (set it at the top of run_perf.sh)"
            while read -r D kv; do
                want_dim "$D" || continue
                local name="${base}_D$D"; [ "$kv" = shared ] && name="${name}_kvshared"
                echo "$name|$py|benchmark_flashmask.py --fm_version $FM_VERSION --head_dim $D --kv_mode $kv --series $base"
            done < <(fm_configs) ;;
        fa4|cudnn)
            if [ "$series" = fa4 ]; then py=$PY_FA4; else py=$PY_CUDNN; fi
            [ -n "$py" ] || die "series $series needs PY_$(echo "$series" | tr a-z A-Z) (set it at the top of run_perf.sh)"
            local script=benchmark_fa4_mask_mod.py
            [ "$series" = cudnn ] && script=benchmark_cudnn_flex_mask_mod.py
            for D in $(competitor_dims); do
                want_dim "$D" || continue
                echo "${series}_D$D|$py|$script --head_dim $D --series $series"
            done ;;
        *) die "unknown series '$series' (choose from fm4 fm4_base fa4 cudnn)" ;;
    esac
}

# ---- aq helpers ----
active_tasks() {   # ids of this user's queued / running tasks, one per line
    aq ls --mine --json 2>/dev/null | python3 -c '
import json, re, sys
text = sys.stdin.read()
try:
    data = json.loads(text)
except ValueError:
    data = None
ids = set()
def walk(x):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in ("id", "task_id") and isinstance(v, (int, str)):
                ids.add(str(v).replace("t-", ""))
            walk(v)
    elif isinstance(x, list):
        for v in x:
            walk(v)
walk(data) if data is not None else ids.update(re.findall(r"t-(\d+)", text))
print("\n".join(sorted(ids)))'
}

bench_cmd() {      # py args [--allow_handoff] -> the benchmark command line of a task
    local py=$1 args=$2 allow=$3
    echo "'$py' $args --current_time $RUN --results_dir '$RESULTS' --no-overwrite --dedup_static_masks --interfere_grace $INTERFERE_GRACE $allow ${EXTRA[*]}"
}

moves() {         # name -> handoffs since the task was last submitted or resumed by hand
    [ -f "$RUN_DIR/handoffs.tsv" ] || { echo 0; return; }
    awk -F'\t' -v n="$1" '$1 == n {c = ($2 == "handoff") ? c + 1 : 0} END {print c + 0}' "$RUN_DIR/handoffs.tsv"
}

submit_task() {    # name py args kind(start|handoff) -> appends to jobs.tsv; 1 if aq submit fails
    local name=$1 py=$2 args=$3 kind=${4:-start} out task log m allow=""
    m=0; [ "$kind" = handoff ] && m=$(( $(moves "$name") + 1 ))
    [ "$m" -lt "$HANDOFF_MAX" ] && allow=--allow_handoff
    # Exit 75: another process keeps this GPU -> hand the rest to a new aq task.
    local cmd="cd '$REPO' && $(bench_cmd "$py" "$args" "$allow"); rc=\$?; \
[ \$rc = 75 ] && exec env RESULTS='$RESULTS' INTERFERE_GRACE='$INTERFERE_GRACE' HANDOFF_MAX='$HANDOFF_MAX' \
bash '$REPO/run_perf.sh' _handoff '$RUN' '$name' '$py' '$args'; exit \$rc"
    # shellcheck disable=SC2086
    out=$(aq submit -g 1 $AQ_ARGS -- bash -c "$cmd" 2>&1 < /dev/null) || { echo "$out"; echo "error: aq submit failed for $name" >&2; return 1; }
    task=$(grep -o 't-[0-9]*' <<< "$out" | head -1)
    log=$(sed -n 's/.*日志: *//p' <<< "$out" | head -1)
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "${task#t-}" "${log:-?}" "$py" "$args" >> "$RUN_DIR/jobs.tsv"
    printf '%s\t%s\n' "$name" "$kind" >> "$RUN_DIR/handoffs.tsv"
    echo "  $name -> ${task:-?}  ${log}"
}

# ---- CSV progress of one task: "done expected" ----
progress() {   # series_dir D extra-args...
    local dir=$1 D=$2; shift 2
    python3 - "$dir" "$D" "$@" <<'PY'
import argparse, os, sys
sys.path.insert(0, os.getcwd())
import bench_common as bc
d, D = sys.argv[1], int(sys.argv[2])
p = argparse.ArgumentParser()
p.add_argument("--examples", nargs="+", default=["all"])
p.add_argument("--first_sample_only", action="store_true")
p.add_argument("--seqlens", type=int, nargs="+", default=None)
a, _ = p.parse_known_args(sys.argv[3:])
paths = [os.path.join(d, os.path.basename(bc.csv_path("", "", "", c))) for c in
         bc.iter_cases([D], a.examples, True, a.first_sample_only, a.seqlens)]
print(sum(os.path.exists(q) for q in paths), len(paths))
PY
}

task_series_dir() {   # name args -> results/<run>/<series dir>
    local name=$1 args=$2 series
    series=$(sed -n 's/.*--series \([^ ]*\).*/\1/p' <<< "$args")
    [[ "$name" == *_kvshared ]] && series="${series}_kvshared"
    echo "$RUN_DIR/$series"
}

latest_jobs() {   # last jobs.tsv line per task name, in submit order
    awk -F'\t' '{last[$1] = $0; if (!($1 in seen)) {seen[$1] = 1; order[n++] = $1}}
                END {for (i = 0; i < n; i++) print last[order[i]]}' "$RUN_DIR/jobs.tsv"
}

load_run() {
    RUN=$1
    [ -n "$RUN" ] || die "give the run id (a directory under $RESULTS)"
    RUN_DIR="$RESULTS/$RUN"
    [ -f "$RUN_DIR/jobs.tsv" ] || die "$RUN_DIR/jobs.tsv not found"
    EXTRA=()
    [ -f "$RUN_DIR/extra_args" ] && read -ra EXTRA < "$RUN_DIR/extra_args"
}

check_shared_paths() {   # --any sends tasks to other machines: everything must be on shared storage
    [[ " $AQ_ARGS " == *" --any "* ]] || return 0
    local p
    for p in "$REPO" "$RESULTS"; do
        [[ "$p" == "$SHARED"* ]] || die "--any needs $p on $SHARED (other machines cannot see it)"
    done
    for p in "$@"; do
        [[ "$(command -v "$p")" == "$SHARED"* ]] || echo "warning: $p is not on $SHARED; other machines must have the same python" >&2
    done
}

print_plot_hint() {
    local dirs=()
    for d in "$RUN_DIR"/*/; do [ -f "$d/meta.json" ] && dirs+=("${d%/}"); done
    [ ${#dirs[@]} -gt 0 ] && echo "plot: python plot_radar.py --series ${dirs[*]}"
}

cmd=submit
case "$1" in status|resume|cancel|check|check_result|_handoff) cmd=$1; shift ;; esac

case "$cmd" in
submit)
    series=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do
        [[ " ${series[*]} " == *" $1 "* ]] || series+=("$1"); shift
    done
    [ "$1" = "--" ] && shift
    EXTRA=("$@")
    [ ${#series[@]} -eq 0 ] && series=(fm4)
    jobs=()
    for s in "${series[@]}"; do
        out=$(tasks_for "$s") || exit 1
        while IFS= read -r line; do [ -n "$line" ] && jobs+=("$line"); done <<< "$out"
    done
    [ ${#jobs[@]} -gt 0 ] || die "no task left (HEAD_DIMS='$HEAD_DIMS')"
    # Biggest head dims first, across series, so the long tasks get GPUs first.
    mapfile -t jobs < <(printf '%s\n' "${jobs[@]}" | awk -F'|' '{match($1, /_D[0-9]+/); print substr($1, RSTART + 2, RLENGTH - 2) "\t" $0}' \
                         | sort -t$'\t' -k1,1nr -s | cut -f2-)
    pys=(); for j in "${jobs[@]}"; do IFS='|' read -r _ py _ <<< "$j"; pys+=("$py"); done
    check_shared_paths "${pys[@]}"

    RUN="${RUN:-$(date +%Y%m%d_%H%M%S)}"
    RUN_DIR="$RESULTS/$RUN"
    mkdir -p "$RUN_DIR"
    if [ -f "$RUN_DIR/extra_args" ]; then
        [ "$(cat "$RUN_DIR/extra_args")" = "${EXTRA[*]}" ] || \
            die "run $RUN was submitted with benchmark args '$(cat "$RUN_DIR/extra_args")', not '${EXTRA[*]}'"
    else
        echo "${EXTRA[*]}" > "$RUN_DIR/extra_args"
    fi
    echo "$AQ_ARGS" > "$RUN_DIR/aq_args"
    echo "run $RUN: ${#jobs[@]} task(s) -> $RUN_DIR"
    for j in "${jobs[@]}"; do
        IFS='|' read -r name py args <<< "$j"
        submit_task "$name" "$py" "$args" || exit 1
    done
    echo "status: bash run_perf.sh status $RUN"
    ;;
status)
    load_run "$1"
    mapfile -t active < <(active_tasks)
    printf '%-24s %-8s %-9s %-9s %-6s %s\n' task aq state csv moves log
    while IFS=$'\t' read -r name task log py args; do
        D=$(grep -o -- '--head_dim [0-9]*' <<< "$args" | cut -d' ' -f2)
        read -r ok total < <(progress "$(task_series_dir "$name" "$args")" "$D" "${EXTRA[@]}")
        if [ "$ok" = "$total" ]; then state=done
        elif [[ " ${active[*]} " == *" $task "* ]]; then state=active
        else state=stopped; fi
        printf '%-24s %-8s %-9s %-9s %-6s %s\n' "$name" "t-$task" "$state" "$ok/$total" "$(moves "$name")" "$log"
    done < <(latest_jobs)
    echo "moves = hand-offs to another GPU because another process kept this one"
    echo "stopped = no longer in aq and CSVs missing: bash run_perf.sh resume $RUN"
    print_plot_hint
    ;;
resume)
    load_run "$1"
    echo "$AQ_ARGS" > "$RUN_DIR/aq_args"
    mapfile -t active < <(active_tasks)
    n=0
    while IFS=$'\t' read -r name task log py args; do
        D=$(grep -o -- '--head_dim [0-9]*' <<< "$args" | cut -d' ' -f2)
        read -r ok total < <(progress "$(task_series_dir "$name" "$args")" "$D" "${EXTRA[@]}")
        [ "$ok" = "$total" ] && continue
        [[ " ${active[*]} " == *" $task "* ]] && { echo "  $name still in aq (t-$task)"; continue; }
        check_shared_paths "$py"
        submit_task "$name" "$py" "$args" || exit 1; n=$((n + 1))
    done < <(latest_jobs)
    echo "resubmitted $n task(s)"
    ;;
cancel)
    load_run "$1"
    mapfile -t active < <(active_tasks)
    while IFS=$'\t' read -r name task _; do
        [[ " ${active[*]} " == *" $task "* ]] && { echo "  cancel $name (t-$task)"; aq cancel "$task" < /dev/null; }
    done < <(latest_jobs)
    ;;
_handoff)   # run name py args: called inside a task whose GPU another process keeps (exit 75)
    load_run "$1"; name=$2; py=$3; args=$4
    # Of the original aq args only --any and --until still apply; --at / --in / --for were for the first start.
    AQ_ARGS=""; a=(); [ -f "$RUN_DIR/aq_args" ] && read -ra a < "$RUN_DIR/aq_args"
    for ((i = 0; i < ${#a[@]}; i++)); do
        case "${a[i]}" in
            --any) AQ_ARGS+=" --any" ;;
            --until) AQ_ARGS+=" --until ${a[i + 1]}"; i=$((i + 1)) ;;
        esac
    done
    echo "[handoff] $name: another process keeps this GPU, submitting a successor (move $(( $(moves "$name") + 1 ))/$HANDOFF_MAX)"
    # This task still holds its GPU, so aq gives the successor another one (or queues it).
    if submit_task "$name" "$py" "$args" handoff; then
        exit 0
    fi
    echo "[handoff] $name: aq submit failed, waiting for this GPU instead"
    cd "$REPO" && eval "exec $(bench_cmd "$py" "$args" "")"
    ;;
check)
    # One 1-GPU task per (backend, head_dim), all writing --check_masks jsonl into one
    # directory; the numpy reference is shared through its acc_ref/ (file-locked).
    # The check does not time anything, so other processes on the GPU do not matter.
    [ "$1" = "--" ] && shift
    if [ "$FM_VERSION" = "3" ]; then dims="${CHECK_HEAD_DIMS:-128 256}"; else dims="${CHECK_HEAD_DIMS:-128 192 256}"; fi
    seqlens="${CHECK_SEQLENS:-4096}"
    id="${CHECK_ID:-$(date +%Y%m%d_%H%M%S)}"
    OUT="$REPO/check_masks/$id"
    [ -z "$CHECK_ID" ] || [ -f "$OUT/jobs.tsv" ] || die "CHECK_ID=$CHECK_ID: $OUT/jobs.tsv not found"
    mkdir -p "$OUT"
    check_shared_paths "$PY_FM" $PY_FM_BASE $PY_FA4 $PY_CUDNN
    n=0
    for spec in "flashmask|$PY_FM|benchmark_flashmask.py --fm_version $FM_VERSION --kv_mode split" \
                "flashmask_base|$PY_FM_BASE|benchmark_flashmask.py --fm_version $FM_VERSION --kv_mode split" \
                "fa4|$PY_FA4|benchmark_fa4_mask_mod.py" \
                "cudnn|$PY_CUDNN|benchmark_cudnn_flex_mask_mod.py"; do
        IFS='|' read -r backend py script <<< "$spec"
        [ -z "$CHECK_BACKENDS" ] || [[ " $CHECK_BACKENDS " == *" $backend "* ]] || continue
        [ -n "$py" ] || { echo "  skip $backend (no python set)"; continue; }
        for D in $dims; do
            f="$OUT/${backend}_D$D"
            # Rerun into CHECK_ID: drop this task's old results and jobs.tsv line.
            rm -f "$f.jsonl" "$f.log" "$f.done"
            [ -f "$OUT/jobs.tsv" ] && { grep -v "^${backend}_D$D"$'\t' "$OUT/jobs.tsv" > "$OUT/jobs.tsv.tmp"; mv "$OUT/jobs.tsv.tmp" "$OUT/jobs.tsv"; }
            cmd="cd '$REPO' && '$py' $script --head_dim $D --seqlens $seqlens --first_sample_only --check_masks '$f.jsonl' $* > '$f.log' 2>&1 && touch '$f.done'"
            # shellcheck disable=SC2086
            out=$(aq submit -g 1 $AQ_ARGS -- bash -c "$cmd" 2>&1 < /dev/null) || { echo "$out"; die "aq submit failed for $backend D=$D"; }
            task=$(grep -o 't-[0-9]*' <<< "$out" | head -1)
            printf '%s\t%s\t%s\n' "${backend}_D$D" "${task#t-}" "$f.log" >> "$OUT/jobs.tsv"
            echo "  ${backend}_D$D -> ${task:-?}  $f.log"; n=$((n + 1))
        done
    done
    [ "$n" -gt 0 ] || die "no backend to check (PY_FM is required)"
    echo "check $id: $n task(s) -> $OUT"
    echo "result: bash run_perf.sh check_result $id"
    ;;
check_result)
    id=$1; OUT="$REPO/check_masks/$id"
    [ -f "$OUT/jobs.tsv" ] || die "$OUT/jobs.tsv not found"
    mapfile -t active < <(active_tasks)
    pending=0; failed=0
    printf '%-22s %-8s %-8s %s\n' task aq state log
    while IFS=$'\t' read -r name task log; do
        if [ -f "$OUT/$name.done" ]; then state=done
        elif [[ " ${active[*]} " == *" $task "* ]]; then state=active; pending=1
        else state=failed; failed=1; fi
        printf '%-22s %-8s %-8s %s\n' "$name" "t-$task" "$state" "$log"
    done < "$OUT/jobs.tsv"
    [ "$pending" = 1 ] && { echo "still running, ask again later"; exit 0; }
    [ "$failed" = 1 ] && die "some tasks failed (no .done): see their logs; rerun bash run_perf.sh check"
    # flashmask first: it is the reference of the comparison.
    files=()
    for backend in flashmask flashmask_base fa4 cudnn; do
        parts=("$OUT/${backend}"_D*.jsonl)
        [ -f "${parts[0]}" ] || continue
        cat "${parts[@]}" > "$OUT/$backend.jsonl"
        files+=("$OUT/$backend.jsonl")
    done
    [ "${files[0]}" = "$OUT/flashmask.jsonl" ] || die "no flashmask result to compare against"
    python3 check_bench_consistency.py "${files[@]}"
    ;;
esac
