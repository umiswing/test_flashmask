#!/bin/bash
# =======================================================
# Bitwise determinism (aadiff) of flashmask fwd out + bwd dq / dk / dv, as aq tasks.
#
#   bash run_aadiff.sh                 # compare with the committed flashmask_gt_fa{N}.json
#   REGEN_GT=1 bash run_aadiff.sh      # precision changed on purpose: record a new GT per shard,
#                                      # compare against it, and (all shards OK) write it back
#   bash run_aadiff.sh status <id>     # per shard: done / active / failed, log
#   bash run_aadiff.sh result <id>     # summary; exit 1 on any failure
#
#   PY        python with flashmask (default python3)
#   AQ_ARGS   extra aq submit args, e.g. "--any"
#   FA_VERSION  4 (SM100, default) or 3 (SM90): picks the shards and the GT file
#
# One 1-GPU task per (d, dv, kv_mode) shard. A case missing from the GT fails. Results in
# aadiff/<id>/: <shard>.log (pytest), <shard>.xml (junit), <shard>.rc, and with REGEN_GT=1
# <shard>.record.log and gt_<shard>.json.
# =======================================================
set -o pipefail
cd "$(dirname "$0")" || exit 1
REPO=$(pwd)
PY="${PY:-python3}"
FA_VERSION="${FA_VERSION:-4}"
GT="$REPO/flashmask_gt_fa$FA_VERSION.json"

die() { echo "error: $*" >&2; exit 1; }

shards() {   # d,dv,kv_mode (same lists as test_md5sum.py)
    local s
    for s in 64,64 80,80 128,128 192,128 256,256; do echo "$s,split"; done
    if [ "$FA_VERSION" = "4" ]; then
        echo "512,512,split"; echo "576,512,split"; echo "512,512,shared"; echo "576,512,shared"
    fi
}

active_tasks() {   # ids of this user's queued / running tasks
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

load() {
    id=$1; OUT="$REPO/aadiff/$id"
    [ -n "$id" ] && [ -f "$OUT/jobs.tsv" ] || die "no aadiff run '$id' ($OUT/jobs.tsv)"
}

cmd=submit
case "$1" in status|result) cmd=$1; shift ;; esac

case "$cmd" in
submit)
    [ -f "$GT" ] || [ "$REGEN_GT" = 1 ] || die "$GT not found; REGEN_GT=1 records one"
    id=$(date +%Y%m%d_%H%M%S)
    OUT="$REPO/aadiff/$id"
    mkdir -p "$OUT"
    mode=check; [ "$REGEN_GT" = 1 ] && mode=regen
    echo "$mode" > "$OUT/mode"
    # The GT is recorded and checked under the same flags (see AADIFF_FLAGS in test_md5sum.py).
    flags="export FLAGS_cudnn_deterministic=1 FLAGS_alloc_fill_value=255 FLAGS_use_system_allocator=1 FLAGS_check_cuda_error=1"
    for s in $(shards); do
        n=${s//,/_}; f="$OUT/$n"
        gt="$GT"; record=""
        if [ "$mode" = regen ]; then
            gt="$OUT/gt_$n.json"
            record="AADIFF_SHARD=$s '$PY' test_md5sum.py record --out '$gt' > '$f.record.log' 2>&1 || { echo 2 > '$f.rc'; touch '$f.done'; exit 1; }; "
        fi
        cmd="cd '$REPO' && $flags && ${record}AADIFF_SHARD=$s AADIFF_GT='$gt' '$PY' -m pytest test_md5sum.py -q -p no:cacheprovider --junitxml '$f.xml' > '$f.log' 2>&1; rc=\$?; echo \$rc > '$f.rc'; touch '$f.done'; exit \$rc"
        # shellcheck disable=SC2086
        out=$(aq submit -g 1 $AQ_ARGS -- bash -c "$cmd" 2>&1 < /dev/null) || { echo "$out"; die "aq submit failed for $s"; }
        task=$(grep -o 't-[0-9]*' <<< "$out" | head -1)
        printf '%s\t%s\t%s\n' "$n" "${task#t-}" "$f.log" >> "$OUT/jobs.tsv"
        echo "  $n -> ${task:-?}  $f.log"
    done
    echo "aadiff $id ($mode, $(wc -l < "$OUT/jobs.tsv") shards) -> $OUT"
    echo "result: bash run_aadiff.sh result $id"
    ;;
status|result)
    load "$1"
    mode=$(cat "$OUT/mode")
    mapfile -t active < <(active_tasks)
    pending=0; failed=0
    printf '%-20s %-8s %-8s %s\n' shard aq state log
    while IFS=$'\t' read -r n task log; do
        if [ -f "$OUT/$n.done" ]; then
            rc=$(cat "$OUT/$n.rc" 2>/dev/null || echo 1)
            if [ "$rc" = 0 ]; then state=ok; elif [ "$rc" = 2 ]; then state=record_failed; failed=1; else state=FAIL; failed=1; fi
        elif [[ " ${active[*]} " == *" $task "* ]]; then state=active; pending=1
        else state=lost; failed=1; fi
        printf '%-20s %-8s %-8s %s\n' "$n" "t-$task" "$state" "$log"
    done < "$OUT/jobs.tsv"
    [ "$cmd" = status ] && exit 0
    [ "$pending" = 1 ] && { echo "still running, ask again later"; exit 0; }
    python3 - "$OUT" <<'PY'
import glob, os, sys
import xml.etree.ElementTree as ET
out = sys.argv[1]
tot = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
fails = []
for x in sorted(glob.glob(os.path.join(out, "*.xml"))):
    for suite in ET.parse(x).getroot().iter("testsuite"):
        for k in tot:
            tot[k] += int(suite.get(k, 0))
        for case in suite.iter("testcase"):
            for bad in list(case.findall("failure")) + list(case.findall("error")):
                fails.append((case.get("name"), (bad.get("message") or "").strip()))
print(f"tests={tot['tests']} passed={tot['tests'] - tot['failures'] - tot['errors'] - tot['skipped']} "
      f"failed={tot['failures']} errors={tot['errors']} skipped={tot['skipped']}")
for name, msg in fails[:40]:
    print(f"  FAIL {name}\n    " + msg.replace("\n", "\n    ")[:600])
if len(fails) > 40:
    print(f"  ... {len(fails) - 40} more, see {out}/*.log")
PY
    for n in $(cut -f1 "$OUT/jobs.tsv"); do
        [ -f "$OUT/$n.record.log" ] && [ "$(cat "$OUT/$n.rc" 2>/dev/null)" = 2 ] && \
            { echo "  RECORD FAILED $n:"; tail -5 "$OUT/$n.record.log" | sed 's/^/    /'; }
    done
    [ "$failed" = 1 ] && die "FAIL (aadiff $id)"
    if [ "$mode" = regen ]; then
        # Every shard recorded and re-checked: merge the shard GTs and replace the committed one.
        python3 - "$OUT" "$GT" <<'PY'
import glob, json, os, shutil, sys
out, gt = sys.argv[1], sys.argv[2]
merged, metas = {}, []
for f in sorted(glob.glob(os.path.join(out, "gt_*.json"))):
    with open(f) as fh:
        d = json.load(fh)
    metas.append(d.pop("gt_meta", {}))
    merged.update(d)
meta = dict(metas[0]) if metas else {}
meta["shard"] = None
meta["shards"] = [m.get("shard") for m in metas]
merged["gt_meta"] = meta
backup = os.path.exists(gt)
if backup:
    shutil.copy2(gt, gt + ".bak")
tmp = gt + ".tmp"
with open(tmp, "w") as fh:
    json.dump(merged, fh, indent=2)
os.replace(tmp, gt)
print(f"wrote {gt}: {len(merged) - 1} cases" + (f" (old file: {gt}.bak)" if backup else ""))
PY
    fi
    echo "OK (aadiff $id, $mode)"
    ;;
esac
