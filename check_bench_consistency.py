"""Check that benchmark_flashmask / fa4_mask_mod / cudnn_flex_mask_mod measure the same thing.

Reads the --check_masks jsonl of each backend and compares, per (S, sample, D, mask):
the mask fingerprint (visible queries per key and keys per query), the reported
sparsity, the shapes and the iteration counts. flashmask is the reference; a key a
competitor did not run (unsupported head_dim, mask_func too large) is listed, not failed.

It also checks every backend's output check ("acc"): the timed fwd / bwd run once on
shared inputs against one numpy reference (bench_common.run_accuracy). A failed or
crashed run fails; a record without one (S > ACC_MAX_S, not run) is listed.

    python check_bench_consistency.py flashmask.jsonl fa4.jsonl cudnn.jsonl
"""
import json
import sys

FIELDS = ("B", "H", "HKV", "DV", "causal", "sparsity", "iters", "fingerprint")


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            # flashmask writes one record per kv_mode; the mask does not depend on it.
            if r.get("kv_mode", "split") != "split":
                continue
            out[(r["S"], r["idx"], r["D"], r["mask"])] = r
    return out


def fmt(key):
    return f"S={key[0]} sample={key[1]} D={key[2]} {key[3]}"


def acc_failure(acc):
    """Why an output check failed, or None if it passed."""
    if acc.get("ok"):
        return None
    if "error" in acc:
        return acc["error"]
    parts = []
    for name in ("out", "dq", "dk", "dv"):
        t = acc.get(name, {})
        if t.get("ok", True):
            continue
        if "error" in t:
            parts.append(f"{name}: {t['error']}")
        elif t.get("nonfinite"):
            parts.append(f"{name}: {t['nonfinite']} NaN/inf (unwritten?)")
        else:
            parts.append(f"{name}: err {t['err']:.3g} > tol {t['tol']:.3g}")
    return "; ".join(parts)


def check_acc(path, recs):
    failed, no_acc = [], []
    for key, r in sorted(recs.items()):
        acc = r.get("acc")
        if acc is None:
            if r["fingerprint"] is not None:
                no_acc.append(key)
            continue
        why = acc_failure(acc)
        if why:
            failed.append((key, why))
    backend = next(iter(recs.values()))["backend"] if recs else path
    print(f"== {backend} outputs ({path}): {len(recs) - len(no_acc)} checked, "
          f"{len(failed)} failed, {len(no_acc)} without output check")
    for key, why in failed[:30]:
        print(f"  ACC FAIL {fmt(key)}: {why}")
    for key in no_acc[:5]:
        print(f"  no acc   {fmt(key)}")
    return bool(failed)


def main(paths):
    ref_path, others = paths[0], paths[1:]
    ref = load(ref_path)
    failed = False
    for path in others:
        got = load(path)
        backend = next(iter(got.values()))["backend"] if got else path
        mismatches, missing, not_run = [], [], []
        for key, r in sorted(got.items()):
            if key not in ref:
                mismatches.append((key, "not in the reference"))
                continue
            if r["fingerprint"] is None:
                not_run.append(key)
                continue
            diff = [f for f in FIELDS if r[f] != ref[key][f]]
            if diff:
                mismatches.append((key, ", ".join(diff)))
        missing = sorted(k for k in ref if k not in got and k[2] in {k2[2] for k2 in got})
        print(f"== {backend} ({path}) vs {ref[next(iter(ref))]['backend']}: "
              f"{len(got)} records, {len(mismatches)} mismatches, "
              f"{len(not_run)} not run, {len(missing)} missing")
        for key, what in mismatches[:30]:
            print(f"  MISMATCH {fmt(key)}: {what}")
        for key in not_run[:10]:
            print(f"  not run  {fmt(key)}")
        for key in missing[:10]:
            print(f"  missing  {fmt(key)}")
        failed |= bool(mismatches or missing)
    for path in paths:
        failed |= check_acc(path, ref if path == ref_path else load(path))
    print("FAIL" if failed else "OK: every backend measures the same masks, shapes and iteration "
          "counts, and its outputs match the reference")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
