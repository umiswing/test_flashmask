"""pytest-xdist GPU binding, compile-key scheduling and a resumable done-ledger. Driven by run.sh.

GPU binding: run.sh starts one xdist worker per card in $GPUS, and worker gw<i>
gets the i-th card. It has to run before paddle initialises CUDA; conftest is
imported before the test modules, and this file does not import paddle.

Scheduling: the cute kernels are compiled once per process per compile key, and
the key is set by (d, dv, kv_mode, nheads / nheads_kv, mask kind, sink dtype, ...),
not by the seqlens. _group_of maps a test_flashmask node id to (d, dv, kv_mode,
nheads / nheads_kv), and a custom xdist scheduler sends each group's tests to one
worker as a unit, so every key compiles once instead of once per worker. Mask kind
and sink dtype stay inside the group: several masks share a key (only causal and
the bound count matter), and sink_dtypeN in the id numbers the shape row, not the
dtype. The full suite has 96 groups (60 that are not all skips), enough for 8 cards. (xdist's own --dist loadgroup does not
work with pytest 9: it renames items through `_nodeid`, which pytest 9 ignores.)
Groups run largest first (estimated by b * h * sq * sk * (d + dv)) to keep the
tail short; the order is deterministic, so all workers collect the same list.

Ledger ($FM_TEST_LEDGER): the controller appends `outcome<TAB>nodeid<TAB>seconds`
once a test's teardown is reported, one flushed line per test, so a kill loses at
most the tests that were running. A test that took its worker down (CUDA fault)
is recorded as crashed. The ledger is only read back when asked for:
$FM_TEST_RESUME=1 deselects every node id in it, $FM_TEST_RERUN_FAILED=1 runs
only the failed / error / crashed ones.
"""
import os
import re

import pytest

_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _worker and os.environ.get("GPUS"):
    _gpus = os.environ["GPUS"].split()
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus[int(_worker[2:])]

_LEDGER = os.environ.get("FM_TEST_LEDGER")
_RESUME = os.environ.get("FM_TEST_RESUME") == "1"
_RERUN_FAILED = os.environ.get("FM_TEST_RERUN_FAILED") == "1"
_BAD = ("failed", "error", "crashed")

# test_flashmask ids: [gen_startend_row_indices<mask>-b-sq-sk-h-hkv-h_mask-sink_dtype<i>-
#                      d<d>-dv<dv>-<kv_mode>-<fa_version>-dtype<i>]
_ID_RE = re.compile(
    r"\[gen_startend_row_indices(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-sink_dtype(\d+)"
    r"-d(\d+)-dv(\d+)-(\w+)-(\d+)-dtype(\d+)\]$"
)


def _group_of(nodeid):
    """(compile group, estimated cost) of a node id; other tests are their own group."""
    m = _ID_RE.search(nodeid)
    if m is None:
        return nodeid, 0
    _, b, sq, sk, h, hkv, _, _, d, dv, kv_mode, fa, dtype = m.groups()
    group = f"d{d}-dv{dv}-{kv_mode}-qhpk{int(h) // int(hkv)}-fa{fa}-dtype{dtype}"
    return group, int(b) * int(h) * int(sq) * int(sk) * (int(d) + int(dv))


def _read_ledger():
    """nodeid -> (last recorded outcome, seconds)."""
    done = {}
    if _LEDGER and os.path.exists(_LEDGER):
        with open(_LEDGER) as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 2:
                    continue
                seconds = float(cols[2]) if len(cols) > 2 and cols[2] else 0.0
                done[cols[1]] = (cols[0], seconds)
    return done


def _is_controller(config):
    return not hasattr(config, "workerinput")


def pytest_xdist_make_scheduler(config, log):
    from xdist.scheduler import LoadScopeScheduling

    class CompileGroupScheduling(LoadScopeScheduling):
        def _split_scope(self, nodeid):
            return _group_of(nodeid)[0]

    return CompileGroupScheduling(config, log)


def pytest_collection_modifyitems(config, items):
    # Runs on every worker (and in a plain run). The controller only appends to the
    # ledger after all workers have collected, so they all read the same file.
    if _LEDGER and (_RESUME or _RERUN_FAILED):
        done = _read_ledger()
        if _RERUN_FAILED:
            bad = {k for k, v in done.items() if v[0] in _BAD}
            skip_it = lambda nodeid: nodeid not in bad
        else:
            skip_it = lambda nodeid: nodeid in done
        keep, skip = [], []
        for item in items:
            (skip if skip_it(item.nodeid) else keep).append(item)
        if skip:
            config.hook.pytest_deselected(items=skip)
            items[:] = keep

    # The scheduler hands out groups in collection order: largest groups first, tests
    # keep their order inside a group.
    group_cost, first_pos = {}, {}
    for i, item in enumerate(items):
        group, cost = _group_of(item.nodeid)
        group_cost[group] = group_cost.get(group, 0) + cost
        first_pos.setdefault(group, i)
    items.sort(key=lambda it: (
        -group_cost[_group_of(it.nodeid)[0]], first_pos[_group_of(it.nodeid)[0]]
    ))


_outcomes = {}
_seconds = {}
_ledger_file = None


def pytest_runtest_logreport(report):
    global _ledger_file
    if not _LEDGER:
        return
    # In an xdist run the controller sees every worker's reports; workers write nothing.
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    nodeid = report.nodeid
    prev = _outcomes.get(nodeid, "passed")
    if report.failed:
        cur = "failed" if report.when == "call" else "error"
    elif report.skipped and prev == "passed":
        cur = "skipped"
    else:
        cur = prev
    _outcomes[nodeid] = cur
    _seconds[nodeid] = _seconds.get(nodeid, 0.0) + (getattr(report, "duration", 0.0) or 0.0)
    # A crashed worker sends no teardown; xdist reports the test with when="???".
    if report.when not in ("setup", "call", "teardown") and report.failed:
        cur = "crashed"
    elif report.when != "teardown":
        return
    if _ledger_file is None:
        _ledger_file = open(_LEDGER, "a", buffering=1)
    _outcomes.pop(nodeid, None)
    _ledger_file.write(f"{cur}\t{nodeid}\t{_seconds.pop(nodeid, 0.0):.3f}\n")
    _ledger_file.flush()
    os.fsync(_ledger_file.fileno())


def pytest_terminal_summary(terminalreporter, config):
    if not _LEDGER or not _is_controller(config):
        return
    counts, group_seconds, group_tests = {}, {}, {}
    done = _read_ledger()
    for nodeid, (outcome, seconds) in done.items():
        counts[outcome] = counts.get(outcome, 0) + 1
        group = _group_of(nodeid)[0]
        group_seconds[group] = group_seconds.get(group, 0.0) + seconds
        group_tests[group] = group_tests.get(group, 0) + 1
    tr = terminalreporter
    tr.write_sep("=", f"ledger {_LEDGER} (all runs)")
    tr.write_line("  ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "empty")
    total = sum(s for _, s in done.values())
    tr.write_line(f"test time (sum over workers): {total / 3600:.2f} h")
    if group_seconds:
        tr.write_line("slowest groups (seconds, tests, group):")
        for group in sorted(group_seconds, key=group_seconds.get, reverse=True)[:10]:
            tr.write_line(f"  {group_seconds[group]:9.1f}  {group_tests[group]:5d}  {group}")
    bad = [n for n, v in done.items() if v[0] in _BAD]
    for nodeid in bad[:50]:
        tr.write_line(f"  {done[nodeid][0].upper()} {nodeid}")
    if len(bad) > 50:
        tr.write_line(f"  ... {len(bad) - 50} more, see {_LEDGER}")
