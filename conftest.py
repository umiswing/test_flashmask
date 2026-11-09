"""pytest-xdist GPU binding and a resumable done-ledger. Driven by run.sh.

GPU binding: each xdist worker (gw0, gw1, ...) gets one card from $GPUS, round
robin, so `-n 2*len(GPUS)` puts two workers on every card. It has to run before
paddle initialises CUDA; conftest is imported before the test modules, and this
file does not import paddle.

Ledger ($FM_TEST_LEDGER): the controller appends `outcome<TAB>nodeid` once a
test's teardown is reported, one flushed line per test, so a kill loses at most
the tests that were running. A test that took its worker down (CUDA fault) is
recorded as crashed. The ledger is only read back when asked for:
$FM_TEST_RESUME=1 deselects every node id in it, $FM_TEST_RERUN_FAILED=1 runs
only the failed / error / crashed ones.
"""
import os

_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _worker and os.environ.get("GPUS"):
    _gpus = os.environ["GPUS"].split()
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus[int(_worker[2:]) % len(_gpus)]

_LEDGER = os.environ.get("FM_TEST_LEDGER")
_RESUME = os.environ.get("FM_TEST_RESUME") == "1"
_RERUN_FAILED = os.environ.get("FM_TEST_RERUN_FAILED") == "1"
_BAD = ("failed", "error", "crashed")


def _read_ledger():
    """nodeid -> last recorded outcome."""
    done = {}
    if _LEDGER and os.path.exists(_LEDGER):
        with open(_LEDGER) as f:
            for line in f:
                outcome, sep, nodeid = line.rstrip("\n").partition("\t")
                if sep:
                    done[nodeid] = outcome
    return done


def _is_controller(config):
    return not hasattr(config, "workerinput")


def pytest_collection_modifyitems(config, items):
    # Runs on every worker (and in a plain run). The controller only appends to the
    # ledger after all workers have collected, so they all read the same file.
    if not _LEDGER or not (_RESUME or _RERUN_FAILED):
        return
    done = _read_ledger()
    if _RERUN_FAILED:
        bad = {k for k, v in done.items() if v in _BAD}
        skip_it = lambda nodeid: nodeid not in bad
    else:
        skip_it = lambda nodeid: nodeid in done
    keep, skip = [], []
    for item in items:
        (skip if skip_it(item.nodeid) else keep).append(item)
    if skip:
        config.hook.pytest_deselected(items=skip)
        items[:] = keep


_outcomes = {}
_ledger_file = None


def pytest_runtest_logreport(report):
    global _ledger_file
    if not _LEDGER:
        return
    # In an xdist run the controller sees every worker's reports; workers write nothing.
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    prev = _outcomes.get(report.nodeid, "passed")
    if report.failed:
        cur = "failed" if report.when == "call" else "error"
    elif report.skipped and prev == "passed":
        cur = "skipped"
    else:
        cur = prev
    _outcomes[report.nodeid] = cur
    # A crashed worker sends no teardown; xdist reports the test with when="???".
    if report.when not in ("setup", "call", "teardown") and report.failed:
        cur = "crashed"
    elif report.when != "teardown":
        return
    if _ledger_file is None:
        _ledger_file = open(_LEDGER, "a", buffering=1)
    _outcomes.pop(report.nodeid, None)
    _ledger_file.write(f"{cur}\t{report.nodeid}\n")
    _ledger_file.flush()
    os.fsync(_ledger_file.fileno())


def pytest_terminal_summary(terminalreporter, config):
    if not _LEDGER or not _is_controller(config):
        return
    counts = {}
    done = _read_ledger()
    for outcome in done.values():
        counts[outcome] = counts.get(outcome, 0) + 1
    tr = terminalreporter
    tr.write_sep("=", f"ledger {_LEDGER} (all runs)")
    tr.write_line("  ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "empty")
    bad = [n for n, o in done.items() if o in _BAD]
    for nodeid in bad[:50]:
        tr.write_line(f"  {done[nodeid].upper()} {nodeid}")
    if len(bad) > 50:
        tr.write_line(f"  ... {len(bad) - 50} more, see {_LEDGER}")
