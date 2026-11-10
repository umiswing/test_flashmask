"""Shared, framework-free pieces of the competitor benchmarks.

benchmark_flashmask.py (paddle), benchmark_fa4_mask_mod.py and
benchmark_cudnn_flex_mask_mod.py (torch) run in three different python
environments, so everything that has to be identical across them lives here and
depends on numpy only:

- the cases: kernel_test_seq_info.txt, D -> (DV, H), B, the mask list and the
  --dedup_static_masks / --first_sample_only selection (iter_cases);
- the masks: one flashmask startend layout per (case, mask) plus the parameters
  the mask_mods are built from (build_mask). The random masks draw from a seed
  derived from (mask, S[, sample]), not from global RNG state, so the order in
  which a script visits masks, kv modes or samples does not change them;
- the sparsity (element_sparsity) and the iteration counts (bench_iters);
- a mask fingerprint (fingerprint) that check_bench_consistency.py compares
  across the three --check_masks outputs, and the numpy reference the outputs of
  each backend's timed fwd / bwd path are checked against (run_accuracy).
"""
import contextlib
import hashlib
import json
import os
import zlib
from typing import Dict, List, NamedTuple, Optional

import numpy as np

SEQ_INFO_PATH = "kernel_test_seq_info.txt"

# The order is the CSV row order of every backend.
MASKS = (
    "Full",
    "Causal",
    "Sliding Window",
    "Global Sliding Window",
    "Causal Document Mask",
    "Document Mask",
    "Share Question Mask",
    "Causal Blockwise Mask",
    "Prefix LM Document Mask",
    "Prefix LM Causal Mask",
    "QK-sparse Mask",
    "Random Eviction Mask",
)

# Masks that depend on S alone: identical on every sample of one "Total length" block,
# so --dedup_static_masks measures them on the first sample only.
S_ONLY_MASKS = ("Full", "Causal", "Sliding Window", "Global Sliding Window",
                "Prefix LM Causal Mask", "Random Eviction Mask")

# Total tokens per case: B = TOTAL_TOKENS // S.
TOTAL_TOKENS = 128 * 1024


class Sample(NamedTuple):
    S: int
    idx: int            # position in kernel_test_seq_info.txt, used in CSV names
    first_of_S: bool    # Sample 1 of its "Total length" block
    prefix_doc_seq_lens: tuple   # ((prefix_len, doc_len), ...)
    qksparse_mask: tuple


class Case(NamedTuple):
    sample: Sample
    B: int
    H: int
    HKV: int
    D: int
    DV: int
    masks: tuple        # masks to run, in MASKS order


def load_seq_info(path: str = SEQ_INFO_PATH) -> List[Sample]:
    samples, total, raw = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "Total length" in line:
                total = int(line.split(":")[1].split(",")[0].strip())
                continue
            body, _, flags = line.split(":", 1)[1].partition("#")
            docs = tuple(tuple(int(v) for v in d) for d in eval(body.strip()))
            qk = tuple(int(v) for v in eval(flags.strip()))
            doc_total = sum(l for _, l in docs)
            # The mask generators pad a short layout, each in its own way, so a layout
            # that does not cover S silently measures a different mask.
            assert doc_total == total, (
                f"{path}: documents sum to {doc_total}, not Total length {total}: {line[:60]}"
            )
            raw.append((total, docs, qk))
    seen = set()
    for i, (S, docs, qk) in enumerate(raw):
        samples.append(Sample(S, i, S not in seen, docs, qk))
        seen.add(S)
    return samples


def head_config(D: int):
    """(DV, H) for a head_dim; HKV = H unless a script overrides it."""
    if D == 192:
        return 128, 16
    if D == 576:
        return 512, 8
    return D, 4096 // D


def select_masks(sample: Sample, examples=("all",), dedup_static_masks=False) -> tuple:
    names = MASKS if "all" in examples else tuple(m for m in MASKS if m in examples)
    unknown = [e for e in examples if e != "all" and e not in MASKS]
    if unknown:
        raise ValueError(f"unknown mask(s) {unknown}; choose from {list(MASKS)} or 'all'")
    if dedup_static_masks and not sample.first_of_S:
        names = tuple(m for m in names if m not in S_ONLY_MASKS)
    return names


def iter_cases(head_dims, examples=("all",), dedup_static_masks=False,
               first_sample_only=False, seqlens=None, path: str = SEQ_INFO_PATH):
    """Cases in run order: head_dim outer, samples inner. Skips samples left without masks.

    seqlens: only these "Total length" blocks (None: all).
    """
    samples = load_seq_info(path)
    if seqlens:
        unknown = set(seqlens) - {s.S for s in samples}
        if unknown:
            raise ValueError(f"no 'Total length' block for {sorted(unknown)} in {path}")
    for D in head_dims:
        DV, H = head_config(D)
        for sample in samples:
            if first_sample_only and not sample.first_of_S:
                continue
            if seqlens and sample.S not in seqlens:
                continue
            masks = select_masks(sample, examples, dedup_static_masks)
            if not masks:
                continue
            yield Case(sample, TOTAL_TOKENS // sample.S, H, H, D, DV, masks)


# ---------------------------------------------------------------------------
# Result layout: <results_dir>/<run>/<series>/{B}_{S}_{H}_{HKV}_{D}_{DV}_{idx}.csv
# plus <series>/meta.json. One directory per measured thing, so plot_radar picks
# series by path and results of different runs can be compared directly.
# ---------------------------------------------------------------------------

RESULTS_DIR = "results"
CSV_HEADERS = ["Operation", "FW Time (ms)", "BW Time (ms)", "TOTAL Time (ms)", "FW FLOPs",
               "BW FLOPs", "TOTAL FLOPs", "FW TFLOPs/s", "BW TFLOPs/s", "TOTAL TFLOPs/s", "Sparsity"]


def series_dir(results_dir, run, series):
    return os.path.join(results_dir, run, series)


def csv_path(results_dir, run, series, case: Case):
    s = case.sample
    return os.path.join(series_dir(results_dir, run, series),
                        f"{case.B}_{s.S}_{case.H}_{case.HKV}_{case.D}_{case.DV}_{s.idx}.csv")


def package_info(module_name):
    """Version, location and git commit (if it sits in a checkout) of an imported package."""
    import importlib
    import subprocess
    info = {"name": module_name}
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:  # noqa: BLE001 - recorded, not fatal
        info["error"] = repr(e)
        return info
    info["version"] = getattr(mod, "__version__", None)
    path = os.path.dirname(os.path.abspath(getattr(mod, "__file__", "") or "."))
    info["path"] = path
    try:
        info["git_commit"] = subprocess.run(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        info["git_commit"] = None
    return info


def write_meta(results_dir, run, series, backend, packages=(), **extra):
    """Write <series>/meta.json once per series (the first job of a run creates it).
    Only series-wide info goes here; per-job args live in jobs.tsv and the aq log."""
    import datetime
    import platform
    import sys
    d = series_dir(results_dir, run, series)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "meta.json")
    if os.path.exists(path):
        return path
    meta = {
        "series": series, "backend": backend, "run": run,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(), "python": sys.executable,
        "packages": [package_info(p) for p in packages],
    }
    meta.update(extra)
    tmp = f"{path}.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, path)
    return path


def write_csv(path, rows):
    """tsv like the old CSVs (tabulate), written atomically so a killed job leaves no half file."""
    from tabulate import tabulate
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(tabulate(rows, headers=CSV_HEADERS, tablefmt="tsv"))
    os.replace(tmp, path)


def run_masks_to_csv(out, masks, run_mask, overwrite=True, own_mem=None, sampler=None, **interfere):
    """Time masks into the CSV out; run_mask(mask) -> CSV row or None (not run).

    --no-overwrite skips an existing CSV. Each mask goes through measure_clean, so a
    measurement another GPU process disturbed is never written; interfere is passed on
    (grace, max_wait, allow_handoff). The CSV is written once the whole case is done.
    Returns the rows, or None if skipped.
    """
    if not overwrite and os.path.exists(out):
        print(f"{out} exists, skipping (--no-overwrite).")
        return None
    rows = []
    for ex in masks:
        row = measure_clean(lambda: run_mask(ex), own_mem, label=f"{os.path.basename(out)} {ex}",
                            sampler=sampler, **interfere)
        if row is not None:
            rows.append(row)
    write_csv(out, rows)
    return rows


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

class MaskSpec(NamedTuple):
    name: str
    causal: bool
    startend: Optional[np.ndarray]   # flashmask layout [B, H_mask, S, n_bounds] int32, None = no mask
    params: dict                     # what the mask_mods are built from


def _seed(mask: str, sample: Sample) -> int:
    # S-only masks share one draw per S, so --dedup_static_masks measures the same mask.
    key = f"{mask}|{sample.S}" if mask in S_ONLY_MASKS else f"{mask}|{sample.S}|{sample.idx}"
    return zlib.crc32(key.encode())


def _col(x, B):
    """[S] per-key bound -> [B, 1, S, 1] int32."""
    x = np.asarray(x, dtype=np.int32).reshape(1, 1, -1, 1)
    return np.ascontiguousarray(np.repeat(x, B, axis=0))


def _stack(cols, B):
    return np.concatenate([_col(c, B) for c in cols], axis=-1)


def split_sequence(sequence_length: int, rs: np.random.RandomState) -> List[int]:
    """One question plus 2-6 answers, every part at least one token long."""
    if sequence_length < 3:
        raise ValueError("a document needs at least 3 tokens: one question and two answers")
    num_answers = int(rs.randint(2, 7))
    parts = num_answers + 1
    extra = rs.multinomial(sequence_length - parts, [1.0 / parts] * parts)
    return [int(1 + e) for e in extra]


def maskout_pairs(sample: Sample) -> List[tuple]:
    doc_seq_lens = [l for _, l in sample.prefix_doc_seq_lens]
    if sum(sample.qksparse_mask) == 0:
        return [(1024, 538), (2358, 1700)]
    pairs, offset = [], 0
    for is_maskout, doc in zip(sample.qksparse_mask, doc_seq_lens):
        if is_maskout:
            pairs.append((offset, doc))
        offset += doc
    return pairs


def eviction_rows(B: int, H: int, S: int, start_row: int, rs: np.random.RandomState) -> np.ndarray:
    """[B, H, S] int32: key j is visible to query rows < rows[j] (S + 1 = never evicted)."""
    out = np.empty((B, H, S), dtype=np.int32)
    index = np.arange(start_row, S)
    for b in range(B):
        for h in range(H):
            rows = np.full(S, S + 1, dtype=np.int32)
            pos = rs.choice(S - 1, S - start_row, replace=False)
            pos = np.concatenate([pos[pos < index - 1], pos[pos >= index - 1]])
            rows[pos] = index
            out[b, h] = rows
    return out


def prefix_lm_document_startend(prefix_doc_seq_lens, S, B) -> np.ndarray:
    """[B, 1, S, 2] (LTS, UTE): documents, each with a bidirectional prefix."""
    docs = [l for _, l in prefix_doc_seq_lens]
    assert sum(docs) == S
    ends = np.cumsum(docs)
    starts = ends - np.asarray(docs)
    ute = np.concatenate([
        np.concatenate([np.full(p, st), np.arange(st + p, st + l)])
        for (p, l), st in zip(prefix_doc_seq_lens, starts)
    ])
    return _stack([np.repeat(ends, docs), ute], B)


def build_mask(name: str, case: Case) -> MaskSpec:
    """The flashmask layout of one mask, the single definition every backend runs."""
    sample, B, S = case.sample, case.B, case.sample.S
    docs = [l for _, l in sample.prefix_doc_seq_lens]
    ends = np.cumsum(docs)
    starts = ends - np.asarray(docs)
    doc_end = np.repeat(ends, docs)        # per key: end of its document
    doc_start = np.repeat(starts, docs)    # per key: start of its document
    window = int(S * 0.0625)

    if name == "Full":
        return MaskSpec(name, False, None, {})
    if name == "Causal":
        return MaskSpec(name, True, None, {})
    if name == "Sliding Window":
        lts = np.minimum(np.arange(window, S + window), S)
        return MaskSpec(name, True, _stack([lts], B), {"window_size": window})
    if name == "Global Sliding Window":
        g, l, r = 16, window, window
        lts = np.minimum(np.arange(l + 1, S + l + 1), S)
        lts[:g] = 0
        lte = np.full(S, S)
        lte[:g] = 0
        uts = np.full(S, g)
        uts[:g + r + 1] = 0
        ute = np.arange(-r, S - r)
        ute[:g + r + 1] = 0
        return MaskSpec(name, False, _stack([lts, lte, uts, ute], B),
                        {"global_token": g, "window_size": (l, r)})
    if name == "Causal Document Mask":
        return MaskSpec(name, True, _stack([doc_end], B), {"doc_seq_lens": docs})
    if name == "Document Mask":
        return MaskSpec(name, False, _stack([doc_end, doc_start], B), {"doc_seq_lens": docs})
    if name == "Share Question Mask":
        rs = np.random.RandomState(_seed(name, sample))
        share_docs = [split_sequence(d, rs) for d in docs]
        lts, off = [], 0
        for parts in share_docs:
            doc_len = sum(parts)
            for k, part in enumerate(parts):
                # Question tokens are visible to the whole document, answer tokens to their answer.
                lts.extend([off + (doc_len if k == 0 else part)] * part)
                off += part
        return MaskSpec(name, True, _stack([lts], B), {"doc_seq_lens": share_docs})
    if name == "Causal Blockwise Mask":
        assert len(docs) >= 3
        lte = np.concatenate([np.full(ends[-2], ends[-2]), np.full(docs[-1], ends[-1])])
        return MaskSpec(name, True, _stack([doc_end, lte], B), {"doc_seq_lens": docs})
    if name == "Prefix LM Document Mask":
        return MaskSpec(name, False, prefix_lm_document_startend(sample.prefix_doc_seq_lens, S, B),
                        {"doc_seq_lens": [tuple(d) for d in sample.prefix_doc_seq_lens]})
    if name == "Prefix LM Causal Mask":
        p = int(S * 0.5)
        ute = np.concatenate([np.zeros(p), np.arange(p, S)])
        return MaskSpec(name, False, _stack([np.full(S, S), ute], B), {"prefix_length": p})
    if name == "QK-sparse Mask":
        pairs = maskout_pairs(sample)
        lts, lte, last = [], [], 0
        for offset, n in pairs:
            assert offset >= last
            lts.extend(range(last, offset))
            lte.extend(range(last, offset))
            lts.extend(range(offset, offset + n))
            lte.extend([offset + n] * n)
            last = offset + n
        lts.extend(range(last, S))
        lte.extend(range(last, S))
        return MaskSpec(name, True, _stack([lts, lte], B), {"maskout_pair": pairs})
    if name == "Random Eviction Mask":
        rs = np.random.RandomState(_seed(name, sample))
        rows = eviction_rows(B, case.HKV, S, S // 2, rs)
        return MaskSpec(name, True, rows[..., None], {"start_row": S // 2, "rows": rows})
    raise ValueError(f"unknown mask {name!r}")


# ---------------------------------------------------------------------------
# Sparsity and fingerprint
# ---------------------------------------------------------------------------

def _masked_intervals(causal, startend, S, SKV):
    """Per key column, the three masked row ranges [s, e), each [B, H_mask, SKV]."""
    fm = np.asarray(startend).astype(np.int64)
    assert fm.shape[2] == SKV, f"startend_row_indices seqlen {fm.shape[2]} != SKV {SKV}"
    shape, nb = fm.shape[:3], fm.shape[-1]
    zero, full = np.zeros(shape, np.int64), np.full(shape, S, np.int64)
    # Empty intervals for whatever this layout does not carry.
    LTS, LTE, UTS, UTE = full, full, zero, zero
    if nb == 4:
        LTS, LTE, UTS, UTE = fm[..., 0], fm[..., 1], fm[..., 2], fm[..., 3]
    elif nb == 2 and causal:
        LTS, LTE = fm[..., 0], fm[..., 1]
    elif nb == 2:
        LTS, UTE = fm[..., 0], fm[..., 1]
    else:
        assert nb == 1, f"unsupported n_bounds={nb}"
        LTS = fm[..., 0]
    col = np.arange(SKV, dtype=np.int64)
    causal_end = np.broadcast_to(col - (SKV - S) if causal else np.zeros(SKV, np.int64), shape)
    ivs = [(zero, causal_end), (LTS, LTE), (UTS, UTE)]
    return [(np.clip(s, 0, S), np.clip(e, 0, S)) for s, e in ivs]


def _length(iv):
    return np.maximum(iv[1] - iv[0], 0)


def _intersect(a, b):
    return np.maximum(a[0], b[0]), np.minimum(a[1], b[1])


def _union_terms(A, B, C):
    """Inclusion-exclusion of three intervals as (sign, interval) pairs."""
    AB, AC, BC = _intersect(A, B), _intersect(A, C), _intersect(B, C)
    return [(1, A), (1, B), (1, C), (-1, AB), (-1, AC), (-1, BC), (1, _intersect(AB, C))]


def element_sparsity(causal, startend, S, SKV) -> float:
    """Fraction of masked (q, k) elements, independent of any kernel's tile sizes.

    FLOPs = density * dense FLOPs, density being the exact fraction of visible elements.
    For key column j the masked rows are the union of [0, j - (SKV - S)) (causal, bottom-right
    aligned), [LTS_j, LTE_j) and [UTS_j, UTE_j). q heads broadcast uniformly over mask heads,
    so the ratio over the mask's [B, H_mask] equals the ratio over [B, H].
    """
    if startend is None:
        if not causal:
            return 0.0
        visible = np.clip(np.arange(S, dtype=np.int64) + SKV - S + 1, 0, SKV).sum()
        return float(1.0 - visible / (S * SKV))
    terms = _union_terms(*_masked_intervals(causal, startend, S, SKV))
    masked = sum(sign * _length(iv) for sign, iv in terms)
    return float(masked.sum() / (masked.size * S))


def visible_counts(causal, startend, B, S, SKV):
    """(col_visible, row_visible), each [B, H_mask, S]: visible queries per key, keys per query."""
    if startend is None:
        startend = np.full((B, 1, SKV, 1), S)  # LTS = S: no masked rows besides causal
    ivs = _masked_intervals(causal, startend, S, SKV)
    terms = _union_terms(*ivs)
    col_masked = sum(sign * _length(iv) for sign, iv in terms)
    shape = col_masked.shape
    # Row counts: every masked interval of every column adds +1 on [s, e) of its rows.
    diff = np.zeros(shape[:2] + (S + 1,), np.int64)
    bi, hi = np.indices(shape[:2])
    bi = np.broadcast_to(bi[..., None], shape)
    hi = np.broadcast_to(hi[..., None], shape)
    for sign, (s, e) in terms:
        keep = e > s
        np.add.at(diff, (bi[keep], hi[keep], s[keep]), sign)
        np.add.at(diff, (bi[keep], hi[keep], e[keep]), -sign)
    row_masked = np.cumsum(diff, axis=-1)[..., :S]
    return S - col_masked, SKV - row_masked


def fingerprint(col_visible, row_visible) -> dict:
    """Hash of the per-key and per-query visible counts, [B, H_mask, S] each.

    Batches and heads that are all identical are collapsed first, so a mask broadcast over
    B or H hashes the same as its single copy.
    """
    col = np.asarray(col_visible, dtype=np.int64)
    row = np.asarray(row_visible, dtype=np.int64)
    for axis in (1, 0):
        if col.shape[axis] > 1 and (col == col.take([0], axis)).all() and (row == row.take([0], axis)).all():
            col, row = col.take([0], axis), row.take([0], axis)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(col).tobytes())
    h.update(np.ascontiguousarray(row).tobytes())
    return {"shape": list(col.shape), "visible": int(col.sum()), "sha256": h.hexdigest()}


# ---------------------------------------------------------------------------
# Output check: every backend runs its timed fwd / bwd path once on the same inputs
# and is compared with one numpy reference built from the flashmask layout, so a
# kernel that skips blocks, drops the mask or leaves outputs unwritten fails even
# when its mask_mod alone looks right. B and H are shrunk (S, D and the mask are
# not) so the dense reference fits; it runs on the CPU, hence ACC_MAX_S.
# ---------------------------------------------------------------------------

ACC_B, ACC_H = 2, 2
ACC_MAX_S = 8192
ACC_TENSORS = ("out", "dq", "dk", "dv")


def acc_case(case: Case) -> Case:
    return case._replace(B=ACC_B, H=ACC_H, HKV=ACC_H)


def acc_enabled(case: Case) -> bool:
    return case.sample.S <= ACC_MAX_S


def _bf16(x):
    """Round float32 to the nearest bf16 value (kept as float32)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    u = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


def acc_inputs(case: Case, mask: str):
    """q, k, v, dout in [B, S, H, D] layout, bf16 values as float32, seeded per (mask, S, sample, D)."""
    S = case.sample.S
    rs = np.random.RandomState(zlib.crc32(f"acc|{mask}|{S}|{case.sample.idx}|{case.D}".encode()))
    shapes = [(case.B, S, case.H, case.D), (case.B, S, case.HKV, case.D),
              (case.B, S, case.HKV, case.DV), (case.B, S, case.H, case.DV)]
    return [_bf16(rs.standard_normal(s).astype(np.float32)) for s in shapes]


def dense_visible(causal, startend, B, S):
    """bool [B, H_mask, S_q, S_k] of the flashmask layout."""
    if startend is None:
        startend = np.full((B, 1, S, 1), S)
    rows = np.arange(S).reshape(S, 1)
    masked = None
    for s, e in _masked_intervals(causal, startend, S, S):
        m = (s[..., None, :] <= rows) & (rows < e[..., None, :])
        masked = m if masked is None else masked | m
    return ~masked


def attention_ref(q, k, v, dout, visible, bf16=False):
    """out, dq, dk, dv of masked softmax attention, float32; bf16=True rounds every matmul
    result to bf16, the naive-bf16 error the kernels are allowed a multiple of.
    A query row with no visible key gets out = 0 and no gradient."""
    r = _bf16 if bf16 else (lambda x: x)
    B, S, H, D = q.shape
    HKV, DV, Hm = k.shape[2], v.shape[-1], visible.shape[1]
    scale = np.float32(1.0 / np.sqrt(D))
    out = np.zeros((B, S, H, DV), np.float32)
    dq = np.zeros_like(q)
    dk = np.zeros_like(k)
    dv = np.zeros_like(v)
    for b in range(B):
        for h in range(H):
            kh, mh = h // (H // HKV), h * Hm // H
            Q, K, V, dO, vis = q[b, :, h], k[b, :, kh], v[b, :, kh], dout[b, :, h], visible[b, mh]
            s = np.where(vis, r(Q @ K.T) * scale, -np.inf)
            m = s.max(-1, keepdims=True)
            p = np.exp(s - np.where(np.isfinite(m), m, 0))
            l = p.sum(-1, keepdims=True)
            p = r(p / np.where(l > 0, l, 1))
            o = r(p @ V)
            out[b, :, h] = o
            dv[b, :, kh] += r(p.T @ dO)
            ds = r(p * (r(dO @ V.T) - (dO * o).sum(-1, keepdims=True)))
            dq[b, :, h] = r(ds @ K) * scale
            dk[b, :, kh] += r(ds.T @ Q) * scale
    return {"out": out, "dq": dq, "dk": dk, "dv": dv}


def reference(case: Case, spec: MaskSpec, cache_dir=None):
    """(ref, ref_err, atol): the float32 reference outputs and, per tensor, the naive bf16
    error and the float32 rounding slack. ~1 s per mask per head at S=4K on the CPU, so
    with cache_dir the first backend saves it and the others load it."""
    path = None
    if cache_dir:
        s = case.sample
        name = spec.name.replace(" ", "_")
        path = os.path.join(cache_dir, f"{name}_{s.S}_{s.idx}_{case.B}_{case.H}_{case.HKV}_{case.D}_{case.DV}.npz")
        if os.path.exists(path):
            z = np.load(path)
            return ({n: z[n] for n in ACC_TENSORS},
                    {n: float(z[f"{n}_ref_err"]) for n in ACC_TENSORS},
                    {n: float(z[f"{n}_atol"]) for n in ACC_TENSORS})
    q, k, v, dout = acc_inputs(case, spec.name)
    vis = dense_visible(spec.causal, spec.startend, case.B, case.sample.S)
    ref = attention_ref(q, k, v, dout, vis)
    naive = attention_ref(q, k, v, dout, vis, bf16=True)
    ref_err = {n: float(np.abs(naive[n] - ref[n]).max()) for n in ACC_TENSORS}
    atol = {n: 2 * float(np.abs(ref[n] + np.float32(0.3) - np.float32(0.3) - ref[n]).max())
            for n in ACC_TENSORS}
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.npz"
        np.savez(tmp, **ref, **{f"{n}_ref_err": ref_err[n] for n in ACC_TENSORS},
                 **{f"{n}_atol": atol[n] for n in ACC_TENSORS})
        os.replace(tmp, path)
    return ref, ref_err, atol


def accuracy(case: Case, spec: MaskSpec, got, cache_dir=None) -> dict:
    """Compare one backend's outputs (dict of float32 arrays, ACC_TENSORS) with the reference.

    Same rule as test_flashmask.py: max |err| <= 2 * max |naive bf16 err| + atol.
    Any NaN / inf (the outputs are NaN-poisoned before the run) fails.
    """
    if cache_dir:
        # Backends checked in parallel (run_perf.sh check) share the cache: the first one
        # computes a reference, the others wait for it instead of computing it again.
        import fcntl
        os.makedirs(cache_dir, exist_ok=True)
        s = case.sample
        lock = os.path.join(cache_dir, f"{spec.name.replace(' ', '_')}_{s.S}_{s.idx}_{case.D}.lock")
        with open(lock, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            ref, ref_err, atol = reference(case, spec, cache_dir)
    else:
        ref, ref_err, atol = reference(case, spec, cache_dir)
    res, ok = {}, True
    for name in ACC_TENSORS:
        g = np.asarray(got[name], dtype=np.float32)
        if g.shape != ref[name].shape:
            res[name] = {"ok": False, "error": f"shape {list(g.shape)} != {list(ref[name].shape)}"}
            ok = False
            continue
        bad = int((~np.isfinite(g)).sum())
        tol = 2 * ref_err[name] + atol[name]
        err = None if bad else float(np.abs(g - ref[name]).max())
        t_ok = bad == 0 and err <= tol
        res[name] = {"ok": t_ok, "err": err, "tol": tol, "ref_err": ref_err[name], "nonfinite": bad}
        ok &= t_ok
    res["ok"] = ok
    return res


def run_accuracy(case: Case, mask: str, run_once, check_masks_path=None) -> Optional[dict]:
    """accuracy() of run_once(acc_case, acc_spec, inputs) -> outputs, None when S is too large.

    The reference is cached next to the --check_masks file (acc_ref/), so the backends of
    one check_bench_consistency.sh run share it. An exception in the backend is recorded
    as a failure instead of stopping the check.
    """
    if not acc_enabled(case):
        return None
    c = acc_case(case)
    spec = build_mask(mask, c)
    try:
        got = run_once(c, spec, acc_inputs(c, mask))
    except Exception as e:  # noqa: BLE001 - reported by check_bench_consistency.py
        return {"ok": False, "error": repr(e)[:500]}
    if got is None:   # the backend does not run this mask (e.g. cuDNN mask_func too large)
        return None
    cache_dir = (os.path.join(os.path.dirname(os.path.abspath(check_masks_path)), "acc_ref")
                 if check_masks_path else None)
    return accuracy(c, spec, got, cache_dir)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def bench_iters(S: int):
    """(warmup, timed) runs for one fwd or bwd measurement, the same for every backend."""
    if S <= 32 * 1024:
        return 10, 50
    if S <= 64 * 1024:
        return 5, 20
    return 2, 10


def do_bench(fn, S, synchronize, new_event):
    """Mean ms of fn over bench_iters(S) timed runs, after one untimed call and the warmup.

    synchronize / new_event are the framework's device sync and timing Event factory
    (paddle.device.synchronize / paddle.device.Event, torch.cuda.synchronize / torch.cuda.Event).
    """
    import gc
    n_warmup, n_repeat = bench_iters(S)
    fn()
    for _ in range(n_warmup):
        fn()
    synchronize()
    start = [new_event(enable_timing=True) for _ in range(n_repeat)]
    end = [new_event(enable_timing=True) for _ in range(n_repeat)]
    gc.collect()
    gc.disable()
    try:
        for i in range(n_repeat):
            start[i].record()
            fn()
            end[i].record()
    finally:
        gc.enable()
    synchronize()
    return float(np.mean([s.elapsed_time(e) for s, e in zip(start, end)]))


# ---------------------------------------------------------------------------
# Interference: another process on our GPU while a mask is timed. A disturbed
# measurement is thrown away, never written. We poll until the GPU is ours again and
# re-measure in-process (no recompile); if the other process stays longer than the
# grace period, the benchmark exits with HANDOFF_EXIT and run_perf.sh hands the rest
# of the task to a new aq task on another GPU (or the queue).
# ---------------------------------------------------------------------------

HANDOFF_EXIT = 75                      # EX_TEMPFAIL: "rerun me elsewhere"
INTERFERE_POLL = 2.0                   # seconds between checks while waiting
INTERFERE_MEM_BYTES = 1 << 30          # memory rule: this much beyond our own baseline
_mem_baseline = [None]                 # min of (GPU used - own reserved) seen in this process
_sampler_cache = []


def _visible_gpu():
    """First entry of CUDA_VISIBLE_DEVICES (set by aq): ('uuid', 'GPU-..') or ('index', n)."""
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    return ("uuid", dev) if not dev.isdigit() else ("index", int(dev))


def _nvml_sampler():
    import pynvml
    pynvml.nvmlInit()
    kind, dev = _visible_gpu()
    if kind == "uuid":
        h = next(h for h in (pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount()))
                 if dev in str(pynvml.nvmlDeviceGetUUID(h)))
    else:
        h = pynvml.nvmlDeviceGetHandleByIndex(dev)

    def sample():
        procs = [(p.pid, p.usedGpuMemory or 0) for p in pynvml.nvmlDeviceGetComputeProcesses(h)]
        return procs, pynvml.nvmlDeviceGetMemoryInfo(h).used
    return sample


def _smi_sampler():
    import subprocess
    kind, dev = _visible_gpu()
    sel = ["-i", str(dev)]

    def smi(query):
        return subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"] + sel,
                              capture_output=True, text=True, timeout=10, check=True).stdout
    smi("--query-gpu=memory.used")   # fails here, not in the thread, if unusable

    def sample():
        procs = []
        for line in smi("--query-compute-apps=pid,used_memory").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit():
                procs.append((int(parts[0]), int(float(parts[1]) * (1 << 20)) if parts[1].replace(".", "").isdigit() else 0))
        used = int(float(smi("--query-gpu=memory.used").split()[0]) * (1 << 20))
        return procs, used
    return sample


def gpu_sampler():
    """() -> ([(pid, bytes)] compute processes on our GPU, used bytes); None if neither
    pynvml nor nvidia-smi works. Read-only: it never picks or changes the GPU."""
    if not _sampler_cache:
        sampler = None
        for make in (_nvml_sampler, _smi_sampler):
            try:
                sampler = make()
                sampler()
                break
            except Exception:  # noqa: BLE001 - try the next way
                sampler = None
        if sampler is None:
            print("[interfere] no pynvml / nvidia-smi: interference is not checked")
        _sampler_cache.append(sampler)
    return _sampler_cache[0]


class GpuGuard:
    """Samples our GPU in a thread while the block runs; .interfered / .sources after exit.

    Interference: a compute process other than this one. If the pid namespace differs
    from NVML's (our pid not in the list), more than one process counts. If no process
    is listed at all (common in containers), GPU memory beyond our own (own_mem():
    allocator reserved bytes) and beyond this process's smallest such overhead by more
    than INTERFERE_MEM_BYTES counts.
    """

    def __init__(self, own_mem=None, interval=0.2, sampler=None):
        self.own_mem = own_mem
        self.interval = interval
        self.sampler = sampler if sampler is not None else gpu_sampler()
        self.sources = []

    def check(self):
        """Why the GPU is not ours right now, or None."""
        procs, used = self.sampler()
        me = os.getpid()
        pids = [p for p, _ in procs]
        if pids:
            others = [(p, m) for p, m in procs if p != me] if me in pids else (procs if len(procs) > 1 else [])
            if others:
                return f"processes {[(p, round(m / 2**30, 2)) for p, m in others]} (pid, GiB)"
            return None
        excess = used - (self.own_mem() if self.own_mem else 0)
        base = _mem_baseline[0]
        if base is None or excess < base:
            _mem_baseline[0] = base = excess
        if excess - base > INTERFERE_MEM_BYTES:
            return f"{(excess - base) / 2**30:.2f} GiB GPU memory not ours"
        return None

    def _loop(self):
        import time
        while not self._stop.is_set():
            try:
                why = self.check()
            except Exception as e:  # noqa: BLE001 - sampling must not break timing
                why, self.sampler = None, None
                print(f"[interfere] sampling failed, stop checking: {e!r}")
            if why:
                self.sources.append({"t": round(time.time(), 1), "what": why})
            if self.sampler is None:
                return
            self._stop.wait(self.interval)

    def __enter__(self):
        import threading
        self._stop = threading.Event()
        self._thread = None
        if self.sampler is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
        return False

    @property
    def interfered(self):
        return bool(self.sources)


class Handoff(Exception):
    """Another process keeps the GPU: leave it to a new aq task (see run_main)."""


class InterferenceTimeout(Exception):
    """Another process kept the GPU longer than max_wait and handing off is not allowed."""


def measure_clean(fn, own_mem=None, grace=60.0, max_wait=1800.0, allow_handoff=False, label="",
                  sampler=None, sleep=None, clock=None):
    """fn() from a run no other GPU process disturbed.

    Before each run, wait until the GPU is ours; a disturbed run is discarded and
    re-run. Once interference has lasted grace seconds: raise Handoff if allowed,
    else keep waiting up to max_wait and then raise InterferenceTimeout.
    """
    import time
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    guard = GpuGuard(own_mem, sampler=sampler)
    if guard.sampler is None:
        return fn()
    first_seen = None

    def give_up_if_due(why):
        waited = clock() - first_seen
        if allow_handoff and waited >= grace:
            raise Handoff(f"{label}: {why} for {waited:.0f}s")
        if waited >= max_wait:
            raise InterferenceTimeout(f"{label}: {why} for {waited:.0f}s")

    while True:
        try:
            why = guard.check()
        except Exception as e:  # noqa: BLE001 - sampling must not stop the benchmark
            print(f"[interfere] sampling failed, measuring unchecked: {e!r}")
            return fn()
        if why:
            if first_seen is None:
                first_seen = clock()
                print(f"[interfere] {label}: waiting, {why}")
            give_up_if_due(why)
            sleep(INTERFERE_POLL)
            continue
        with GpuGuard(own_mem, sampler=guard.sampler) as g:
            result = fn()
        if not g.interfered:
            if first_seen is not None:
                print(f"[interfere] {label}: clean after {clock() - first_seen:.0f}s")
            return result
        why = g.sources[0]["what"]
        print(f"[interfere] {label}: run disturbed by {why}, discarded")
        if first_seen is None:
            first_seen = clock()
        give_up_if_due(why)


def run_main(main, **kwargs):
    """main(**kwargs); exit HANDOFF_EXIT on Handoff so run_perf.sh moves the task."""
    import sys
    try:
        main(**kwargs)
    except Handoff as e:
        print(f"[interfere] handing off: {e}")
        sys.exit(HANDOFF_EXIT)


# ---------------------------------------------------------------------------
# --check_masks records
# ---------------------------------------------------------------------------

def check_record(backend, case: Case, spec: MaskSpec, fp: dict, kv_mode="split", acc=None) -> dict:
    S = case.sample.S
    return {
        "backend": backend,
        "S": S, "idx": case.sample.idx, "D": case.D, "mask": spec.name, "kv_mode": kv_mode,
        "B": case.B, "H": case.H, "HKV": case.HKV, "DV": case.DV, "causal": spec.causal,
        "sparsity": element_sparsity(spec.causal, spec.startend, S, S),
        "iters": list(bench_iters(S)),
        "fingerprint": fp,
        "acc": acc,   # run_accuracy() of the shrunk case, None = not run
    }


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# torch side of --check_masks (FA4, cuDNN). torch is imported lazily so that the
# paddle environment can import this module.
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def torch_poison_empty():
    """Hand out torch.empty / empty_like float buffers as NaN, so output a kernel never
    writes shows up as NaN (fresh memory is often zero, which matches masked rows)."""
    import torch
    orig = torch.empty, torch.empty_like

    def poisoned(f):
        def g(*args, **kwargs):
            t = f(*args, **kwargs)
            if t.is_floating_point():
                t.fill_(float("nan"))
            return t
        return g

    torch.empty, torch.empty_like = poisoned(orig[0]), poisoned(orig[1])
    try:
        yield
    finally:
        torch.empty, torch.empty_like = orig


def torch_outputs(out, dq, dk, dv):
    """ACC_TENSORS dict of float32 numpy arrays from torch tensors."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {n: t.detach().float().cpu().numpy() for n, t in zip(ACC_TENSORS, (out, dq, dk, dv))}


def torch_mask_mod_counts(flex_mask_mod, causal, B, H_mask, S, device="cuda", row_chunk=None):
    """(col_visible, row_visible) [B, H_mask, S] of a flex mask_mod, evaluated in row chunks."""
    import torch
    if flex_mask_mod is None:
        return visible_counts(causal, None, B, S, S)
    kv_idx = torch.arange(S, device=device).view(1, 1, S)
    h_idx = torch.arange(H_mask, device=device).view(H_mask, 1, 1)
    row_chunk = row_chunk or max(1, (1 << 28) // (H_mask * S))
    col = torch.zeros((B, H_mask, S), dtype=torch.int64, device=device)
    row = torch.zeros((B, H_mask, S), dtype=torch.int64, device=device)
    for b in range(B):
        b_idx = torch.tensor(b, device=device)
        for r0 in range(0, S, row_chunk):
            r1 = min(r0 + row_chunk, S)
            q_idx = torch.arange(r0, r1, device=device).view(1, -1, 1)
            vis = torch.broadcast_to(flex_mask_mod(b_idx, h_idx, q_idx, kv_idx), (H_mask, r1 - r0, S))
            col[b] += vis.sum(1)
            row[b, :, r0:r1] = vis.sum(-1)
    return col.cpu().numpy(), row.cpu().numpy()


def torch_mask_func_counts(mask_func, B, S):
    """(col_visible, row_visible) [B, H_mask, S] of a cuDNN Flex mask_func [H_mask, nfunc, B*S].

    Row r = b * S + q sees keys [0, F0) U [F1, F2) U [F3, F4) U ...
    """
    import torch
    mf = mask_func.to(torch.int64).clamp(0, S)
    H_mask, nfunc, _ = mf.shape
    starts = [torch.zeros_like(mf[:, 0])] + [mf[:, i] for i in range(1, nfunc - 1, 2)]
    ends = [mf[:, 0]] + [mf[:, i + 1] for i in range(1, nfunc - 1, 2)]
    row = torch.zeros((H_mask, B * S), dtype=torch.int64, device=mf.device)
    diff = torch.zeros((H_mask, B, S + 1), dtype=torch.int64, device=mf.device)
    for s, e in zip(starts, ends):
        e = torch.maximum(e, s)
        row += e - s
        live = (e > s).to(torch.int64).view(H_mask, B, S)
        diff.scatter_add_(-1, s.view(H_mask, B, S), live)
        diff.scatter_add_(-1, e.view(H_mask, B, S), -live)
    col = diff.cumsum(-1)[..., :S].permute(1, 0, 2)
    row = row.view(H_mask, B, S).permute(1, 0, 2)
    return col.cpu().numpy(), row.cpu().numpy()
