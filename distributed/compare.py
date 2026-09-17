"""Accuracy comparison for the FM-4 overlap CP test.

The overlap kernel reduces dK/dV in a different order than the all-gather reference,
so bf16 elements legitimately differ by a few ULPs (double-rounding of two fp32
reduction orders). Longer reductions (large head dim, kv_shared) produce more such
2-ULP hits, so a fixed relative band mis-scales across configs. The check therefore
measures each element's error in **bf16 ULPs of the reference** and:

  * declares an element mismatched only when
    ``|actual - desire| > atol + ulp_k * ulp(ref)``  where ``ulp(ref) =
    2^(floor(log2|ref|) - 7)`` (bf16 has 7 mantissa bits); the ``atol`` term guards
    near-zero refs (a real bug that writes gradient where ref~=0 is caught there), and
  * fails only when the *fraction* of mismatched elements exceeds ``ratio_limit``.

Reduction noise is <= ~1 ULP for d<=256 but grows to ~4 ULP for d=512 (kv_shared
merges dK+dV, ~4x the reduction length). So the default band is tight (``ulp_k=2``);
only the d=512 run widens dk/dv to ``ulp_k=6`` (the test sets ``FM4_TOL_DK`` when
kv_shared). A real bug -- large multi-ULP diffs and/or spurious gradient on near-zero
refs -- is still counted at either band and trips the tight ``ratio_limit``. This
deliberately fixes a dev-harness bug whose tensor path printed mismatches but always
returned ``True``.

Every knob is overridable at runtime, e.g. ``FM4_TOL_DK=6,1e-2,2e-4``.
"""

import os

import numpy as np
import paddle

# name -> (ulp_k, atol, ratio_limit).  ulp_k = allowed error in bf16 ULPs of the ref.
# Tight by default; the test widens dk/dv to 6 ULP only for the d=512 (kv_shared) run.
TOLERANCES = {
    "out": (2, 2e-3, 1e-5),
    "dq": (2, 1e-2, 1e-5),
    "dk": (2, 1e-2, 2e-4),
    "dv": (2, 1e-2, 2e-4),
}


def tolerance_for(name):
    """Return ``(ulp_k, atol, ratio_limit)`` for ``name``, applying any env override.

    Override with ``FM4_TOL_<NAME>="ulp_k,atol,ratio"`` (name upper-cased), e.g.
    ``FM4_TOL_DK=6,1e-2,2e-4``.
    """
    ulp_k, atol, ratio = TOLERANCES[name]
    override = os.environ.get(f"FM4_TOL_{name.upper()}")
    if override:
        parts = [p.strip() for p in override.split(",")]
        assert len(parts) == 3, f"FM4_TOL_{name.upper()} must be 'ulp_k,atol,ratio', got {override!r}"
        ulp_k, atol, ratio = (float(p) for p in parts)
    return ulp_k, atol, ratio


def _bf16_ulp(ref_abs):
    """bf16 ULP at each |ref|: 2^(floor(log2|ref|) - 7). Clip tiny refs so log2 is
    finite; there the atol term dominates anyway."""
    safe = paddle.clip(ref_abs, min=2.0 ** -24)
    return paddle.pow(paddle.full_like(safe, 2.0), paddle.floor(paddle.log2(safe)) - 7)


def check_accuracy(actual, desire, name, rank=0):
    """Compare two tensors under this test's ULP-band + ratio policy.

    Returns True on pass; on failure prints a structural diagnostic.
    """
    ulp_k, atol, ratio_limit = tolerance_for(name)

    if list(actual.shape) != list(desire.shape):
        print(f"[FAIL] {name}: shape mismatch {actual.shape} vs {desire.shape}")
        return False

    a = actual.detach().cast("float32")
    d = desire.detach().cast("float32")
    abs_diff = paddle.abs(a - d)
    ref_abs = paddle.abs(d)
    band = atol + ulp_k * _bf16_ulp(ref_abs)
    mismatch = abs_diff > band

    total = int(np.prod(actual.shape)) or 1
    mismatch_count = int(paddle.sum(mismatch.cast("int64")).item())
    ratio = mismatch_count / total

    if mismatch_count == 0:
        return True

    max_abs = float(paddle.max(abs_diff).item())
    # relative error only over finite, non-zero references (near-zero refs give
    # meaninglessly huge relative errors and are covered by the atol band above)
    rel_mask = (ref_abs > 0) & paddle.isfinite(abs_diff)
    if bool(paddle.any(rel_mask).item()):
        rel = paddle.where(rel_mask, abs_diff / ref_abs, paddle.zeros_like(abs_diff))
        max_rel = float(paddle.max(rel).item())
    else:
        max_rel = float("nan")

    passed = ratio <= ratio_limit
    tag = "WARN" if passed else "FAIL"
    print(
        f"[r{rank}][{tag}] {name}: {mismatch_count}/{total} ({ratio:.2e}) exceed "
        f"atol+{ulp_k:g}*ulp(ref) (atol={atol}); limit={ratio_limit:.1e}. "
        f"max_abs={max_abs:.6g}, max_rel={max_rel:.6g}"
    )

    if not passed:
        _diagnose(actual, a, d, abs_diff, mismatch, atol, name, rank)

    return passed


def _diagnose(actual, a, d, abs_diff, mismatch, atol, name, rank):
    """Print the STRUCTURE of a mismatch so we can tell reduction noise apart
    from a real misalignment/spurious-gradient bug.

    Reduction-order noise is spread thinly and only where |ref| is non-trivial.
    A misalignment (grad landing on wrong seq positions) or spurious gradient
    concentrates on positions where the reference is ~0, and/or clusters in a
    few heads / sequence blocks. The breakdown below surfaces exactly that.
    """
    ref_abs = paddle.abs(d)
    mm = mismatch
    mm_cnt = paddle.sum(mm.cast("int64"))
    # share of mismatches sitting where the reference is ~0 (ref-near-zero):
    # high => overlap writes gradient where there should be ~none (misalignment)
    near_zero = mm & (ref_abs <= atol)
    nz_share = float((paddle.sum(near_zero.cast("float32")) / paddle.maximum(mm_cnt.cast("float32"), paddle.to_tensor(1.0))).item())
    print(f"       [diag {name}] mismatches with |ref|<=atol: {nz_share:.1%}")

    shape = list(actual.shape)
    if len(shape) == 4:  # (b, s, h, d)
        b, s, h, dd = shape
        per_head = paddle.sum(mm.cast("float32"), axis=[0, 1, 3]) / (b * s * dd)
        hv = per_head.numpy()
        print(f"       [diag {name}] per-head mismatch frac: min={hv.min():.2e} max={hv.max():.2e} "
              f"nonzero_heads={int((hv > 0).sum())}/{h}")
        seq_any = paddle.sum(mm.cast("float32"), axis=[0, 2, 3]) > 0
        print(f"       [diag {name}] seq positions with any mismatch: "
              f"{int(paddle.sum(seq_any.cast('int64')).item())}/{s}")

